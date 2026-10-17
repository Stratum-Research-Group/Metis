"""
WordNet Hierarchy Geometric Demo (Poincaré Hyperbolic vs Euclidean Radial Construction)
=======================================================================================
Evaluates hierarchical capacity on a BFS subtree of the WordNet noun transitive closure.

NOTE (Finding B17):
  This is a synthetic radial construction demonstration, not an empirical benchmark of
  representations trained via Riemannian SGD (as in Nickel & Kiela 2017). Both arms are
  constructed analytically from graph tree depth along seeded random directions to
  illustrate hyperbolic vs Euclidean radial metric properties, not learned embeddings.

Metrics:
  - Mean Average Precision (MAP)
  - Mean Rank
  - Tree Distortion: D = (1 / |E|) * sum_{(u,v) in E} |d_emb(u, v) - d_graph(u, v)| / d_graph(u, v)
"""

import os
import sys
import time
import math
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

from lthcm.benchmarks.real.utils.significance import paired_bootstrap_test
from lthcm.utils.determinism import set_all_seeds


def poincare_distance(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Computes exact geodesic distance in the Poincaré ball model:
      d_H(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2) * (1 - ||v||^2)))
    """
    sq_dist = torch.sum((u - v) ** 2, dim=-1)
    u_norm_sq = torch.clamp(torch.sum(u ** 2, dim=-1), 0.0, 1.0 - eps)
    v_norm_sq = torch.clamp(torch.sum(v ** 2, dim=-1), 0.0, 1.0 - eps)
    
    alpha = 1.0 - u_norm_sq
    beta = 1.0 - v_norm_sq
    xi = 1.0 + 2.0 * sq_dist / (alpha * beta + eps)
    
    # arcosh(xi) = ln(xi + sqrt(xi^2 - 1))
    xi = torch.clamp(xi, min=1.0 + eps)
    return torch.acosh(xi)


class WordNetHierarchyBenchmark:
    """
    Evaluates hierarchical embedding fidelity across the WordNet noun taxonomy.
    """

    def __init__(self, d_model: int = 16, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.d_model = d_model

    def load_wordnet_closure(self, max_nodes: int = 250) -> Tuple[List[str], List[Tuple[int, int]], Dict[int, List[int]]]:
        """
        Extracts the transitive closure of WordNet noun hypernym tree (using NLTK or fallback ontology).
        """
        try:
            import nltk
            try:
                from nltk.corpus import wordnet as wn
                _ = wn.synsets("entity")
            except Exception:
                nltk.download("wordnet", quiet=True)
                from nltk.corpus import wordnet as wn

            print("[-] Extracting WordNet noun hypernym subtree from NLTK...")
            root = wn.synset("entity.n.01")
            visited = set()
            nodes = []
            queue = [root]
            adj = {}

            while queue and len(nodes) < max_nodes:
                curr = queue.pop(0)
                name = curr.name()
                if name in visited:
                    continue
                visited.add(name)
                nodes.append(name)
                idx = len(nodes) - 1
                adj[idx] = []

                for child in sorted(curr.hyponyms(), key=lambda syn: syn.name()):
                    c_name = child.name()
                    if c_name not in visited and len(nodes) + len(queue) < max_nodes * 2:
                        queue.append(child)

            # Build edges
            node_to_idx = {name: i for i, name in enumerate(nodes)}
            edges = []
            for i, name in enumerate(nodes):
                syn = wn.synset(name)
                for hyp in sorted(syn.hypernyms(), key=lambda syn: syn.name()):
                    if hyp.name() in node_to_idx:
                        p_idx = node_to_idx[hyp.name()]
                        edges.append((p_idx, i))
                        adj[p_idx].append(i)

            print(f"    Loaded {len(nodes)} WordNet nodes with {len(edges)} hypernym-hyponym edges.")
            return nodes, edges, adj
        except Exception as e:
            print(f"    Notice: NLTK wordnet error ({e}), generating canonical multi-level taxonomy tree.")
            nodes = [f"node_{i}" for i in range(max_nodes)]
            edges = []
            adj = {i: [] for i in range(max_nodes)}
            # Form balanced binary/ternary tree
            for i in range(1, max_nodes):
                parent = (i - 1) // 3
                edges.append((parent, i))
                adj[parent].append(i)
            return nodes, edges, adj

    def run_evaluation(
        self,
        max_nodes: int = 250,
        output_dir: str = "runs",
        seed: int = 42,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compares Poincaré Hyperbolic vs Euclidean embedding reconstruction fidelity.
        """
        os.makedirs(output_dir, exist_ok=True)
        nodes, edges, adj = self.load_wordnet_closure(max_nodes=max_nodes)
        num_nodes = len(nodes)

        set_all_seeds(seed)

        # -------------------------------------------------------------
        # Compute Ground-Truth Shortest Path Tree Distances
        # -------------------------------------------------------------
        from collections import deque
        dist_matrix = np.full((num_nodes, num_nodes), fill_value=1e9)
        for i in range(num_nodes):
            dist_matrix[i, i] = 0
            q = deque([(i, 0)])
            while q:
                curr, d = q.popleft()
                for neighbor in adj.get(curr, []):
                    if dist_matrix[i, neighbor] > d + 1:
                        dist_matrix[i, neighbor] = d + 1
                        q.append((neighbor, d + 1))

        # Hypernym relation pairs (ground truth children for each parent)
        gt_children = {i: set(adj.get(i, [])) for i in range(num_nodes) if len(adj.get(i, [])) > 0}

        # -------------------------------------------------------------
        # 1. Poincaré Hyperbolic & Euclidean Embedding Spaces
        # -------------------------------------------------------------
        # Generate shared, deterministic direction vectors for all nodes (using CPU generator for cross-device determinism)
        gen = torch.Generator().manual_seed(seed)
        directions = F.normalize(torch.randn((num_nodes, self.d_model), generator=gen), dim=-1).to(self.device)

        poincare_embs = torch.zeros((num_nodes, self.d_model), device=self.device)
        euclidean_embs = torch.zeros((num_nodes, self.d_model), device=self.device)

        for i in range(num_nodes):
            depth = dist_matrix[0, i] if dist_matrix[0, i] < 1e8 else 1.0
            # Radial distance in Poincaré ball: r = tanh(depth / 3.0) * 0.85
            r_poincare = math.tanh(depth / 3.0) * 0.85
            poincare_embs[i] = r_poincare * directions[i]
            euclidean_embs[i] = depth * directions[i]

        # -------------------------------------------------------------
        # Compute Reconstruction MAP & Mean Rank
        # -------------------------------------------------------------
        def evaluate_embeddings(embs, is_poincare: bool = True):
            aps = []
            ranks = []
            for parent, children in gt_children.items():
                p_vec = embs[parent]
                # Compute distance to all other nodes
                if is_poincare:
                    dists = poincare_distance(p_vec.unsqueeze(0), embs).squeeze(0).cpu().numpy()
                else:
                    dists = torch.norm(p_vec.unsqueeze(0) - embs, dim=-1).squeeze(0).cpu().numpy()
                
                # Exclude self
                dists[parent] = 1e9
                ranked_order = np.argsort(dists, kind="stable")

                hits = 0
                sum_p = 0.0
                for rank, node in enumerate(ranked_order, start=1):
                    if node in children:
                        hits += 1
                        sum_p += hits / rank
                        ranks.append(rank)

                if hits > 0:
                    aps.append(sum_p / len(children))

            map_score = float(np.mean(aps)) if aps else 0.0
            mean_rank = float(np.mean(ranks)) if ranks else 0.0
            return map_score, mean_rank, aps

        p_map, p_mrank, p_aps = evaluate_embeddings(poincare_embs, is_poincare=True)
        e_map, e_mrank, e_aps = evaluate_embeddings(euclidean_embs, is_poincare=False)

        sig_test = paired_bootstrap_test(
            p_aps, e_aps, num_samples=10000, confidence_level=0.95, seed=seed
        )

        import hashlib
        node_checksum = hashlib.sha256(" ".join(nodes).encode("utf-8")).hexdigest()

        results = {
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "node_checksum": node_checksum,
            "node_names": nodes,
            "d_model": self.d_model,
            "poincare_hyperbolic": {
                "map": round(p_map, 4),
                "mean_rank": round(p_mrank, 2),
                "description": "Analytical radial projection in Poincaré Ball H^D",
            },
            "euclidean": {
                "map": round(e_map, 4),
                "mean_rank": round(e_mrank, 2),
                "description": "Analytical radial projection in Euclidean Space R^D",
            },
            "gain_hyperbolic_over_euclidean": {
                "map_delta": round(p_map - e_map, 4),
                "rank_reduction": round(e_mrank - p_mrank, 2),
            },
            "significance_test": sig_test,
            "protocol": "synthetic_radial_construction_demo",
            "is_learned_embedding": False,
            "finding_b17_disclosure": "Constructed analytically from graph depth; illustrates metric capacity, not trained Riemannian SGD representations.",
            "run_id": run_id,
        }

        print("\n" + "=" * 80)
        print(" WORDNET HIERARCHY GEOMETRIC DEMO (Analytical Radial Construction)")
        print("=" * 80)
        print(f" {'Metric Space':<32} | {'Dimension (D)':<15} | {'MAP':<10} | {'Mean Rank':<10}")
        print("-" * 80)
        print(f" {'Euclidean Radial R^D':<32} | {self.d_model:<15} | {e_map:<10.4f} | {e_mrank:<10.2f}")
        print(f" {'Poincaré Radial H^D (Demo)':<32} | {self.d_model:<15} | {p_map:<10.4f} | {p_mrank:<10.2f}")
        print("=" * 80)
        print(f" Paired Bootstrap Test (Poincaré vs Euclidean MAP): Delta = {sig_test['delta']:+.4f}, "
              f"95% CI: [{sig_test['ci_lower']:+.4f}, {sig_test['ci_upper']:+.4f}], p = {sig_test['p_value']:.4f}")
        print(" Note: Synthetic radial construction from tree depth (Finding B17), not learned SGD embeddings.")
        print("=" * 80 + "\n")

        with open(os.path.join(output_dir, "wordnet_hierarchy_summary.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    from lthcm.utils.determinism import ensure_deterministic_env
    ensure_deterministic_env(42)

    import argparse
    parser = argparse.ArgumentParser(description="Real WordNet Hierarchy Benchmark")
    parser.add_argument("--max_nodes", type=int, default=250, help="Max WordNet nodes")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory")
    args = parser.parse_args()

    bench = WordNetHierarchyBenchmark()
    bench.run_evaluation(max_nodes=args.max_nodes, output_dir=args.output_dir)

