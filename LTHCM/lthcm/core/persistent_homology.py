"""
Persistent Homology & Betti Filtration Guard (Topological Data Analysis)
========================================================================
Audits retrieved knowledge manifolds for epistemic completeness, structural
voids, and conceptual fragmentation using Vietoris-Rips persistent homology
over discrete cochains in embedding space.

Mathematical Framework:
  1. Filtered simplicial complex: VR_ε(X) = { σ ⊆ X | ∀ u, v ∈ σ, d(u, v) ≤ ε }
  2. Persistent Betti numbers across scale ε:
       β_0(ε): Count of disconnected epistemic clusters (knowledge islands).
       β_1(ε): Count of 1-dimensional topological cycles (hollow reasoning loops).
       β_2(ε): Count of 2-dimensional voids (missing intermediate premises).
  3. Epistemic Completeness Score:
       C(X) = exp(-α * Σ lifetime(β_1) / d_med - γ * (β_0(ε*) - 1)) ∈ (0, 1]

Reference: Tier 1 #2 (Persistent Homology Guard) in impro.md.
"""

import math
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import torch


class PersistentHomologyGuard:
    """
    Computes persistent homology barcodes and audits candidate passage
    embeddings for topological voids and epistemic fragmentation.
    """

    def __init__(
        self,
        distance_metric: str = "euclidean",
        max_edge_length: Optional[float] = None,
        significance_ratio: float = 0.20,
    ):
        """
        Args:
            distance_metric: 'euclidean' or 'cosine'.
            max_edge_length: Maximum filtration threshold ε_max. If None, uses max pairwise distance.
            significance_ratio: Ratio of lifetime (death - birth) to birth threshold
                                for a 1-cycle to be classified as a genuine epistemic void.
        """
        self.distance_metric = distance_metric.lower()
        self.max_edge_length = max_edge_length
        self.significance_ratio = significance_ratio

    def compute_distance_matrix(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Computes pairwise distance matrix for candidate vectors.
        """
        if isinstance(embeddings, torch.Tensor):
            X = embeddings.detach().cpu().float().numpy()
        else:
            X = np.asarray(embeddings, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = len(X)
        if n <= 1:
            return np.zeros((n, n), dtype=float)

        if self.distance_metric == "cosine":
            # Normalize vectors
            norms = np.linalg.norm(X, axis=-1, keepdims=True)
            norms[norms == 0] = 1e-12
            X_norm = X / norms
            cos_sim = np.dot(X_norm, X_norm.T)
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            D = 1.0 - cos_sim
            np.fill_diagonal(D, 0.0)
            return np.maximum(D, 0.0)
        else:
            # Euclidean ground metric
            diff = X[:, None, :] - X[None, :, :]
            D = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(D, 0.0)
            return D

    def compute_persistence(
        self,
        embeddings: torch.Tensor,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Builds the Vietoris-Rips complex and computes persistent homology
        in dimensions 0 (components) and 1 (cycles/voids) via Z_2 boundary reduction.
        """
        D = self.compute_distance_matrix(embeddings)
        n = len(D)

        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(n)]

        if n <= 1:
            return {
                "num_documents": n,
                "intervals_dim0": [(0.0, float("inf"))] if n == 1 else [],
                "intervals_dim1": [],
                "betti_0_profile": 1 if n == 1 else 0,
                "betti_1_profile": 0,
                "epistemic_completeness": 1.0,
                "is_complete": True,
                "status": "COMPLETE",
                "voids": [],
                "recommendation": "Single or empty document set.",
            }

        # Determine filtration scale
        upper_tri = D[np.triu_indices(n, k=1)]
        median_dist = float(np.median(upper_tri)) if len(upper_tri) > 0 else 1.0
        max_dist = self.max_edge_length or (float(np.max(upper_tri)) if len(upper_tri) > 0 else 1.0)

        # 1. Enumerate Simplices
        simplices: List[Tuple[float, int, Tuple[int, ...]]] = []
        
        # 0-simplices (vertices born at 0.0)
        for i in range(n):
            simplices.append((0.0, 0, (i,)))

        # 1-simplices (edges born at distance)
        for i in range(n):
            for j in range(i + 1, n):
                val = float(D[i, j])
                if val <= max_dist:
                    simplices.append((val, 1, (i, j)))

        # 2-simplices (triangles born at max boundary edge)
        # Cap to reasonable complexity if N is larger
        max_triangles = 15000
        triangle_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = D[i, j]
                if d_ij > max_dist:
                    continue
                for k in range(j + 1, n):
                    val = float(max(d_ij, D[j, k], D[i, k]))
                    if val <= max_dist:
                        simplices.append((val, 2, (i, j, k)))
                        triangle_count += 1
                        if triangle_count >= max_triangles:
                            break
                if triangle_count >= max_triangles:
                    break
            if triangle_count >= max_triangles:
                break

        # Sort simplices by: (filtration_value, dimension, vertices)
        simplices.sort(key=lambda s: (s[0], s[1], s[2]))
        s_to_idx = {s[2]: idx for idx, s in enumerate(simplices)}

        # 2. Build Boundary Matrix B over Z_2
        B: List[Set[int]] = []
        for idx, (val, dim, verts) in enumerate(simplices):
            faces: Set[int] = set()
            if dim == 1:
                faces.add(s_to_idx[(verts[0],)])
                faces.add(s_to_idx[(verts[1],)])
            elif dim == 2:
                faces.add(s_to_idx[(verts[0], verts[1])])
                faces.add(s_to_idx[(verts[1], verts[2])])
                faces.add(s_to_idx[(verts[0], verts[2])])
            B.append(faces)

        # 3. Standard Boundary Matrix Reduction (R = B * V)
        low_to_col: Dict[int, int] = {}
        intervals_0: List[Tuple[float, float]] = []
        intervals_1: List[Tuple[float, float, Tuple[int, ...]]] = []

        for j in range(len(simplices)):
            while B[j]:
                low = max(B[j])
                if low in low_to_col:
                    k = low_to_col[low]
                    B[j] = B[j] ^ B[k]  # XOR / symmetric difference over Z_2
                else:
                    low_to_col[low] = j
                    birth = simplices[low][0]
                    death = simplices[j][0]
                    dim = simplices[low][1]
                    if death > birth:
                        if dim == 0:
                            intervals_0.append((birth, death))
                        elif dim == 1:
                            killer_verts = simplices[j][2] if simplices[j][1] == 2 else ()
                            all_verts = tuple(sorted(set(simplices[low][2] + killer_verts)))
                            intervals_1.append((birth, death, all_verts))
                    break

        # Unpaired features (essential homology classes, live to infinity)
        paired_births = set(low_to_col.keys())
        for i, (val, dim, verts) in enumerate(simplices):
            if dim == 0 and i not in paired_births and not B[i] and i not in low_to_col.values():
                intervals_0.append((val, float("inf")))
            elif dim == 1 and i not in paired_births and not B[i] and i not in low_to_col.values():
                intervals_1.append((val, float("inf"), verts))

        # 4. Extract Diagnostic Voids and Completeness Metrics
        # Compute characteristic scale from local nearest-neighbor density
        if n > 1:
            D_no_diag = D.copy()
            np.fill_diagonal(D_no_diag, np.inf)
            min_dists = np.min(D_no_diag, axis=1)
            finite_mins = min_dists[np.isfinite(min_dists)]
            knn_scale = float(np.median(finite_mins)) if len(finite_mins) > 0 else median_dist
            characteristic_eps = max(knn_scale * 1.5, 1e-4)
        else:
            characteristic_eps = median_dist

        betti_0_at_eps = sum(1 for (b, d) in intervals_0 if b <= characteristic_eps < d)
        
        voids = []
        total_dim1_persistence = 0.0

        for b, d, verts in intervals_1:
            lifetime = (d - b) if math.isfinite(d) else (max_dist - b)
            if lifetime > 0:
                total_dim1_persistence += lifetime
                # Significant void check: lifetime relative to birth
                is_significant = (b > 0 and (lifetime / b) >= self.significance_ratio) or math.isinf(d)
                if is_significant:
                    participating_docs = [doc_ids[v] for v in verts if v < len(doc_ids)]
                    voids.append({
                        "birth": round(b, 4),
                        "death": round(d, 4) if math.isfinite(d) else "inf",
                        "lifetime": round(lifetime, 4),
                        "participating_docs": participating_docs,
                    })

        # Betti-1 count at characteristic scale
        betti_1_at_eps = sum(1 for (b, d, _) in intervals_1 if b <= characteristic_eps < d)

        # Epistemic Completeness Score C in [0, 1]
        norm_factor = max(median_dist, 1e-6)
        penalty_voids = min(2.0, total_dim1_persistence / norm_factor)
        penalty_clusters = max(0, betti_0_at_eps - 1) * 0.35
        completeness_score = float(np.clip(math.exp(-penalty_voids - penalty_clusters), 0.0, 1.0))

        # Status and Recommendations
        if betti_0_at_eps > 1:
            status = "FRAGMENTED_CLUSTERS"
            rec = f"Retrieved set splits into {betti_0_at_eps} disconnected conceptual islands at median scale ε={median_dist:.3f}. Suggest bridging query expansion."
        elif len(voids) > 0:
            status = "KNOWLEDGE_VOID_DETECTED"
            doc_str = ", ".join(voids[0]["participating_docs"])
            rec = f"Topological reasoning loop detected between [{doc_str}] without internal connecting premise. Intermediate evidence is missing."
        else:
            status = "COMPLETE"
            rec = "Knowledge manifold is topologically coherent and contractible (no persistent voids or disconnected islands)."

        return {
            "num_documents": n,
            "characteristic_epsilon": round(characteristic_eps, 4),
            "betti_0_profile": int(betti_0_at_eps),
            "betti_1_profile": int(betti_1_at_eps),
            "epistemic_completeness": round(completeness_score, 4),
            "is_complete": bool(completeness_score >= 0.70 and len(voids) == 0 and betti_0_at_eps <= 1),
            "status": status,
            "voids": voids,
            "recommendation": rec,
            "intervals_dim0": [(round(b, 4), round(d, 4) if math.isfinite(d) else "inf") for (b, d) in intervals_0],
            "intervals_dim1": [(round(b, 4), round(d, 4) if math.isfinite(d) else "inf") for (b, d, _) in intervals_1],
        }

    def audit_retrieved_context(
        self,
        embeddings: torch.Tensor,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        High-level wrapper for LTHCMEngine pipeline integration.
        """
        return self.compute_persistence(embeddings, doc_ids=doc_ids)
