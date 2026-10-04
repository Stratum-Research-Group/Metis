"""
Real MuSiQue Multi-Hop Benchmark (Shortcut-Free Multi-Hop Retrieval)
===================================================================
Evaluates multi-hop retrieval on MuSiQue (Trivedi et al., 2022).
MuSiQue is deliberately constructed to prevent single-hop shortcut reasoning,
ensuring that connected topological reasoning across disparate passages
is strictly required to achieve joint retrieval recall.

Compares:
  1. Flat Dense Single-Shot (FAISS)
  2. Flat Dense 2-Step Iterative Retrieval
  3. LTHCM Wormhole Graph Retrieval with build_edges()
  4. LTHCM Random Edges Control
  5. LTHCM Oracle Edges Upper Bound
"""

import os
import sys
import time
import math
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from lthcm.core.wormhole import WormholeGraphIndex
from lthcm.graph.build_edges import (
    build_edges,
    generate_random_edges,
    generate_oracle_edges,
)
from lthcm.benchmarks.real.utils.significance import paired_bootstrap_test
from lthcm.utils.determinism import set_all_seeds


class MuSiQueRealBenchmark:
    """
    Evaluates multi-hop retrieval across shortcut-free MuSiQue questions.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = SentenceTransformer(model_name)
        if hasattr(self.encoder, "eval"):
            self.encoder.eval()
        self.d_model = self.encoder.get_embedding_dimension()
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/datasets")

    def load_data(
        self,
        max_questions: Optional[int] = None,
        allow_synthetic_fallback: bool = False,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Loads MuSiQue dataset from HuggingFace (dgslibisey/MuSiQue) or falls back to synthetic multi-hop pool if explicitly allowed.
        """
        try:
            from datasets import load_dataset
            print("[-] Loading MuSiQue dataset (dgslibisey/MuSiQue)...")
            ds = load_dataset("dgslibisey/MuSiQue", split="validation", cache_dir=self.cache_dir)
            cases = []
            for ex in ds:
                q_id = str(ex["id"])
                question = str(ex["question"]).strip()
                paragraphs = ex.get("paragraphs", [])
                titles = [p.get("title", f"p_{i}") for i, p in enumerate(paragraphs)]
                texts = [f"{t}. {p.get('paragraph_text', '')}".strip() for t, p in zip(titles, paragraphs)]
                gold_indices = [i for i, p in enumerate(paragraphs) if p.get("is_supporting", False)]
                gold_titles = [titles[i] for i in gold_indices]

                cases.append({
                    "id": q_id,
                    "question": question,
                    "titles": titles,
                    "paragraphs": texts,
                    "gold_titles": gold_titles,
                    "gold_indices": gold_indices,
                })
                if max_questions and len(cases) >= max_questions:
                    break
            print(f"    Loaded {len(cases)} MuSiQue questions.")
            return cases, False
        except Exception as e:
            if not allow_synthetic_fallback:
                raise RuntimeError(
                    f"Failed to load real MuSiQue dataset from HuggingFace ({e}). "
                    "To run with synthetic demonstration fixtures for offline testing, pass allow_synthetic_fallback=True."
                )
            print(f"    [WARNING] Using SYNTHETIC FALLBACK data for smoke testing. These metrics do NOT represent real MuSiQue numbers! ({e})")
            from lthcm.benchmarks.synthetic.demo_multihop import SyntheticMultiHopDemo
            demo = SyntheticMultiHopDemo()
            raw_cases, distractors = demo.generate_multihop_dataset()
            cases = []
            for i, c in enumerate(raw_cases):
                titles = ["Doc_Hop1", "Doc_Hop2"] + [f"Distractor_{j}" for j in range(len(distractors[:8]))]
                texts = [c["doc_hop1"]["text"], c["doc_hop2"]["text"]] + distractors[:8]
                cases.append({
                    "id": f"musique_synth_{i}",
                    "question": c["query"],
                    "titles": titles,
                    "paragraphs": texts,
                    "gold_titles": ["Doc_Hop1", "Doc_Hop2"],
                    "gold_indices": [0, 1],
                })
            return cases, True

    def run_evaluation(
        self,
        max_questions: Optional[int] = None,
        output_dir: str = "runs",
        seed: int = 42,
        allow_synthetic_fallback: bool = False,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        set_all_seeds(seed)
        os.makedirs(output_dir, exist_ok=True)
        cases, is_synthetic = self.load_data(max_questions=max_questions, allow_synthetic_fallback=allow_synthetic_fallback)
        
        arms = ["flat_1shot", "flat_2step", "lthcm_mined", "lthcm_random", "lthcm_oracle"]
        metrics = {arm: {"joint@2": [], "recall@2": [], "ndcg@10": []} for arm in arms}

        print(f"[-] Evaluating MuSiQue shortcut-free multi-hop across {len(cases)} questions...")
        t0 = time.time()

        for idx, ex in enumerate(cases):
            question = ex["question"]
            titles = ex["titles"]
            paras = ex["paragraphs"]
            gold_indices = ex["gold_indices"]

            P_embs = self.encoder.encode(paras, normalize_embeddings=True, show_progress_bar=False).astype("float32")
            Q_emb = self.encoder.encode([question], normalize_embeddings=True, show_progress_bar=False).astype("float32")[0]

            # 1. Flat 1-shot
            sims_1shot = np.dot(P_embs, Q_emb)
            ranked_1shot = list(np.argsort(sims_1shot, kind='stable')[::-1])
            self._record(metrics["flat_1shot"], ranked_1shot, gold_indices)

            # 2. Flat 2-step iterative
            top1_idx = int(np.argsort(sims_1shot, kind='stable')[::-1][0])
            iter_q = f"{question} {paras[top1_idx]}"
            Q_step2 = self.encoder.encode([iter_q], normalize_embeddings=True, show_progress_bar=False).astype("float32")[0]
            sims_2step = np.dot(P_embs, Q_step2)
            sims_2step[top1_idx] = -1e9
            ranked_2step = [top1_idx] + [i for i in np.argsort(sims_2step, kind='stable')[::-1] if i != top1_idx]
            self._record(metrics["flat_2step"], ranked_2step, gold_indices)

            # Pool docs
            pool_docs = [{"id": f"p_{i}", "title": titles[i], "text": paras[i]} for i in range(len(titles))]
            gold_pairs = [(f"p_{i}", f"p_{j}") for i in gold_indices for j in gold_indices if i < j]

            # 3. LTHCM Mined (Graph-driven 2-step)
            mined_edges = build_edges(pool_docs, method="title_mention+ner", max_out=4)
            ranked_mined = self._rank_wormhole(pool_docs, mined_edges, P_embs, Q_emb, sims_1shot, sims_2step=sims_2step, top1_idx=top1_idx)
            self._record(metrics["lthcm_mined"], ranked_mined, gold_indices)

            # 4. LTHCM Random (Random control 2-step)
            rand_edges = generate_random_edges(pool_docs, mined_edges, seed=seed + idx)
            ranked_rand = self._rank_wormhole(pool_docs, rand_edges, P_embs, Q_emb, sims_1shot, sims_2step=sims_2step, top1_idx=top1_idx)
            self._record(metrics["lthcm_random"], ranked_rand, gold_indices)

            # 5. LTHCM Oracle (Oracle control 2-step)
            oracle_edges = generate_oracle_edges(gold_pairs)
            ranked_oracle = self._rank_wormhole(pool_docs, oracle_edges, P_embs, Q_emb, sims_1shot, sims_2step=sims_2step, top1_idx=top1_idx)
            self._record(metrics["lthcm_oracle"], ranked_oracle, gold_indices)

        summary = {arm: {k: round(float(np.mean(v)), 4) for k, v in metrics[arm].items()} for arm in arms}
        sig_test_random = paired_bootstrap_test(
            metrics["lthcm_mined"]["joint@2"],
            metrics["lthcm_random"]["joint@2"],
            num_samples=10000,
            seed=seed,
        )
        sig_test_flat2step = paired_bootstrap_test(
            metrics["lthcm_mined"]["joint@2"],
            metrics["flat_2step"]["joint@2"],
            num_samples=10000,
            seed=seed,
        )

        results = {
            "dataset": "MuSiQue",
            "run_id": run_id,
            "is_synthetic": is_synthetic,
            "dataset_source": "synthetic_fallback" if is_synthetic else "dgslibisey/MuSiQue",
            "total_questions": len(cases),
            "elapsed_seconds": round(time.time() - t0, 2),
            "summary": summary,
            "significance_mined_vs_random_joint@2": sig_test_random,
            "significance_mined_vs_flat2step_joint@2": sig_test_flat2step,
        }

        with open(os.path.join(output_dir, "musique_summary.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _rank_wormhole(self, docs, edges, P_embs, Q_emb, base_sims, sims_2step=None, top1_idx=None):
        if sims_2step is not None and top1_idx is not None:
            # Graph-driven 2-step iterative retrieval:
            # After Hop 1 identifies top1_idx, traverse mined wormhole edges from top seed doc.
            # Edge-conditioned expansion re-ranks the 2nd step candidates.
            id_to_idx = {d["id"]: i for i, d in enumerate(docs)}
            step2_scores = sims_2step.copy()
            for src, dst, rel, perm in edges:
                if src == f"p_{top1_idx}":
                    if dst in id_to_idx:
                        t_idx = id_to_idx[dst]
                        step2_scores[t_idx] += 0.25 * perm
                elif dst == f"p_{top1_idx}":
                    if src in id_to_idx:
                        s_idx = id_to_idx[src]
                        step2_scores[s_idx] += 0.20 * perm
            step2_scores[top1_idx] = -1e9
            step2_ranked = [i for i in np.argsort(step2_scores, kind='stable')[::-1] if i != top1_idx]
            return [top1_idx] + step2_ranked

        id_to_idx = {d["id"]: i for i, d in enumerate(docs)}
        wormhole_index = WormholeGraphIndex(d_model=self.d_model)
        for d, emb in zip(docs, P_embs):
            wormhole_index.add_node(d["id"], torch.from_numpy(emb), d["text"])
        for src, dst, rel, perm in edges:
            wormhole_index.add_wormhole(src, dst, rel, permeability=perm)

        seed_idx = int(np.argsort(base_sims, kind='stable')[::-1][0])
        bridges = wormhole_index.get_bridges(docs[seed_idx]["id"], min_permeability=0.35)
        scores = base_sims.copy()
        for b in bridges:
            tid = b["target"]
            if tid in id_to_idx:
                t_idx = id_to_idx[tid]
                b_sim = float(np.dot(Q_emb, P_embs[t_idx]))
                scores[t_idx] += 0.35 * float(base_sims[seed_idx]) * b["permeability"] * max(0.0, b_sim)

        return list(np.argsort(scores, kind="stable")[::-1])

    def _record(self, metrics_dict, ranked_indices, gold_indices):
        gold_set = set(gold_indices)
        top2 = set(ranked_indices[:2])
        joint2 = 1.0 if gold_set.issubset(top2) else 0.0
        rec2 = len(top2 & gold_set) / max(1, len(gold_set))
        
        # Standard binary NDCG@10 on distinct paragraph units
        dcg = sum((2.0 ** (1.0 if p_idx in gold_set else 0.0) - 1.0) / math.log2(i + 2.0) 
                  for i, p_idx in enumerate(ranked_indices[:10]))
        idcg = sum((2.0 ** 1.0 - 1.0) / math.log2(i + 2.0) 
                   for i in range(min(10, len(gold_set))))
        ndcg = min(1.0, (dcg / idcg)) if idcg > 0.0 else 0.0

        metrics_dict["joint@2"].append(joint2)
        metrics_dict["recall@2"].append(rec2)
        metrics_dict["ndcg@10"].append(ndcg)


if __name__ == "__main__":
    from lthcm.utils.determinism import ensure_deterministic_env
    ensure_deterministic_env(42)

    import argparse
    parser = argparse.ArgumentParser(description="MuSiQue Multi-Hop Benchmark")
    parser.add_argument("--max_questions", type=int, default=100, help="Number of questions to evaluate")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory")
    args = parser.parse_args()

    bench = MuSiQueRealBenchmark()
    bench.run_evaluation(max_questions=args.max_questions, output_dir=args.output_dir)
