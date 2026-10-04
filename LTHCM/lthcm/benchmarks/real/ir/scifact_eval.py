"""
Real BEIR SciFact Scientific Information Retrieval Benchmark
============================================================
Evaluates real community SciFact corpus (5,183 docs, 300 test queries)
using standard pytrec_eval metrics per Thakur et al. (2021) and the honest
evaluation protocol in BenchmarkingLTHCMhonestly.md:

  1. Flat Dense FAISS (MiniLM-L6-v2, reference nDCG@10 ~ 0.645)
  2. Lexical BM25 (rank_bm25, reference nDCG@10 ~ 0.652)
  3. LTHCM-Flat (POVM + Stochastic Resonance + Fisher-Rao, zero edges)
  4. LTHCM-Random-Edges (Control: matching degree distribution)
  5. LTHCM-Mined-Edges (build_edges() strictly from unlabelled corpus text)
  6. LTHCM-Oracle-Edges (Upper Bound: gold co-relevant pairs strictly labeled ORACLE)

Statistical significance: Paired bootstrap test (10,000 resamples, 95% CI).
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
import faiss
import pytrec_eval
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from lthcm.core.superposition import SuperpositionMemoryStore
from lthcm.core.fisher_rao import FisherRaoMetric
from lthcm.core.stochastic_resonance import StochasticResonanceQueryLayer
from lthcm.core.wormhole import WormholeGraphIndex
from lthcm.graph.build_edges import (
    build_edges,
    compute_edge_statistics,
    generate_random_edges,
    generate_oracle_edges,
    save_edges_to_file,
)
from lthcm.benchmarks.real.utils.significance import paired_bootstrap_test
from lthcm.utils.determinism import set_all_seeds


class SciFactRealBenchmark:
    """
    Evaluates real community SciFact dataset under standard and topological regimes.
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
        self, max_queries: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, int]]]:
        """
        Loads the real SciFact dataset from HuggingFace BeIR/scifact.
        """
        from datasets import load_dataset
        import pandas as pd

        print("[-] Loading real BeIR/scifact corpus and test qrels...")
        corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus", cache_dir=self.cache_dir)
        queries_ds = load_dataset("BeIR/scifact", "queries", split="queries", cache_dir=self.cache_dir)
        
        # Load qrels
        qrels_df = pd.read_csv("hf://datasets/BeIR/scifact-qrels/test.tsv", sep="\t", dtype=str)
        qrels: Dict[str, Dict[str, int]] = {}
        for q, d, s in qrels_df.itertuples(index=False):
            qrels.setdefault(str(q), {})[str(d)] = int(s)

        # Match test queries
        qtext_map = {str(q["_id"]): str(q["text"]) for q in queries_ds if str(q["_id"]) in qrels}
        qids = sorted(list(qtext_map.keys()))

        if max_queries and max_queries < len(qids):
            qids = qids[:max_queries]
            qtext_map = {q: qtext_map[q] for q in qids}
            qrels = {q: qrels[q] for q in qids}

        # Corpus docs: title + " " + text
        corpus_docs = []
        for d in corpus_ds:
            did = str(d["_id"])
            title = str(d.get("title", "")).strip()
            text = str(d.get("text", "")).strip()
            combined_text = f"{title} {text}".strip()
            corpus_docs.append({
                "id": did,
                "title": title,
                "text": combined_text,
            })

        print(f"    Loaded {len(corpus_docs)} corpus passages, {len(qids)} test queries.")
        return corpus_docs, qtext_map, qrels

    def run_evaluation(
        self,
        max_queries: Optional[int] = None,
        output_dir: str = "runs",
        seed: int = 42,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full SciFact evaluation comparing the 4 arms + BM25 baseline.
        """
        set_all_seeds(seed)
        os.makedirs(output_dir, exist_ok=True)
        corpus_docs, queries, qrels = self.load_data(max_queries=max_queries)
        qids = list(queries.keys())
        dids = [d["id"] for d in corpus_docs]
        dtext = [d["text"] for d in corpus_docs]

        # -------------------------------------------------------------
        # 1. Encode Corpus and Queries
        # -------------------------------------------------------------
        print(f"[-] Encoding {len(corpus_docs)} documents with MiniLM-L6-v2 (D={self.d_model})...")
        t0 = time.time()
        D = self.encoder.encode(dtext, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        Q = self.encoder.encode([queries[q] for q in qids], batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        print(f"    Encoding finished in {time.time() - t0:.2f}s.")

        # Evaluator
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {"ndcg_cut.10", "recall.10", "recall.100", "map_cut.100", "recip_rank"}
        )

        results = {
            "dataset": "beir/scifact",
            "is_synthetic": False,
            "total_queries": len(qids),
        }

        # -------------------------------------------------------------
        # 2. Lexical Baseline: BM25 (with punctuation stripping)
        # -------------------------------------------------------------
        print("[-] Running Lexical Baseline: Okapi BM25...")
        import string
        def tokenize_bm25(text: str) -> List[str]:
            clean = text.lower().translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
            return [w for w in clean.split() if w]

        tokenized_corpus = [tokenize_bm25(doc) for doc in dtext]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_run = {}
        for q in qids:
            q_tok = tokenize_bm25(queries[q])
            scores = bm25.get_scores(q_tok)
            top_k_idx = np.argsort(scores)[::-1][:100]
            bm25_run[q] = {dids[i]: float(scores[i]) for i in top_k_idx}
        bm25_per_q = evaluator.evaluate(bm25_run)
        results["bm25"] = self._aggregate_metrics(bm25_per_q)

        # -------------------------------------------------------------
        # 3. Dense Baseline: Flat FAISS Exact Inner Product
        # -------------------------------------------------------------
        print("[-] Running Dense Baseline: Flat FAISS IndexFlatIP...")
        faiss_index = faiss.IndexFlatIP(self.d_model)
        faiss_index.add(D)
        S, I = faiss_index.search(Q, 100)
        flat_run = {q: {dids[j]: float(s) for s, j in zip(S[n], I[n])} for n, q in enumerate(qids)}
        flat_per_q = evaluator.evaluate(flat_run)
        results["flat_dense"] = self._aggregate_metrics(flat_per_q)

        # -------------------------------------------------------------
        # 3b. R3: HyDE-Expanded Dense Retrieval
        # -------------------------------------------------------------
        print("[-] Running R3: HyDE-Expanded Dense Retrieval...")
        from lthcm.core.query_expansion import HyDEExpander
        hyde = HyDEExpander(self.encoder, alpha=0.5, num_expansions=3)
        Q_hyde = np.array([
            hyde.expand_query_embedding(queries[q], Q[n])
            for n, q in enumerate(qids)
        ], dtype="float32")
        S_hyde, I_hyde = faiss_index.search(Q_hyde, 100)
        hyde_run = {q: {dids[j]: float(s) for s, j in zip(S_hyde[n], I_hyde[n])} for n, q in enumerate(qids)}
        hyde_per_q = evaluator.evaluate(hyde_run)
        results["flat_hyde"] = self._aggregate_metrics(hyde_per_q)

        # -------------------------------------------------------------
        # 4. Arm 1: LTHCM-Flat (POVM + Stochastic Resonance + Fisher-Rao, No Edges)
        # -------------------------------------------------------------
        print("[-] Running Arm 1: LTHCM-Flat (Multi-Facet POVM + Fisher-Rao, zero edges)...")
        stochastic_res = StochasticResonanceQueryLayer(d_model=self.d_model).to(self.device)
        fisher_rao = FisherRaoMetric(d_model=self.d_model).to(self.device)
        
        # Estimate Fisher-Rao uncertainty weights from corpus variance
        with torch.no_grad():
            D_tensor = torch.from_numpy(D).to(self.device)
            fisher_rao.estimate_from_data(D_tensor)

        # Apply stochastic resonance to query embeddings
        with torch.no_grad():
            set_all_seeds(seed)
            Q_tensor = torch.from_numpy(Q).to(self.device)
            Q_enhanced = stochastic_res(Q_tensor, noise_scale=0.03, num_samples=3).cpu().numpy().astype("float32")

        S_enh, I_enh = faiss_index.search(Q_enhanced, 100)
        
        # Fisher-Rao manifold reranking on candidate pool (top-50)
        lthcm_flat_run = {}
        with torch.no_grad():
            for n, q in enumerate(qids):
                cand_indices = I_enh[n][:50]
                cand_sims = S_enh[n][:50]
                q_vec = torch.from_numpy(Q_enhanced[n:n+1]).to(self.device)
                cand_vecs = torch.from_numpy(D[cand_indices]).to(self.device)
                
                # Fisher-Rao manifold similarity scores
                fr_sims = fisher_rao(q_vec, cand_vecs).squeeze(0).cpu().numpy()
                fr_scores = []
                for idx, c_idx in enumerate(cand_indices):
                    combined = 0.65 * cand_sims[idx] + 0.35 * max(0.0, float(fr_sims[idx]))
                    fr_scores.append((dids[c_idx], float(combined)))
                
                # Merge with remainder of top-100
                cand_set = {dids[ci] for ci in cand_indices}
                for idx, c_idx in enumerate(I_enh[n][50:], start=50):
                    did = dids[c_idx]
                    if did not in cand_set:
                        fr_scores.append((did, float(S_enh[n][idx])))

                fr_scores.sort(key=lambda x: x[1], reverse=True)
                lthcm_flat_run[q] = {did: score for did, score in fr_scores}

        lthcm_flat_per_q = evaluator.evaluate(lthcm_flat_run)
        results["lthcm_flat"] = self._aggregate_metrics(lthcm_flat_per_q)

        # -------------------------------------------------------------
        # 5. Build Wormhole Graph Edges (Corpus Text Mining)
        # -------------------------------------------------------------
        print("[-] Building Wormhole Graph Edges (method='title_mention+ner', max_out=5)...")
        mined_edges = build_edges(corpus_docs, method="title_mention+ner", max_out=5)
        edge_stats = compute_edge_statistics(mined_edges, total_docs=len(corpus_docs))
        save_edges_to_file(mined_edges, os.path.join(output_dir, "scifact_mined_edges.json"), stats=edge_stats)
        print(f"    Mined {len(mined_edges)} edges across {edge_stats['docs_with_edges']} docs (mean out-degree: {edge_stats['mean_out_degree']}).")

        # -------------------------------------------------------------
        # 6. Arm 3: LTHCM-Mined-Edges
        # -------------------------------------------------------------
        print("[-] Running Arm 3: LTHCM-Mined-Edges (Relational Wormholes)...")
        wormhole_index = WormholeGraphIndex(d_model=self.d_model)
        for d, emb in zip(corpus_docs, D):
            wormhole_index.add_node(d["id"], torch.from_numpy(emb), d["text"])
        for src, dst, rel, perm in mined_edges:
            wormhole_index.add_wormhole(src, dst, rel, permeability=perm)

        lthcm_mined_run = self._run_wormhole_reranking(lthcm_flat_run, wormhole_index, Q, qids, dids, D)
        mined_per_q = evaluator.evaluate(lthcm_mined_run)
        results["lthcm_mined_graph"] = self._aggregate_metrics(mined_per_q)

        # -------------------------------------------------------------
        # 7. Arm 2: LTHCM-Random-Edges Control
        # -------------------------------------------------------------
        print("[-] Running Arm 2: LTHCM-Random-Edges (Degree-matched control)...")
        random_edges = generate_random_edges(corpus_docs, mined_edges, seed=seed)
        rand_index = WormholeGraphIndex(d_model=self.d_model)
        for d, emb in zip(corpus_docs, D):
            rand_index.add_node(d["id"], torch.from_numpy(emb), d["text"])
        for src, dst, rel, perm in random_edges:
            rand_index.add_wormhole(src, dst, rel, permeability=perm)

        lthcm_random_run = self._run_wormhole_reranking(lthcm_flat_run, rand_index, Q, qids, dids, D)
        random_per_q = evaluator.evaluate(lthcm_random_run)
        results["lthcm_random_graph"] = self._aggregate_metrics(random_per_q)

        # -------------------------------------------------------------
        # 8. Arm 4: LTHCM-Oracle-Edges (Upper Bound using gold qrels)
        # -------------------------------------------------------------
        print("[-] Running Arm 4: LTHCM-Oracle-Edges (Upper bound from gold co-relevance)...")
        gold_pairs = []
        for q, rel_docs in qrels.items():
            pos_dids = [did for did, score in rel_docs.items() if score > 0]
            for i in range(len(pos_dids)):
                for j in range(i + 1, len(pos_dids)):
                    gold_pairs.append((pos_dids[i], pos_dids[j]))
        oracle_edges = generate_oracle_edges(gold_pairs, permeability=1.0)
        oracle_index = WormholeGraphIndex(d_model=self.d_model)
        for d, emb in zip(corpus_docs, D):
            oracle_index.add_node(d["id"], torch.from_numpy(emb), d["text"])
        for src, dst, rel, perm in oracle_edges:
            oracle_index.add_wormhole(src, dst, rel, permeability=perm)

        lthcm_oracle_run = self._run_wormhole_reranking(lthcm_flat_run, oracle_index, Q, qids, dids, D)
        oracle_per_q = evaluator.evaluate(lthcm_oracle_run)
        results["lthcm_oracle_graph"] = self._aggregate_metrics(oracle_per_q)

        # -------------------------------------------------------------
        # 9. R2: Cross-Encoder Reranking (2×2 Ablation Grid)
        # -------------------------------------------------------------
        print("[-] Running R2: Cross-Encoder Reranking (2×2 ablation grid)...")
        from lthcm.core.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(device=str(self.device))

        corpus_text_map = {d["id"]: d["text"] for d in corpus_docs}
        query_text_map = {q: queries[q] for q in qids}

        # Arm R2a: Flat + CrossEncoder Rerank (no LTHCM geometry)
        flat_rerank_run = reranker.rerank_run(query_text_map, flat_run, corpus_text_map, top_k=50)
        flat_rerank_per_q = evaluator.evaluate(flat_rerank_run)
        results["flat_rerank"] = self._aggregate_metrics(flat_rerank_per_q)

        # Arm R2b: LTHCM Mined + CrossEncoder Rerank (geometry + reranker)
        lthcm_mined_rerank_run = reranker.rerank_run(query_text_map, lthcm_mined_run, corpus_text_map, top_k=50)
        lthcm_mined_rerank_per_q = evaluator.evaluate(lthcm_mined_rerank_run)
        results["lthcm_mined_rerank"] = self._aggregate_metrics(lthcm_mined_rerank_per_q)

        reranker.clear_cache()

        # -------------------------------------------------------------
        # 10. Statistical Significance: Paired Bootstrap (10,000 resamples)
        # -------------------------------------------------------------
        print("[-] Computing Paired Bootstrap Significance (10,000 resamples, 95% CI)...")
        flat_ndcg = [flat_per_q[q]["ndcg_cut_10"] for q in qids]
        hyde_ndcg = [hyde_per_q[q]["ndcg_cut_10"] for q in qids]
        random_ndcg = [random_per_q[q]["ndcg_cut_10"] for q in qids]
        mined_ndcg = [mined_per_q[q]["ndcg_cut_10"] for q in qids]
        flat_rerank_ndcg = [flat_rerank_per_q[q]["ndcg_cut_10"] for q in qids]
        mined_rerank_ndcg = [lthcm_mined_rerank_per_q[q]["ndcg_cut_10"] for q in qids]

        # Primary hypothesis test: Mined vs Random Control (isolates topological shortcuts from random density)
        sig_test_random = paired_bootstrap_test(mined_ndcg, random_ndcg, num_samples=10000, confidence_level=0.95, seed=seed)
        # Secondary test: Mined vs Flat FAISS
        sig_test_flat = paired_bootstrap_test(mined_ndcg, flat_ndcg, num_samples=10000, confidence_level=0.95, seed=seed)

        # R3 HyDE significance test
        sig_hyde_vs_flat = paired_bootstrap_test(hyde_ndcg, flat_ndcg, num_samples=10000, confidence_level=0.95, seed=seed)

        # R2 2×2 ablation significance tests
        # Does reranking improve flat? (flat+rerank vs flat)
        sig_rerank_flat = paired_bootstrap_test(flat_rerank_ndcg, flat_ndcg, num_samples=10000, confidence_level=0.95, seed=seed)
        # Does reranking improve LTHCM mined? (mined+rerank vs mined)
        sig_rerank_mined = paired_bootstrap_test(mined_rerank_ndcg, mined_ndcg, num_samples=10000, confidence_level=0.95, seed=seed)
        # Key ablation: Does LTHCM+rerank beat flat+rerank? (geometry earns its place)
        sig_lthcm_rerank_vs_flat_rerank = paired_bootstrap_test(mined_rerank_ndcg, flat_rerank_ndcg, num_samples=10000, confidence_level=0.95, seed=seed)

        results["significance_mined_vs_random_ndcg@10"] = sig_test_random
        results["significance_mined_vs_flat_ndcg@10"] = sig_test_flat
        results["significance_hyde_vs_flat_ndcg@10"] = sig_hyde_vs_flat
        results["significance_flat_rerank_vs_flat_ndcg@10"] = sig_rerank_flat
        results["significance_mined_rerank_vs_mined_ndcg@10"] = sig_rerank_mined
        results["significance_lthcm_rerank_vs_flat_rerank_ndcg@10"] = sig_lthcm_rerank_vs_flat_rerank

        results["summary"] = {
            "bm25": results["bm25"],
            "bm25_lexical": results["bm25"],
            "flat_dense": results["flat_dense"],
            "flat_hyde": results["flat_hyde"],
            "flat_rerank": results["flat_rerank"],
            "lthcm_flat": results["lthcm_flat"],
            "lthcm_random": results["lthcm_random_graph"],
            "lthcm_graph": results["lthcm_mined_graph"],
            "lthcm_mined_graph": results["lthcm_mined_graph"],
            "lthcm_mined_rerank": results["lthcm_mined_rerank"],
            "lthcm_oracle_graph": results["lthcm_oracle_graph"],
        }

        print("\n" + "=" * 80)
        print(" REAL SCIFACT COMMUNITY BENCHMARK SUMMARY")
        print("=" * 80)
        print(f" {'System':<28} | {'nDCG@10':<9} | {'Recall@10':<9} | {'Recall@100':<10} | {'MAP@100':<8} | {'MRR':<8}")
        print("-" * 80)
        for name, key in [
            ("BM25 (rank_bm25)", "bm25"),
            ("Flat FAISS (MiniLM-L6-v2)", "flat_dense"),
            ("Flat + HyDE (R3)", "flat_hyde"),
            ("Flat + CrossEncoder (R2)", "flat_rerank"),
            ("LTHCM-Flat (No edges)", "lthcm_flat"),
            ("LTHCM-Random-Edges (Control)", "lthcm_random_graph"),
            ("LTHCM-Mined-Edges (Graph)", "lthcm_mined_graph"),
            ("LTHCM-Mined + CrossEncoder", "lthcm_mined_rerank"),
            ("LTHCM-Oracle (Upper Bound)", "lthcm_oracle_graph"),
        ]:
            m = results[key]
            print(f" {name:<28} | {m['ndcg@10']:<9.4f} | {m['recall@10']:<9.4f} | {m['recall@100']:<10.4f} | {m['map@100']:<8.4f} | {m['recip_rank']:<8.4f}")
        print("=" * 80)
        print(f" Paired Bootstrap (Mined vs Random nDCG@10): Delta = {sig_test_random['delta']:+.4f}, "
              f"95% CI: [{sig_test_random['ci_lower']:+.4f}, {sig_test_random['ci_upper']:+.4f}], p = {sig_test_random['p_value']:.4f} "
              f"({'Significant' if sig_test_random['is_significant'] else 'Not Significant'})")
        print(f" Paired Bootstrap (Mined vs Flat FAISS nDCG@10): Delta = {sig_test_flat['delta']:+.4f}, "
              f"95% CI: [{sig_test_flat['ci_lower']:+.4f}, {sig_test_flat['ci_upper']:+.4f}], p = {sig_test_flat['p_value']:.4f}")
        print(f" Paired Bootstrap (Flat+HyDE vs Flat FAISS nDCG@10): Delta = {sig_hyde_vs_flat['delta']:+.4f}, "
              f"95% CI: [{sig_hyde_vs_flat['ci_lower']:+.4f}, {sig_hyde_vs_flat['ci_upper']:+.4f}], p = {sig_hyde_vs_flat['p_value']:.4f}")
        print("-" * 80)
        print(" R2 2×2 Ablation Grid (flat vs LTHCM × no-rerank vs rerank):")
        print(f"   Flat+Rerank vs Flat:       Delta = {sig_rerank_flat['delta']:+.4f}, "
              f"95% CI: [{sig_rerank_flat['ci_lower']:+.4f}, {sig_rerank_flat['ci_upper']:+.4f}], p = {sig_rerank_flat['p_value']:.4f}")
        print(f"   LTHCM+Rerank vs LTHCM:    Delta = {sig_rerank_mined['delta']:+.4f}, "
              f"95% CI: [{sig_rerank_mined['ci_lower']:+.4f}, {sig_rerank_mined['ci_upper']:+.4f}], p = {sig_rerank_mined['p_value']:.4f}")
        print(f"   LTHCM+Rerank vs Flat+Rerank: Delta = {sig_lthcm_rerank_vs_flat_rerank['delta']:+.4f}, "
              f"95% CI: [{sig_lthcm_rerank_vs_flat_rerank['ci_lower']:+.4f}, {sig_lthcm_rerank_vs_flat_rerank['ci_upper']:+.4f}], "
              f"p = {sig_lthcm_rerank_vs_flat_rerank['p_value']:.4f}")
        print("=" * 80 + "\n")

        results["run_id"] = run_id

        # Save runs
        with open(os.path.join(output_dir, "scifact_summary.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _run_wormhole_reranking(
        self,
        base_run: Dict[str, Dict[str, float]],
        wormhole_index: WormholeGraphIndex,
        Q: np.ndarray,
        qids: List[str],
        dids: List[str],
        D: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Applies topological wormhole bridge reranking to a candidate run."""
        reranked_run = {}
        did_to_idx = {did: idx for idx, did in enumerate(dids)}

        for n, q in enumerate(qids):
            q_run = sorted(base_run[q].items(), key=lambda x: x[1], reverse=True)
            top_seed_id, top_score = q_run[0]
            
            # Query wormhole index for outgoing bridges
            bridges = wormhole_index.get_bridges(top_seed_id, min_permeability=0.50)
            
            new_scores = {did: s for did, s in q_run}
            q_vec = Q[n]

            # Re-wormholing score: seed score * permeability * bridge similarity
            for b in bridges:
                target_id = b["target"]
                if target_id in did_to_idx:
                    t_vec = D[did_to_idx[target_id]]
                    bridge_sim = float(np.dot(q_vec, t_vec))
                    perm = b["permeability"]
                    bridge_bonus = 0.15 * perm * max(0.0, bridge_sim)
                    new_scores[target_id] = new_scores.get(target_id, 0.0) + bridge_bonus

            # Sort new run
            sorted_run = sorted(new_scores.items(), key=lambda x: x[1], reverse=True)[:100]
            reranked_run[q] = {did: score for did, score in sorted_run}

        return reranked_run

    def _aggregate_metrics(self, per_query_eval: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Averages metrics across queries."""
        ndcg_10 = np.mean([v.get("ndcg_cut_10", 0.0) for v in per_query_eval.values()])
        rec_10 = np.mean([v.get("recall_10", 0.0) for v in per_query_eval.values()])
        rec_100 = np.mean([v.get("recall_100", 0.0) for v in per_query_eval.values()])
        map_100 = np.mean([v.get("map_cut_100", 0.0) for v in per_query_eval.values()])
        mrr = np.mean([v.get("recip_rank", 0.0) for v in per_query_eval.values()])

        return {
            "ndcg@10": round(float(ndcg_10), 4),
            "recall@10": round(float(rec_10), 4),
            "recall@100": round(float(rec_100), 4),
            "map@100": round(float(map_100), 4),
            "recip_rank": round(float(mrr), 4),
        }


if __name__ == "__main__":
    from lthcm.utils.determinism import ensure_deterministic_env
    ensure_deterministic_env(42)

    import argparse
    parser = argparse.ArgumentParser(description="Real BEIR SciFact Evaluation")
    parser.add_argument("--max_queries", type=int, default=None, help="Max queries to evaluate (e.g. 50 for fast smoke test)")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory for runs and stats")
    args = parser.parse_args()

    bench = SciFactRealBenchmark()
    bench.run_evaluation(max_queries=args.max_queries, output_dir=args.output_dir)
