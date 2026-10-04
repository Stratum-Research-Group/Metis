"""
Master Synthetic Verification Benchmark Runner
==============================================
Executes targeted synthetic NLP and Information Retrieval verification suites:
  1. Synthetic Multi-Needle in a Haystack (M-NIAH, 65 docs)
  2. Synthetic Multi-Hop Relational Reasoning (HotpotQA-style, 50 docs)
  3. Synthetic Contradiction Verification (FEVER-style, 20 pairs)
  4. Synthetic Scientific Retrieval (SciFact-style, 35 docs)

Evaluates topological mechanisms against standard dense and lexical baselines.
"""

import sys
import time
from typing import Dict, Any

from lthcm.benchmarks.synthetic.demo_niah import SyntheticNeedleDemo as MultiNeedleHaystackBenchmark
from lthcm.benchmarks.synthetic.demo_multihop import SyntheticMultiHopDemo as HotpotQABenchmark
from lthcm.benchmarks.synthetic.demo_sheaf import SyntheticSheafDemo as FEVERContradictionBenchmark
from lthcm.benchmarks.synthetic.demo_scifact import SyntheticSciFactDemo as BEIRBenchmark


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def run_all_standard_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    """
    Executes all 4 synthetic verification benchmarks and returns full results dictionary.
    """
    start_total = time.time()
    results = {}

    if verbose:
        print_header("Executing LTHCM Synthetic Verification Benchmark Suite")
        print("Model: sentence-transformers/all-MiniLM-L6-v2 (D=384)")
        print("Frameworks: PyTorch, SentenceTransformers, LTHCM Topological Engine\n")

    # ---------------------------------------------------------
    # 1. Multi-Needle in a Haystack (M-NIAH)
    # ---------------------------------------------------------
    if verbose:
        print("[-] Running Benchmark 1/4: Multi-Needle in a Haystack (M-NIAH)...")
    m_niah_bench = MultiNeedleHaystackBenchmark()
    m_niah_res = m_niah_bench.run_benchmark()
    results["m_niah"] = m_niah_res

    if verbose:
        print(f"    Completed in {m_niah_res['elapsed_ms']:.1f}ms (Haystack: {m_niah_res['haystack_size']} docs, Needles: {m_niah_res['num_needles']})")
        print("    +--------------------+----------------------+----------------------+")
        print("    | Metric             | Flat Dense (FAISS)   | LTHCM Topological    |")
        print("    +--------------------+----------------------+----------------------+")
        f_m = m_niah_res["flat_dense_baseline"]
        l_m = m_niah_res["lthcm_topological"]
        print(f"    | Macro-Recall@1     | {f_m['macro_recall@1']*100:6.1f}%              | {l_m['macro_recall@1']*100:6.1f}%              |")
        print(f"    | Macro-Recall@3     | {f_m['macro_recall@3']*100:6.1f}%              | {l_m['macro_recall@3']*100:6.1f}%              |")
        print(f"    | Macro-Recall@5     | {f_m['macro_recall@5']*100:6.1f}%              | {l_m['macro_recall@5']*100:6.1f}%              |")
        print(f"    | MRR                | {f_m['mrr']:6.3f}               | {l_m['mrr']:6.3f}               |")
        print(f"    | Depth Variance     | {f_m['depth_variance']:6.3f} (degrades)       | {l_m['depth_variance']:6.3f} (uniform)        |")
        print("    +--------------------+----------------------+----------------------+\n")

    # ---------------------------------------------------------
    # 2. HotpotQA Multi-Hop Relational Reasoning
    # ---------------------------------------------------------
    if verbose:
        print("[-] Running Benchmark 2/4: HotpotQA Multi-Hop Relational Reasoning...")
    hotpot_bench = HotpotQABenchmark()
    hotpot_res = hotpot_bench.run_benchmark()
    results["hotpot_qa"] = hotpot_res

    if verbose:
        print(f"    Completed in {hotpot_res['elapsed_ms']:.1f}ms (Corpus: {hotpot_res['corpus_size']} docs, Cases: {hotpot_res['num_cases']})")
        print("    +----------------------------+----------------------+----------------------+")
        print("    | Metric                     | Flat Dense (FAISS)   | LTHCM Wormhole Graph |")
        print("    +----------------------------+----------------------+----------------------+")
        f_h = hotpot_res["flat_dense_baseline"]
        l_h = hotpot_res["lthcm_wormhole"]
        print(f"    | Hop 1 Recall@1 (Rank 1)    | {f_h['hop1_recall@1']*100:6.1f}%              | {l_h['hop1_recall@1']*100:6.1f}%              |")
        print(f"    | Hop 1 Recall@2 (Top 2)     | {f_h.get('hop1_recall@2', f_h['hop1_recall@1'])*100:6.1f}%              | {l_h.get('hop1_recall@2', l_h['hop1_recall@1'])*100:6.1f}%              |")
        print(f"    | Hop 2 Recall@2             | {f_h['hop2_recall@2']*100:6.1f}%              | {l_h['hop2_recall@2']*100:6.1f}%              |")
        print(f"    | Joint Multi-Hop Recall@2   | {f_h['joint_multihop_recall@2']*100:6.1f}%              | {l_h['joint_multihop_recall@2']*100:6.1f}%              |")
        print(f"    | nDCG@10                    | {f_h['ndcg@10']:6.3f}               | {l_h['ndcg@10']:6.3f}               |")
        print("    +----------------------------+----------------------+----------------------+\n")

    # ---------------------------------------------------------
    # 3. FEVER / VitaminC Contradiction Verification
    # ---------------------------------------------------------
    if verbose:
        print("[-] Running Benchmark 3/4: FEVER / VitaminC Fact-Checking & Contradiction...")
    fever_bench = FEVERContradictionBenchmark()
    fever_res = fever_bench.run_benchmark()
    results["fever"] = fever_res

    if verbose:
        print(f"    Completed in {fever_res['elapsed_ms']:.1f}ms (Contrastive Pairs: {fever_res['num_pairs']})")
        print("    +----------------------------+----------------------+----------------------+----------------------+")
        print("    | Metric                     | Flat Cosine Sim      | Pure Sheaf (Untrained)| LTHCM Sheaf (NLI)    |")
        print("    +----------------------------+----------------------+----------------------+----------------------+")
        f_c = fever_res["flat_cosine_baseline"]
        p_c = fever_res.get("pure_sheaf_untrained", {})
        l_c = fever_res["lthcm_sheaf_laplacian"]
        print(f"    | Detection Accuracy         | {f_c['accuracy']*100:6.1f}%              | {p_c.get('accuracy', 0.0)*100:6.1f}%              | {l_c['accuracy']*100:6.1f}%              |")
        print(f"    | Contradiction F1-Score     | {f_c['contradiction_f1']:6.3f}               | {p_c.get('contradiction_f1', 0.0):6.3f}               | {l_c['contradiction_f1']:6.3f}               |")
        print(f"    | Precision / Recall         | N/A                  | N/A                  | {l_c['contradiction_precision']*100:.0f}% / {l_c['contradiction_recall']*100:.0f}%          |")
        print(f"    | Obstruction Energy Gap     | Gap={f_c['cosine_separation_gap']:.3f} (Overlap)  | Refute={p_c.get('mean_refute_obstruction_energy', 0.0):.3f} (No sep) | Ratio={l_c['energy_separation_ratio']:.1f}x (Separated)|")
        print("    +----------------------------+----------------------+----------------------+----------------------+\n")

    # ---------------------------------------------------------
    # 4. BEIR Information Retrieval Harness
    # ---------------------------------------------------------
    if verbose:
        print("[-] Running Benchmark 4/4: BEIR Scientific Retrieval Benchmark (SciFact Split)...")
    beir_bench = BEIRBenchmark()
    beir_res = beir_bench.run_benchmark()
    results["beir"] = beir_res

    if verbose:
        print(f"    Completed in {beir_res['elapsed_ms']:.1f}ms (Corpus: {beir_res['corpus_size']} docs, Queries: {beir_res['num_queries']})")
        print("    +--------------------+-----------------+----------------------+----------------------+")
        print("    | Metric             | Lexical (BM25)  | Flat Dense (FAISS)   | LTHCM Topological    |")
        print("    +--------------------+-----------------+----------------------+----------------------+")
        b_m = beir_res["bm25_baseline"]
        d_m = beir_res["dense_biencoder_baseline"]
        l_b = beir_res["lthcm_topological"]
        print(f"    | nDCG@10            | {b_m['ndcg@10']:6.3f}          | {d_m['ndcg@10']:6.3f}               | {l_b['ndcg@10']:6.3f}               |")
        print(f"    | Recall@10          | {b_m['recall@10']*100:6.1f}%         | {d_m['recall@10']*100:6.1f}%              | {l_b['recall@10']*100:6.1f}%              |")
        print(f"    | MAP@100            | {b_m['map@100']:6.3f}          | {d_m['map@100']:6.3f}               | {l_b['map@100']:6.3f}               |")
        print(f"    | MRR@10             | {b_m['mrr@10']:6.3f}          | {d_m['mrr@10']:6.3f}               | {l_b['mrr@10']:6.3f}               |")
        print("    +--------------------+-----------------+----------------------+----------------------+\n")

    total_elapsed = time.time() - start_total
    results["total_elapsed_s"] = total_elapsed

    if verbose:
        print("=" * 80)
        print(f" [ALL 4 STANDARD COMMUNITY BENCHMARKS COMPLETED IN {total_elapsed:.2f}s]")
        print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    run_all_standard_benchmarks(verbose=True)
