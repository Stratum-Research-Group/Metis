"""
Master Benchmark Fleet Runner for LTHCM
=======================================
Unified entrypoint for running synthetic smoke tests, core community benchmarks,
and extended mathematical frontier evaluations across five core claims:

  1. Multi-Hop Reasoning (Wormholes): HotpotQA, MuSiQue, 2WikiMultiHopQA
  2. Contradiction Verification (Sheaf): VitaminC, FEVER
  3. Scientific Information Retrieval: BEIR SciFact
  4. Long-Context Needle-in-a-Haystack: RULER
  5. Hyperbolic Hierarchical Geometry: WordNet Noun Subtree

Usage:
  python -m lthcm.benchmarks.run_benchmarks --suite=synthetic
  python -m lthcm.benchmarks.run_benchmarks --suite=core --smoke
  python -m lthcm.benchmarks.run_benchmarks --suite=all
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any

from lthcm.benchmarks.run_standard_benchmarks import run_all_standard_benchmarks
from lthcm.benchmarks.real.ir.scifact_eval import SciFactRealBenchmark
from lthcm.benchmarks.real.multihop.hotpot_real_eval import HotpotQARealBenchmark
from lthcm.benchmarks.real.multihop.musique_eval import MuSiQueRealBenchmark
from lthcm.benchmarks.real.multihop.twowiki_eval import TwoWikiRealBenchmark
from lthcm.benchmarks.real.contradiction.vitaminc_eval import VitaminCRealBenchmark
from lthcm.benchmarks.synthetic.demo_sheaf import SyntheticSheafDemo as FEVERContradictionBenchmark
from lthcm.benchmarks.real.hierarchy.wordnet_eval import WordNetHierarchyBenchmark
from lthcm.benchmarks.real.niah.ruler_eval import RULERBenchmark


def run_fleet(
    suite: str = "synthetic",
    smoke: bool = True,
    scale: str = "smoke",
    output_dir: str = "runs",
) -> Dict[str, Any]:
    """
    Executes benchmark suites according to specified depth and scope.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    start_time = time.time()

    # Determine sample limits
    if scale == "improv2":
        max_q = 300
        max_hotpot = 1000
        max_vit = 1000
        max_musique = 100
        max_twowiki = 100
        max_nodes = 250
        num_dist = 150
        num_needles = 10
    elif scale == "full" or not smoke:
        max_q = None
        max_hotpot = None
        max_vit = None
        max_musique = None
        max_twowiki = None
        max_nodes = 250
        num_dist = 150
        num_needles = 10
    else:  # smoke
        max_q = 25
        max_hotpot = 25
        max_vit = 30
        max_musique = 20
        max_twowiki = 20
        max_nodes = 100
        num_dist = 60
        num_needles = 5

    print("=" * 80)
    print(f" LTHCM BENCHMARK FLEET RUNNER: suite='{suite}', scale='{scale}' (smoke={smoke})")
    print("=" * 80 + "\n")

    # -------------------------------------------------------------
    # 1. Synthetic Smoke Suite
    # -------------------------------------------------------------
    if suite in ("synthetic", "all"):
        print("\n[>>>] EXECUTING SUITE: SYNTHETIC SMOKE DEMOS [<<<]")
        synthetic_res = run_all_standard_benchmarks(verbose=True)
        all_results["synthetic"] = synthetic_res

    # -------------------------------------------------------------
    # 2. Core Real Community Benchmarks
    # -------------------------------------------------------------
    if suite in ("core", "all"):
        print("\n[>>>] EXECUTING SUITE: CORE REAL COMMUNITY BENCHMARKS [<<<]")
        
        # SciFact
        print(f"\n[-] Running Real BEIR SciFact Benchmark (max_queries={max_q})...")
        scifact_bench = SciFactRealBenchmark()
        all_results["real_scifact"] = scifact_bench.run_evaluation(max_queries=max_q, output_dir=output_dir)

        # HotpotQA
        print(f"\n[-] Running Real HotpotQA Multi-Hop Benchmark (max_questions={max_hotpot})...")
        hotpot_bench = HotpotQARealBenchmark()
        all_results["real_hotpot"] = hotpot_bench.run_evaluation(max_questions=max_hotpot, output_dir=output_dir)

        # VitaminC
        print(f"\n[-] Running Real VitaminC Contradiction Verification (max_pairs={max_vit})...")
        vit_bench = VitaminCRealBenchmark()
        all_results["real_vitaminc"] = vit_bench.run_evaluation(max_pairs=max_vit, output_dir=output_dir)

    # -------------------------------------------------------------
    # 3. Extended Real Fleet
    # -------------------------------------------------------------
    if suite in ("extended", "all"):
        print("\n[>>>] EXECUTING SUITE: EXTENDED REAL BENCHMARKS [<<<]")

        # MuSiQue
        print(f"\n[-] Running MuSiQue Shortcut-Free Multi-Hop Benchmark (max_questions={max_musique})...")
        musique_bench = MuSiQueRealBenchmark()
        all_results["real_musique"] = musique_bench.run_evaluation(max_questions=max_musique, output_dir=output_dir)

        # 2WikiMultiHopQA
        print(f"\n[-] Running 2WikiMultiHopQA Multi-Hop Benchmark (max_questions={max_twowiki})...")
        twowiki_bench = TwoWikiRealBenchmark()
        all_results["real_twowiki"] = twowiki_bench.run_evaluation(max_questions=max_twowiki, output_dir=output_dir)

        # FEVER Fact-Checking / Contradiction Verification
        print(f"\n[-] Running FEVER Fact-Checking & Contradiction Verification...")
        fever_bench = FEVERContradictionBenchmark()
        all_results["fever_contradiction"] = fever_bench.run_benchmark()

        # WordNet
        print(f"\n[-] Running WordNet Hyperbolic Hierarchy Benchmark (max_nodes={max_nodes})...")
        wordnet_bench = WordNetHierarchyBenchmark()
        all_results["real_wordnet"] = wordnet_bench.run_evaluation(max_nodes=max_nodes, output_dir=output_dir)

        # RULER
        print(f"\n[-] Running RULER Long-Context NIAH Benchmark (distractors={num_dist}, needles={num_needles})...")
        ruler_bench = RULERBenchmark()
        all_results["real_ruler"] = ruler_bench.run_evaluation(
            num_distractors=num_dist, num_needles=num_needles, output_dir=output_dir
        )

    total_elapsed = time.time() - start_time
    all_results["total_elapsed_seconds"] = round(total_elapsed, 2)

    summary_file = os.path.join(output_dir, "benchmark_run_latest.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f" BENCHMARK FLEET EXECUTION COMPLETE ({total_elapsed:.2f}s)")
    print(f" Summary saved to: {summary_file}")
    print("=" * 80 + "\n")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Benchmark Fleet Runner for LTHCM")
    parser.add_argument(
        "--suite",
        type=str,
        default="synthetic",
        choices=["synthetic", "core", "extended", "all"],
        help="Benchmark suite to execute",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=["smoke", "improv2", "full"],
        help="Dataset scale: smoke (fast verification), improv2 (audit reference scale), or full (uncapped)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Run in fast smoke mode with truncated query samples for rapid verification",
    )
    parser.add_argument(
        "--improv2",
        action="store_true",
        default=False,
        help="Run at improv2.md reference scale (SciFact 300, HotpotQA 1000, VitaminC 1000)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Run full datasets (warning: compute-intensive)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs",
        help="Directory to save output run files and statistics",
    )
    args = parser.parse_args()

    scale = args.scale
    if scale is None:
        if args.improv2:
            scale = "improv2"
        elif args.full:
            scale = "full"
        elif args.smoke:
            scale = "smoke"
        else:
            scale = "smoke"

    run_fleet(suite=args.suite, smoke=(scale == "smoke"), scale=scale, output_dir=args.output_dir)
