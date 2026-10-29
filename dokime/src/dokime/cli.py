"""Command line interface for Dokime evaluation harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from dokime import __version__
from dokime.client import HestiaClient
from dokime.compare import CompareError, run_comparison, run_cross_model_comparison
from dokime.config import RunConfig
from dokime.runner import run_evaluation
from dokime.tasks.registry import list_task_info


def format_stat_value(
    val: float | None,
    sign: bool = False,
    default_precision: int = 4,
    max_precision: int = 8,
) -> str:
    """Format float ensuring that non-zero values do not display as all-zero."""
    if val is None:
        return "N/A"
    if val == 0.0 or not np.isfinite(val):
        sign_str = "+" if sign else ""
        return f"{sign_str}0.{'0' * default_precision}"

    sign_fmt = "+" if sign else ""
    if round(val, default_precision) != 0.0:
        return f"{val:{sign_fmt}.{default_precision}f}"

    for p in range(default_precision + 1, max_precision + 1):
        if round(val, p) != 0.0:
            return f"{val:{sign_fmt}.{p}f}"

    # For nonzero values below the configured decimal cap, use scientific notation
    # to expose the sign and nonzero magnitude.
    return f"{val:{sign_fmt}.{default_precision}e}"


def format_interval(
    ci_lower: float | None,
    ci_upper: float | None,
    sign: bool = True,
    default_precision: int = 4,
    max_precision: int = 8,
) -> str:
    """Format a confidence interval [lower, upper] ensuring significant bounds are not shown as all-zero."""
    if ci_lower is None or ci_upper is None:
        return "[N/A, N/A]"

    # If any non-zero bound is below the decimal cap, format with scientific notation
    # so that a significant nonzero interval never reads as all-zero.
    needs_scientific = False
    for val in (ci_lower, ci_upper):
        if val != 0.0 and np.isfinite(val) and round(val, max_precision) == 0.0:
            needs_scientific = True
            break

    if needs_scientific:
        low_str = format_stat_value(
            ci_lower,
            sign=sign,
            default_precision=default_precision,
            max_precision=max_precision,
        )
        high_str = format_stat_value(
            ci_upper,
            sign=sign,
            default_precision=default_precision,
            max_precision=max_precision,
        )
        return f"[{low_str}, {high_str}]"

    precision = default_precision
    for val in (ci_lower, ci_upper):
        if val != 0.0 and np.isfinite(val) and round(val, precision) == 0.0:
            for p in range(precision + 1, max_precision + 1):
                if round(val, p) != 0.0:
                    precision = max(precision, p)
                    break

    sign_fmt = "+" if sign else ""
    return f"[{ci_lower:{sign_fmt}.{precision}f}, {ci_upper:{sign_fmt}.{precision}f}]"


def create_parser() -> argparse.ArgumentParser:
    """Create root argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="dokime",
        description="Dokime: measurement layer for local AI stack.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: tasks
    tasks_parser = subparsers.add_parser(
        "tasks",
        help="List all discovered tasks and their item counts.",
    )

    # Subcommand: run
    run_parser = subparsers.add_parser(
        "run",
        help="Execute an evaluation run across models and tasks.",
    )
    run_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to TOML configuration file.",
    )
    run_parser.add_argument(
        "--model",
        "--models",
        dest="models",
        action="append",
        default=None,
        help="Model identifier(s) to evaluate. Can be specified multiple times.",
    )
    run_parser.add_argument(
        "--task",
        "--tasks",
        dest="tasks",
        action="append",
        default=None,
        help="Task identifier(s) to evaluate. Can be specified multiple times.",
    )
    run_parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Path to prompt template file or inline template string.",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic reproducibility (default: 42).",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.0).",
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum generation tokens.",
    )
    run_parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Hestia API base URL (overrides HESTIA_URL env var).",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store run artifacts (default: 'runs').",
    )

    # Subcommand: compare
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare evaluation runs and flag regressions with bootstrap CIs.",
    )
    compare_parser.add_argument(
        "run_a",
        type=str,
        help="Baseline run_id or path to results.jsonl (or single matrix run for --cross-model).",
    )
    compare_parser.add_argument(
        "run_b",
        type=str,
        nargs="?",
        default=None,
        help="Candidate run_id or path to results.jsonl (required for ordinary compare).",
    )
    compare_parser.add_argument(
        "--cross-model",
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        default=None,
        help="Compare two models within a single matrix run for the given --task.",
    )
    compare_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to scope pairing. Required if either run contains multiple models.",
    )
    compare_parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Optional task ID to slice comparison (required for --cross-model).",
    )
    compare_parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Base directory containing runs (default: 'runs').",
    )

    # Subcommand: explain (Sufficient Input Subsets)
    explain_parser = subparsers.add_parser(
        "explain",
        help="Find Sufficient Input Subsets (SIS) rationales for prompts (Carter et al., Google Research).",
    )
    explain_parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Input text or prompt to explain.",
    )
    explain_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to file containing input text to explain.",
    )
    explain_parser.add_argument(
        "--unit",
        choices=["sentence", "paragraph", "word"],
        default="sentence",
        help="Granularity of explanation units (default: 'sentence').",
    )
    explain_parser.add_argument(
        "--tau",
        type=float,
        default=0.8,
        help="Sufficiency threshold tau (default: 0.8).",
    )
    explain_parser.add_argument(
        "--target-keyword",
        type=str,
        default=None,
        help="Target keyword to evaluate presence / frequency for sufficiency.",
    )
    explain_parser.add_argument(
        "--disjoint",
        action="store_true",
        help="Find collection of disjoint sufficient input subsets.",
    )
    explain_parser.add_argument(
        "--max-subsets",
        type=int,
        default=3,
        help="Maximum number of disjoint subsets to uncover (default: 3).",
    )

    # Subcommand: ann-benchmark (Google Research ScaNN vs FAISS vs Exact)
    ann_parser = subparsers.add_parser(
        "ann-benchmark",
        help="Benchmark vector retrieval backends (ScaNN Tree-AH, FAISS, Exact).",
    )
    ann_parser.add_argument(
        "--num-docs",
        type=int,
        default=1000,
        help="Number of document vectors to index (default: 1000).",
    )
    ann_parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Embedding vector dimension (default: 64).",
    )
    ann_parser.add_argument(
        "--num-queries",
        type=int,
        default=50,
        help="Number of queries to benchmark (default: 50).",
    )
    ann_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of nearest neighbors to retrieve (default: 10).",
    )
    ann_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic vector generation (default: 42).",
    )
    ann_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save benchmark report JSON.",
    )

    # Subcommand: symbolon-benchmark (Symbolon candidate capture & fusion study)
    symbolon_parser = subparsers.add_parser(
        "symbolon-benchmark",
        help="Benchmark Symbolon discrete code retrieval and candidate fusion against flat baselines.",
    )
    symbolon_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of nearest neighbors to retrieve (default: 3).",
    )
    symbolon_parser.add_argument(
        "--level",
        type=int,
        default=4,
        help="Symbolon discrete codebook level to evaluate (default: 4).",
    )
    symbolon_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save benchmark report JSON.",
    )

    # Subcommand: graph-rollout (EoG repeated rollout diagnostic)
    rollout_parser = subparsers.add_parser(
        "graph-rollout",
        help="Run repeated rollout evaluations over executable graph reasoning cases.",
    )
    rollout_parser.add_argument(
        "--attempts",
        "-n",
        type=int,
        default=5,
        help="Number of rollout attempts per case (default: 5).",
    )
    rollout_parser.add_argument(
        "--policy",
        type=str,
        default="retrieval_baseline",
        choices=["retrieval_baseline", "oracle", "live_hestia"],
        help="Execution policy (default: retrieval_baseline).",
    )
    rollout_parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder-7b",
        help="Model name identifier (default: qwen2.5-coder-7b).",
    )
    rollout_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to export CaseAttemptRecord JSON array for Paideia boundary sampling.",
    )

    return parser


def handle_tasks() -> int:
    """Handle 'dokime tasks' subcommand."""
    tasks = list_task_info()
    if not tasks:
        print("No tasks registered.")
        return 0

    print("Discovered tasks:")
    print(f"{'TASK ID':<25} {'ITEMS':<10} {'DATASET PATH'}")
    print("-" * 65)
    for t in tasks:
        print(f"{t['task_id']:<25} {t['items']:<10} {t['dataset_path']}")
    return 0


def handle_run(args: argparse.Namespace) -> int:
    """Handle 'dokime run' subcommand."""
    # 1. Base config from TOML if specified
    if args.config:
        try:
            config = RunConfig.from_toml_file(args.config)
        except Exception as e:
            print(f"Error loading configuration: {e}", file=sys.stderr)
            return 1
    else:
        config = RunConfig()

    # 2. Merge CLI overrides
    try:
        config = config.merge_cli_args(
            models=args.models,
            tasks=args.tasks,
            prompt_template=args.prompt_template,
            seed=args.seed,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            url=args.url,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"Error in arguments: {e}", file=sys.stderr)
        return 1

    # Auto-detect model from serving layer if none specified (Item 6 & Round 2 Item 1)
    client = HestiaClient(base_url=config.url)
    if not config.models:
        try:
            usable_chat = client.get_auto_selectable_chat_models()
            if usable_chat:
                chosen = usable_chat[0]
                print(f"Auto-selected served chat model: {chosen}")
                config.models = [chosen]
            else:
                all_meta = client.get_models_metadata()
                if all_meta:
                    chat_meta = [m for m in all_meta if m.get("type") in ("chat", None) and m.get("name")]
                    failed_chat = [m["name"] for m in chat_meta if m.get("state") == "failed"]
                    if failed_chat:
                        print(
                            f"Error: No usable chat models available on Hestia. Registered chat model(s) are in 'failed' state: {', '.join(failed_chat)}. "
                            "Hestia cannot load models in failed state.",
                            file=sys.stderr,
                        )
                    else:
                        non_chat = [f"{m['name']} ({m.get('type')})" for m in all_meta]
                        print(
                            f"Error: No chat models available on Hestia. Registered models are non-chat: {', '.join(non_chat)}. "
                            "Cannot auto-select a model for chat evaluation.",
                            file=sys.stderr,
                        )
                else:
                    print("Error: No models registered on Hestia serving layer.", file=sys.stderr)
                return 1
        except Exception as e:
            print(f"Error: No model specified and could not query Hestia: {e}", file=sys.stderr)
            return 1

    print("== Dokime Evaluation Run ==")
    print(f"Models:       {config.models}")
    print(f"Tasks:        {config.tasks}")
    print(f"Seed:         {config.seed}")
    print(f"Temperature:  {config.temperature}")
    if config.prompt_template_path:
        print(f"Prompt Tmpl:  {config.prompt_template_path}")
    print("Executing evaluation matrix...")

    try:
        results = run_evaluation(config=config, client=client)
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        return 1

    print("\n== Run Completed ==")
    print(f"Run ID:       {results['run_id']}")
    print(f"Total Queries:{results['total_queries']}")
    print(f"Artifacts:    {results['output_dir']}")
    print("\nSummary Statistics (Point Estimate & 95% Bootstrap CI, 10k resamples):")
    for cell, cell_info in results["stats"].items():
        st = cell_info["stats"]
        success_q = cell_info.get("successful_queries", st["count"])
        failed_q = cell_info.get("failed_queries", 0)
        total_q = cell_info.get("total_queries", success_q + failed_q)
        if st.get("mean") is None:
            print(
                f"  {cell:<30} Mean: not-measured  "
                f"95% CI: N/A  "
                f"(n={success_q}/{total_q}, failures={failed_q})"
            )
        else:
            mean_str = format_stat_value(st['mean'], sign=False)
            ci_str = format_interval(st['ci_lower'], st['ci_upper'], sign=False)
            print(
                f"  {cell:<30} Mean: {mean_str}  "
                f"95% CI: {ci_str}  "
                f"(n={success_q}/{total_q}, failures={failed_q})"
            )

    return 0


def handle_compare(args: argparse.Namespace) -> int:
    """Handle 'dokime compare' subcommand."""
    is_cross_model = bool(args.cross_model)

    if is_cross_model:
        if args.run_b is not None:
            print("Compare error: --cross-model cannot be combined with a second run ID.", file=sys.stderr)
            return 2
        if args.model is not None:
            print("Compare error: --cross-model cannot be combined with --model.", file=sys.stderr)
            return 2
        if not args.task:
            print("Compare error: --task <task_id> is required for --cross-model compare.", file=sys.stderr)
            return 2
        model_a, model_b = args.cross_model
        if model_a == model_b:
            print(
                f"Compare error: --cross-model requires two distinct model IDs, got '{model_a}' for both.",
                file=sys.stderr,
            )
            return 2
    else:
        if args.run_b is None:
            print(
                "Compare error: Two run IDs are required for ordinary compare (or use --cross-model for single-run compare).",
                file=sys.stderr,
            )
            return 2

    try:
        if is_cross_model:
            model_a, model_b = args.cross_model
            report = run_cross_model_comparison(
                run_ref=args.run_a,
                model_a=model_a,
                model_b=model_b,
                task=args.task,
                base_dir=args.output_dir,
                write_report=True,
            )
        else:
            report = run_comparison(
                run_a_ref=args.run_a,
                run_b_ref=args.run_b,
                model=args.model,
                task=args.task,
                base_dir=args.output_dir,
                write_report=True,
            )
    except CompareError as ce:
        print(f"Compare error: {ce}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error during comparison: {e}", file=sys.stderr)
        return 1

    sig = report["significance"]
    base_st = report["baseline_stats"]
    cand_st = report["candidate_stats"]
    verdict = sig["verdict"]

    if is_cross_model:
        print("== Dokime Cross-Model Compare ==")
        print(f"Run ID:            {report['run_id']}")
        print(f"Model A (Baseline):{report['model_a']}")
        print(f"Model B (Candidate):{report['model_b']}")
        print(f"Task:              {report['task_id']}")
        print(f"Paired Successes:  {report['paired_success_count']}")
        print(f"Model A Failures:  {report['failures_model_a']}")
        print(f"Model B Failures:  {report['failures_model_b']}")
        print("-" * 55)
        print(f"Model A Score:     {format_stat_value(base_st['mean'], sign=False)}  95% CI: {format_interval(base_st['ci_lower'], base_st['ci_upper'], sign=False)}")
        print(f"Model B Score:     {format_stat_value(cand_st['mean'], sign=False)}  95% CI: {format_interval(cand_st['ci_lower'], cand_st['ci_upper'], sign=False)}")
        print(f"Delta (B - A):     {format_stat_value(sig['delta'], sign=True)}  95% CI: {format_interval(sig['ci_lower'], sig['ci_upper'], sign=True)}")
        print(f"p-value:           {format_stat_value(sig['p_value'], sign=False)}")
        print("-" * 55)
    else:
        print("== Dokime Regression Compare ==")
        print(f"Baseline Run (A):  {report['run_baseline']}")
        print(f"Candidate Run (B): {report['run_candidate']}")
        print(f"Model Scoped:      {report['model_id']}")
        if report["task_filter"]:
            print(f"Task Scoped:       {report['task_filter']}")
        print(f"Paired Successes:  {report['paired_success_count']}")
        print(f"Baseline Failures: {report['failures_baseline']}")
        print(f"Candidate Failures:{report['failures_candidate']}")
        print(f"Unpaired Baseline: {report.get('unpaired_baseline', 0)} (baseline-only)")
        print(f"Unpaired Candidate: {report.get('unpaired_candidate', 0)} (candidate-only)")
        print("-" * 55)
        print(f"Baseline Score:    {format_stat_value(base_st['mean'], sign=False)}  95% CI: {format_interval(base_st['ci_lower'], base_st['ci_upper'], sign=False)}")
        print(f"Candidate Score:   {format_stat_value(cand_st['mean'], sign=False)}  95% CI: {format_interval(cand_st['ci_lower'], cand_st['ci_upper'], sign=False)}")
        print(f"Delta (B - A):     {format_stat_value(sig['delta'], sign=True)}  95% CI: {format_interval(sig['ci_lower'], sig['ci_upper'], sign=True)}")
        print(f"p-value:           {format_stat_value(sig['p_value'], sign=False)}")
        print("-" * 55)

    if verdict == "IMPROVEMENT":
        flag = "[IMPROVEMENT] CI excludes zero (strictly positive)"
    elif verdict == "REGRESSION":
        flag = "[REGRESSION] CI excludes zero (strictly negative)"
    else:
        flag = "[NO SIGNIFICANT CHANGE] CI includes zero"

    print(f"Verdict:           {flag}")
    diffs = report["query_diffs"]
    print(f"Query Breakdown:   {diffs['improved_count']} improved, {diffs['regressed_count']} regressed, {diffs['unchanged_count']} unchanged")

    if "report_path" in report:
        print(f"Report written:    {report['report_path']}")

    return 0


def handle_explain(args: argparse.Namespace) -> int:
    """Handle 'dokime explain' subcommand using Sufficient Input Subsets."""
    from dokime.explainers.sis import (
        SISCollectionResult,
        SISResult,
        explain_text_sufficiency,
    )

    # Read input text
    text: str | None = args.text
    if args.file:
        path = Path(args.file)
        if not path.is_file():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8")

    if not text or not text.strip():
        print("Error: must provide non-empty input via --text or --file.", file=sys.stderr)
        return 1

    # Define score function
    if args.target_keyword:
        target = args.target_keyword.lower()

        def score_fn(cand_text: str) -> float:
            low = cand_text.lower()
            return 1.0 if target in low else 0.0

    else:
        # Default length/density ratio
        initial_len = max(len(text), 1)

        def score_fn(cand_text: str) -> float:
            return min(len(cand_text) / initial_len, 1.0)

    try:
        res = explain_text_sufficiency(
            text=text,
            score_fn=score_fn,
            tau=args.tau,
            unit=args.unit,
            find_disjoint=args.disjoint,
            max_subsets=args.max_subsets,
        )
    except Exception as e:
        print(f"Error computing Sufficient Input Subsets: {e}", file=sys.stderr)
        return 1

    print("== Dokime Sufficient Input Subsets (SIS) ==")
    if isinstance(res, SISCollectionResult):
        print(f"Threshold (tau):    {res.threshold}")
        print(f"Total Input Items:  {res.total_items} ({args.unit}s)")
        print(f"Disjoint Subsets:   {res.total_subsets}")
        print(f"Residual Score:     {res.residual_score}")
        print("-" * 55)
        for i, sub in enumerate(res.subsets, 1):
            print(f"Subset #{i}: {len(sub.subset_indices)}/{sub.total_items} items (Score: {sub.final_score:.4f}, Reduction: {sub.reduction_ratio:.1%})")
            print(f"  Indices: {sub.subset_indices}")
            print(f"  Text:    \"{sub.reconstructed_text}\"")
    else:
        print(f"Threshold (tau):    {res.threshold}")
        print(f"Total Input Items:  {res.total_items} ({args.unit}s)")
        print(f"Subset Size:        {len(res.subset_indices)} ({args.unit}s)")
        print(f"Is Sufficient:      {res.is_sufficient}")
        print(f"Initial Score:      {res.initial_score:.4f}")
        print(f"Final Score:        {res.final_score:.4f}")
        print(f"Reduction Ratio:    {res.reduction_ratio:.1%}")
        print("-" * 55)
        print("Sufficient Input Subset:")
        print(f"  Indices: {res.subset_indices}")
        print(f"  Text:    \"{res.reconstructed_text}\"")

    return 0


def handle_ann_benchmark(args: argparse.Namespace) -> int:
    """Handle 'dokime ann-benchmark' subcommand comparing vector retrieval backends."""
    import json
    from dokime.tasks.ann_benchmark import ANNBenchmarkConfig, ANNBenchmarkSuite

    config = ANNBenchmarkConfig(
        num_documents=args.num_docs,
        d_model=args.dim,
        num_queries=args.num_queries,
        top_k=args.top_k,
        seed=args.seed,
    )

    print(f"[*] Running ANN Vector Benchmark (Docs={args.num_docs}, Dim={args.dim}, Queries={args.num_queries}, Top-K={args.top_k})...")
    suite = ANNBenchmarkSuite(config=config)
    report = suite.run()

    print()
    print(report.format_table())
    print()

    if args.output:
        out_dict = {
            "num_documents": report.num_documents,
            "d_model": report.d_model,
            "num_queries": report.num_queries,
            "top_k": report.top_k,
            "algorithms": {
                name: {
                    "algorithm_name": r.algorithm_name,
                    "build_time_s": r.build_time_s,
                    "mean_latency_ms": r.mean_latency_ms,
                    "p50_latency_ms": r.p50_latency_ms,
                    "p95_latency_ms": r.p95_latency_ms,
                    "p99_latency_ms": r.p99_latency_ms,
                    "qps": r.qps,
                    "recall_at_1": r.recall_at_1,
                    "recall_at_k": r.recall_at_k,
                    "speedup_vs_exact": r.speedup_vs_exact,
                }
                for name, r in report.algorithms.items()
            },
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)
        print(f"[+] Report saved to {out_path}")

    return 0


def handle_symbolon_benchmark(args: argparse.Namespace) -> int:
    """Handle 'dokime symbolon-benchmark' subcommand."""
    from dokime.tasks.symbolon_benchmark import SymbolonRetrievalBenchmarkSuite

    print(f"[*] Running Symbolon Retrieval Benchmark (Top-K={args.top_k}, Level={args.level})...\n")
    suite = SymbolonRetrievalBenchmarkSuite(top_k=args.top_k, level=args.level)
    report = suite.run_benchmark()

    print(report.to_markdown())

    if args.output:
        out_dict = {
            "total_queries": report.total_queries,
            "top_k": report.top_k,
            "level_evaluated": report.level_evaluated,
            "decision_verdict": report.decision_verdict,
            "decision_rationale": report.decision_rationale,
            "arms": {
                name: {
                    "arm_name": arm.arm_name,
                    "mean_recall": arm.mean_recall,
                    "contradiction_retention": arm.contradiction_retention,
                    "proxy_agreement": arm.proxy_agreement,
                    "candidate_expansion": arm.candidate_expansion,
                    "total_dropped_gold": arm.total_dropped_gold,
                    "delta_vs_baseline": arm.delta_vs_baseline,
                }
                for name, arm in report.arms.items()
            },
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)
        print(f"\n[+] Report saved to {out_path}")

    return 0


def handle_graph_rollout(args: argparse.Namespace) -> int:
    """Handle 'dokime graph-rollout' subcommand."""
    from dokime.tasks.executable_graph import ExecutableGraphRolloutSuite

    print(f"[*] Running Executable Graph Rollouts (Policy={args.policy}, Attempts={args.attempts})...\n")
    client = None
    if args.policy == "live_hestia":
        client = HestiaClient()

    suite = ExecutableGraphRolloutSuite()
    records, report = suite.run_rollouts(
        num_attempts=args.attempts,
        policy=args.policy,
        model_name=args.model,
        client=client,
    )

    print(report.to_markdown())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"\n[+] Saved {len(records)} attempt records to {out_path}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "tasks":
        return handle_tasks()
    elif args.subcommand == "run":
        return handle_run(args)
    elif args.subcommand == "compare":
        return handle_compare(args)
    elif args.subcommand == "explain":
        return handle_explain(args)
    elif args.subcommand == "ann-benchmark":
        return handle_ann_benchmark(args)
    elif args.subcommand == "symbolon-benchmark":
        return handle_symbolon_benchmark(args)
    elif args.subcommand == "graph-rollout":
        return handle_graph_rollout(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
