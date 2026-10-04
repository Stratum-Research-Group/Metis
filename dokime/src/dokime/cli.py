"""Command line interface for Dokime evaluation harness."""

from __future__ import annotations

import argparse
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
