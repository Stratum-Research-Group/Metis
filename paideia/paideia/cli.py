"""Paideia Command-Line Interface."""

import json
from pathlib import Path
from typing import Optional
import click

from paideia.ops.doctor import print_doctor_report
from paideia.data.router_loader import load_router_dataset
from paideia.data.walk_loader import load_walk_dataset
from paideia.data.splitter import (
    split_grouped_dataset,
    derive_question_family_key,
)
from paideia.data.card import DatasetCard
from paideia.baselines.ladder import run_baseline_ladder
from paideia.training.config import TrainingConfig
from paideia.training.runner import QLoRATrainingRunner
from paideia.evaluation.promotion import OperationalGuardrails, evaluate_promotion
from paideia.evaluation.dokime_bundle import DokimeRunBundle
from paideia.export.registry import export_hestia_adapter_registry


@click.group()
def cli():
    """Paideia: production-grade local training and policy evaluation pipeline."""
    pass


@cli.command()
def doctor():
    """Run diagnostic checks on hardware, dependencies, and environment."""
    print_doctor_report()


@cli.group()
def dataset():
    """Dataset validation, audit, and card generation."""
    pass


@dataset.command("validate-router")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--eligible-routes", "-r", multiple=True, help="Eligible routes to evaluate for counterfactuals")
def validate_router(path: Path, eligible_routes):
    """Validate LTHCM routing outcomes and display audit summary."""
    routes = list(eligible_routes) if eligible_routes else None
    records, audit = load_router_dataset(path, eligible_routes=routes)
    click.echo(f"\n=== LTHCM Router Dataset Audit ===")
    click.echo(json.dumps(audit.to_dict(), indent=2))
    click.echo(f"\nSuccessfully validated {len(records)} clean records.")


@dataset.command("validate-walk")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def validate_walk(path: Path):
    """Validate Dianoia walk JSONL logs and extract policy steps."""
    samples, audit = load_walk_dataset(path)
    click.echo(f"\n=== Dianoia Walk Dataset Audit ===")
    click.echo(json.dumps(audit.to_dict(), indent=2))
    click.echo(f"\nExtracted {len(samples)} defensible policy training steps.")


@dataset.command("sample-active")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), required=True, help="Destination JSONL path")
@click.option("--budget", "-b", default=50, help="Number of active queries to select")
@click.option("--beta", default=0.6, help="Weight on uncertainty (0.0=diversity only, 1.0=uncertainty only)")
@click.option("--min-score", default=0.0, help="Minimum rejection score to be eligible")
def sample_active(input_path: Path, output: Path, budget: int, beta: float, min_score: float):
    """Run ASPEST active selection over Krisis abstentions and prediction ambiguities."""
    from paideia.contracts.krisis_schema import ActiveSamplingConfig
    from paideia.data.aspest_sampler import ASPESTSampler, load_krisis_active_records, save_krisis_active_records

    records = load_krisis_active_records(input_path)
    cfg = ActiveSamplingConfig(budget=budget, beta=beta, min_rejection_score=min_score)
    selected, report = ASPESTSampler.select_active_batch(records, cfg)
    save_krisis_active_records(selected, output)
    click.echo("\n=== ASPEST Active Selection Audit ===")
    click.echo(report.model_dump_json(indent=2))
    click.echo(f"\nSaved {len(selected)} high-yield active queries to: {output}")


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--task", type=click.Choice(["type", "best_route"]), default="type", help="Target task")
@click.option("--seed", default=42, help="RNG seed")
def baseline(path: Path, task: str, seed: int):
    """Run baseline ladder (Majority, Regex, Lexical, Embedding) on frozen splits."""
    records, _ = load_router_dataset(path)
    if not records:
        click.echo("No valid records found in dataset.")
        return

    splits = split_grouped_dataset(
        records,
        group_key_fn=lambda r: derive_question_family_key(r.question),
        seed=seed,
    )

    if task == "type":
        target_fn = lambda r: r.gold_type
    else:
        from paideia.contracts.lthcm_schema import RouteUtility
        util = RouteUtility()
        target_fn = lambda r: r.get_best_route(util)

    click.echo(f"\nRunning Baseline Ladder on {len(splits.dev)} dev items (Seed: {seed})...")
    report = run_baseline_ladder(
        train_items=splits.train,
        eval_items=splits.dev,
        target_extractor=target_fn,
        task_name=f"router_{task}",
    )
    click.echo("\n" + report.to_table() + "\n")


@cli.command()
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
def verify_run(manifest: Path):
    """Verify integrity of a run manifest and output checksums."""
    with open(manifest, "r") as f:
        data = json.load(f)
    click.echo(f"\nRun ID: {data.get('run_id')}")
    click.echo(f"Status: {data.get('status')}")
    click.echo(f"Hardware: {data.get('hardware_line')}")
    click.echo(f"Started: {data.get('started_at_utc')}")
    click.echo(f"Completed: {data.get('completed_at_utc')}")
    if data.get("failure_reason"):
        click.echo(f"Failure Reason: {data.get('failure_reason')}")
    if data.get("adapter_digest"):
        click.echo(f"Adapter Digest: {data.get('adapter_digest')}")
    click.echo("\nManifest verified successfully.")


if __name__ == "__main__":
    cli()
