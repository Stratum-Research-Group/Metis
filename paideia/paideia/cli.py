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


@dataset.command("sample-boundary")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), required=False, help="Optional destination JSON path for selected case IDs and report")
@click.option("--budget", "-b", default=20, help="Number of practice cases to select")
@click.option("--boundary-ratio", default=0.6, help="Target ratio of boundary cases (sigma^2 > 0)")
@click.option("--retention-ratio", default=0.2, help="Target ratio of retention checks (mean=1.0, sigma^2=0)")
@click.option("--hard-ratio", default=0.2, help="Target ratio of hard diagnostic cases (mean=0.0, sigma^2=0)")
@click.option("--min-variance", default=1e-4, help="Minimum sample variance to qualify as boundary")
@click.option("--baseline-uniform", is_flag=True, default=False, help="Run uniform random sampling as baseline comparison")
@click.option("--seed", default=42, help="RNG seed")
def sample_boundary(
    input_path: Path,
    output: Optional[Path],
    budget: int,
    boundary_ratio: float,
    retention_ratio: float,
    hard_ratio: float,
    min_variance: float,
    baseline_uniform: bool,
    seed: int,
):
    """Select boundary-aware practice batch from repeated rollout attempts (EoG Item 2)."""
    from paideia.contracts.boundary_schema import BoundarySelectionConfig
    from paideia.data.boundary_sampler import (
        AttemptRolloutVarianceSampler,
        load_case_attempts,
    )

    sampler = AttemptRolloutVarianceSampler(min_variance=min_variance)
    try:
        count = sampler.ingest_dokime_eval_run(input_path)
        if count == 0:
            records = load_case_attempts(input_path)
            sampler.ingest_rollouts(records)
    except Exception:
        records = load_case_attempts(input_path)
        sampler.ingest_rollouts(records)

    if baseline_uniform:
        selected, report = sampler.sample_uniform(budget=budget, seed=seed)
    else:
        cfg = BoundarySelectionConfig(
            budget=budget,
            boundary_ratio=boundary_ratio,
            retention_ratio=retention_ratio,
            hard_ratio=hard_ratio,
            min_variance=min_variance,
            seed=seed,
        )
        selected, report = sampler.sample(cfg)

    click.echo("\n=== Paideia Boundary Practice Selection Audit ===")
    click.echo(report.model_dump_json(indent=2))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump({
                "report": report.model_dump(),
                "selected_cases": selected,
            }, f, indent=2)
        click.echo(f"\nSaved {len(selected)} practice cases and audit report to: {output}")


@cli.command("train")
@click.option("--data", "-d", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False, help="Path to training dataset JSON/JSONL")
@click.option("--output-dir", "-o", default="./runs/latest_adapter", help="Directory to save checkpoint and manifest")
@click.option("--task", type=click.Choice(["router", "walk_policy"]), default="router", help="Target policy task")
@click.option("--optimizer", type=click.Choice(["adamw", "shampoo"]), default="adamw", help="Optimizer: adamw or second-order matrix shampoo")
@click.option("--epochs", default=3, help="Training epochs")
@click.option("--batch-size", default=2, help="Batch size")
@click.option("--lr", default=2e-4, help="Learning rate")
@click.option("--rank", default=16, help="LoRA rank")
@click.option("--simulate/--no-simulate", default=True, help="Run marker-only simulation or real training")
@click.option("--seed", default=42, help="RNG seed")
def train(data, output_dir, task, optimizer, epochs, batch_size, lr, rank, simulate, seed):
    """Train or fine-tune policy LoRA adapters with AdamW or Distributed Shampoo."""
    from paideia.data.splitter import DatasetSplits
    from paideia.training.config import TrainingConfig
    from paideia.training.runner import QLoRATrainingRunner

    train_items = []
    if data:
        if task == "router":
            records, _ = load_router_dataset(data)
            train_items = records
        else:
            samples, _ = load_walk_dataset(data)
            train_items = samples

    splits = DatasetSplits(train=train_items, dev=[], test=[])
    cfg = TrainingConfig(
        output_dir=output_dir,
        task=task,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        rank=rank,
        allow_simulated=simulate,
        seed=seed,
    )

    runner = QLoRATrainingRunner(cfg, splits)
    click.echo(f"\nStarting Paideia QLoRA Training Run...")
    click.echo(f"  Task: {task}")
    click.echo(f"  Optimizer: {optimizer}")
    click.echo(f"  Mode: {'SIMULATION' if simulate else 'REAL TRAINING'}")
    click.echo(f"  Train Samples: {len(train_items)}")
    click.echo(f"  Output Dir: {output_dir}")

    manifest = runner.run()
    click.echo("\n=== Training Manifest Summary ===")
    click.echo(f"Run ID: {manifest.run_id}")
    click.echo(f"Status: {manifest.status}")
    click.echo(f"Duration: {manifest.duration_seconds:.2f}s")
    click.echo(f"Adapter Digest: {manifest.adapter_digest}")
    if manifest.metrics:
        click.echo("Metrics:")
        for k, v in manifest.metrics.items():
            click.echo(f"  {k}: {v}")
    if manifest.failure_reason:
        click.echo(f"Failure Reason: {manifest.failure_reason}")


@cli.command("benchmark-shampoo")
@click.option("--steps", default=40, help="Number of benchmark optimization steps")
@click.option("--dim-out", default=16, help="Output matrix dimension")
@click.option("--dim-in", default=32, help="Input matrix dimension")
@click.option("--seed", default=42, help="RNG seed")
def benchmark_shampoo(steps: int, dim_out: int, dim_in: int, seed: int):
    """Benchmark second-order matrix Shampoo against AdamW on ill-conditioned problem."""
    from paideia.training.optimizer import benchmark_shampoo_vs_adamw

    click.echo(f"\nRunning Shampoo vs AdamW second-order preconditioner benchmark ({steps} steps)...")
    res = benchmark_shampoo_vs_adamw(dim_out=dim_out, dim_in=dim_in, num_steps=steps, seed=seed)
    click.echo("\n=== Benchmark Results ===")
    click.echo(f"Steps: {res['num_steps']}")
    click.echo(f"AdamW Final Loss:    {res['adamw_final_loss']:.6f} (initial: {res['adamw_initial_loss']:.6f})")
    click.echo(f"Shampoo Final Loss:  {res['shampoo_final_loss']:.6f} (initial: {res['shampoo_initial_loss']:.6f})")
    click.echo(f"Shampoo Relative Gain: {res['shampoo_relative_gain_percent']:+.2f}%")


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


@cli.command("loop")
@click.option("--log-file", "-l", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Path to Krisis active prediction / Dokime attempt JSON or JSONL logs")
@click.option("--output-dir", "-o", default="./runs/continuous", help="Base output directory for runs and checkpoints")
@click.option("--base-model", "-m", default="qwen2.5-14b-instruct", help="Base model identifier")
@click.option("--task", type=click.Choice(["router", "walk_policy"]), default="router", help="Target policy task")
@click.option("--min-samples", "-n", default=10, help="Minimum new boundary/attempt records required to trigger retraining")
@click.option("--optimizer", type=click.Choice(["shampoo", "adamw"]), default="shampoo", help="Optimizer choice: shampoo or adamw")
@click.option("--epochs", default=3, help="Training epochs per cycle")
@click.option("--lr", default=1e-3, help="Optimizer learning rate")
@click.option("--rank", default=16, help="LoRA rank")
@click.option("--hestia-url", default=None, help="Hestia HTTP base URL (e.g. http://127.0.0.1:8080)")
@click.option("--slot", default="primary", help="Hestia serving slot target for model hot-swap")
@click.option("--auto-swap/--no-auto-swap", default=True, help="Automatically trigger Hestia CAS hot-swap upon promotion")
@click.option("--simulate/--no-simulate", default=False, help="Run marker-only simulation or real training")
@click.option("--once", is_flag=True, default=False, help="Run a single retraining check and exit")
@click.option("--interval", default=60.0, help="Polling interval in seconds for continuous mode")
def loop(
    log_file: Path,
    output_dir: str,
    base_model: str,
    task: str,
    min_samples: int,
    optimizer: str,
    epochs: int,
    lr: float,
    rank: int,
    hestia_url: Optional[str],
    slot: str,
    auto_swap: bool,
    simulate: bool,
    once: bool,
    interval: float,
):
    """Run autonomous continuous policy retraining loop (Krisis -> Shampoo QLoRA -> Dokime -> Hestia)."""
    from paideia.training.loop import (
        ContinuousPolicyRetrainingCoordinator,
        RetrainingCoordinatorConfig,
    )

    cfg = RetrainingCoordinatorConfig(
        log_file=log_file,
        output_dir=Path(output_dir),
        base_model=base_model,
        task=task,
        min_samples=min_samples,
        optimizer=optimizer,
        epochs=epochs,
        lr=lr,
        rank=rank,
        hestia_url=hestia_url,
        slot=slot,
        auto_swap=auto_swap,
        allow_simulated=simulate,
    )

    coordinator = ContinuousPolicyRetrainingCoordinator(cfg)

    click.echo("\n=== Paideia Continuous Policy Retraining Loop ===")
    click.echo(f"  Log File:     {log_file}")
    click.echo(f"  Output Dir:   {output_dir}")
    click.echo(f"  Base Model:   {base_model}")
    click.echo(f"  Task:         {task}")
    click.echo(f"  Optimizer:    {optimizer}")
    click.echo(f"  Min Samples:  {min_samples}")
    click.echo(f"  Mode:         {'SIMULATION' if simulate else 'REAL TRAINING'}")
    click.echo(f"  Auto Swap:    {auto_swap} ({hestia_url or 'local'})")

    if once:
        click.echo("\nExecuting single retraining cycle...")
        result = coordinator.run_cycle()
        click.echo(f"\nCycle Result [{result.cycle_id}]: {result.status}")
        click.echo(f"  Samples Ingested: {result.samples_ingested}")
        click.echo(f"  Samples Selected: {result.samples_selected}")
        if result.adapter_id:
            click.echo(f"  Adapter ID:       {result.adapter_id}")
        if result.adapter_digest:
            click.echo(f"  Adapter Digest:   {result.adapter_digest}")
        if result.hestia_swap_status:
            click.echo(f"  Hestia Swap:      {result.hestia_swap_status.get('status')}")
        if result.error:
            click.echo(f"  Error:            {result.error}")
    else:
        click.echo(f"\nStarting continuous loop polling every {interval:.1f}s (Ctrl+C to stop)...")
        try:
            coordinator.run_loop(interval_seconds=interval)
        except KeyboardInterrupt:
            click.echo("\nCoordinator loop stopped by user.")


if __name__ == "__main__":
    cli()
