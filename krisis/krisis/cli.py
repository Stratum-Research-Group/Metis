"""Command-line interface for Krisis answer selection engine."""

from pathlib import Path
import json
import sys
import click
import jsonschema
from tabulate import tabulate

from krisis.audit.replay import FixtureReplayRunner
from krisis.engine.selector import KrisisSelector

DEFAULT_POLICY_PATH = Path("krisis-policy-0.1.json")
DEFAULT_SCHEMA_PATH = Path("krisis-policy.schema.json")
DEFAULT_FIXTURES_PATH = Path("krisis-fixtures-0.1.json")


@click.group()
@click.version_option(package_name="krisis")
def cli() -> None:
    """Krisis: Candidate answer selector and eligibility gating engine."""
    pass


@cli.command("validate-schema")
@click.option("--policy", "policy_path", type=click.Path(exists=True, path_type=Path), default=DEFAULT_POLICY_PATH)
@click.option("--schema", "schema_path", type=click.Path(exists=True, path_type=Path), default=DEFAULT_SCHEMA_PATH)
def validate_schema(policy_path: Path, schema_path: Path) -> None:
    """Validate a policy JSON file against its JSON Schema."""
    click.echo(f"Validating {policy_path} against {schema_path}...")
    try:
        policy_data = json.loads(policy_path.read_text(encoding="utf-8"))
        schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=policy_data, schema=schema_data)
        click.secho(f"\n[+] SUCCESS: {policy_path.name} is fully valid according to {schema_path.name}!", fg="green", bold=True)
    except jsonschema.ValidationError as exc:
        click.secho(f"\n[-] Schema Validation FAILED: {exc.message}", fg="red", bold=True)
        click.echo(f"Path in document: {' -> '.join(str(p) for p in exc.path)}")
        sys.exit(1)
    except Exception as exc:
        click.secho(f"\n[-] Error loading files: {exc}", fg="red")
        sys.exit(1)


@cli.command("replay")
@click.option("--fixtures", "fixtures_path", type=click.Path(exists=True, path_type=Path), default=DEFAULT_FIXTURES_PATH)
@click.option("--policy", "policy_path", type=click.Path(exists=True, path_type=Path), default=DEFAULT_POLICY_PATH)
def replay(fixtures_path: Path, policy_path: Path) -> None:
    """Replay pure-policy test fixtures and verify exact byte-and-score determinism."""
    click.echo(f"Replaying fixtures from {fixtures_path} using policy {policy_path}...\n")
    runner = FixtureReplayRunner(policy_path=policy_path, fixtures_path=fixtures_path)
    summary = runner.run_all()

    table_data = []
    for r in summary["results"]:
        status_str = click.style("PASS", fg="green", bold=True) if r["passed"] else click.style("FAIL", fg="red", bold=True)
        diff_str = "; ".join(r["diffs"]) if r["diffs"] else "-"
        table_data.append([r["id"], r["name"][:38], status_str, diff_str[:40]])

    click.echo(tabulate(table_data, headers=["ID", "Fixture Name", "Status", "Diffs / Mismatches"], tablefmt="rounded"))

    click.echo(f"\nSummary: {summary['passed_count']} / {summary['total']} fixtures passed.")
    if summary["all_passed"]:
        click.secho("[+] 100% REPLAY SUCCESS: All fixtures verified against policy manifest!", fg="green", bold=True)
    else:
        click.secho(f"[-] FAILED: {summary['failed_count']} fixture(s) had discrepancies.", fg="red", bold=True)
        sys.exit(1)


@cli.command("evaluate")
@click.argument("request_json", type=click.Path(exists=True, path_type=Path))
@click.option("--policy", "policy_path", type=click.Path(exists=True, path_type=Path), default=DEFAULT_POLICY_PATH)
def evaluate(request_json: Path, policy_path: Path) -> None:
    """Evaluate candidate answers from a request JSON file."""
    req_data = json.loads(request_json.read_text(encoding="utf-8"))
    selector = KrisisSelector(policy_path)
    decision = selector.evaluate(req_data)

    click.echo(json.dumps(decision.model_dump(exclude_none=True), indent=2))


@cli.command("inspect-policy")
@click.option("--policy", "policy_path", type=click.Path(exists=True, path_type=Path), default=DEFAULT_POLICY_PATH)
def inspect_policy(policy_path: Path) -> None:
    """Inspect rules, profiles, and thresholds of a policy manifest."""
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    click.secho(f"\nPolicy Manifest: {data.get('manifest_id')} (Status: {data.get('status')})", bold=True)

    # Profiles
    profiles = data.get("profiles", {})
    prof_table = []
    for pid, p in profiles.items():
        tau = p.get("tau") or "-"
        per_claim = p.get("per_claim_min") or "-"
        prof_table.append([pid, p.get("kind"), tau, per_claim, ", ".join(p.get("hard_gates", []))[:35]])
    click.echo("\n--- Profiles ---")
    click.echo(tabulate(prof_table, headers=["Profile ID", "Kind", "Tau", "Per-Claim Min", "Hard Gates"], tablefmt="rounded"))

    # Reason Precedence
    prec = data.get("reason_precedence", [])
    click.echo(f"\n--- Terminal Reason Precedence ({len(prec)} reasons) ---")
    click.echo(" > ".join(prec))


if __name__ == "__main__":
    cli()
