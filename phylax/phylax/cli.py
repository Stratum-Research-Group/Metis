"""Phylax Command-Line Interface."""

import json
from pathlib import Path
import sys
import click
from tabulate import tabulate

from phylax.approvals.manager import ApprovalManager
from phylax.audit.logger import AuditLogger
from phylax.audit.verifier import AuditVerifier
from phylax.contracts.enums import Effect, ResourceClass, Verb
from phylax.engine.manager import PolicyManager
from phylax.gateway.gateway import GuardedGateway
from phylax.linter.linter import LintSeverity, PolicyLinter

DEFAULT_POLICY_PATH = Path("policies/default_policy.yaml")
DEFAULT_AUDIT_PATH = Path(".phylax/audit.jsonl")
DEFAULT_APPROVAL_PATH = Path(".phylax/approvals.json")


@click.group()
@click.version_option(package_name="phylax")
def cli() -> None:
    """Phylax: Local policy and guarded capability gateway."""
    pass


@cli.command("validate")
@click.argument("policy_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=DEFAULT_POLICY_PATH)
def validate_policy(policy_path: Path) -> None:
    """Validate and lint a Phylax policy YAML file."""
    click.echo(f"Validating policy file: {policy_path}")
    linter = PolicyLinter(workspace_root=Path.cwd())
    policy, issues = linter.lint_file(policy_path)

    has_errors = any(i.severity == LintSeverity.ERROR for i in issues)
    has_warnings = any(i.severity == LintSeverity.WARNING for i in issues)

    if issues:
        table_data = [
            [i.severity.value, i.rule_id or "-", i.field or "-", i.message]
            for i in issues
        ]
        click.echo(tabulate(table_data, headers=["Severity", "Rule ID", "Field", "Message"], tablefmt="rounded"))

    if has_errors:
        click.secho(f"\nPolicy validation FAILED with {sum(1 for i in issues if i.severity == LintSeverity.ERROR)} error(s).", fg="red", bold=True)
        sys.exit(1)
    elif has_warnings:
        click.secho(f"\nPolicy is valid with {sum(1 for i in issues if i.severity == LintSeverity.WARNING)} warning(s).", fg="yellow")
    else:
        click.secho(f"\nPolicy is perfectly valid! (Version: {policy.version}, Rules: {len(policy.rules)})", fg="green", bold=True)


@cli.command("doctor")
@click.option("--policy", "policy_path", type=click.Path(path_type=Path), default=DEFAULT_POLICY_PATH)
@click.option("--audit-log", "audit_path", type=click.Path(path_type=Path), default=DEFAULT_AUDIT_PATH)
def doctor(policy_path: Path, audit_path: Path) -> None:
    """Check health of Phylax policy, directories, and audit chain."""
    click.secho("Running Phylax Health Check...", bold=True)
    all_ok = True

    # 1. Policy check
    if not policy_path.exists():
        click.secho(f"[-] Policy file missing: {policy_path}", fg="red")
        all_ok = False
    else:
        linter = PolicyLinter()
        policy, issues = linter.lint_file(policy_path)
        errors = [i for i in issues if i.severity == LintSeverity.ERROR]
        if errors:
            click.secho(f"[-] Policy file has {len(errors)} error(s)", fg="red")
            all_ok = False
        else:
            click.secho(f"[+] Policy file healthy: {policy_path} ({len(policy.rules)} rules)", fg="green")

    # 2. Audit log check
    if not audit_path.exists():
        click.secho(f"[i] Audit log not initialized yet: {audit_path}", fg="cyan")
    else:
        verifier = AuditVerifier(audit_path)
        res = verifier.verify()
        if res.is_valid:
            click.secho(f"[+] Audit log verified: {res.total_records} records, chain intact", fg="green")
        else:
            click.secho(f"[-] Audit log corrupted at record {res.broken_record_id}: {res.error_message}", fg="red")
            all_ok = False

    # 3. Submodule check
    for sub in ["LTHCM", "dianoia"]:
        sub_dir = Path(f"../{sub}")
        if sub_dir.exists() and (sub_dir / ".git").exists():
            click.secho(f"[+] Workspace submodule present (read-only): {sub}", fg="green")
        else:
            click.secho(f"[i] Submodule {sub} located at workspace root", fg="cyan")

    if all_ok:
        click.secho("\nPhylax environment is HEALTHY and ready for enforcement.", fg="green", bold=True)
    else:
        click.secho("\nPhylax health check detected ISSUES.", fg="red", bold=True)
        sys.exit(1)


@cli.command("decide")
@click.option("--subject", required=True, help="Requesting component (e.g. kybernetes, stoma)")
@click.option("--verb", type=click.Choice(["read", "write", "execute", "send"]), required=True)
@click.option("--resource-class", type=click.Choice(["file", "tool", "network", "system"]), required=True)
@click.option("--target", required=True, help="Target path, tool name, or destination")
@click.option("--policy", "policy_path", type=click.Path(exists=True, path_type=Path), default=DEFAULT_POLICY_PATH)
@click.option("--audit-log", "audit_path", type=click.Path(path_type=Path), default=DEFAULT_AUDIT_PATH)
def decide(
    subject: str,
    verb: str,
    resource_class: str,
    target: str,
    policy_path: Path,
    audit_path: Path,
) -> None:
    """Evaluate an authorization decision and record it to the audit log."""
    pm = PolicyManager(policy_path)
    audit = AuditLogger(audit_path)
    gateway = GuardedGateway(pm, audit)

    auth_rec = gateway.decide(
        subject=subject,
        verb=Verb(verb),
        resource_class=ResourceClass(resource_class),
        target=target,
    )

    color = "green" if auth_rec.effect == Effect.ALLOW else ("yellow" if auth_rec.effect == Effect.ASK else "red")
    click.secho(f"\nDECISION: {auth_rec.effect.value.upper()}", fg=color, bold=True)
    click.echo(f"  Decision ID:        {auth_rec.decision_id}")
    click.echo(f"  Subject:            {auth_rec.subject}")
    click.echo(f"  Verb:               {auth_rec.verb.value}")
    click.echo(f"  Canonical Resource: {auth_rec.canonical_resource}")
    click.echo(f"  Matched Rule:       {auth_rec.matched_rule_id or 'default_deny'}")
    click.echo(f"  Reason:             {auth_rec.reason}")
    if auth_rec.approval_id:
        click.echo(f"  Pending Approval:   {auth_rec.approval_id}")


@cli.group("audit")
def audit_group() -> None:
    """Audit log inspection and cryptographic verification."""
    pass


@audit_group.command("verify")
@click.option("--log-file", type=click.Path(path_type=Path), default=DEFAULT_AUDIT_PATH)
def audit_verify(log_file: Path) -> None:
    """Verify cryptographic integrity of audit log chain."""
    verifier = AuditVerifier(log_file)
    res = verifier.verify()
    if res.is_valid:
        click.secho(f"Audit log integrity VERIFIED. {res.total_records} records checked.", fg="green", bold=True)
        click.echo(f"Head Hash: {res.head_hash}")
    else:
        click.secho(f"Audit chain BROKEN at record {res.broken_record_id}!", fg="red", bold=True)
        click.echo(f"Error: {res.error_message}")
        sys.exit(1)


@audit_group.command("list")
@click.option("--log-file", type=click.Path(path_type=Path), default=DEFAULT_AUDIT_PATH)
@click.option("--limit", type=int, default=20)
@click.option("--subject", help="Filter by subject")
@click.option("--effect", type=click.Choice(["allow", "deny", "ask"]), help="Filter by effect")
def audit_list(log_file: Path, limit: int, subject: str | None, effect: str | None) -> None:
    """List recent audit entries with optional filters."""
    if not log_file.exists():
        click.echo("No audit records found.")
        return

    records = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                try:
                    records.append(json.loads(stripped))
                except Exception:
                    pass

    if subject:
        records = [r for r in records if r.get("subject") == subject]
    if effect:
        records = [r for r in records if r.get("effect") == effect]

    records = records[-limit:]

    table_data = []
    for r in records:
        ts = r.get("timestamp_utc", "")[:19]
        table_data.append(
            [
                r.get("record_id"),
                ts,
                r.get("subject"),
                r.get("verb"),
                r.get("effect"),
                r.get("canonical_resource", "")[:35],
                r.get("matched_rule_id"),
                r.get("execution_status") or "-",
            ]
        )

    click.echo(
        tabulate(
            table_data,
            headers=["ID", "Timestamp (UTC)", "Subject", "Verb", "Effect", "Resource", "Rule", "Status"],
            tablefmt="rounded",
        )
    )


@cli.group("approvals")
def approvals_group() -> None:
    """Manage human-in-the-loop approvals."""
    pass


@approvals_group.command("list")
@click.option("--storage", type=click.Path(path_type=Path), default=DEFAULT_APPROVAL_PATH)
def list_approvals(storage: Path) -> None:
    """List pending approvals awaiting owner confirmation."""
    mgr = ApprovalManager(storage)
    pending = mgr.list_pending()
    if not pending:
        click.echo("No pending approvals.")
        return

    table_data = []
    for req in pending:
        table_data.append(
            [
                req.approval_id,
                req.subject,
                req.action,
                req.target[:35],
                req.argument_summary[:30],
                req.expiry_utc[:19],
            ]
        )
    click.echo(
        tabulate(
            table_data,
            headers=["Approval ID", "Subject", "Action", "Target", "Summary", "Expires (UTC)"],
            tablefmt="rounded",
        )
    )


@approvals_group.command("approve")
@click.argument("approval_id")
@click.option("--reason", default="Approved via CLI")
@click.option("--storage", type=click.Path(path_type=Path), default=DEFAULT_APPROVAL_PATH)
def approve(approval_id: str, reason: str, storage: Path) -> None:
    """Approve a pending request."""
    mgr = ApprovalManager(storage)
    try:
        res = mgr.resolve_request(approval_id, approved=True, reason=reason)
        click.secho(f"Request {approval_id} APPROVED.", fg="green", bold=True)
    except Exception as exc:
        click.secho(f"Failed to approve {approval_id}: {exc}", fg="red")
        sys.exit(1)


@approvals_group.command("reject")
@click.argument("approval_id")
@click.option("--reason", default="Rejected via CLI")
@click.option("--storage", type=click.Path(path_type=Path), default=DEFAULT_APPROVAL_PATH)
def reject(approval_id: str, reason: str, storage: Path) -> None:
    """Reject a pending request."""
    mgr = ApprovalManager(storage)
    try:
        res = mgr.resolve_request(approval_id, approved=False, reason=reason)
        click.secho(f"Request {approval_id} REJECTED.", fg="yellow", bold=True)
    except Exception as exc:
        click.secho(f"Failed to reject {approval_id}: {exc}", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    cli()
