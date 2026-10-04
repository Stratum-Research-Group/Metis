"""Tests for Phylax CLI commands."""

from pathlib import Path
from click.testing import CliRunner

from phylax.cli import cli


def test_cli_validate():
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", "policies/default_policy.yaml"])
    assert result.exit_code == 0
    assert "Policy is perfectly valid" in result.output


def test_cli_doctor():
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "Phylax environment is HEALTHY" in result.output


def test_cli_decide(tmp_path: Path):
    runner = CliRunner()
    audit_file = tmp_path / "cli_audit.jsonl"
    result = runner.invoke(
        cli,
        [
            "decide",
            "--subject", "kybernetes",
            "--verb", "read",
            "--resource-class", "file",
            "--target", "docs/manual.md",
            "--policy", "policies/default_policy.yaml",
            "--audit-log", str(audit_file),
        ],
    )
    assert result.exit_code == 0
    assert "DECISION: ALLOW" in result.output
    assert audit_file.exists()


def test_cli_audit_verify(tmp_path: Path):
    runner = CliRunner()
    audit_file = tmp_path / "cli_audit.jsonl"
    
    # First make a decision to populate the audit log
    runner.invoke(
        cli,
        [
            "decide",
            "--subject", "kybernetes",
            "--verb", "read",
            "--resource-class", "file",
            "--target", "docs/manual.md",
            "--policy", "policies/default_policy.yaml",
            "--audit-log", str(audit_file),
        ],
    )

    # Now verify it
    result = runner.invoke(cli, ["audit", "verify", "--log-file", str(audit_file)])
    assert result.exit_code == 0
    assert "Audit log integrity VERIFIED" in result.output
