"""Integration tests for Paideia CLI commands."""

from pathlib import Path
from click.testing import CliRunner
import pytest

from paideia.cli import cli

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_cli_doctor():
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "Doctor Readiness Report" in result.output
    assert "Python >= 3.10" in result.output


def test_cli_validate_router():
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "lthcm_sample.json"
    result = runner.invoke(cli, [
        "dataset", "validate-router",
        str(fixture_path),
        "-r", "flat",
        "-r", "mined",
    ])
    assert result.exit_code == 0
    assert "LTHCM Router Dataset Audit" in result.output
    assert "valid_records" in result.output


def test_cli_validate_walk():
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "dianoia_sample.jsonl"
    result = runner.invoke(cli, [
        "dataset", "validate-walk",
        str(fixture_path),
    ])
    assert result.exit_code == 0
    assert "Dianoia Walk Dataset Audit" in result.output
    assert "Extracted 4 defensible policy training steps" in result.output


def test_cli_baseline():
    runner = CliRunner()
    fixture_path = FIXTURES_DIR / "lthcm_sample.json"
    result = runner.invoke(cli, [
        "baseline",
        str(fixture_path),
        "--task", "type",
    ])
    assert result.exit_code == 0
    assert "Running Baseline Ladder" in result.output
    assert "lthcm_regex_router" in result.output


def test_cli_verify_run(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"run_id": "r-1", "status": "COMPLETED", "hardware_line": "RTX 5060 Ti"}')

    runner = CliRunner()
    result = runner.invoke(cli, ["verify-run", "--manifest", str(manifest_path)])
    assert result.exit_code == 0
    assert "Run ID: r-1" in result.output
    assert "COMPLETED" in result.output
