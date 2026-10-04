"""Unit tests for Prosopon CLI commands."""

from pathlib import Path
from click.testing import CliRunner
import pytest

from prosopon.cli import cli


def test_cli_doctor(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Doctor Readiness Report" in result.output
    assert "Python >= 3.10" in result.output


def test_cli_pair(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["pair", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Phone Device Pairing" in result.output
    assert "PIN:" in result.output


def test_cli_revoke(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["revoke", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Successfully revoked" in result.output
