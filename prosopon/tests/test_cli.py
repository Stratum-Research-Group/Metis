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


def test_cli_serve_and_run(tmp_path, monkeypatch):
    """Item 1: serve and run launch uvicorn exactly once with correct parameters and no NameErrors."""
    from unittest.mock import MagicMock
    mock_uvicorn = MagicMock()
    monkeypatch.setattr("uvicorn.run", mock_uvicorn)

    runner = CliRunner()
    # Test serve command
    res_serve = runner.invoke(cli, ["serve", "--port", "9999", "--data-dir", str(tmp_path), "--lan"])
    assert res_serve.exit_code == 0
    assert "Prosopon Local Interface" in res_serve.output
    assert "Mode: LAN (Phone Ready)" in res_serve.output
    assert "Single-Owner Phone Pairing PIN:" in res_serve.output
    assert mock_uvicorn.call_count == 1
    call_args, call_kwargs = mock_uvicorn.call_args
    assert call_kwargs["host"] == "0.0.0.0"
    assert call_kwargs["port"] == 9999

    mock_uvicorn.reset_mock()

    # Test run command alias
    res_run = runner.invoke(cli, ["run", "--port", "9998", "--data-dir", str(tmp_path)])
    assert res_run.exit_code == 0
    assert "Prosopon Local Interface" in res_run.output
    assert mock_uvicorn.call_count == 1
    call_args, call_kwargs = mock_uvicorn.call_args
    assert call_kwargs["host"] == "127.0.0.1"
    assert call_kwargs["port"] == 9998

