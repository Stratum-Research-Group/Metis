"""Tests for the Kybernetes CLI interface."""

from click.testing import CliRunner
from kybernetes.cli import cli


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_doctor():
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "Hestia Model Server" in result.output
    assert "LTHCM / Stoma Memory" in result.output
    assert "Dianoia Deliberation" in result.output
    assert "Phylax Capability Gateway" in result.output
    assert "Krisis Verifier Gates" in result.output


def test_cli_chat_direct():
    runner = CliRunner()
    result = runner.invoke(cli, ["chat", "-q", "Hello there!"])
    assert result.exit_code == 0
    assert "Plan: direct" in result.output
    assert "Turn Completed" in result.output


def test_cli_chat_factual():
    runner = CliRunner()
    result = runner.invoke(cli, ["chat", "-q", "What is Metis architecture?"])
    assert result.exit_code == 0
    assert "evidence citations" in result.output
    assert "Turn Completed" in result.output


def test_cli_lochos():
    runner = CliRunner()
    result = runner.invoke(cli, ["lochos", "ps"])
    assert result.exit_code == 0
    assert "Lochos Squad" in result.output
    assert "coder@metis-sovereign-squad" in result.output
    assert "reviewer@metis-sovereign-squad" in result.output

    res_q = runner.invoke(cli, ["lochos", "queue"])
    assert res_q.exit_code == 0

