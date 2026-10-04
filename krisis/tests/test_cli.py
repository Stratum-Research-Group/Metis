"""Tests for Krisis CLI commands."""

import json
from pathlib import Path
from click.testing import CliRunner

from krisis.cli import cli


def test_cli_validate_schema():
    runner = CliRunner()
    result = runner.invoke(cli, ["validate-schema"])
    assert result.exit_code == 0
    assert "is fully valid according to" in result.output


def test_cli_replay():
    runner = CliRunner()
    result = runner.invoke(cli, ["replay"])
    assert result.exit_code == 0
    assert "100% REPLAY SUCCESS" in result.output


def test_cli_inspect_policy():
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect-policy"])
    assert result.exit_code == 0
    assert "krisis-policy-0.1" in result.output
    assert "grounded_v1" in result.output


def test_cli_evaluate(tmp_path: Path):
    req_file = tmp_path / "req.json"
    req_data = {
        "question_class": "closed_form_number",
        "requires_grounding": False,
        "capabilities": {"verifier_active": False, "logprobs": False},
        "candidates": [
            {"id": "c1", "gate_results": {"access": True, "provenance": True, "contradiction": True, "citation_resolution": True}}
        ]
    }
    req_file.write_text(json.dumps(req_data))

    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", str(req_file)])
    assert result.exit_code == 0
    res_data = json.loads(result.output)
    assert res_data["status"] == "selected"
    assert res_data["selected_id"] == "c1"
    assert res_data["profile"] == "eligibility_baseline"
