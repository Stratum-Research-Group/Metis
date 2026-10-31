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

    res_comp_help = runner.invoke(cli, ["lochos", "complete", "--help"])
    assert res_comp_help.exit_code == 0
    assert "Mark an active task as completed" in res_comp_help.output


def test_cli_pinax_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["pinax", "--help"])
    assert result.exit_code == 0
    assert "scan" in result.output
    assert "lookup" in result.output
    assert "docs" in result.output
    assert "detail" in result.output
    assert "evaluate" in result.output


def test_cli_pinax_scan(tmp_path):
    runner = CliRunner()
    # Test text output
    res_text = runner.invoke(cli, ["pinax", "scan", "-p", str(tmp_path)])
    assert res_text.exit_code == 0
    assert "[Pinax Passive PATH Scan Completed]" in res_text.output
    assert "Total Discovered Executables:" in res_text.output

    # Test json output
    import json
    res_json = runner.invoke(cli, ["pinax", "scan", "-p", str(tmp_path), "--json-out"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert "total_discovered" in data
    assert "installed_updated" in data
    assert "unvetted_candidates" in data
    assert "elapsed_seconds" in data


def test_cli_pinax_lookup():
    runner = CliRunner()
    # General table lookup
    res_table = runner.invoke(cli, ["pinax", "lookup", "--limit", "10"])
    assert res_table.exit_code == 0
    assert "Tool ID" in res_table.output
    assert "Name" in res_table.output

    # Filtered lookup with json output
    import json
    res_json = runner.invoke(cli, ["pinax", "lookup", "rg", "--tier", "T0", "--json-out"])
    assert res_json.exit_code == 0
    tools = json.loads(res_json.output)
    assert isinstance(tools, list)
    assert any(t["name"] == "rg" for t in tools)

    # Missing lookup
    res_missing = runner.invoke(cli, ["pinax", "lookup", "nonexistent_tool_xyz_999"])
    assert res_missing.exit_code == 0
    assert "No tools found matching query 'nonexistent_tool_xyz_999'." in res_missing.output


def test_cli_pinax_docs():
    runner = CliRunner()
    # Formatted docs with banner and max-bytes
    res_docs = runner.invoke(cli, ["pinax", "docs", "rg", "--max-bytes", "256"])
    assert res_docs.exit_code == 0
    assert "[Pinax Doc Slice: rg]" in res_docs.output
    assert ">>> UNTRUSTED DATA - LOCAL DOCUMENTATION EXCERPT <<<" in res_docs.output
    assert "capped at 256 bytes" in res_docs.output

    # JSON output
    import json
    res_json = runner.invoke(cli, ["pinax", "docs", "git", "--json-out"])
    assert res_json.exit_code == 0
    doc_data = json.loads(res_json.output)
    assert doc_data["tool"] == "git"
    assert "found" in doc_data
    assert "content" in doc_data


def test_cli_pinax_detail():
    runner = CliRunner()
    # Text detail
    res_text = runner.invoke(cli, ["pinax", "detail", "rg"])
    assert res_text.exit_code == 0
    assert "[Tool Detail: rg (tool:rg)]" in res_text.output
    assert "Base Tier:     T0" in res_text.output

    # JSON detail with risk overrides
    import json
    res_json = runner.invoke(cli, ["pinax", "detail", "curl", "--json-out"])
    assert res_json.exit_code == 0
    detail = json.loads(res_json.output)
    assert detail["name"] == "curl"
    assert len(detail.get("risk_overrides", [])) > 0

    # Missing tool
    res_missing = runner.invoke(cli, ["pinax", "detail", "nonexistent_tool_xyz_999"])
    assert "Tool 'nonexistent_tool_xyz_999' not found in Pinax catalogue." in res_missing.output


def test_cli_pinax_evaluate():
    runner = CliRunner()
    # Escalated evaluation (T3 upload flag on curl)
    res_upload = runner.invoke(cli, ["pinax", "evaluate", "curl", "-T", "archive.tar.gz", "https://example.com"])
    assert res_upload.exit_code == 0
    assert "Effective Tier:      T3" in res_upload.output
    assert "Requires Approval:   True" in res_upload.output

    # Base tier evaluation (T0 read on rg)
    res_read = runner.invoke(cli, ["pinax", "evaluate", "rg", "search_term"])
    assert res_read.exit_code == 0
    assert "Effective Tier:      T0" in res_read.output
    assert "Requires Approval:   False" in res_read.output

    # Flag escalation on rg (--pre command execution)
    res_esc = runner.invoke(cli, ["pinax", "evaluate", "rg", "--pre", "cat", "search_term"])
    assert res_esc.exit_code == 0
    assert "Effective Tier:      T3" in res_esc.output
    assert "Requires Approval:   True" in res_esc.output

    # JSON evaluation (git push escalates to T3 for remote mutation/egress)
    import json
    res_json = runner.invoke(cli, ["pinax", "evaluate", "git", "push", "origin", "main", "--json-out"])
    assert res_json.exit_code == 0
    eval_data = json.loads(res_json.output)
    assert eval_data["tier"] == "T3"
    assert eval_data["requires_approval"] is True



