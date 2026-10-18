"""Tests for TerminalCodingAgent and terminal-native MSS CLI commands."""

import asyncio
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock
from click.testing import CliRunner
import pytest

# Ensure Metis root is on python path for importing harness.skill_learner
sys.path.insert(0, str(Path(__file__).parents[2]))

from kybernetes.cli import cli
from kybernetes.engine.terminal_agent import TerminalCodingAgent


def test_terminal_agent_action_extraction():
    agent = TerminalCodingAgent(workspace_root=Path.cwd())

    # Case 1: ```done marker
    text1 = "I found the issue.\n```done\nTask complete: fixed typo in parser.py\n```"
    act1, done1, msg1 = agent._extract_action(text1)
    assert act1 is None
    assert done1 is True
    assert "fixed typo in parser.py" in msg1

    # Case 2: ```tool_call block
    text2 = (
        "Let me edit the file.\n"
        "```tool_call\n"
        "{\n"
        '  "tool_name": "edit_file_block",\n'
        '  "arguments": {"path": "main.py", "search_block": "foo", "replace_block": "bar"}\n'
        "}\n"
        "```"
    )
    act2, done2, msg2 = agent._extract_action(text2)
    assert done2 is False
    assert act2 is not None
    assert act2["tool_name"] == "edit_file_block"
    assert act2["arguments"]["path"] == "main.py"

    # Case 3: Natural language done
    text3 = "All steps complete and verified with pytest."
    act3, done3, msg3 = agent._extract_action(text3)
    assert done3 is True
    assert act3 is None


def test_terminal_agent_diff_formatting(tmp_path: Path):
    target = tmp_path / "sample.py"
    target.write_text("def hello():\n    return 'old'\n", encoding="utf-8")

    agent = TerminalCodingAgent(workspace_root=tmp_path)
    diff = agent._format_diff_preview(
        "sample.py",
        "return 'old'",
        "return 'new'",
    )
    assert "-return 'old'" in diff or "old" in diff
    assert "+return 'new'" in diff or "new" in diff


def test_terminal_agent_autonomous_run(tmp_path: Path):
    # Set up git repo
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)

    test_file = tmp_path / "greeting.py"
    test_file.write_text("def greet(): return 'hi'\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Baseline"], cwd=str(tmp_path), check=True)

    # Mock Hestia adapter stream
    mock_hestia = MagicMock()

    turn_count = 0

    async def mock_stream(*args, **kwargs):
        nonlocal turn_count
        turn_count += 1
        if turn_count == 1:
            # Edit file
            payload = (
                "```tool_call\n"
                "{\n"
                '  "tool_name": "edit_file_block",\n'
                '  "arguments": {\n'
                '    "path": "greeting.py",\n'
                '    "search_block": "return \'hi\'",\n'
                '    "replace_block": "return \'hello world\'"\n'
                "  }\n"
                "}\n"
                "```"
            )
        else:
            payload = "```done\nTask complete: updated greeting message\n```"
        yield {"delta": payload}

    mock_hestia.stream_chat = mock_stream
    mock_hestia.default_model = "gemma-4-26b"

    agent = TerminalCodingAgent(
        workspace_root=tmp_path,
        hestia_adapter=mock_hestia,
        auto_approve=True,
        max_steps=5,
    )

    result = agent.run("Update greeting message in greeting.py")
    assert result.success is True
    assert result.total_steps == 2
    assert "return 'hello world'" in test_file.read_text(encoding="utf-8")
    assert len(result.history) == 1
    assert result.history[0].tool_name == "edit_file_block"
    assert result.history[0].passed_validation is True
    assert (tmp_path / ".metis" / "skills").exists()


def test_terminal_agent_artifact_spillover_and_context_retention(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)

    large_file = tmp_path / "large.txt"
    large_text = "LINE " * 400  # ~2000 chars
    large_file.write_text(large_text, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Init"], cwd=str(tmp_path), check=True)

    captured_prompts = []
    turn_count = 0

    mock_hestia = MagicMock()

    async def mock_stream(*args, **kwargs):
        nonlocal turn_count
        captured_prompts.append(kwargs.get("prompt", ""))
        turn_count += 1
        if turn_count == 1:
            payload = (
                "```tool_call\n"
                "{\n"
                '  "tool_name": "read_file_range",\n'
                '  "arguments": {"path": "large.txt", "start_line": 1, "end_line": 400}\n'
                "}\n"
                "```"
            )
        else:
            payload = "```done\nTask complete: inspected large file\n```"
        yield {"delta": payload}

    mock_hestia.stream_chat = mock_stream
    mock_hestia.default_model = "gemma-4-26b"

    agent = TerminalCodingAgent(
        workspace_root=tmp_path,
        hestia_adapter=mock_hestia,
        auto_approve=True,
        max_steps=5,
    )

    res = agent.run("Inspect large file")
    assert res.success is True
    assert len(res.history) == 1
    assert res.history[0].tool_name == "read_file_range"

    # Verify artifact was persisted on disk due to truncation
    artifacts_dir = tmp_path / ".metis" / "artifacts"
    assert artifacts_dir.exists()
    artifact_files = list(artifacts_dir.glob("tool_out_*.txt"))
    assert len(artifact_files) == 1
    assert "LINE " in artifact_files[0].read_text(encoding="utf-8")

    # Verify turn 2 prompt contained full conversation context including previous tool execution
    assert len(captured_prompts) == 2
    turn2_prompt = captured_prompts[1]
    assert "Tool 'read_file_range' executed with result:" in turn2_prompt
    assert "tool_out_" in turn2_prompt


def test_cli_code_help():
    runner = CliRunner()
    res = runner.invoke(cli, ["code", "--help"])
    assert res.exit_code == 0
    assert "--goal" in res.output
    assert "--max-steps" in res.output
    assert "--auto-approve" in res.output
    assert "--test-cmd" in res.output


def test_cli_lochos_run_no_claimed_task(tmp_path: Path):
    runner = CliRunner()
    # When coder has no claimed task
    res = runner.invoke(cli, ["lochos", "run", "--seat", "coder"])
    assert res.exit_code == 0
    assert "has no active or claimed task in the queue" in res.output
