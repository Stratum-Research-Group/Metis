"""Unit tests for CodingAgentHarness loop."""

from pathlib import Path
import subprocess
import pytest

from kybernetes.contracts.enums import TurnOutcomeStatus
from kybernetes.engine.checkpoint import GitMilestoneCheckpointer
from kybernetes.engine.coding_agent import CodingAgentHarness
from kybernetes.tools.registry import ToolRegistry


def test_coding_agent_loop_success(tmp_path: Path):
    # Set up repo
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)

    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(a, b): return a\n", encoding="utf-8")
    subprocess.run(["git", "add", "calc.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Baseline"], cwd=str(tmp_path), check=True)

    tools = ToolRegistry(allowed_roots=[tmp_path])
    harness = CodingAgentHarness(workspace_root=tmp_path, tool_registry=tools)

    # Simulated action sequence: edit file to fix bug
    def action_generator(step, history):
        if step == 1:
            return {
                "tool_name": "edit_file_block",
                "arguments": {
                    "path": str(target_file),
                    "search_block": "return a",
                    "replace_block": "return a + b",
                },
            }
        return None

    def verifier():
        return "return a + b" in target_file.read_text(encoding="utf-8")

    res = harness.run_autonomous_loop(
        goal="Fix addition bug in calc.py",
        action_generator=action_generator,
        verifier=verifier,
    )

    assert res.success is True
    assert res.status == TurnOutcomeStatus.VERIFIED
    assert res.total_steps == 1
    assert res.backtracks_executed == 0
    assert "return a + b" in target_file.read_text(encoding="utf-8")


def test_coding_agent_loop_with_rollback(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)

    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(a, b): return a\n", encoding="utf-8")
    subprocess.run(["git", "add", "calc.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Baseline"], cwd=str(tmp_path), check=True)

    tools = ToolRegistry(allowed_roots=[tmp_path])
    harness = CodingAgentHarness(workspace_root=tmp_path, tool_registry=tools)

    # Step 1 makes an edit that fails verification, triggering rollback.
    # Step 2 makes the correct edit that passes.
    def action_generator(step, history):
        if step == 1:
            return {
                "tool_name": "edit_file_block",
                "arguments": {
                    "path": str(target_file),
                    "search_block": "return a",
                    "replace_block": "return 0",
                },
            }
        elif step == 2:
            return {
                "tool_name": "edit_file_block",
                "arguments": {
                    "path": str(target_file),
                    "search_block": "return a",
                    "replace_block": "return a + b",
                },
            }
        return None

    def verifier():
        content = target_file.read_text(encoding="utf-8")
        # Only passes if "return a + b" is present
        return "return a + b" in content

    res = harness.run_autonomous_loop(
        goal="Fix addition bug in calc.py",
        action_generator=action_generator,
        verifier=verifier,
    )

    assert res.success is True
    assert res.status == TurnOutcomeStatus.VERIFIED
    assert res.total_steps == 2
    assert res.backtracks_executed == 1
    assert "return a + b" in target_file.read_text(encoding="utf-8")


def test_stream_flow_monitor_early_abort():
    from kybernetes.engine.coding_agent import EarlyFlowAbort, StreamFlowMonitor

    monitor = StreamFlowMonitor(max_repetition_threshold=3)

    # Clean stream
    for chunk in ["def ", "calculate", "(x, y):", "\n    ", "return x + y"]:
        monitor.feed(chunk)

    # Repetitive hallucination loop
    with pytest.raises(EarlyFlowAbort, match="repetition loop"):
        for _ in range(10):
            monitor.feed("repetitive_token_loop_")

