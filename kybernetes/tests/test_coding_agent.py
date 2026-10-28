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

    tools = ToolRegistry(allowed_roots=[tmp_path], enforce_approvals=False)
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

    tools = ToolRegistry(allowed_roots=[tmp_path], enforce_approvals=False)
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


def test_coding_agent_requires_approval_blocks_mutations(tmp_path: Path):
    """Verify Finding A2: CodingAgentHarness awaiting approval halts and never executes mutation."""
    from kybernetes.contracts.enums import AgentSemanticState

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)

    target_file = tmp_path / "secret.py"
    target_file.write_text("API_KEY = 'old_secret'\n", encoding="utf-8")
    subprocess.run(["git", "add", "secret.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Baseline"], cwd=str(tmp_path), check=True)

    tools = ToolRegistry(allowed_roots=[tmp_path])
    harness = CodingAgentHarness(workspace_root=tmp_path, tool_registry=tools)

    def action_generator(step, history):
        if step == 1:
            return {
                "tool_name": "write_file",
                "arguments": {
                    "path": str(target_file),
                    "content": "API_KEY = 'hacked'\n",
                    "overwrite": True,
                },
                "requires_approval": True,
            }
        return None

    res = harness.run_autonomous_loop(
        goal="Rewrite secret.py",
        action_generator=action_generator,
        verifier=lambda: True,
    )

    # Must be blocked and denied
    assert res.success is False
    assert res.final_state == AgentSemanticState.AWAITING_APPROVAL
    assert harness.state_machine.current_state == AgentSemanticState.AWAITING_APPROVAL
    # The file MUST NOT be modified on disk (Finding A2 fix)
    assert target_file.read_text(encoding="utf-8") == "API_KEY = 'old_secret'\n"


def test_tool_registry_rejects_unapproved_and_fake_token_mutations(tmp_path: Path):
    """Verify Finding A1: ToolRegistry.execute fails closed without valid Phylax approval."""
    from kybernetes.contracts.tool_spec import ToolCallRequest

    tools = ToolRegistry(allowed_roots=[tmp_path])
    target_file = tmp_path / "unapproved.txt"

    # 1. Unapproved mutating tool call with approval_id=None must fail
    req_none = ToolCallRequest(
        tool_id="write_file",
        arguments={"path": str(target_file), "content": "malicious payload"},
        approval_id=None,
    )
    res_none = tools.execute(req_none)
    assert res_none.success is False
    assert "requires verified Phylax approval token" in res_none.error
    assert not target_file.exists()

    # 2. Fabricated token must fail closed
    req_fake = ToolCallRequest(
        tool_id="write_file",
        arguments={"path": str(target_file), "content": "malicious payload"},
        approval_id="appr-fake123",
    )
    res_fake = tools.execute(req_fake)
    assert res_fake.success is False
    assert "Invalid or unverified Phylax approval token" in res_fake.error
    assert not target_file.exists()


def test_autonomous_mode_forbids_self_approval(tmp_path: Path):
    """Verify Finding 2: Autonomous mode must fail closed and never self-approve mutating tools."""
    from kybernetes.contracts.enums import AgentSemanticState

    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)

    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(a, b): return a\n", encoding="utf-8")
    subprocess.run(["git", "add", "calc.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Baseline"], cwd=str(tmp_path), check=True)

    tools = ToolRegistry(allowed_roots=[tmp_path], enforce_approvals=True)
    harness = CodingAgentHarness(workspace_root=tmp_path, tool_registry=tools)

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

    res = harness.run_autonomous_loop(
        goal="Autonomous edit without approval token",
        action_generator=action_generator,
        verifier=lambda: True,
    )

    assert res.success is False
    assert res.final_state == AgentSemanticState.AWAITING_APPROVAL
    assert "autonomous self-approval is forbidden" in res.history[0].error
    # File must not be modified
    assert target_file.read_text(encoding="utf-8") == "def add(a, b): return a\n"

