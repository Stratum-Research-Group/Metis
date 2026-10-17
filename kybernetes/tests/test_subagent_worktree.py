"""Tests for SubagentManager background execution in isolated git worktrees."""

from pathlib import Path
import subprocess
import time
import pytest

from kybernetes.contracts.enums import AgentSemanticState
from kybernetes.engine.checkpoint import GitMilestoneCheckpointer
from kybernetes.engine.subagent import SubagentManager


@pytest.fixture
def git_workspace(tmp_path: Path) -> Path:
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)
    base_file = tmp_path / "main.py"
    base_file.write_text("print('baseline')\n", encoding="utf-8")
    subprocess.run(["git", "add", "main.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(tmp_path), check=True)
    return tmp_path


def test_subagent_fork_isolated_worktree(git_workspace: Path):
    checkpointer = GitMilestoneCheckpointer(git_workspace)
    manager = SubagentManager(workspace_root=git_workspace, checkpointer=checkpointer)

    def subagent_action(worktree_dir: Path):
        # Modify a file inside the isolated worktree
        feature_file = worktree_dir / "feature.py"
        feature_file.write_text("def subagent_feature(): return 42\n", encoding="utf-8")
        subprocess.run(["git", "add", "feature.py"], cwd=str(worktree_dir), check=True)
        subprocess.run(["git", "commit", "-m", "Add feature from subagent"], cwd=str(worktree_dir), check=True)
        return {"success": True, "output": "Created feature.py"}

    subagent_id = "test-worker-1"
    task = manager.fork_subagent(
        subagent_id=subagent_id,
        description="Build auxiliary feature in isolated worktree",
        action_fn=subagent_action,
        isolated=True,
    )

    # Wait for completion
    for _ in range(50):
        if task.state_machine.is_terminal():
            break
        time.sleep(0.05)

    assert task.state_machine.current_state == AgentSemanticState.COMPLETED
    assert task.success is True
    assert "Created feature.py" in task.output

    # Check that main workspace did NOT get modified yet (isolated worktree)
    assert not (git_workspace / "feature.py").exists()

    # Now merge subagent
    merged = manager.merge_subagent(subagent_id)
    assert merged is True
    assert (git_workspace / "feature.py").exists()


def test_subagent_discard_cleans_up_worktree(git_workspace: Path):
    checkpointer = GitMilestoneCheckpointer(git_workspace)
    manager = SubagentManager(workspace_root=git_workspace, checkpointer=checkpointer)

    def failed_action(worktree_dir: Path):
        # Deliberately fails
        raise ValueError("Simulated failure in subagent")

    subagent_id = "test-failed-worker"
    task = manager.fork_subagent(
        subagent_id=subagent_id,
        description="Failing speculative experiment",
        action_fn=failed_action,
        isolated=True,
    )

    for _ in range(50):
        if task.state_machine.is_terminal():
            break
        time.sleep(0.05)

    assert task.state_machine.current_state == AgentSemanticState.FAILED
    assert task.success is False
    assert "Simulated failure" in (task.error or "")

    # Clean up and discard
    discarded = manager.discard_subagent(subagent_id)
    assert discarded is True
    assert not (git_workspace / ".metis" / "worktrees" / subagent_id).exists()


def test_subagent_task_persistence_across_restart(git_workspace: Path):
    """Verify subagent task registry persists on disk and reloads across restarts (Finding B5)."""
    checkpointer = GitMilestoneCheckpointer(git_workspace)
    manager = SubagentManager(workspace_root=git_workspace, checkpointer=checkpointer)

    def persistent_action(worktree_dir: Path):
        return {"success": True, "output": "Persistent output 12345"}

    subagent_id = "test-persistent-subagent"
    task = manager.fork_subagent(
        subagent_id=subagent_id,
        description="Persistent background task",
        action_fn=persistent_action,
        isolated=False,
    )

    for _ in range(50):
        if task.state_machine.is_terminal():
            break
        time.sleep(0.05)

    assert task.success is True

    # Check on-disk persistence file exists
    persisted_file = git_workspace / ".metis" / "subagents" / f"{subagent_id}.json"
    assert persisted_file.exists()

    # Re-instantiate a fresh SubagentManager against the same workspace
    new_manager = SubagentManager(workspace_root=git_workspace, checkpointer=checkpointer)
    loaded_task = new_manager.get_task(subagent_id)
    assert loaded_task is not None
    assert loaded_task.subagent_id == subagent_id
    assert loaded_task.description == "Persistent background task"
    assert loaded_task.success is True
    assert loaded_task.output == "Persistent output 12345"
    assert loaded_task.state_machine.current_state == AgentSemanticState.COMPLETED

    # Clean up
    assert new_manager.discard_subagent(subagent_id) is True
    assert not persisted_file.exists()
    assert new_manager.get_task(subagent_id) is None
