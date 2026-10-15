"""Unit tests for GitMilestoneCheckpointer."""

from pathlib import Path
import subprocess
import pytest

from kybernetes.engine.checkpoint import GitMilestoneCheckpointer


def test_checkpoint_creation_and_listing(tmp_path: Path):
    # Initialize a temporary git repository
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)

    test_file = tmp_path / "hello.txt"
    test_file.write_text("initial content", encoding="utf-8")
    subprocess.run(["git", "add", "hello.txt"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(tmp_path), check=True)

    checkpointer = GitMilestoneCheckpointer(workspace_root=tmp_path)
    assert checkpointer.is_git_repo() is True

    cp = checkpointer.create_checkpoint("Pre-edit baseline")
    assert cp.id.startswith("cp-")
    assert cp.commit_sha != ""
    assert len(checkpointer.list_checkpoints()) == 1


def test_rollback_on_failure(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)

    code_file = tmp_path / "app.py"
    code_file.write_text("def run(): return 42\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Working version"], cwd=str(tmp_path), check=True)

    checkpointer = GitMilestoneCheckpointer(workspace_root=tmp_path)
    cp = checkpointer.create_checkpoint("Before buggy patch")

    # Simulate agent introducing a broken bug
    code_file.write_text("def run(): syntax error here\n", encoding="utf-8")
    assert "syntax error" in code_file.read_text(encoding="utf-8")

    # Trigger rollback
    success = checkpointer.rollback(cp)
    assert success is True
    assert code_file.read_text(encoding="utf-8") == "def run(): return 42\n"


def test_worktree_sandbox_lifecycle(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)

    base_file = tmp_path / "main.py"
    base_file.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "main.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Base"], cwd=str(tmp_path), check=True)

    checkpointer = GitMilestoneCheckpointer(workspace_root=tmp_path)

    # 1. Create worktree sandbox for subagent
    wt_dir = checkpointer.create_worktree_sandbox("agent-42")
    assert wt_dir.exists()
    assert (wt_dir / "main.py").exists()

    # 2. Subagent makes isolated edit
    (wt_dir / "subagent_work.txt").write_text("speculative result", encoding="utf-8")
    assert not (tmp_path / "subagent_work.txt").exists()  # Main workspace untouched!

    # 3. Prune worktree
    pruned = checkpointer.prune_worktree_sandbox(wt_dir)
    assert pruned is True
    assert not wt_dir.exists()

