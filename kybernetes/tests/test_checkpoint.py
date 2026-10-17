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


def test_rollback_preserves_pre_existing_untracked_files_and_cleans_new_ones(tmp_path: Path):
    """Finding A7 regression: git clean -fd must not delete pre-existing untracked user files."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)

    tracked_file = tmp_path / "main.py"
    tracked_file.write_text("initial code\n", encoding="utf-8")
    subprocess.run(["git", "add", "main.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(tmp_path), check=True)

    # Pre-existing untracked files before checkpoint is created
    user_notes = tmp_path / "notes.txt"
    user_notes.write_text("important user notes\n", encoding="utf-8")
    skills_dir = tmp_path / ".metis" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skills_dir / "learned_skill.md"
    skill_file.write_text("learned skill content\n", encoding="utf-8")

    checkpointer = GitMilestoneCheckpointer(workspace_root=tmp_path)
    cp = checkpointer.create_checkpoint("Baseline before turn")

    # Agent performs edits during turn:
    # 1. Modifies tracked file
    tracked_file.write_text("broken agent change\n", encoding="utf-8")
    # 2. Generates a new temporary untracked artifact
    temp_artifact = tmp_path / "temp_agent_file.py"
    temp_artifact.write_text("import sys\n", encoding="utf-8")

    # Execute rollback
    success = checkpointer.rollback(cp)
    assert success is True

    # Tracked file must be restored
    assert tracked_file.read_text(encoding="utf-8") == "initial code\n"
    # Agent-created untracked file must be removed
    assert not temp_artifact.exists()
    # Pre-existing untracked files MUST NOT be deleted!
    assert user_notes.exists()
    assert user_notes.read_text(encoding="utf-8") == "important user notes\n"
    assert skill_file.exists()
    assert skill_file.read_text(encoding="utf-8") == "learned skill content\n"


def test_commit_milestone_stages_newly_created_files(tmp_path: Path):
    """Finding A7 regression: commit_milestone must stage new agent-created files (not git add -u)."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)

    existing = tmp_path / "existing.py"
    existing.write_text("v1\n", encoding="utf-8")
    subprocess.run(["git", "add", "existing.py"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=str(tmp_path), check=True)

    checkpointer = GitMilestoneCheckpointer(workspace_root=tmp_path)

    # Agent modifies existing file and creates a brand new file
    existing.write_text("v2\n", encoding="utf-8")
    new_file = tmp_path / "new_module.py"
    new_file.write_text("print('hello')\n", encoding="utf-8")

    committed = checkpointer.commit_milestone("Agent completed task")
    assert committed is True

    # Verify both files are committed in git log
    result = subprocess.run(
        ["git", "show", "--name-only", "--oneline", "HEAD"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "existing.py" in result.stdout
    assert "new_module.py" in result.stdout


