"""Odyssey-inspired Git Milestone Checkpointing and Verification Rollback.
Prevents long-horizon context drift and working-tree corruption during autonomous coding.
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from typing import List, Optional


@dataclass
class MilestoneCheckpoint:
    id: str
    timestamp: float
    description: str
    stash_ref: Optional[str] = None
    commit_sha: Optional[str] = None
    untracked_files: List[str] = None


class GitMilestoneCheckpointer:
    """Manages transactional filesystem checkpoints using local git operations."""

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self._checkpoints: List[MilestoneCheckpoint] = []

    def _run_git(self, args: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git"] + args,
            cwd=str(self.workspace_root),
            capture_output=True,
            text=True,
            check=False,
        )

    def is_git_repo(self) -> bool:
        res = self._run_git(["rev-parse", "--is-inside-work-tree"])
        return res.returncode == 0 and res.stdout.strip() == "true"

    def get_head_sha(self) -> str:
        res = self._run_git(["rev-parse", "HEAD"])
        return res.stdout.strip() if res.returncode == 0 else ""

    def _get_untracked_files(self) -> List[str]:
        res = self._run_git(["status", "--porcelain", "-uall"])
        untracked = []
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if line.startswith("?? "):
                    path_str = line[3:].strip()
                    if path_str.startswith(".metis") or path_str.startswith(".git"):
                        continue
                    untracked.append(path_str)
        return untracked

    def create_checkpoint(self, description: str) -> MilestoneCheckpoint:
        """Create a point-in-time milestone before mutating operations."""
        checkpoint_id = f"cp-{int(time.time() * 1000)}-{len(self._checkpoints) + 1}"
        head_sha = self.get_head_sha()

        # Check if there are uncommitted changes to stash as clean baseline
        stash_ref = None
        untracked_files = []
        if self.is_git_repo():
            untracked_files = self._get_untracked_files()
            stash_msg = f"metis-checkpoint-{checkpoint_id}"
            stash_res = self._run_git(["stash", "create", stash_msg])
            if stash_res.returncode == 0 and stash_res.stdout.strip():
                stash_ref = stash_res.stdout.strip()

        checkpoint = MilestoneCheckpoint(
            id=checkpoint_id,
            timestamp=time.time(),
            description=description,
            stash_ref=stash_ref,
            commit_sha=head_sha,
            untracked_files=untracked_files,
        )
        self._checkpoints.append(checkpoint)
        return checkpoint

    def rollback(self, checkpoint: MilestoneCheckpoint) -> bool:
        """Revert working tree to the state at checkpoint time upon test failure without destroying untracked files."""
        if not self.is_git_repo():
            return False

        # Reset unstaged modifications to tracked files
        self._run_git(["checkout", "--", "."])

        # Remove ONLY newly created untracked files that were not present at checkpoint time (Finding A7)
        current_untracked = set(self._get_untracked_files())
        baseline_untracked = set(checkpoint.untracked_files or [])
        newly_created = current_untracked - baseline_untracked

        for rel_path in newly_created:
            if rel_path.startswith(".metis") or rel_path.startswith(".git"):
                continue
            target = (self.workspace_root / rel_path).resolve()
            try:
                target.relative_to(self.workspace_root)
                if target.is_file() or target.is_symlink():
                    target.unlink()
                elif target.is_dir():
                    import shutil
                    shutil.rmtree(target, ignore_errors=True)
            except Exception:
                pass

        # If we had a specific stash reference, restore it
        if checkpoint.stash_ref:
            res = self._run_git(["stash", "apply", checkpoint.stash_ref])
            return res.returncode == 0
        elif checkpoint.commit_sha:
            res = self._run_git(["checkout", checkpoint.commit_sha, "--", "."])
            return res.returncode == 0

        return True

    def commit_milestone(self, message: str, files: Optional[List[str]] = None) -> bool:
        """Stage and commit verified changes after test gates pass, including new files (Finding A7)."""
        if not self.is_git_repo():
            return False

        if files:
            for f in files:
                self._run_git(["add", f])
        else:
            add_res = self._run_git(["add", "."])
            if add_res.returncode != 0:
                self._run_git(["add", "-A"])

        commit_res = self._run_git(["commit", "-m", f"[Metis Milestone] {message}"])
        return commit_res.returncode == 0

    def list_checkpoints(self) -> List[MilestoneCheckpoint]:
        return list(self._checkpoints)

    def create_worktree_sandbox(self, subagent_id: str) -> Path:
        """Create an isolated git worktree for a parallel subagent.
        Prevents speculative changes from polluting the user's active editor workspace.
        """
        if not self.is_git_repo():
            raise RuntimeError("Worktree sandboxes require a valid git repository.")

        worktree_dir = self.workspace_root / ".metis" / "worktrees" / subagent_id
        worktree_dir.parent.mkdir(parents=True, exist_ok=True)
        if worktree_dir.exists():
            self.prune_worktree_sandbox(worktree_dir)

        branch_name = f"subagent/{subagent_id}"
        res = self._run_git(["worktree", "add", "-B", branch_name, str(worktree_dir), "HEAD"])
        if res.returncode != 0:
            raise RuntimeError(f"Failed to create git worktree sandbox: {res.stderr}")
        return worktree_dir

    def prune_worktree_sandbox(self, worktree_path: Path) -> bool:
        """Cleanly remove and prune an isolated subagent worktree."""
        if not self.is_git_repo():
            return False
        res = self._run_git(["worktree", "remove", "--force", str(worktree_path)])
        self._run_git(["worktree", "prune"])
        return res.returncode == 0

    def merge_worktree_sandbox(self, subagent_id: str) -> bool:
        """Merge verified subagent branch into active branch."""
        if not self.is_git_repo():
            return False
        branch_name = f"subagent/{subagent_id}"
        res = self._run_git(["merge", "--ff-only", branch_name])
        if res.returncode != 0:
            res = self._run_git(["merge", "-m", f"Merge verified worktree for {subagent_id}", branch_name])
        return res.returncode == 0

