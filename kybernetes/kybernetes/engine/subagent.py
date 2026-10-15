"""OpenCode 2.0-inspired Subagent Worktree Forking & Parallel Worker Manager.
Spins up autonomous background subagents in isolated git worktrees so speculative changes,
heavy test suites, or multi-file audits never lock or dirty the developer's working tree.
"""

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from kybernetes.contracts.enums import AgentSemanticState
from kybernetes.engine.checkpoint import GitMilestoneCheckpointer
from kybernetes.engine.state import AgentStateMachine


@dataclass
class SubagentTask:
    """Represents an isolated, forked background subagent task."""
    subagent_id: str
    description: str
    worktree_path: Optional[Path] = None
    branch_name: Optional[str] = None
    state_machine: AgentStateMachine = field(default_factory=AgentStateMachine)
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    success: Optional[bool] = None
    output: str = ""
    diff: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subagent_id": self.subagent_id,
            "description": self.description,
            "current_state": self.state_machine.current_state.value,
            "worktree_path": str(self.worktree_path) if self.worktree_path else None,
            "branch_name": self.branch_name,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "success": self.success,
            "output_preview": self.output[:500] if self.output else "",
            "diff_preview": self.diff[:500] if self.diff else "",
            "error": self.error,
        }


class SubagentManager:
    """
    Manages the lifecycle of background subagents running in isolated git worktrees.
    Supports asynchronous forking, diff extraction, merging, and clean worktree pruning.
    """

    def __init__(self, workspace_root: Optional[Path] = None, checkpointer: Optional[GitMilestoneCheckpointer] = None):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.checkpointer = checkpointer or GitMilestoneCheckpointer(self.workspace_root)
        self._tasks: Dict[str, SubagentTask] = {}
        self._lock = threading.Lock()

    def get_task(self, subagent_id: str) -> Optional[SubagentTask]:
        with self._lock:
            return self._tasks.get(subagent_id)

    def list_tasks(self) -> List[SubagentTask]:
        with self._lock:
            return list(self._tasks.values())

    def fork_subagent(
        self,
        subagent_id: str,
        description: str,
        action_fn: Callable[[Path], Dict[str, Any]],
        isolated: bool = True,
    ) -> SubagentTask:
        """
        Fork a subagent to run asynchronously in a background worker thread.
        If isolated=True, runs within an isolated git worktree sandbox.
        """
        worktree_path: Optional[Path] = None
        branch_name: Optional[str] = None

        if isolated and self.checkpointer.is_git_repo():
            worktree_path = self.checkpointer.create_worktree_sandbox(subagent_id)
            branch_name = f"subagent/{subagent_id}"
        else:
            worktree_path = self.workspace_root

        task = SubagentTask(
            subagent_id=subagent_id,
            description=description,
            worktree_path=worktree_path,
            branch_name=branch_name,
        )

        with self._lock:
            self._tasks[subagent_id] = task

        # Launch background runner thread
        thread = threading.Thread(
            target=self._run_subagent_worker,
            args=(task, action_fn),
            daemon=True,
            name=f"Subagent-{subagent_id}",
        )
        thread.start()
        return task

    def _run_subagent_worker(
        self,
        task: SubagentTask,
        action_fn: Callable[[Path], Dict[str, Any]],
    ) -> None:
        """Worker thread entrypoint for subagent execution."""
        task.state_machine.transition(
            AgentSemanticState.PLANNING,
            reason=f"Forked background subagent: {task.description}",
        )

        target_dir = task.worktree_path or self.workspace_root
        try:
            task.state_machine.transition(
                AgentSemanticState.MUTATING,
                reason="Subagent executing assigned workload in sandbox",
            )
            result = action_fn(target_dir)

            task.state_machine.transition(
                AgentSemanticState.VERIFYING,
                reason="Subagent verifying outputs and inspecting git diff",
            )

            # Collect git diff if in a worktree
            diff_text = ""
            if task.worktree_path and task.branch_name:
                diff_res = subprocess.run(
                    ["git", "diff", "HEAD~1..HEAD"],
                    cwd=str(task.worktree_path),
                    capture_output=True,
                    text=True,
                )
                if diff_res.returncode == 0 and diff_res.stdout:
                    diff_text = diff_res.stdout
                else:
                    uncommitted_diff = subprocess.run(
                        ["git", "diff"],
                        cwd=str(task.worktree_path),
                        capture_output=True,
                        text=True,
                    )
                    diff_text = uncommitted_diff.stdout

            task.output = str(result.get("output", ""))
            task.diff = diff_text
            task.success = bool(result.get("success", True))
            task.finished_at = time.time()

            final_state = AgentSemanticState.COMPLETED if task.success else AgentSemanticState.FAILED
            task.state_machine.transition(
                final_state,
                reason="Subagent finished workload",
            )

        except Exception as e:
            task.finished_at = time.time()
            task.success = False
            task.error = f"{type(e).__name__}: {str(e)}"
            task.state_machine.transition(
                AgentSemanticState.FAILED,
                reason=f"Subagent failed: {str(e)}",
            )

    def merge_subagent(self, subagent_id: str) -> bool:
        """Merge verified subagent branch into active branch and prune worktree."""
        task = self.get_task(subagent_id)
        if not task:
            raise KeyError(f"Subagent '{subagent_id}' not found.")

        if not task.state_machine.is_terminal():
            raise RuntimeError(f"Subagent '{subagent_id}' is still in-flight ({task.state_machine.current_state.value}).")

        if not task.success:
            raise RuntimeError(f"Cannot merge failed subagent '{subagent_id}'.")

        if not task.branch_name:
            return True

        merged = self.checkpointer.merge_worktree_sandbox(subagent_id)
        if merged and task.worktree_path:
            self.checkpointer.prune_worktree_sandbox(task.worktree_path)
        return merged

    def discard_subagent(self, subagent_id: str) -> bool:
        """Prune and clean up a subagent worktree without merging."""
        task = self.get_task(subagent_id)
        if not task:
            return False

        if task.worktree_path and self.checkpointer.is_git_repo():
            self.checkpointer.prune_worktree_sandbox(task.worktree_path)
            # Delete branch
            subprocess.run(
                ["git", "branch", "-D", f"subagent/{subagent_id}"],
                cwd=str(self.workspace_root),
                capture_output=True,
            )
        return True
