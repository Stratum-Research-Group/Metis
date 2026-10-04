"""Atomic policy manager with hot-reloading and thread safety."""

from pathlib import Path
import threading
from typing import Callable

from phylax.contracts.policy import PolicyDocument
from phylax.engine.evaluator import PolicyEvaluator
from phylax.linter.linter import LintSeverity, PolicyLinter


class PolicyManager:
    """Manages policy lifecycle, atomic reloading, and thread-safe evaluator access."""

    def __init__(self, policy_path: Path | str, workspace_root: Path | None = None):
        self.policy_path = Path(policy_path)
        self.workspace_root = workspace_root or Path.cwd()
        self._lock = threading.Lock()
        self._linter = PolicyLinter(workspace_root=self.workspace_root)
        self._policy: PolicyDocument | None = None
        self._evaluator: PolicyEvaluator | None = None
        self.reload()

    def reload(self) -> None:
        """Atomically reload and validate policy YAML.
        
        Raises ValueError if policy fails syntax or linter error checks.
        """
        policy, issues = self._linter.lint_file(self.policy_path)
        errors = [i for i in issues if i.severity == LintSeverity.ERROR]
        if errors:
            err_msg = "; ".join(str(e) for e in errors)
            raise ValueError(f"Failed to load policy due to lint errors: {err_msg}")

        if not policy:
            raise ValueError(f"Failed to load policy from {self.policy_path}")

        new_evaluator = PolicyEvaluator(policy=policy, workspace_root=self.workspace_root)

        with self._lock:
            self._policy = policy
            self._evaluator = new_evaluator

    @property
    def policy(self) -> PolicyDocument:
        with self._lock:
            assert self._policy is not None
            return self._policy

    @property
    def evaluator(self) -> PolicyEvaluator:
        with self._lock:
            assert self._evaluator is not None
            return self._evaluator
