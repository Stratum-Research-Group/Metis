"""Safe sandboxed bash executor bounded to workspace roots and strict execution timeouts."""

import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional

from kybernetes.contracts.enums import ToolSideEffectClass
from kybernetes.contracts.tool_spec import ToolDefinition
from kybernetes.execution import ExecutionBackend, get_default_execution_backend


class SafeSandboxedBash:
    """Safe shell executor bounded to authorized workspace roots and strict execution limits."""

    DEFAULT_TIMEOUT_SECONDS = 30
    MAX_TIMEOUT_SECONDS = 120
    MAX_OUTPUT_CHARS = 50000

    def __init__(
        self,
        allowed_roots: Optional[List[Path]] = None,
        backend: Optional[ExecutionBackend] = None,
    ):
        self.allowed_roots = [r.resolve() for r in (allowed_roots or [Path.cwd()])]
        self.backend = backend or get_default_execution_backend()

    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="sandboxed_bash",
            version="1.0.0",
            description="Execute a non-interactive shell command in a workspace directory with bounded timeout and output truncation.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command line to execute",
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory relative to authorized workspace (default root)",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (default 30, max 120)",
                    },
                },
                "required": ["command"],
            },
            returns_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "exit_code": {"type": "integer"},
                    "stdout": {"type": "string"},
                    "stderr": {"type": "string"},
                    "duration_seconds": {"type": "number"},
                    "truncated": {"type": "boolean"},
                },
            },
            declared_capabilities=["execute_command", "bash"],
            side_effect_class=ToolSideEffectClass.SYSTEM_EXECUTION,
            timeout_ms=120000,
        )

    def _is_path_allowed(self, target_path: Path) -> bool:
        resolved = target_path.resolve()
        for root in self.allowed_roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _build_sanitized_env(self, target_cwd: Path) -> Dict[str, str]:
        """Construct a minimal, sanitized environment stripping secrets and auth tokens."""
        safe_keys = {
            "PATH", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL",
            "TERM", "TMPDIR", "TZ", "PWD"
        }
        clean_env = {k: v for k, v in os.environ.items() if k in safe_keys}

        # Inherit non-secret development/toolchain vars
        for k, v in os.environ.items():
            k_upper = k.upper()
            if any(term in k_upper for term in ("TOKEN", "KEY", "SECRET", "PASSWORD", "AUTH", "CREDENTIAL", "ADMIN", "METIS_")):
                continue
            if k.startswith(("PYTHON", "CARGO", "RUST", "GO", "GIT", "VIRTUAL_ENV")):
                clean_env[k] = v

        clean_env["PAGER"] = "cat"
        clean_env["PYTHONUNBUFFERED"] = "1"
        clean_env["TERM"] = "dumb"

        # Confine HOME into workspace rather than leaking host user directory
        clean_env["HOME"] = str(target_cwd)

        # Confine TMPDIR into workspace .metis/tmp if possible
        ws_tmp = target_cwd / ".metis" / "tmp"
        try:
            ws_tmp.mkdir(parents=True, exist_ok=True)
            clean_env["TMPDIR"] = str(ws_tmp)
        except Exception:
            pass

        return clean_env

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not command or not command.strip():
            raise ValueError("Command cannot be empty")

        target_cwd = Path(cwd) if cwd else self.allowed_roots[0]
        if not target_cwd.is_absolute():
            target_cwd = (self.allowed_roots[0] / target_cwd).resolve()
        else:
            target_cwd = target_cwd.resolve()

        if not self._is_path_allowed(target_cwd):
            raise PermissionError(f"CWD {target_cwd} is outside authorized roots: {self.allowed_roots}")

        if not target_cwd.exists() or not target_cwd.is_dir():
            raise FileNotFoundError(f"Working directory does not exist: {target_cwd}")

        timeout = min(
            max(1, timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS),
            self.MAX_TIMEOUT_SECONDS,
        )

        clean_env = self._build_sanitized_env(target_cwd)
        res = self.backend.execute(
            command=command,
            cwd=target_cwd,
            env=clean_env,
            timeout_seconds=timeout,
        )

        return {
            "command": res.command,
            "exit_code": res.exit_code,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "duration_seconds": res.duration_seconds,
            "truncated": res.truncated,
            "backend": res.backend,
        }
