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


class SafeSandboxedBash:
    """Safe shell executor bounded to authorized workspace roots and strict execution limits."""

    DEFAULT_TIMEOUT_SECONDS = 30
    MAX_TIMEOUT_SECONDS = 120
    MAX_OUTPUT_CHARS = 50000

    def __init__(self, allowed_roots: Optional[List[Path]] = None):
        self.allowed_roots = [r.resolve() for r in (allowed_roots or [Path.cwd()])]

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
            timeout_ms=125000,
            idempotent=False,
        )

    def _is_path_allowed(self, path: Path) -> bool:
        resolved = path.resolve()
        for root in self.allowed_roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _truncate_output(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.MAX_OUTPUT_CHARS:
            return text, False
        head_chars = self.MAX_OUTPUT_CHARS // 2
        tail_chars = self.MAX_OUTPUT_CHARS // 2
        omitted = len(text) - head_chars - tail_chars
        truncated_text = (
            f"{text[:head_chars]}\n\n"
            f"[... Metis Output Truncated: {omitted} characters omitted ...]\n\n"
            f"{text[-tail_chars:]}"
        )
        return truncated_text, True

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

    def _build_sanitized_env(self, target_cwd: Path) -> Dict[str, str]:
        """Construct a minimal, sanitized environment stripping secrets and auth tokens."""
        safe_keys = {
            "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL",
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
        start_time = time.monotonic()
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(target_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
            env=clean_env,
        )

        try:
            stdout_raw, stderr_raw = proc.communicate(timeout=timeout)
            duration = time.monotonic() - start_time
            stdout_clean, trunc_out = self._truncate_output(stdout_raw or "")
            stderr_clean, trunc_err = self._truncate_output(stderr_raw or "")

            return {
                "command": command,
                "exit_code": proc.returncode,
                "stdout": stdout_clean,
                "stderr": stderr_clean,
                "duration_seconds": round(duration, 3),
                "truncated": trunc_out or trunc_err,
            }
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start_time
            # Terminate the entire process group to ensure child processes are killed
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            out, err = proc.communicate()
            stdout_clean, _ = self._truncate_output(out or "")
            return {
                "command": command,
                "exit_code": -1,
                "stdout": stdout_clean,
                "stderr": f"Command timed out after {timeout} seconds",
                "duration_seconds": round(duration, 3),
                "truncated": False,
            }
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
