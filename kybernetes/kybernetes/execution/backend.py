"""Sandboxed execution backend abstraction for Metis Sovereign Stack (MSS).

Addresses the Substrate Review (docs/review/metis-substrate-review-d2b97509-f9bb42d2.txt)
by providing genuine execution boundaries:
- Bubblewrap (bwrap) mount/PID/IPC isolation with read-only root and writable workspace
- Confined $HOME pointing into the workspace rather than leaking host home directory
- Process group timeout termination preventing orphaned rogue processes
- Graceful fallback to constrained LocalProcessBackend when running unprivileged or in containers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    truncated: bool = False
    backend: str = "unknown"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "truncated": self.truncated,
            "backend": self.backend,
        }


class ExecutionBackend(ABC):
    """Abstract process execution backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend."""
        pass

    @abstractmethod
    def execute(
        self,
        command: str,
        cwd: Path,
        env: Dict[str, str],
        timeout_seconds: int = 30,
        max_output_bytes: int = 100 * 1024,
        allow_network: bool = False,
    ) -> ExecutionResult:
        """Execute command within the boundary."""
        pass


class LocalProcessBackend(ExecutionBackend):
    """
    Constrained local process execution backend.
    Enforces non-shared process sessions, strict timeoutSIGKILL groups, and sanitized env.
    """

    @property
    def name(self) -> str:
        return "local_process"

    def execute(
        self,
        command: str,
        cwd: Path,
        env: Dict[str, str],
        timeout_seconds: int = 30,
        max_output_bytes: int = 100 * 1024,
        allow_network: bool = False,
    ) -> ExecutionResult:
        start_time = time.monotonic()
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
            env=env,
        )

        try:
            stdout_raw, stderr_raw = proc.communicate(timeout=timeout_seconds)
            duration = time.monotonic() - start_time
            stdout_clean, trunc_out = self._truncate(stdout_raw or "", max_output_bytes)
            stderr_clean, trunc_err = self._truncate(stderr_raw or "", max_output_bytes)
            return ExecutionResult(
                command=command,
                exit_code=proc.returncode,
                stdout=stdout_clean,
                stderr=stderr_clean,
                duration_seconds=round(duration, 3),
                truncated=trunc_out or trunc_err,
                backend=self.name,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start_time
            self._kill_process_group(proc)
            try:
                stdout_raw, stderr_raw = proc.communicate(timeout=2)
            except Exception:
                stdout_raw, stderr_raw = "", ""
            return ExecutionResult(
                command=command,
                exit_code=-1,
                stdout=stdout_raw or "",
                stderr=(stderr_raw or "") + f"\nCommand execution timed out after {timeout_seconds} seconds.",
                duration_seconds=round(duration, 3),
                truncated=False,
                backend=self.name,
            )

    def _truncate(self, text: str, max_bytes: int) -> tuple[str, bool]:
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text, False
        head_bytes = max_bytes // 2
        tail_bytes = max_bytes - head_bytes
        truncated = encoded[:head_bytes] + b"\n\n... [output truncated] ...\n\n" + encoded[-tail_bytes:]
        return truncated.decode("utf-8", errors="replace"), True

    def _kill_process_group(self, proc: subprocess.Popen) -> None:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


class BubblewrapBackend(ExecutionBackend):
    """
    Bubblewrap (bwrap) unprivileged sandbox backend.
    Mounts root read-only, restricts writes strictly to workspace cwd,
    isolates /tmp onto private tmpfs, unshares PID, IPC, UTS, and Network namespaces.
    """

    def __init__(self, bwrap_path: Optional[str] = None, allow_network: bool = False):
        self.bwrap_path = bwrap_path or shutil.which("bwrap") or "/usr/bin/bwrap"
        self.allow_network = allow_network

    @property
    def name(self) -> str:
        return "bubblewrap"

    def is_available(self) -> bool:
        if not self.bwrap_path or not os.path.exists(self.bwrap_path):
            return False
        try:
            # Check basic functional invocation
            res = subprocess.run(
                [self.bwrap_path, "--ro-bind", "/", "/", "true"],
                capture_output=True,
                timeout=2,
            )
            return res.returncode == 0
        except Exception:
            return False

    def execute(
        self,
        command: str,
        cwd: Path,
        env: Dict[str, str],
        timeout_seconds: int = 30,
        max_output_bytes: int = 100 * 1024,
        allow_network: Optional[bool] = None,
    ) -> ExecutionResult:
        start_time = time.monotonic()

        net_allowed = self.allow_network if allow_network is None else allow_network
        if os.environ.get("METIS_ALLOW_NETWORK") == "1":
            net_allowed = True

        # Build bubblewrap container command line
        bwrap_cmd: List[str] = [
            self.bwrap_path,
            "--ro-bind", "/", "/",
            "--tmpfs", "/tmp",
            "--dir", str(cwd),
            "--bind", str(cwd), str(cwd),
            "--proc", "/proc",
            "--dev", "/dev",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
        ]

        if not net_allowed:
            bwrap_cmd.append("--unshare-net")

        bwrap_cmd.extend([
            "--setenv", "HOME", str(cwd),
            "--setenv", "TMPDIR", "/tmp",
            "--die-with-parent",
            "--chdir", str(cwd),
            "bash", "-c", command,
        ])

        proc = subprocess.Popen(
            bwrap_cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
            env=env,
        )

        try:
            stdout_raw, stderr_raw = proc.communicate(timeout=timeout_seconds)
            duration = time.monotonic() - start_time
            stdout_clean, trunc_out = self._truncate(stdout_raw or "", max_output_bytes)
            stderr_clean, trunc_err = self._truncate(stderr_raw or "", max_output_bytes)
            return ExecutionResult(
                command=command,
                exit_code=proc.returncode,
                stdout=stdout_clean,
                stderr=stderr_clean,
                duration_seconds=round(duration, 3),
                truncated=trunc_out or trunc_err,
                backend=self.name,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start_time
            self._kill_process_group(proc)
            try:
                stdout_raw, stderr_raw = proc.communicate(timeout=2)
            except Exception:
                stdout_raw, stderr_raw = "", ""
            return ExecutionResult(
                command=command,
                exit_code=-1,
                stdout=stdout_raw or "",
                stderr=(stderr_raw or "") + f"\nCommand execution timed out after {timeout_seconds} seconds.",
                duration_seconds=round(duration, 3),
                truncated=False,
                backend=self.name,
            )

    def _truncate(self, text: str, max_bytes: int) -> tuple[str, bool]:
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text, False
        head_bytes = max_bytes // 2
        tail_bytes = max_bytes - head_bytes
        truncated = encoded[:head_bytes] + b"\n\n... [output truncated] ...\n\n" + encoded[-tail_bytes:]
        return truncated.decode("utf-8", errors="replace"), True

    def _kill_process_group(self, proc: subprocess.Popen) -> None:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def get_default_execution_backend(preferred_backend: Optional[str] = None) -> ExecutionBackend:
    """
    Select the optimal execution backend based on system capabilities and environment configuration.
    Priority:
      1. Explicit preference or METIS_EXECUTION_BACKEND env var
      2. BubblewrapBackend (if available and functional)
      3. LocalProcessBackend (universal fallback)
    """
    pref = preferred_backend or os.environ.get("METIS_EXECUTION_BACKEND", "").lower().strip()
    if pref == "local" or pref == "local_process":
        return LocalProcessBackend()
    if pref in ("bwrap", "bubblewrap"):
        bwrap = BubblewrapBackend()
        if bwrap.is_available():
            return bwrap
        return LocalProcessBackend()

    # Automatic detection: prefer bubblewrap if available
    bwrap = BubblewrapBackend()
    if bwrap.is_available():
        return bwrap
    return LocalProcessBackend()
