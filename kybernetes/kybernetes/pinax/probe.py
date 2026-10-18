"""Pinax Tool Prober: active, side-effect-free health probing state machine.
Implements Phase 1 / Agent-Reach model:
- 5-state probe outcome: ok, missing, broken, timeout, error
- Specifically detects stale .venv shims (shim exists but python interpreter was removed)
- Sanitizes environment (strips tokens, secrets, credentials)
- Strict timeouts and allowlist of side-effect-free flags (--version, -V)
"""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from kybernetes.pinax.models import ProbeResult, ProbeStatus


class PinaxProber:
    """
    Executes controlled, side-effect-free health probes against inventoried tools.
    Never executes arbitrary commands or untested flags.
    """

    ALLOWED_PROBE_FLAGS: Set[str] = {
        "--version",
        "-V",
        "-v",
        "version",
    }
    DEFAULT_TIMEOUT_SECONDS: float = 2.0
    MAX_OUTPUT_BYTES: int = 4096

    @classmethod
    def check_stale_shim(cls, filepath: str) -> Tuple[bool, Optional[str]]:
        """
        Inspects script shebang to verify interpreter exists on disk.
        Detects the common failure mode where a .venv/bin/tool shim exists
        but its Python virtual environment was removed or broken.
        """
        p = Path(filepath)
        try:
            resolved = p.resolve(strict=True)
        except (FileNotFoundError, OSError):
            return True, f"File '{filepath}' cannot be resolved"

        if not resolved.is_file():
            return False, None

        try:
            with open(resolved, "rb") as f:
                header = f.read(512)
            if header.startswith(b"#!"):
                first_line = header.split(b"\n")[0].decode("utf-8", errors="replace")
                shebang_content = first_line[2:].strip()
                parts = shebang_content.split()
                if not parts:
                    return True, "Empty shebang in executable script"

                interp = parts[0]
                if interp == "/usr/bin/env" and len(parts) > 1:
                    target_cmd = parts[1]
                    if not shutil.which(target_cmd):
                        return True, f"Shebang interpreter '{target_cmd}' via /usr/bin/env not found on PATH"
                elif not os.path.exists(interp):
                    return True, f"Stale shim: interpreter '{interp}' does not exist on disk"
        except Exception as e:
            return False, f"Could not inspect script header: {e}"

        return False, None

    @classmethod
    def build_sanitized_probe_env(cls) -> Dict[str, str]:
        """Constructs a minimal, secure environment completely stripped of secrets."""
        safe_keys = {
            "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL",
            "TERM", "TMPDIR", "TZ"
        }
        clean_env = {k: v for k, v in os.environ.items() if k in safe_keys}

        # Inherit non-secret toolchain vars (e.g. PYTHONHOME, CARGO_HOME)
        for k, v in os.environ.items():
            k_upper = k.upper()
            if any(term in k_upper for term in ("TOKEN", "KEY", "SECRET", "PASSWORD", "AUTH", "CREDENTIAL", "ADMIN", "METIS_")):
                continue
            if k.startswith(("PYTHON", "CARGO", "RUST", "GO", "GIT")):
                clean_env[k] = v

        clean_env["PAGER"] = "cat"
        clean_env["PYTHONUNBUFFERED"] = "1"
        clean_env["TERM"] = "dumb"
        return clean_env

    @classmethod
    def probe_command(
        cls,
        tool_id: str,
        invocation_path: Optional[str],
        args: Optional[List[str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> ProbeResult:
        """
        Executes a side-effect-free health probe and returns a typed ProbeResult.
        Status classifications:
        - ok: exit code 0
        - missing: invocation path does not exist
        - broken: stale venv shim, missing interpreter, or dynamic linker failure
        - timeout: execution took longer than timeout limit
        - error: non-zero exit code or OS launch failure
        """
        start_time = time.monotonic()
        timeout = max(0.2, min(timeout_seconds or cls.DEFAULT_TIMEOUT_SECONDS, 10.0))

        if not invocation_path:
            return ProbeResult(
                tool_id=tool_id,
                status=ProbeStatus.MISSING,
                detail="Invocation path is empty or undefined",
                duration_ms=0.0,
            )

        inv_p = Path(invocation_path)
        if not inv_p.exists():
            return ProbeResult(
                tool_id=tool_id,
                status=ProbeStatus.MISSING,
                detail=f"Target file '{invocation_path}' not found on filesystem",
                duration_ms=0.0,
            )

        # Check for stale venv shim before running
        is_stale, stale_reason = cls.check_stale_shim(invocation_path)
        if is_stale:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            return ProbeResult(
                tool_id=tool_id,
                status=ProbeStatus.BROKEN,
                exit_code=None,
                detail=stale_reason or "Stale shim detected",
                is_stale_shim=True,
                duration_ms=duration_ms,
            )

        # Validate probe arguments: default to --version, restrict to allowlist
        probe_args = args or ["--version"]
        for arg in probe_args:
            if arg not in cls.ALLOWED_PROBE_FLAGS:
                probe_args = ["--version"]
                break

        cmd = [str(inv_p.resolve())] + probe_args
        clean_env = cls.build_sanitized_probe_env()

        try:
            proc = subprocess.Popen(
                cmd,
                env=clean_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            try:
                stdout_data, stderr_data = proc.communicate(timeout=timeout)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                # Terminate entire process group
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.communicate()
                duration_ms = (time.monotonic() - start_time) * 1000.0
                return ProbeResult(
                    tool_id=tool_id,
                    status=ProbeStatus.TIMEOUT,
                    exit_code=None,
                    duration_ms=duration_ms,
                    detail=f"Probe exceeded timeout limit of {timeout}s",
                )

            duration_ms = (time.monotonic() - start_time) * 1000.0
            stdout_trunc = stdout_data[:cls.MAX_OUTPUT_BYTES]
            stderr_trunc = stderr_data[:cls.MAX_OUTPUT_BYTES]

            if exit_code == 0:
                return ProbeResult(
                    tool_id=tool_id,
                    status=ProbeStatus.OK,
                    exit_code=0,
                    stdout=stdout_trunc,
                    stderr=stderr_trunc,
                    duration_ms=duration_ms,
                    detail="Probe exited cleanly with status 0",
                )

            # Check if stderr points to broken interpreter or missing dynamic lib
            err_lower = stderr_trunc.lower()
            if any(term in err_lower for term in ("no such file or directory", "cannot execute", "command not found", "error while loading shared libraries")):
                return ProbeResult(
                    tool_id=tool_id,
                    status=ProbeStatus.BROKEN,
                    exit_code=exit_code,
                    stdout=stdout_trunc,
                    stderr=stderr_trunc,
                    duration_ms=duration_ms,
                    detail=f"Binary broken or missing shared libraries (exit {exit_code})",
                )

            return ProbeResult(
                tool_id=tool_id,
                status=ProbeStatus.ERROR,
                exit_code=exit_code,
                stdout=stdout_trunc,
                stderr=stderr_trunc,
                duration_ms=duration_ms,
                detail=f"Probe exited with non-zero status {exit_code}",
            )

        except PermissionError as pe:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            return ProbeResult(
                tool_id=tool_id,
                status=ProbeStatus.BROKEN,
                duration_ms=duration_ms,
                detail=f"Permission denied executing target: {pe}",
            )
        except OSError as oe:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            return ProbeResult(
                tool_id=tool_id,
                status=ProbeStatus.ERROR,
                duration_ms=duration_ms,
                detail=f"OS error during probe launch: {oe}",
            )
