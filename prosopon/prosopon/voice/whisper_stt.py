"""Local whisper.cpp Speech-to-Text invocation and honest fallback."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple


class STTUnavailableError(Exception):
    """Raised when whisper binary is not installed and mock mode is disabled."""
    pass


class STTExecutionError(Exception):
    """Raised when transcription subprocess fails or times out."""
    pass


class WhisperSTTEngine:
    """Invokes local whisper.cpp binary or honest fallback mock for speech transcription."""

    @staticmethod
    def _resolve_binary(binary_name: str) -> Optional[str]:
        """Resolves whisper binary from env, path, or standard workstation locations."""
        # 1. Environment variable override
        env_val = os.environ.get("WHISPER_BINARY")
        if env_val:
            p_env = Path(env_val).expanduser()
            if p_env.is_file() and os.access(p_env, os.X_OK):
                return str(p_env.resolve())
            resolved = shutil.which(env_val)
            if resolved:
                return resolved

        # 2. Passed binary name
        if binary_name and binary_name != "nonexistent_whisper_binary":
            resolved = shutil.which(binary_name)
            if resolved:
                return resolved
            p_bin = Path(binary_name).expanduser()
            if p_bin.is_file() and os.access(p_bin, os.X_OK):
                return str(p_bin.resolve())

        if binary_name == "nonexistent_whisper_binary":
            return None

        # 3. Common binary aliases
        for alias in ("whisper.cpp", "whisper-cli", "whisper", "main"):
            res = shutil.which(alias)
            if res:
                return res

        # 4. Standard workstation locations
        home = Path.home()
        candidates = [
            home / ".unsloth" / "whisper.cpp" / "build" / "bin" / "whisper-cli",
            home / ".unsloth" / "whisper.cpp" / "build" / "bin" / "main",
            home / ".local" / "bin" / "whisper-cli",
            home / ".local" / "bin" / "whisper.cpp",
            home / ".local" / "bin" / "whisper",
            Path("/usr/local/bin/whisper-cli"),
            Path("/usr/local/bin/whisper.cpp"),
        ]
        for c in candidates:
            if c.is_file() and os.access(c, os.X_OK):
                return str(c.resolve())

        return None

    def __init__(
        self,
        binary_name: str = "whisper.cpp",
        model_path: str = "base.en",
        mock_mode: bool = False,
    ) -> None:
        self.binary_path = self._resolve_binary(binary_name)
        self.model_path = os.environ.get("WHISPER_MODEL_PATH") or model_path
        self.mock_mode = mock_mode

    def is_available(self) -> bool:
        return self.binary_path is not None or self.mock_mode

    async def transcribe(self, wav_path: Path, timeout_seconds: float = 30.0) -> Tuple[str, float, bool]:
        """Transcribe 16kHz WAV file.

        Returns:
            Tuple of (transcribed text, elapsed duration in milliseconds, is_simulated).
        """
        t0 = time.perf_counter()

        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        if self.binary_path:
            cmd = [
                self.binary_path,
                "-m", self.model_path,
                "-f", str(wav_path),
                "--no-timestamps",
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
                try:
                    proc.kill()
                    await proc.wait()
                except OSError:
                    pass
                raise STTExecutionError(
                    f"whisper transcription timed out or cancelled after {timeout_seconds}s"
                ) from exc

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace").strip()
                raise STTExecutionError(f"whisper failed with exit code {proc.returncode}: {err[:200]}")

            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            text = stdout.decode("utf-8", errors="replace").strip()
            return text, t_elapsed_ms, False
        else:
            if not self.mock_mode:
                raise STTUnavailableError(
                    "whisper speech-to-text binary not found on PATH. Live transcription is unavailable."
                )
            await asyncio.sleep(0.05)
            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return (
                "[SIMULATED TRANSCRIPT] What are the latest findings in the LTHCM benchmark?",
                t_elapsed_ms,
                True,
            )
