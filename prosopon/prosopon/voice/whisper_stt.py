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

    def __init__(
        self,
        binary_name: str = "whisper.cpp",
        model_path: str = "base.en",
        mock_mode: bool = False,
    ) -> None:
        self.binary_path = shutil.which(binary_name) or shutil.which("whisper-cli") or shutil.which("whisper")
        self.model_path = model_path
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
