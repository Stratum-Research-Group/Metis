"""Local piper Text-to-Speech synthesis and honest fallback."""

from __future__ import annotations

import asyncio
import io
import shutil
import struct
import time
import wave
from pathlib import Path
from typing import Optional, Tuple


class TTSUnavailableError(Exception):
    """Raised when piper binary is not installed and mock mode is disabled."""
    pass


class TTSExecutionError(Exception):
    """Raised when synthesis subprocess fails or times out."""
    pass


class PiperTTSEngine:
    """Invokes local piper TTS binary or honest fallback waveform generator."""

    def __init__(
        self,
        binary_name: str = "piper",
        model_path: str = "en_US-lessac-medium",
        mock_mode: bool = False,
    ) -> None:
        self.binary_path = shutil.which(binary_name)
        self.model_path = model_path
        self.mock_mode = mock_mode

    def is_available(self) -> bool:
        return self.binary_path is not None or self.mock_mode

    async def synthesize(self, text: str, timeout_seconds: float = 30.0) -> Tuple[bytes, float, bool]:
        """Synthesize speech from text.

        Returns:
            Tuple of (WAV audio bytes, synthesis latency in milliseconds, is_simulated).
        """
        t0 = time.perf_counter()

        if self.binary_path:
            cmd = [
                self.binary_path,
                "--model", self.model_path,
                "--output_stdout",
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=text.encode("utf-8")),
                    timeout=timeout_seconds,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
                try:
                    proc.kill()
                    await proc.wait()
                except OSError:
                    pass
                raise TTSExecutionError(
                    f"piper synthesis timed out or cancelled after {timeout_seconds}s"
                ) from exc

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace").strip()
                raise TTSExecutionError(f"piper failed with exit code {proc.returncode}: {err[:200]}")

            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return stdout, t_elapsed_ms, False
        else:
            if not self.mock_mode:
                raise TTSUnavailableError(
                    "piper text-to-speech binary not found on PATH. Live synthesis is unavailable."
                )
            await asyncio.sleep(0.05)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                frames = struct.pack("<500h", *([0] * 500))
                wf.writeframes(frames)
            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return buf.getvalue(), t_elapsed_ms, True
