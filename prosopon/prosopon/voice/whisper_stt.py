"""Local whisper.cpp Speech-to-Text invocation and fallback."""

import asyncio
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple


class WhisperSTTEngine:
    """Invokes local whisper.cpp binary or fallback mock for speech transcription."""
    def __init__(self, binary_name: str = "whisper.cpp", model_path: str = "base.en"):
        self.binary_path = shutil.which(binary_name) or shutil.which("whisper-cli") or shutil.which("whisper")
        self.model_path = model_path

    def is_available(self) -> bool:
        return self.binary_path is not None

    async def transcribe(self, wav_path: Path) -> Tuple[str, float]:
        """
        Transcribe 16kHz WAV file.
        
        Returns:
            Tuple of (transcribed text, elapsed duration in milliseconds).
        """
        t0 = time.perf_counter()

        if self.binary_path:
            # Command: whisper -m <model> -f <wav_path> --no-timestamps -nt
            cmd = [
                self.binary_path,
                "-m", self.model_path,
                "-f", str(wav_path),
                "--no-timestamps",
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate()
            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            text = stdout.decode("utf-8", errors="replace").strip()
            return text, t_elapsed_ms
        else:
            # Fallback mock for testing/dev environments without whisper.cpp binary
            await asyncio.sleep(0.1)  # Simulate processing latency
            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return "What are the latest findings in the LTHCM benchmark?", t_elapsed_ms
