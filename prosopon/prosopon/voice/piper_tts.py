"""Local piper Text-to-Speech synthesis and fallback."""

import asyncio
import io
import shutil
import struct
import time
import wave
from pathlib import Path
from typing import Optional, Tuple


class PiperTTSEngine:
    """Invokes local piper TTS binary or fallback waveform generator."""
    def __init__(self, binary_name: str = "piper", model_path: str = "en_US-lessac-medium"):
        self.binary_path = shutil.which(binary_name)
        self.model_path = model_path

    def is_available(self) -> bool:
        return self.binary_path is not None

    async def synthesize(self, text: str) -> Tuple[bytes, float]:
        """
        Synthesize speech from text.
        
        Returns:
            Tuple of (WAV audio bytes, synthesis latency in milliseconds).
        """
        t0 = time.perf_counter()

        if self.binary_path:
            # piper accepts text on stdin and writes wav to stdout or file
            cmd = [
                self.binary_path,
                "--model", self.model_path,
                "--output_stdout",
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await proc.communicate(input=text.encode("utf-8"))
            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return stdout, t_elapsed_ms
        else:
            # Fallback: generate a brief valid 16kHz PCM WAV (silence/mock tone) for dev/test
            await asyncio.sleep(0.05)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                # 0.25 seconds of silence
                frames = struct.pack("<500h", *([0] * 500))
                wf.writeframes(frames)
            t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return buf.getvalue(), t_elapsed_ms
