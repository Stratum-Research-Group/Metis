"""Audio ingestion, container validation, and 16kHz mono WAV transcoding."""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional


class AudioTranscodingError(Exception):
    """Raised when audio conversion or format decoding fails."""
    pass


async def transcode_to_wav(
    audio_bytes: bytes,
    max_duration_seconds: int = 45,
    max_input_bytes: int = 15 * 1024 * 1024,
) -> Path:
    """
    Transcode browser audio (WebM/Opus, MP4, OGG) to 16kHz 16-bit mono WAV for whisper.cpp.
    
    Args:
        audio_bytes: Raw binary payload from MediaRecorder.
        max_duration_seconds: Hard safety cap on voice prompt duration.
        max_input_bytes: Hard safety cap on upload size (15 MB).
        
    Returns:
        Path to temporary 16kHz mono WAV file. Caller is responsible for unlinking when done.
    """
    if len(audio_bytes) > max_input_bytes:
        raise AudioTranscodingError(f"Audio payload ({len(audio_bytes)} bytes) exceeds maximum permitted limit.")
    if len(audio_bytes) == 0:
        raise AudioTranscodingError("Empty audio buffer received.")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise AudioTranscodingError("ffmpeg is required for audio transcoding but was not found on PATH.")

    # Write input buffer to temp file
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as in_file:
        in_file.write(audio_bytes)
        in_path = Path(in_file.name)

    out_path = in_path.with_suffix(".wav")

    # ffmpeg command: -y (overwrite), -t max_duration, -i input, -ar 16000 (16kHz), -ac 1 (mono), -c:a pcm_s16le
    cmd = [
        ffmpeg_bin,
        "-y",
        "-t", str(max_duration_seconds),
        "-i", str(in_path),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace").strip()
            raise AudioTranscodingError(f"ffmpeg conversion failed (code {proc.returncode}): {err_msg[:200]}")
        return out_path
    finally:
        if in_path.exists():
            in_path.unlink()
