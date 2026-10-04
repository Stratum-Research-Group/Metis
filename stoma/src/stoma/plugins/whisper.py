"""
Whisper Transcript Source Plugin.
Consumes whisper.cpp transcript outputs (.vtt, .srt, .json, and timestamped .txt).
Does NOT perform transcription; extracts dialogue turns, timestamps, and speech segments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from stoma.plugins.base import ExtractionResult, RawPassage, SourcePlugin
from stoma.provenance import compute_file_sha256


def _parse_timestamp_seconds(ts_str: str) -> float:
    """Convert HH:MM:SS.mmm or HH:MM:SS,mmm to float seconds."""
    ts_str = ts_str.replace(",", ".").strip()
    parts = ts_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600.0 + float(m) * 60.0 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60.0 + float(s)
    try:
        return float(ts_str)
    except ValueError:
        return 0.0


class WhisperPlugin(SourcePlugin):
    """Plugin to parse whisper.cpp transcript files."""

    name: str = "whisper"
    supported_extensions: Set[str] = {".vtt", ".srt", ".transcript.json", ".transcript.txt"}

    def supports_file(self, path: Path) -> bool:
        ext = path.suffix.lower()
        if ext in {".vtt", ".srt"}:
            return True
        name = path.name.lower()
        if name.endswith(".transcript.json") or name.endswith(".transcript.txt"):
            return True
        # Check if parent directory is transcripts / audio
        parent_names = {p.lower() for p in path.parent.parts}
        if "transcripts" in parent_names or "whisper" in parent_names:
            return ext in {".json", ".txt"}
        return False

    MAX_SAFE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB safe size cap for transcript parsing

    def extract(self, path: Path, rel_path: str) -> ExtractionResult:
        try:
            stat = path.stat()
            file_hash = compute_file_sha256(path)
            if stat.st_size > self.MAX_SAFE_SIZE_BYTES:
                return ExtractionResult(
                    path=path,
                    rel_path=rel_path,
                    content_hash=file_hash,
                    size_bytes=stat.st_size,
                    mtime=stat.st_mtime,
                    passages=[],
                    plugin_name=self.name,
                    is_supported=False,
                    error_message=(
                        f"Whisper transcript size {stat.st_size} bytes exceeds safe 25MB limit. "
                        f"Memory-unbounded transcript parsing unsupported; source BLOCKED to protect process memory."
                    ),
                )
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return self._create_error_result(path, rel_path, f"Failed to read transcript: {e}")

        ext = path.suffix.lower()
        passages: List[RawPassage] = []

        if ext == ".vtt":
            passages = self._parse_vtt(content)
        elif ext == ".srt":
            passages = self._parse_srt(content)
        elif ext == ".json":
            try:
                data = json.loads(content)
                passages = self._parse_json(data)
            except Exception as e:
                return self._create_error_result(path, rel_path, f"Malformed whisper JSON: {e}")
        else:
            passages = self._parse_txt(content)

        if not passages:
            return ExtractionResult(
                path=path,
                rel_path=rel_path,
                content_hash=file_hash,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                passages=[],
                plugin_name=self.name,
                warning_message="Whisper transcript contains no segments.",
            )

        return ExtractionResult(
            path=path,
            rel_path=rel_path,
            content_hash=file_hash,
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
            passages=passages,
            plugin_name=self.name,
        )

    def extract_stream(self, path: Path, rel_path: str) -> Iterator[RawPassage]:
        """Stream passages from Whisper transcript."""
        from stoma.plugins.base import ExtractionError, UnsupportedDocumentError
        res = self.extract(path, rel_path)
        if not res.is_supported:
            raise UnsupportedDocumentError(res.error_message or f"Unsupported Whisper transcript '{rel_path}'")
        if res.error_message:
            raise ExtractionError(res.error_message)
        yield from res.passages

    def _parse_vtt(self, content: str) -> List[RawPassage]:
        passages: List[RawPassage] = []
        # Pattern: 00:00:01.000 --> 00:00:04.500
        time_pat = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})")
        lines = content.splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx].strip()
            m = time_pat.search(line)
            if m:
                start_ts, end_ts = m.group(1), m.group(2)
                start_sec = _parse_timestamp_seconds(start_ts)
                end_sec = _parse_timestamp_seconds(end_ts)
                text_lines = []
                idx += 1
                while idx < len(lines) and lines[idx].strip() and not time_pat.search(lines[idx]):
                    text_lines.append(lines[idx].strip())
                    idx += 1
                text = " ".join(text_lines).strip()
                if text:
                    passages.append(
                        RawPassage(
                            content=text,
                            location={
                                "start_sec": start_sec,
                                "end_sec": end_sec,
                                "timestamp": f"{start_ts} --> {end_ts}",
                            },
                            section=f"[{start_ts} --> {end_ts}]",
                        )
                    )
            else:
                idx += 1
        return passages

    def _parse_srt(self, content: str) -> List[RawPassage]:
        passages: List[RawPassage] = []
        blocks = re.split(r"\n\s*\n", content.strip())
        time_pat = re.compile(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})")

        for block in blocks:
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if not lines:
                continue
            for i, line in enumerate(lines):
                m = time_pat.search(line)
                if m:
                    start_ts, end_ts = m.group(1), m.group(2)
                    text = " ".join(lines[i + 1:]).strip()
                    if text:
                        passages.append(
                            RawPassage(
                                content=text,
                                location={
                                    "start_sec": _parse_timestamp_seconds(start_ts),
                                    "end_sec": _parse_timestamp_seconds(end_ts),
                                    "timestamp": f"{start_ts} --> {end_ts}",
                                },
                                section=f"[{start_ts} --> {end_ts}]",
                            )
                        )
                    break
        return passages

    def _parse_json(self, data: Any) -> List[RawPassage]:
        passages: List[RawPassage] = []
        segments = []
        if isinstance(data, dict):
            if "transcription" in data and isinstance(data["transcription"], list):
                segments = data["transcription"]
            elif "segments" in data and isinstance(data["segments"], list):
                segments = data["segments"]
        elif isinstance(data, list):
            segments = data

        for seg in segments:
            if not isinstance(seg, dict):
                continue
            text = seg.get("text", "").strip()
            if not text:
                continue

            start_sec = 0.0
            end_sec = 0.0
            ts_str = ""

            if "timestamps" in seg and isinstance(seg["timestamps"], dict):
                from_str = seg["timestamps"].get("from", "0")
                to_str = seg["timestamps"].get("to", "0")
                start_sec = _parse_timestamp_seconds(from_str)
                end_sec = _parse_timestamp_seconds(to_str)
                ts_str = f"{from_str} --> {to_str}"
            elif "start" in seg and "end" in seg:
                start_sec = float(seg.get("start", 0.0))
                end_sec = float(seg.get("end", 0.0))
                ts_str = f"{start_sec:.2f}s --> {end_sec:.2f}s"
            elif "offsets" in seg and isinstance(seg["offsets"], dict):
                start_sec = seg["offsets"].get("from", 0) / 1000.0
                end_sec = seg["offsets"].get("to", 0) / 1000.0
                ts_str = f"{start_sec:.2f}s --> {end_sec:.2f}s"

            passages.append(
                RawPassage(
                    content=text,
                    location={
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "timestamp": ts_str,
                    },
                    section=ts_str or "Transcript",
                )
            )
        return passages

    def _parse_txt(self, content: str) -> List[RawPassage]:
        passages: List[RawPassage] = []
        bracket_pat = re.compile(r"^\[(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\]\s*(.*)$")

        for line in content.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            m = bracket_pat.match(line_str)
            if m:
                start_ts, end_ts, text = m.group(1), m.group(2), m.group(3).strip()
                if text:
                    passages.append(
                        RawPassage(
                            content=text,
                            location={
                                "start_sec": _parse_timestamp_seconds(start_ts),
                                "end_sec": _parse_timestamp_seconds(end_ts),
                                "timestamp": f"{start_ts} --> {end_ts}",
                            },
                            section=f"[{start_ts} --> {end_ts}]",
                        )
                    )
            else:
                passages.append(
                    RawPassage(
                        content=line_str,
                        location={"timestamp": "plain"},
                        section="Transcript",
                    )
                )
        return passages
