"""
Plain Text and Markdown Source Plugin.
Extracts sections, headings, paragraphs, and line number ranges.
Implements bounded streaming extraction to prevent high memory usage on large files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, Set

from stoma.plugins.base import ExtractionResult, RawPassage, SourcePlugin
from stoma.provenance import compute_file_sha256


class TextPlugin(SourcePlugin):
    """Plugin for plain text, markdown, and restructured text documents."""

    name: str = "text"
    supported_extensions: Set[str] = {".txt", ".md", ".markdown", ".rst"}

    def extract_stream(self, path: Path, rel_path: str) -> Iterator[RawPassage]:
        """Stream passages from file without loading entire file into memory at once."""
        current_section = "Document"
        current_lines: List[str] = []
        start_line = 1

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for idx, line in enumerate(f, start=1):
                    stripped = line.strip()
                    # Detect Markdown headings
                    if stripped.startswith("#"):
                        if current_lines:
                            text_block = "\n".join(current_lines).strip()
                            if text_block:
                                yield RawPassage(
                                    content=text_block,
                                    location={
                                        "start_line": start_line,
                                        "end_line": idx - 1,
                                        "line_start": start_line,
                                        "line_end": idx - 1,
                                    },
                                    section=current_section,
                                )
                            current_lines = []
                        current_section = stripped.lstrip("#").strip()
                        current_lines.append(line.rstrip("\r\n"))
                        start_line = idx
                    elif not stripped:
                        # Blank line: paragraph boundary if buffer is non-empty
                        if len(current_lines) >= 10:
                            text_block = "\n".join(current_lines).strip()
                            if text_block:
                                yield RawPassage(
                                    content=text_block,
                                    location={
                                        "start_line": start_line,
                                        "end_line": idx,
                                        "line_start": start_line,
                                        "line_end": idx,
                                    },
                                    section=current_section,
                                )
                            current_lines = []
                            start_line = idx + 1
                        else:
                            current_lines.append(line.rstrip("\r\n"))
                    else:
                        current_lines.append(line.rstrip("\r\n"))

                        # Yield in bounded chunks of max 50 lines to keep memory strictly bounded
                        if len(current_lines) >= 50:
                            text_block = "\n".join(current_lines).strip()
                            if text_block:
                                yield RawPassage(
                                    content=text_block,
                                    location={
                                        "start_line": start_line,
                                        "end_line": idx,
                                        "line_start": start_line,
                                        "line_end": idx,
                                    },
                                    section=current_section,
                                )
                            current_lines = []
                            start_line = idx + 1

                # Flush final buffer
                if current_lines:
                    text_block = "\n".join(current_lines).strip()
                    if text_block:
                        end_l = start_line + len(current_lines) - 1
                        yield RawPassage(
                            content=text_block,
                            location={
                                "start_line": start_line,
                                "end_line": end_l,
                                "line_start": start_line,
                                "line_end": end_l,
                            },
                            section=current_section,
                        )
        except Exception as e:
            from stoma.plugins.base import CorruptedDocumentError
            raise CorruptedDocumentError(f"Failed to read text file '{rel_path}': {e}") from e

    def extract(self, path: Path, rel_path: str) -> ExtractionResult:
        try:
            stat = path.stat()
            file_hash = compute_file_sha256(path)
        except Exception as e:
            return self._create_error_result(path, rel_path, f"Failed to access file: {e}")

        if stat.st_size == 0:
            return ExtractionResult(
                path=path,
                rel_path=rel_path,
                content_hash=file_hash,
                size_bytes=0,
                mtime=stat.st_mtime,
                passages=[],
                plugin_name=self.name,
                warning_message="File is empty",
            )

        passages = list(self.extract_stream(path, rel_path))

        if not passages:
            return ExtractionResult(
                path=path,
                rel_path=rel_path,
                content_hash=file_hash,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                passages=[],
                plugin_name=self.name,
                warning_message="File contains only whitespace.",
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
