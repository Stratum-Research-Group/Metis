"""
Code Repository Source Plugin.
Performs language-aware extraction at function and class boundaries where cheap.
Uses Python AST for .py files, and robust regex/structural parsing for other languages.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator, List, Optional, Set

from stoma.plugins.base import ExtractionResult, RawPassage, SourcePlugin
from stoma.provenance import compute_file_sha256


class CodePlugin(SourcePlugin):
    """Plugin for source code repositories with syntax boundary awareness."""

    name: str = "code"
    supported_extensions: Set[str] = {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".java",
        ".sh",
        ".bash",
        ".sql",
    }

    MAX_SAFE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB safe size cap for AST parsing

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
                        f"Code file size {stat.st_size} bytes exceeds safe 25MB AST limit. "
                        f"Memory-unbounded AST parsing unsupported; source BLOCKED to protect process memory."
                    ),
                )
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return self._create_error_result(path, rel_path, f"Failed to read code file: {e}")

        lines = content.splitlines()
        if not lines or not content.strip():
            return ExtractionResult(
                path=path,
                rel_path=rel_path,
                content_hash=file_hash,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                passages=[],
                plugin_name=self.name,
                warning_message="Code file is empty.",
            )

        ext = path.suffix.lower()
        if ext == ".py":
            passages = self._extract_python(content, lines)
        else:
            passages = self._extract_generic_code(content, lines, ext)

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
        """Stream passages from code file."""
        from stoma.plugins.base import ExtractionError, UnsupportedDocumentError
        res = self.extract(path, rel_path)
        if not res.is_supported:
            raise UnsupportedDocumentError(res.error_message or f"Unsupported code file '{rel_path}'")
        if res.error_message:
            raise ExtractionError(res.error_message)
        yield from res.passages

    def _extract_python(self, content: str, lines: List[str]) -> List[RawPassage]:
        """Extract Python code using the standard library ast parser."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._extract_generic_code(content, lines, ".py")

        passages: List[RawPassage] = []
        covered_lines = set()

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
                symbol_name = node.name

                code_block = "\n".join(lines[start - 1 : end])
                passages.append(
                    RawPassage(
                        content=code_block,
                        location={
                            "start_line": start,
                            "end_line": end,
                            "symbol": symbol_name,
                            "symbol_type": symbol_type,
                            "language": "python",
                        },
                        section=f"{symbol_type} {symbol_name}",
                    )
                )
                for line_no in range(start, end + 1):
                    covered_lines.add(line_no)

        # Collect top-level code or docstrings not inside classes/functions
        uncovered_start = None
        uncovered_buffer = []

        for idx, line in enumerate(lines, start=1):
            if idx not in covered_lines:
                if uncovered_start is None:
                    uncovered_start = idx
                uncovered_buffer.append(line)
            else:
                if uncovered_buffer and "\n".join(uncovered_buffer).strip():
                    passages.append(
                        RawPassage(
                            content="\n".join(uncovered_buffer).strip(),
                            location={
                                "start_line": uncovered_start,
                                "end_line": idx - 1,
                                "symbol": "module_scope",
                                "symbol_type": "preamble",
                                "language": "python",
                            },
                            section="module_scope",
                        )
                    )
                uncovered_start = None
                uncovered_buffer = []

        if uncovered_buffer and "\n".join(uncovered_buffer).strip():
            passages.append(
                RawPassage(
                    content="\n".join(uncovered_buffer).strip(),
                    location={
                        "start_line": uncovered_start,
                        "end_line": len(lines),
                        "symbol": "module_scope",
                        "symbol_type": "postscript",
                        "language": "python",
                    },
                    section="module_scope",
                )
            )

        # Sort passages by start_line
        passages.sort(key=lambda p: p.location.get("start_line", 0))
        return passages if passages else [RawPassage(content=content, location={"start_line": 1, "end_line": len(lines)})]

    def _extract_generic_code(self, content: str, lines: List[str], ext: str) -> List[RawPassage]:
        """Split other languages using structural regex patterns."""
        decl_patterns = [
            re.compile(r"^\s*(export\s+)?(function|class|interface|type|const|let)\s+([A-Za-z0-9_]+)"),
            re.compile(r"^\s*func\s+(\([^\)]+\)\s+)?([A-Za-z0-9_]+)"),
            re.compile(r"^\s*(pub\s+)?(fn|struct|enum|trait|impl)\s+([A-Za-z0-9_]+)"),
            re.compile(r"^\s*(public|private|protected|static|\s)+[\w<>\[\]]+\s+([A-Za-z0-9_]+)\s*\("),
        ]

        passages: List[RawPassage] = []
        current_block: List[str] = []
        block_start = 1
        current_symbol = "preamble"

        for idx, line in enumerate(lines, start=1):
            matched_symbol = None
            for pat in decl_patterns:
                m = pat.match(line)
                if m:
                    matched_symbol = m.group(m.lastindex)
                    break

            if matched_symbol and current_block:
                block_text = "\n".join(current_block).strip()
                if block_text:
                    passages.append(
                        RawPassage(
                            content=block_text,
                            location={
                                "start_line": block_start,
                                "end_line": idx - 1,
                                "symbol": current_symbol,
                                "language": ext.lstrip("."),
                            },
                            section=current_symbol,
                        )
                    )
                current_block = [line]
                block_start = idx
                current_symbol = matched_symbol
            else:
                current_block.append(line)

        if current_block:
            block_text = "\n".join(current_block).strip()
            if block_text:
                passages.append(
                    RawPassage(
                        content=block_text,
                        location={
                            "start_line": block_start,
                            "end_line": len(lines),
                            "symbol": current_symbol,
                            "language": ext.lstrip("."),
                        },
                        section=current_symbol,
                    )
                )

        return passages
