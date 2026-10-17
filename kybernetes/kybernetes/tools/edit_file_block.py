"""Safe exact-anchor and hashline search-and-replace block editor bounded to authorized workspace roots."""

import ast
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from kybernetes.contracts.enums import ToolSideEffectClass
from kybernetes.contracts.tool_spec import ToolDefinition
from kybernetes.tools.read_file_range import compute_line_hash

HASHLINE_PATTERN = re.compile(r"^\s*(\d+):([0-9a-fA-F]{4})\|\s?(.*)$")


def parse_hash_anchors(block: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Parse potential hashline prefixes from block.
    Returns (cleaned_block, list of (line_no, hash)).
    """
    cleaned_lines = []
    anchors = []
    has_any_anchor = False
    for line in block.splitlines(keepends=True):
        m = HASHLINE_PATTERN.match(line)
        if m:
            has_any_anchor = True
            line_no = int(m.group(1))
            l_hash = m.group(2).lower()
            text = m.group(3)
            line_ending = "\n" if line.endswith("\n") else ""
            cleaned_lines.append(text + line_ending)
            anchors.append((line_no, l_hash))
        else:
            cleaned_lines.append(line)
    if has_any_anchor:
        return "".join(cleaned_lines), anchors
    return block, []


def validate_syntax(path: Path, content: str) -> List[str]:
    """In-process syntax validation feedback loop (Python, JSON)."""
    diagnostics = []
    suffix = path.suffix.lower()
    if suffix == ".py":
        try:
            ast.parse(content, filename=str(path))
        except SyntaxError as e:
            diagnostics.append(f"Python SyntaxError on line {e.lineno}: {e.msg}")
    elif suffix == ".json":
        try:
            json.loads(content)
        except Exception as e:
            diagnostics.append(f"JSON Parse Error: {e}")
    return diagnostics


class SafeEditBlock:
    """Exact-match search-and-replace block editor with hashline stale-anchor verification."""

    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

    def __init__(self, allowed_roots: Optional[List[Path]] = None):
        self.allowed_roots = [r.resolve() for r in (allowed_roots or [Path.cwd()])]

    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="edit_file_block",
            version="1.2.0",
            description=(
                "Replace an exact block of text within a file using unambiguous search-and-replace matching. "
                "Supports hashline anchor validation (stale-anchor rejection) and post-edit in-process syntax checks."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative or absolute file path to edit",
                    },
                    "search_block": {
                        "type": "string",
                        "description": "Exact text substring to find and replace. Can include hashline 'line:hash|' prefixes",
                    },
                    "replace_block": {
                        "type": "string",
                        "description": "Exact replacement text block",
                    },
                    "expected_start_hash": {
                        "type": "string",
                        "description": "Optional 4-hex-char FNV-1a hash of the first line of the search block for stale-anchor check",
                    },
                    "expected_end_hash": {
                        "type": "string",
                        "description": "Optional 4-hex-char FNV-1a hash of the last line of the search block for stale-anchor check",
                    },
                    "allow_syntax_errors": {
                        "type": "boolean",
                        "description": "If true, persist file changes even if syntax validation detects errors (default false)",
                    },
                },
                "required": ["path", "search_block", "replace_block"],
            },
            returns_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "replaced": {"type": "boolean"},
                    "occurrences": {"type": "integer"},
                    "lines_before": {"type": "integer"},
                    "lines_after": {"type": "integer"},
                    "syntax_valid": {"type": "boolean"},
                    "diagnostics": {"type": "array", "items": {"type": "string"}},
                    "error": {"type": "string"},
                },
            },
            declared_capabilities=["write_file", "edit_code"],
            side_effect_class=ToolSideEffectClass.FILESYSTEM_MUTATION,
            timeout_ms=5000,
            idempotent=False,
        )

    def _is_path_allowed(self, path: Path) -> bool:
        resolved = path.resolve()
        for root in self.allowed_roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def edit_block(
        self,
        path: str,
        search_block: str,
        replace_block: str,
        expected_start_hash: Optional[str] = None,
        expected_end_hash: Optional[str] = None,
        allow_syntax_errors: bool = False,
    ) -> Dict[str, Any]:
        target = Path(path)
        if not target.is_absolute():
            target = (self.allowed_roots[0] / target).resolve()
        else:
            target = target.resolve()

        if not self._is_path_allowed(target):
            raise PermissionError(f"File {target} is outside authorized roots: {self.allowed_roots}")

        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")

        if not search_block:
            raise ValueError("search_block cannot be empty")

        stat = target.stat()
        if stat.st_size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size ({stat.st_size} bytes) exceeds maximum limit ({self.MAX_FILE_SIZE_BYTES} bytes)")

        # Parse potential hashline prefixes from search_block and replace_block
        clean_search, parsed_anchors = parse_hash_anchors(search_block)
        clean_replace, _ = parse_hash_anchors(replace_block)

        with open(target, "r", encoding="utf-8") as f:
            content = f.read()

        file_lines = content.splitlines(keepends=True)

        # 1. Stale-Anchor Verification if hashline anchors were provided in search_block
        if parsed_anchors:
            for line_no, expected_hash in parsed_anchors:
                if not (1 <= line_no <= len(file_lines)):
                    raise ValueError(
                        f"Anchor line number {line_no} is out of range for {target} (file has {len(file_lines)} lines)."
                    )
                actual_line = file_lines[line_no - 1]
                actual_hash = compute_line_hash(actual_line)
                if actual_hash.lower() != expected_hash.lower():
                    raise ValueError(
                        f"Stale-anchor rejection on line {line_no}: expected hash '{expected_hash}', "
                        f"but file has hash '{actual_hash}' (content: '{actual_line.strip()}'). "
                        "File has drifted since last read. Re-read with read_file_range before editing."
                    )

        occurrences = content.count(clean_search)
        if occurrences == 0:
            raise ValueError(
                f"Target search_block was not found in {target}. "
                "Ensure search_block matches the file content exactly, including whitespace and line endings."
            )
        if occurrences > 1:
            raise ValueError(
                f"Target search_block matches {occurrences} times in {target}. "
                "Search block must be unique. Include additional surrounding lines of code to disambiguate."
            )

        # Verify explicit expected_start_hash / expected_end_hash if given
        idx = content.find(clean_search)
        prefix = content[:idx]
        start_line_no = prefix.count("\n") + 1
        search_lines = clean_search.splitlines()
        end_line_no = start_line_no + len(search_lines) - 1

        if expected_start_hash:
            if not (1 <= start_line_no <= len(file_lines)):
                raise ValueError(
                    f"Start line number {start_line_no} is out of range for {target} (file has {len(file_lines)} lines)."
                )
            actual_start_hash = compute_line_hash(file_lines[start_line_no - 1])
            if actual_start_hash.lower() != expected_start_hash.lower():
                raise ValueError(
                    f"Stale-anchor rejection on start line {start_line_no}: expected hash '{expected_start_hash}', "
                    f"found '{actual_start_hash}'. Target file drifted."
                )

        if expected_end_hash:
            if not (1 <= end_line_no <= len(file_lines)):
                raise ValueError(
                    f"End line number {end_line_no} is out of range for {target} (file has {len(file_lines)} lines)."
                )
            actual_end_hash = compute_line_hash(file_lines[end_line_no - 1])
            if actual_end_hash.lower() != expected_end_hash.lower():
                raise ValueError(
                    f"Stale-anchor rejection on end line {end_line_no}: expected hash '{expected_end_hash}', "
                    f"found '{actual_end_hash}'. Target file drifted."
                )

        new_content = content.replace(clean_search, clean_replace, 1)

        # In-process syntax check (Feature C)
        diagnostics = validate_syntax(target, new_content)

        # Gated write on syntax diagnostics (Finding B4)
        if diagnostics and not allow_syntax_errors:
            return {
                "path": str(target),
                "replaced": False,
                "occurrences": 1,
                "lines_before": len(file_lines),
                "lines_after": len(file_lines),
                "syntax_valid": False,
                "diagnostics": diagnostics,
                "error": f"Syntax validation failed: {diagnostics[0]}. Edit was aborted to prevent file corruption. Pass allow_syntax_errors=True to override.",
            }

        # Atomic write via temporary file in the same directory
        temp_dir = target.parent
        with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
            tf.write(new_content)
            temp_path = Path(tf.name)

        temp_path.replace(target)

        return {
            "path": str(target),
            "replaced": True,
            "occurrences": 1,
            "lines_before": len(file_lines),
            "lines_after": len(new_content.splitlines()),
            "syntax_valid": len(diagnostics) == 0,
            "diagnostics": diagnostics,
        }
