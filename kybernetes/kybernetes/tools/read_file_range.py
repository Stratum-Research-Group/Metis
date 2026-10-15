"""Safe line-numbered file range reader bounded to authorized workspace roots."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from kybernetes.contracts.enums import ToolSideEffectClass
from kybernetes.contracts.tool_spec import ToolDefinition


def compute_line_hash(line: str) -> str:
    """Compute a compact 4-hex-char FNV-1a hash of stripped line content."""
    h = 2166136261
    for byte in line.rstrip("\r\n").encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return f"{h & 0xFFFF:04x}"


class SafeReadRange:
    """Read a bounded range of lines from a local file with line numbering and bounds checking."""

    MAX_LINES = 500
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

    def __init__(self, allowed_roots: Optional[List[Path]] = None):
        self.allowed_roots = [r.resolve() for r in (allowed_roots or [Path.cwd()])]

    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="read_file_range",
            version="1.1.0",
            description="Read lines from a file with 1-indexed numbering and FNV-1a line hashes for deterministic hash-anchored editing. Bounded to max 500 lines per call.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative or absolute file path to read",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-indexed, inclusive, default 1)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (1-indexed, inclusive, default 200)",
                    },
                },
                "required": ["path"],
            },
            returns_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "total_lines": {"type": "integer"},
                    "lines": {"type": "array", "items": {"type": "string"}},
                    "hashed_lines": {"type": "array", "items": {"type": "string"}},
                    "content": {"type": "string"},
                },
            },
            declared_capabilities=["read_file"],
            side_effect_class=ToolSideEffectClass.READ_ONLY,
            timeout_ms=5000,
            idempotent=True,
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

    def read_range(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 200,
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

        stat = target.stat()
        if stat.st_size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size ({stat.st_size} bytes) exceeds maximum limit ({self.MAX_FILE_SIZE_BYTES} bytes)")

        if start_line < 1:
            start_line = 1
        if end_line < start_line:
            end_line = start_line

        requested_span = end_line - start_line + 1
        if requested_span > self.MAX_LINES:
            end_line = start_line + self.MAX_LINES - 1

        with open(target, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)
        actual_start = min(start_line, max(1, total_lines))
        actual_end = min(end_line, total_lines)

        sliced_lines = all_lines[actual_start - 1 : actual_end] if total_lines > 0 else []
        
        hashed_lines = []
        numbered_content_parts = []
        for i, line in enumerate(sliced_lines):
            line_no = actual_start + i
            l_hash = compute_line_hash(line)
            clean_line = line.rstrip("\r\n")
            hashed_lines.append(f"{line_no}:{l_hash}| {clean_line}")
            numbered_content_parts.append(f"{line_no}:{l_hash}| {line}")

        numbered_content = "".join(numbered_content_parts)

        return {
            "path": str(target),
            "start_line": actual_start,
            "end_line": actual_end,
            "total_lines": total_lines,
            "lines": [line.rstrip("\r\n") for line in sliced_lines],
            "hashed_lines": hashed_lines,
            "content": numbered_content,
        }
