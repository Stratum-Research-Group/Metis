"""Safe local file search tool integrated with canonical path boundaries and Phylax."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from kybernetes.contracts.enums import ToolSideEffectClass
from kybernetes.contracts.tool_spec import ToolDefinition


class SafeFileSearch:
    """
    Local file search tool strictly bounded to allowed roots.
    Prevents path traversal and checks canonical paths.
    """

    def __init__(self, allowed_roots: Optional[List[Path]] = None):
        self.allowed_roots = [r.resolve() for r in (allowed_roots or [Path.cwd()])]

    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="file_search",
            version="1.0.0",
            description="Search for keyword or pattern matches in local files within allowed workspace roots.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text substring or regex pattern to search for",
                    },
                    "directory": {
                        "type": "string",
                        "description": "Relative or absolute directory path to search within",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of file matches to return (default 20)",
                    },
                },
                "required": ["pattern"],
            },
            returns_schema={
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string"},
                                "line_number": {"type": "integer"},
                                "snippet": {"type": "string"},
                            },
                        },
                    },
                    "total_matches": {"type": "integer"},
                },
            },
            declared_capabilities=["read_file"],
            side_effect_class=ToolSideEffectClass.READ_ONLY,
            timeout_ms=10000,
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

    def search(
        self,
        pattern: str,
        directory: Optional[str] = None,
        max_results: int = 20,
    ) -> Dict[str, Any]:
        """Perform search over allowed local files."""
        target_dir = Path(directory).resolve() if directory else self.allowed_roots[0]

        if not self._is_path_allowed(target_dir):
            raise PermissionError(f"Directory {target_dir} is outside authorized roots: {self.allowed_roots}")

        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {target_dir}")

        matches: List[Dict[str, Any]] = []
        pat_lower = pattern.lower()

        for root, dirs, files in os.walk(target_dir):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", "venv", ".venv")]

            for fname in files:
                if fname.startswith(".") or fname.endswith((".pyc", ".png", ".jpg", ".bin", ".db", ".sqlite")):
                    continue

                fpath = Path(root) / fname
                if not self._is_path_allowed(fpath):
                    continue

                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for l_idx, line in enumerate(f, 1):
                            if pat_lower in line.lower():
                                matches.append(
                                    {
                                        "file": str(fpath.relative_to(self.allowed_roots[0])),
                                        "line_number": l_idx,
                                        "snippet": line.strip()[:200],
                                    }
                                )
                                if len(matches) >= max_results:
                                    return {"matches": matches, "total_matches": len(matches)}
                except (OSError, PermissionError):
                    continue

        return {"matches": matches, "total_matches": len(matches)}
