"""Safe atomic file writer bounded to authorized workspace roots."""

from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

from kybernetes.contracts.enums import ToolSideEffectClass
from kybernetes.contracts.tool_spec import ToolDefinition


class SafeWriteFile:
    """Atomic file writer strictly bounded to authorized workspace roots."""

    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

    def __init__(self, allowed_roots: Optional[List[Path]] = None):
        self.allowed_roots = [r.resolve() for r in (allowed_roots or [Path.cwd()])]

    @classmethod
    def get_definition(cls) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            version="1.0.0",
            description="Create a new file or rewrite an existing file atomically within authorized roots.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative or absolute file path to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write into the file",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Whether to overwrite if the file already exists (default false)",
                    },
                },
                "required": ["path", "content"],
            },
            returns_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "bytes_written": {"type": "integer"},
                    "created": {"type": "boolean"},
                    "overwritten": {"type": "boolean"},
                },
            },
            declared_capabilities=["write_file"],
            side_effect_class=ToolSideEffectClass.FILESYSTEM_MUTATION,
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

    def write(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        target = Path(path)
        if not target.is_absolute():
            target = (self.allowed_roots[0] / target).resolve()
        else:
            target = target.resolve()

        if not self._is_path_allowed(target):
            raise PermissionError(f"Target path {target} is outside authorized roots: {self.allowed_roots}")

        exists = target.exists()
        if exists and not overwrite:
            raise FileExistsError(f"File already exists: {target}. Set overwrite=true to overwrite.")

        encoded_len = len(content.encode("utf-8"))
        if encoded_len > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(f"Content size ({encoded_len} bytes) exceeds maximum limit ({self.MAX_FILE_SIZE_BYTES} bytes)")

        target.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write
        temp_dir = target.parent
        with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
            tf.write(content)
            temp_path = Path(tf.name)

        temp_path.replace(target)

        return {
            "path": str(target),
            "bytes_written": encoded_len,
            "created": not exists,
            "overwritten": exists,
        }
