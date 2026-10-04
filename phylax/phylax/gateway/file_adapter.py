"""Guarded file adapter with symlink protection and TOCTOU mitigations."""

import os
from pathlib import Path
from typing import Any

from phylax.contracts.enums import ResourceClass, Verb
from phylax.gateway.gateway import GuardedGateway


class GuardedFileAdapter:
    """Safe filesystem operations mediated by Phylax gateway."""

    def __init__(self, gateway: GuardedGateway):
        self.gateway = gateway

    def read_text(
        self,
        path: Path | str,
        subject: str = "kybernetes",
        approval_id: str | None = None,
        trace_id: str | None = None,
        encoding: str = "utf-8",
    ) -> str:
        """Safely read text from a file with symlink and TOCTOU protection."""
        target_path = Path(path)

        def _do_read(**kwargs: Any) -> str:
            # Safe open with O_NOFOLLOW to mitigate symlink replacement TOCTOU races
            target_str = str(target_path)
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW

            fd = os.open(target_str, flags)
            try:
                # Descriptor validation
                fstat_res = os.fstat(fd)
                stat_res = os.stat(target_str)
                if fstat_res.st_ino != stat_res.st_ino:
                    raise PermissionError(
                        f"TOCTOU Inode mismatch on '{target_str}': file was swapped during open!"
                    )

                with open(fd, "r", encoding=encoding, closefd=False) as f:
                    return f.read()
            finally:
                os.close(fd)

        result, _ = self.gateway.authorize_and_execute(
            subject=subject,
            verb=Verb.READ,
            resource_class=ResourceClass.FILE,
            target=str(target_path),
            action_fn=_do_read,
            arguments={},
            approval_id=approval_id,
            trace_id=trace_id,
        )
        return result

    def write_text(
        self,
        path: Path | str,
        content: str,
        subject: str = "kybernetes",
        approval_id: str | None = None,
        trace_id: str | None = None,
        encoding: str = "utf-8",
    ) -> int:
        """Safely write text to a file within authorized roots."""
        target_path = Path(path)

        def _do_write(**kwargs: Any) -> int:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            encoded = content.encode(encoding)
            
            # Atomic temp write then rename within same directory
            temp_path = target_path.with_suffix(f".tmp-{os.getpid()}")
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW

            fd = os.open(str(temp_path), flags, 0o600)
            try:
                with open(fd, "wb", closefd=False) as f:
                    f.write(encoded)
                    f.flush()
                    os.fsync(fd)
            finally:
                os.close(fd)

            os.replace(str(temp_path), str(target_path))
            return len(encoded)

        result, _ = self.gateway.authorize_and_execute(
            subject=subject,
            verb=Verb.WRITE,
            resource_class=ResourceClass.FILE,
            target=str(target_path),
            action_fn=_do_write,
            arguments={"content_bytes": len(content.encode(encoding))},
            approval_id=approval_id,
            trace_id=trace_id,
        )
        return result
