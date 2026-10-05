import hashlib
import os
import secrets
import stat
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
                # Descriptor validation: check device, inode, and regular file type
                fstat_res = os.fstat(fd)
                if not stat.S_ISREG(fstat_res.st_mode):
                    raise PermissionError(f"Security violation: target '{target_str}' is not a regular file")
                stat_res = os.stat(target_str)
                if fstat_res.st_ino != stat_res.st_ino or fstat_res.st_dev != stat_res.st_dev:
                    raise PermissionError(
                        f"TOCTOU device/inode mismatch on '{target_str}': file was swapped during open!"
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
        """Safely write text to a file within authorized roots, binding exact content digest."""
        target_path = Path(path)
        encoded = content.encode(encoding)
        content_sha256 = hashlib.sha256(encoded).hexdigest()
        content_bytes = len(encoded)

        def _do_write(**kwargs: Any) -> int:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Verify payload digest matches authorized intent / bound arguments
            expected_digest = kwargs.get("content_sha256", content_sha256)
            if hashlib.sha256(encoded).hexdigest() != expected_digest:
                raise ValueError("Content tampering detected: payload digest does not match authorized digest")

            # Unique exclusive temporary write then rename within same directory
            rand_suffix = secrets.token_hex(6)
            temp_path = target_path.with_name(f".{target_path.name}.tmp-{os.getpid()}-{rand_suffix}")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
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
            arguments={
                "content_bytes": content_bytes,
                "content_sha256": content_sha256,
                "encoding": encoding,
            },
            approval_id=approval_id,
            trace_id=trace_id,
        )
        return result
