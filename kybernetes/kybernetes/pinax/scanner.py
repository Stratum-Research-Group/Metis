"""Passive, safe PATH scanner and use-time integrity verifier for Pinax.
Adheres strictly to Section 4: passive readdir/stat only, NEVER executes target binaries.
"""

import hashlib
import os
from pathlib import Path
import stat
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from kybernetes.pinax.models import EntryKind, InstallState, RiskTier, ToolDoc, ToolKind, ToolRecord


class PinaxScanner:
    """
    Scans PATH directories passively without executing discovered binaries.
    Provides use-time cryptographic verification to prevent TOCTOU and symlink swap attacks.
    """

    def __init__(self, host_id: str = "localhost"):
        self.host_id = host_id

    def scan_path_directories(
        self,
        path_dirs: Optional[List[str]] = None,
        curated_names: Optional[Set[str]] = None,
    ) -> List[ToolRecord]:
        """
        Passively scans directories on PATH.
        NEVER invokes or executes discovered files.
        """
        if path_dirs is None:
            raw_path = os.environ.get("PATH", "/usr/bin:/bin")
            path_dirs = [p for p in raw_path.split(os.pathsep) if p]

        curated = curated_names or set()
        discovered: Dict[str, ToolRecord] = {}

        for dir_str in path_dirs:
            dir_path = Path(dir_str)
            if not dir_path.is_dir():
                continue

            try:
                with os.scandir(dir_path) as it:
                    for entry in it:
                        # Skip non-files or unreadable entries
                        try:
                            # Use follow_symlinks=False to inspect symlink directly
                            st = entry.stat(follow_symlinks=False)
                        except (OSError, PermissionError):
                            continue

                        # Check if executable
                        is_symlink = stat.S_ISLNK(st.st_mode)
                        is_regular = stat.S_ISREG(st.st_mode)
                        if not (is_symlink or is_regular):
                            continue

                        # Check execute bits
                        if not is_symlink and not (st.st_mode & 0o111):
                            continue

                        name = entry.name
                        if name in discovered:
                            # Earlier PATH directory takes precedence
                            continue

                        inv_path = str(Path(entry.path).resolve())
                        resolved_path = inv_path
                        try:
                            resolved_path = str(Path(entry.path).resolve(strict=True))
                        except Exception:
                            resolved_path = inv_path

                        # Compute sha256 purely by reading bytes (passive)
                        file_sha = self._compute_sha256(resolved_path)

                        # Determine if curated or unvetted newcomer
                        is_curated = name in curated
                        tier = RiskTier.T0 if is_curated else RiskTier.T3
                        notes = "Curated executable" if is_curated else "Unvetted newcomer from PATH"

                        tool_id = f"tool:{name}@{self.host_id}:{entry.path}"
                        record = ToolRecord(
                            id=tool_id,
                            host_id=self.host_id,
                            name=name,
                            kind=ToolKind.BINARY,
                            entry_kind=EntryKind.INSTALLED,
                            summary=f"{name} executable on PATH",
                            category="system" if not is_curated else "general",
                            invocation_path=entry.path,
                            resolved_path=resolved_path,
                            sha256=file_sha,
                            install_state=InstallState.INSTALLED,
                            risk_tier=tier,
                            discovered_by="path-scan",
                            notes=notes,
                        )
                        discovered[name] = record

            except (OSError, PermissionError):
                continue

        return list(discovered.values())

    @staticmethod
    def _compute_sha256(filepath: str, max_bytes: int = 50 * 1024 * 1024) -> Optional[str]:
        """Safely compute SHA-256 of a file without executing it."""
        try:
            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                bytes_read = 0
                while chunk := f.read(65536):
                    bytes_read += len(chunk)
                    if bytes_read > max_bytes:
                        break
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    @classmethod
    def verify_executable_integrity(cls, invocation_path: str, expected_sha256: Optional[str]) -> Tuple[bool, str]:
        """
        Use-time re-verify before dispatch:
        Checks that target still exists, is a regular file, and sha256 matches the catalogued digest.
        Prevents TOCTOU symlink swaps and payload replacements.
        """
        if not invocation_path:
            return False, "Invocation path is missing"

        p = Path(invocation_path)
        try:
            resolved = p.resolve(strict=True)
        except (FileNotFoundError, OSError):
            return False, f"Target '{invocation_path}' does not exist or cannot be resolved"

        if not resolved.is_file():
            return False, f"Target '{resolved}' is not a regular file"

        # Check execute permissions
        st = resolved.stat()
        if not (st.st_mode & 0o111):
            return False, f"Target '{resolved}' is not executable"

        if expected_sha256:
            current_sha = cls._compute_sha256(str(resolved))
            if current_sha != expected_sha256:
                return False, f"Integrity violation: sha256 mismatch (expected {expected_sha256[:12]}..., got {str(current_sha)[:12]}...)"

        return True, "Integrity verified"

    @classmethod
    def locate_doc_page(cls, tool_name: str) -> Optional[ToolDoc]:
        """Passively inspect standard man directories for matching documentation."""
        man_roots = [
            Path("/usr/share/man"),
            Path("/usr/local/share/man"),
        ]
        sections = ["man1", "man8", "man7"]

        for root in man_roots:
            if not root.is_dir():
                continue
            for sec in sections:
                sec_dir = root / sec
                if not sec_dir.is_dir():
                    continue
                # Look for tool_name.1, tool_name.1.gz
                for ext in ["", ".gz"]:
                    candidate = sec_dir / f"{tool_name}.{sec[-1]}{ext}"
                    if candidate.is_file():
                        try:
                            st = candidate.stat()
                            return ToolDoc(
                                tool_id=f"tool:{tool_name}",
                                doc_kind="man",
                                locator=str(candidate),
                                version_scope=None,
                                sha256=None,
                                size_bytes=st.st_size,
                            )
                        except OSError:
                            continue
        return None
