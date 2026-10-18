"""
Stoma Manifest-Governed Ingestion Staging Boundary (Phase 2 / WALDO model).
Ensures:
- All ingestion acquisitions stop at a validated, manifest-backed staging directory.
- Adapters strictly refuse to process any file not declared in the manifest.
- Strict path containment: rejects absolute paths, '..' traversal, and symlinks.
- Atomic contribution staging: writes use temporary files + fsync + rename with rollback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SecurityPathError(ValueError):
    """Raised when an untrusted path attempts traversal, absolute escape, or symlink indirection."""
    pass


class StagingValidationError(RuntimeError):
    """Raised when files in staging do not match the declared manifest."""
    pass


@dataclass
class StagedFileRecord:
    """Record of a verified staged file inside an ingestion manifest."""
    rel_path: str
    sha256: str
    size_bytes: int
    staged_at: float = field(default_factory=time.time)
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rel_path": self.rel_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "staged_at": self.staged_at,
            "mime_type": self.mime_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StagedFileRecord:
        return cls(
            rel_path=data["rel_path"],
            sha256=data["sha256"],
            size_bytes=data["size_bytes"],
            staged_at=data.get("staged_at", time.time()),
            mime_type=data.get("mime_type"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StagingManifest:
    """Cryptographically pinned manifest governing an ingestion batch."""
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    source_origin: str = "local"
    files: Dict[str, StagedFileRecord] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at,
            "source_origin": self.source_origin,
            "files": {k: v.to_dict() for k, v in self.files.items()},
        }

    def save(self, manifest_path: Path) -> None:
        """Atomically saves the manifest file."""
        tmp_path = manifest_path.with_name(f".{manifest_path.name}.tmp.{uuid.uuid4().hex}")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, manifest_path)

    @classmethod
    def load(cls, manifest_path: Path) -> StagingManifest:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        files = {k: StagedFileRecord.from_dict(v) for k, v in data.get("files", {}).items()}
        return cls(
            batch_id=data.get("batch_id", str(uuid.uuid4())),
            created_at=data.get("created_at", time.time()),
            source_origin=data.get("source_origin", "local"),
            files=files,
        )


class StagingGate:
    """
    Enforces strict isolation, atomic staging, and manifest verification
    prior to parser ingestion.
    """

    @classmethod
    def compute_sha256(cls, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def verify_path_containment(cls, rel_path: str, staging_dir: Path) -> Path:
        """
        Verifies that rel_path is strictly relative, does not contain '..',
        does not contain symlinks, and stays strictly within staging_dir.
        """
        raw_segments = [s for s in rel_path.replace("\\", "/").split("/") if s]
        if ".." in raw_segments or "." in raw_segments:
            raise SecurityPathError(f"Relative traversal elements ('..' or '.') in '{rel_path}' are forbidden")

        p = Path(rel_path)
        if p.is_absolute():
            raise SecurityPathError(f"Absolute path '{rel_path}' is forbidden in staging")

        resolved_staging = staging_dir.resolve()
        target = (resolved_staging / p).resolve()

        # Check containment
        try:
            target.relative_to(resolved_staging)
        except ValueError:
            raise SecurityPathError(f"Path '{rel_path}' escapes staging root: {staging_dir}")

        # Check parent components for symlinks
        current = staging_dir.resolve()
        parts = p.parts
        for part in parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise SecurityPathError(f"Symlink directory component in path '{current}' is forbidden")

        if target.is_symlink():
            raise SecurityPathError(f"Target path '{target}' is a symlink (symlinks are forbidden in staging)")

        return target

    @classmethod
    def stage_file(
        cls,
        source_path: Path,
        rel_target: str,
        staging_dir: Path,
        manifest: StagingManifest,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StagedFileRecord:
        """
        Stages a source file atomically into the staging directory, computes its SHA-256,
        and registers it in the manifest.
        """
        target_path = cls.verify_path_containment(rel_target, staging_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Source file '{source_path}' does not exist or is not a regular file")

        # Atomic copy: write to temp file then replace
        tmp_target = target_path.with_name(f".{target_path.name}.tmp.{uuid.uuid4().hex}")
        try:
            h = hashlib.sha256()
            size = 0
            with open(source_path, "rb") as src, open(tmp_target, "wb") as dst:
                while chunk := src.read(65536):
                    size += len(chunk)
                    h.update(chunk)
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())

            os.replace(tmp_target, target_path)
            file_sha = h.hexdigest()

            record = StagedFileRecord(
                rel_path=rel_target,
                sha256=file_sha,
                size_bytes=size,
                mime_type=mime_type,
                metadata=metadata or {},
            )
            manifest.files[rel_target] = record
            return record

        except Exception as e:
            if tmp_target.exists():
                try:
                    tmp_target.unlink()
                except Exception:
                    pass
            raise e

    @classmethod
    def validate_staging_dir(
        cls,
        staging_dir: Path,
        manifest: StagingManifest,
        strict_undeclared_rejection: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Validates staging directory against manifest.
        WALDO invariant: Every file on disk must match the manifest. Undeclared files fail validation.
        """
        errors: List[str] = []
        resolved_staging = staging_dir.resolve()

        if not resolved_staging.is_dir():
            return False, [f"Staging directory '{staging_dir}' does not exist"]

        # Collect files on disk
        disk_files: Dict[str, Path] = {}
        for root, _, files in os.walk(resolved_staging):
            for file in files:
                if file.startswith(".") or file == "staging_manifest.json":
                    continue
                full_p = Path(root) / file
                rel_p = str(full_p.relative_to(resolved_staging))
                disk_files[rel_p] = full_p

        # 1. Check all manifest files exist and match hashes
        for rel_path, rec in manifest.files.items():
            if rel_path not in disk_files:
                errors.append(f"Missing staged file declared in manifest: '{rel_path}'")
                continue

            file_on_disk = disk_files[rel_path]
            actual_size = file_on_disk.stat().st_size
            if actual_size != rec.size_bytes:
                errors.append(f"Size mismatch for '{rel_path}': expected {rec.size_bytes}, got {actual_size}")

            actual_sha = cls.compute_sha256(file_on_disk)
            if actual_sha != rec.sha256:
                errors.append(f"Hash mismatch for '{rel_path}': expected {rec.sha256[:12]}..., got {actual_sha[:12]}...")

        # 2. Check for undeclared files on disk
        if strict_undeclared_rejection:
            for rel_path in disk_files:
                if rel_path not in manifest.files:
                    errors.append(f"Undeclared file on disk not authorized in manifest: '{rel_path}'")

        return len(errors) == 0, errors


# Backward-compatible and descriptive alias
ManifestStagingGate = StagingGate

