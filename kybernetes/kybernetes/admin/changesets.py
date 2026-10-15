"""ChangeSet lifecycle, atomic configuration persistence, snapshots, and rollbacks.

Strictly conforms to Section 4.4, Section 5, and Step 8 of the Metis Control Dashboard specification.
Provides atomic writes (tempfile + replace), base_hash optimistic concurrency locks,
pre-write snapshotting, unified diff computation, sensitive modification gating,
and append-only history tracking.
"""

from __future__ import annotations

from datetime import datetime, timezone
import difflib
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import yaml

from kybernetes.admin.descriptors import (
    MODULE_REGISTRY,
    compute_dict_hash,
    redact_sensitive_dict,
)
from kybernetes.admin.validators import ConfigValidationResult, validate_module_configuration


class ChangeSetStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    REJECTED = "rejected"
    WRITTEN = "written"
    RESTART_PENDING = "restart_pending"
    EFFECTIVE_CONFIRMED = "effective_confirmed"
    PARTIALLY_APPLIED = "partially_applied"
    FAILED = "failed"
    ROLLED_BACK_ON_DISK = "rolled_back_on_disk"
    ROLLBACK_EFFECTIVE_CONFIRMED = "rollback_effective_confirmed"


class ChangeSet(BaseModel):
    id: str = Field(..., description="Unique ChangeSet identifier (e.g. cs-20261015-...)")
    module: str = Field(..., description="Module identifier this ChangeSet targets")
    base_hash: str = Field(..., description="Base configuration hash for optimistic concurrency control")
    patch: Dict[str, Any] = Field(..., description="Proposed configuration updates")
    validation_result: Optional[ConfigValidationResult] = Field(
        default=None, description="Result of executing the module native validator"
    )
    requested_by: str = Field(..., description="Actor identity or token fingerprint")
    requested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of creation request",
    )
    status: ChangeSetStatus = Field(..., description="ChangeSet lifecycle state")
    applied_at: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp when written to disk")
    prev_snapshot_id: Optional[str] = Field(default=None, description="Identifier of pre-write snapshot for rollback")
    files_affected: List[str] = Field(default_factory=list, description="List of filesystem paths modified")
    error_message: Optional[str] = Field(default=None, description="Diagnostic error message on failure or rejection")
    diff: Optional[str] = Field(default=None, description="Unified diff of configuration changes")
    restart_required: bool = Field(default=True, description="Whether applying this ChangeSet requires process restart")


class ConfigApplyRequest(BaseModel):
    base_hash: str = Field(..., description="Base configuration hash for optimistic concurrency lock check")
    patch: Dict[str, Any] = Field(..., description="Proposed configuration updates")
    confirm_token: Optional[str] = Field(
        default=None,
        description="Confirmation token required for sensitive changes (e.g., 'CONFIRM')",
    )


# Sensitive configuration keys requiring explicit user confirmation
SENSITIVE_PROSOPON_KEYS = {"host", "port", "lan", "enable_tls", "allowed_origins", "pairing_required"}
SENSITIVE_HESTIA_KEYS = {"listen_host", "listen_port", "backend_port_range"}
SENSITIVE_KYBERNETES_KEYS = {"allowed_origins", "admin_token", "service_token", "admin_token_file"}


def check_sensitive_modification(
    module_id: str,
    patch: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check if the proposed patch touches sensitive security, network, or policy settings."""
    if module_id in ("phylax", "krisis"):
        return True, f"Modifications to '{module_id}' security policy require explicit confirmation."

    patch_keys = set(patch.keys())
    if module_id == "prosopon":
        touched = patch_keys & SENSITIVE_PROSOPON_KEYS
        if touched:
            return True, f"Prosopon network/security parameters modified: {sorted(list(touched))}"
    elif module_id == "hestia":
        touched = patch_keys & SENSITIVE_HESTIA_KEYS
        if touched:
            return True, f"Hestia listener/port parameters modified: {sorted(list(touched))}"
    elif module_id == "kybernetes":
        touched = patch_keys & SENSITIVE_KYBERNETES_KEYS
        if touched:
            return True, f"Kybernetes access/auth parameters modified: {sorted(list(touched))}"

    return False, ""


def compute_unified_diff(current_text: str, new_text: str, filename: str) -> str:
    """Generate a clean unified diff between original on-disk content and proposed content."""
    curr_lines = current_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(
        curr_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    return "".join(diff)


class ChangeSetManager:
    """
    Manages snapshots, atomic file modifications, and ChangeSet history.
    Stores pre-write snapshots under .admin_snapshots/ to enable instant rollback.
    """

    def __init__(self, snapshot_dir: str = ".admin_snapshots"):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._history: Dict[str, List[ChangeSet]] = {}  # module_id -> List[ChangeSet]
        self._all_changesets: Dict[str, ChangeSet] = {}
        self._snapshots: Dict[str, Dict[str, Any]] = {}  # snapshot_id -> metadata
        self._counter: int = 0

    def get_history(self, module_id: Optional[str] = None) -> List[ChangeSet]:
        """Retrieve ChangeSet history in reverse chronological order."""
        if module_id:
            return list(reversed(self._history.get(module_id, [])))
        
        all_cs = list(self._all_changesets.values())
        all_cs.sort(key=lambda cs: cs.requested_at, reverse=True)
        return all_cs

    def get_changeset(self, changeset_id: str) -> Optional[ChangeSet]:
        """Retrieve a specific ChangeSet by identifier."""
        return self._all_changesets.get(changeset_id)

    def _create_snapshot(self, module_id: str, file_path: Path) -> str:
        """Create a pre-write snapshot of the target configuration file."""
        self._counter += 1
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snap_id = f"snap_{module_id}_{now_str}_{self._counter:04d}"

        content_bytes = b""
        exists = file_path.exists()
        if exists:
            content_bytes = file_path.read_bytes()

        snap_file = self.snapshot_dir / f"{snap_id}.bak"
        snap_file.write_bytes(content_bytes)

        self._snapshots[snap_id] = {
            "snapshot_id": snap_id,
            "module_id": module_id,
            "target_path": str(file_path),
            "file_existed": exists,
            "size_bytes": len(content_bytes),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return snap_id

    def _restore_snapshot(self, snapshot_id: str) -> Path:
        """Atomically restore a configuration file from a recorded snapshot."""
        if snapshot_id not in self._snapshots:
            raise KeyError(f"Snapshot '{snapshot_id}' not found")

        meta = self._snapshots[snapshot_id]
        target_path = Path(meta["target_path"])
        snap_file = self.snapshot_dir / f"{snapshot_id}.bak"

        if not meta["file_existed"]:
            # File was initially created by the changeset; rollback deletes it
            if target_path.exists():
                target_path.unlink()
            return target_path

        content_bytes = snap_file.read_bytes()
        self._atomic_write(target_path, content_bytes)
        return target_path

    def _atomic_write(self, target_path: Path, content_bytes: bytes) -> None:
        """Write content atomically via temporary file and replace."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix="tmp_admin_",
            dir=target_path.parent,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content_bytes)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, target_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def get_current_saved_content_and_hash(self, module_id: str) -> Tuple[Dict[str, Any], str, str]:
        """
        Read the current on-disk saved configuration, its raw text, and computed hash.
        Returns (parsed_dict, raw_text, config_hash).
        """
        if module_id not in MODULE_REGISTRY:
            raise KeyError(f"Unknown module '{module_id}'")

        descriptor = MODULE_REGISTRY[module_id]
        if not descriptor.config_path:
            return {}, "", compute_dict_hash({})

        target_path = Path(descriptor.config_path)
        if not target_path.exists():
            return {}, "", compute_dict_hash({})

        raw_text = target_path.read_text(encoding="utf-8")
        parsed: Dict[str, Any] = {}
        try:
            if descriptor.config_format in ("yaml", "pydantic"):
                parsed = yaml.safe_load(raw_text) or {}
            elif descriptor.config_format == "json":
                parsed = json.loads(raw_text) or {}
            elif descriptor.config_format == "toml":
                import tomllib
                parsed = tomllib.loads(raw_text) or {}
        except Exception:
            parsed = {}

        if not isinstance(parsed, dict):
            parsed = {}

        clean_dict = redact_sensitive_dict(parsed)
        config_hash = compute_dict_hash(clean_dict)
        return parsed, raw_text, config_hash

    def serialize_config(self, module_id: str, data: Dict[str, Any]) -> str:
        """Serialize configuration dictionary into module's native file format."""
        descriptor = MODULE_REGISTRY[module_id]
        fmt = descriptor.config_format
        if fmt in ("yaml", "pydantic"):
            return yaml.safe_dump(data, sort_keys=False)
        elif fmt == "json":
            return json.dumps(data, indent=2) + "\n"
        elif fmt == "toml":
            raise NotImplementedError("Writing TOML configurations is not supported in v1")
        else:
            raise ValueError(f"Unsupported configuration format '{fmt}' for serialization")

    def apply(
        self,
        module_id: str,
        base_hash: str,
        patch: Dict[str, Any],
        requested_by: str,
        confirm_token: Optional[str] = None,
    ) -> ChangeSet:
        """
        Execute safe atomic application of a configuration ChangeSet.
        Validates module native rules, enforces optimistic lock (base_hash),
        gates sensitive changes behind confirm_token, snapshots existing file,
        and atomically writes the update to disk.
        """
        if module_id not in MODULE_REGISTRY:
            raise KeyError(f"Unknown module: '{module_id}'")

        descriptor = MODULE_REGISTRY[module_id]
        if not descriptor.config_path:
            raise ValueError(
                f"Module '{module_id}' does not have an on-disk configuration file to modify."
            )

        self._counter += 1
        now_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cs_id = f"cs-{module_id}-{now_ts}-{self._counter:04d}"

        target_path = Path(descriptor.config_path)

        # 1. Inspect current state and check optimistic concurrency lock (base_hash)
        current_data, current_text, current_hash = self.get_current_saved_content_and_hash(module_id)
        if base_hash != current_hash:
            cs_conflict = ChangeSet(
                id=cs_id,
                module=module_id,
                base_hash=base_hash,
                patch=patch,
                validation_result=None,
                requested_by=requested_by,
                status=ChangeSetStatus.REJECTED,
                error_message=(
                    f"Concurrent modification detected: base_hash mismatch "
                    f"(expected '{current_hash}', got '{base_hash}')"
                ),
                restart_required=descriptor.restart_required,
            )
            self._record_changeset(cs_conflict)
            raise ValueError(cs_conflict.error_message)

        # 2. Run module native validation without side effects
        val_result = validate_module_configuration(module_id, patch)
        if not val_result.valid:
            cs_invalid = ChangeSet(
                id=cs_id,
                module=module_id,
                base_hash=base_hash,
                patch=patch,
                validation_result=val_result,
                requested_by=requested_by,
                status=ChangeSetStatus.REJECTED,
                error_message=f"Validation failed: {'; '.join(val_result.errors)}",
                restart_required=descriptor.restart_required,
            )
            self._record_changeset(cs_invalid)
            raise ValueError(cs_invalid.error_message)

        # 3. Format proposed content and compute diff
        new_text = self.serialize_config(module_id, patch)
        diff_text = compute_unified_diff(current_text, new_text, target_path.name)

        # 4. Enforce confirmation token for sensitive modifications
        is_sensitive, reason = check_sensitive_modification(module_id, patch)
        if is_sensitive and (not confirm_token or confirm_token.strip().upper() != "CONFIRM"):
            cs_need_confirm = ChangeSet(
                id=cs_id,
                module=module_id,
                base_hash=base_hash,
                patch=patch,
                validation_result=val_result,
                requested_by=requested_by,
                status=ChangeSetStatus.REJECTED,
                diff=diff_text,
                error_message=(
                    f"Explicit confirmation required: {reason}. "
                    "Review diff and resubmit with confirm_token='CONFIRM'."
                ),
                restart_required=descriptor.restart_required,
            )
            self._record_changeset(cs_need_confirm)
            raise PermissionError(cs_need_confirm.error_message)

        # 5. Snapshot previous file before write
        snapshot_id = self._create_snapshot(module_id, target_path)

        # 6. Atomic write (tempfile + replace)
        try:
            self._atomic_write(target_path, new_text.encode("utf-8"))
        except Exception as exc:
            cs_failed = ChangeSet(
                id=cs_id,
                module=module_id,
                base_hash=base_hash,
                patch=patch,
                validation_result=val_result,
                requested_by=requested_by,
                status=ChangeSetStatus.FAILED,
                prev_snapshot_id=snapshot_id,
                files_affected=[],
                error_message=f"Atomic disk write failure: {exc}",
                diff=diff_text,
                restart_required=descriptor.restart_required,
            )
            self._record_changeset(cs_failed)
            raise IOError(cs_failed.error_message)

        # 7. Record successful application
        # Status reflects whether a restart is pending for effective configuration
        final_status = (
            ChangeSetStatus.RESTART_PENDING
            if descriptor.restart_required
            else ChangeSetStatus.WRITTEN
        )
        applied_at = datetime.now(timezone.utc).isoformat()
        cs_success = ChangeSet(
            id=cs_id,
            module=module_id,
            base_hash=base_hash,
            patch=patch,
            validation_result=val_result,
            requested_by=requested_by,
            status=final_status,
            applied_at=applied_at,
            prev_snapshot_id=snapshot_id,
            files_affected=[str(target_path)],
            diff=diff_text,
            restart_required=descriptor.restart_required,
        )
        self._record_changeset(cs_success)
        return cs_success

    def rollback(self, module_id: str, changeset_id: str, requested_by: str) -> ChangeSet:
        """
        Restore configuration from the snapshot preceding a specific ChangeSet.
        Creates a new ChangeSet with status 'rolled_back_on_disk'.
        """
        if module_id not in MODULE_REGISTRY:
            raise KeyError(f"Unknown module: '{module_id}'")

        target_cs = self.get_changeset(changeset_id)
        if not target_cs or target_cs.module != module_id:
            raise KeyError(f"ChangeSet '{changeset_id}' not found for module '{module_id}'")

        if not target_cs.prev_snapshot_id:
            raise ValueError(f"ChangeSet '{changeset_id}' has no recorded snapshot for rollback.")

        descriptor = MODULE_REGISTRY[module_id]

        # Snapshot current state before rolling back so this rollback itself can be examined
        target_path = Path(descriptor.config_path) if descriptor.config_path else Path("unknown")
        current_data, current_text, current_hash = self.get_current_saved_content_and_hash(module_id)
        post_rollback_snap_id = self._create_snapshot(module_id, target_path)

        # Perform restore
        restored_path = self._restore_snapshot(target_cs.prev_snapshot_id)

        # Inspect restored content to produce rollback patch record
        restored_data, restored_text, restored_hash = self.get_current_saved_content_and_hash(module_id)
        diff_text = compute_unified_diff(current_text, restored_text, restored_path.name)

        self._counter += 1
        now_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rb_id = f"cs-rollback-{module_id}-{now_ts}-{self._counter:04d}"

        rb_cs = ChangeSet(
            id=rb_id,
            module=module_id,
            base_hash=current_hash,
            patch=restored_data,
            validation_result=None,
            requested_by=requested_by,
            status=ChangeSetStatus.ROLLED_BACK_ON_DISK,
            applied_at=datetime.now(timezone.utc).isoformat(),
            prev_snapshot_id=post_rollback_snap_id,
            files_affected=[str(restored_path)],
            diff=diff_text,
            restart_required=descriptor.restart_required,
        )
        self._record_changeset(rb_cs)
        return rb_cs

    def _record_changeset(self, cs: ChangeSet) -> None:
        """Store ChangeSet in history index."""
        if cs.module not in self._history:
            self._history[cs.module] = []
        self._history[cs.module].append(cs)
        self._all_changesets[cs.id] = cs


# Shared singleton manager
changeset_manager = ChangeSetManager()
