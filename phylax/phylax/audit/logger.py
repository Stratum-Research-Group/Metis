"""Fail-closed append-only audit logger with cryptographic hash chaining."""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

from phylax.contracts.audit import AuditRecord
from phylax.contracts.decision import AuthorizationRecord, ExecutionReceipt
from phylax.contracts.policy import RedactionConfig

GENESIS_HASH = "0" * 64


class PhylaxAuditFailureError(RuntimeError):
    """Raised when writing to the audit log fails, triggering a fail-closed deny."""
    pass


class AuditLogger:
    """Manages an append-only, tamper-evident JSONL audit log with fail-closed semantics."""

    def __init__(
        self,
        log_path: Path | str,
        redaction_config: RedactionConfig | None = None,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.redaction_config = redaction_config or RedactionConfig()
        self._lock = threading.Lock()
        self._last_hash = self._get_initial_head_hash()
        self._last_id = self._get_initial_last_id()

    def _get_initial_head_hash(self) -> str:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return GENESIS_HASH
        
        last_line = ""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        last_line = stripped
            if last_line:
                data = json.loads(last_line)
                return data.get("record_hash", GENESIS_HASH)
        except Exception:
            return GENESIS_HASH
        return GENESIS_HASH

    def _get_initial_last_id(self) -> int:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return 0
        last_line = ""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        last_line = stripped
            if last_line:
                data = json.loads(last_line)
                return int(data.get("record_id", 0))
        except Exception:
            return 0
        return 0

    def redact_details(self, details: dict[str, Any] | None) -> dict[str, Any]:
        """Redact sensitive keys and content per redaction config."""
        if not details:
            return {}

        redacted: dict[str, Any] = {}
        mask_patterns = [p.lower() for p in self.redaction_config.mask_patterns]
        replacement = self.redaction_config.mask_replacement

        for k, v in details.items():
            k_lower = k.lower()
            if any(pat in k_lower for pat in mask_patterns):
                redacted[k] = replacement
            elif self.redaction_config.redact_content and k_lower in ("content", "payload", "body", "secret"):
                redacted[k] = f"{replacement} (length={len(str(v))})"
            elif isinstance(v, dict):
                redacted[k] = self.redact_details(v)
            else:
                redacted[k] = v

        return redacted

    def log_event(
        self,
        auth_record: AuthorizationRecord,
        receipt: ExecutionReceipt | None = None,
        context: dict[str, Any] | None = None,
    ) -> AuditRecord:
        """Write an audit entry with hash chaining.
        
        CRITICAL: If writing or flushing to disk fails, raises PhylaxAuditFailureError.
        """
        with self._lock:
            next_id = self._last_id + 1
            now_utc = datetime.now(timezone.utc).isoformat()
            prev_hash = self._last_hash

            redacted_info = self.redact_details(context)
            if auth_record.argument_digest:
                redacted_info["argument_digest"] = auth_record.argument_digest

            exec_status = None
            if receipt:
                exec_status = "EXECUTED_SUCCESS" if receipt.success else f"EXECUTED_FAILED: {receipt.error_message}"
            elif auth_record.effect.value == "deny":
                exec_status = "BLOCKED_DENIED"
            elif auth_record.effect.value == "ask":
                exec_status = "PENDING_APPROVAL"

            # Compute content hash
            canonical_content = json.dumps(
                {
                    "record_id": next_id,
                    "timestamp_utc": now_utc,
                    "prev_hash": prev_hash,
                    "subject": auth_record.subject,
                    "verb": auth_record.verb.value,
                    "resource_class": auth_record.resource_class.value,
                    "canonical_resource": auth_record.canonical_resource,
                    "effect": auth_record.effect.value,
                    "matched_rule_id": auth_record.matched_rule_id,
                    "policy_version": auth_record.policy_version,
                    "trace_id": auth_record.trace_id,
                    "execution_status": exec_status,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            record_hash = hashlib.sha256(
                f"{prev_hash}:{canonical_content}".encode("utf-8")
            ).hexdigest()

            record = AuditRecord(
                record_id=next_id,
                timestamp_utc=now_utc,
                prev_hash=prev_hash,
                record_hash=record_hash,
                subject=auth_record.subject,
                verb=auth_record.verb.value,
                resource_class=auth_record.resource_class.value,
                canonical_resource=auth_record.canonical_resource,
                effect=auth_record.effect.value,
                matched_rule_id=auth_record.matched_rule_id,
                policy_version=auth_record.policy_version,
                trace_id=auth_record.trace_id,
                execution_receipt_id=receipt.receipt_id if receipt else None,
                execution_status=exec_status,
                redacted_details=redacted_info,
            )

            # Persist to disk with fail-closed guarantee
            line = record.model_dump_json() + "\n"
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        f.write(line)
                        f.flush()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception as exc:
                # FAIL-CLOSED: audit write failure MUST abort the operation
                raise PhylaxAuditFailureError(
                    f"Fail-Closed Triggered: Audit log write failed to {self.log_path}: {exc}"
                ) from exc

            # Update in-memory chain state only upon verified disk flush
            self._last_id = next_id
            self._last_hash = record_hash
            return record
