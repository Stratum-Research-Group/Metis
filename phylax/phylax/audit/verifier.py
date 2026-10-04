"""Audit chain verification utility."""

import hashlib
import json
from pathlib import Path

from phylax.contracts.audit import AuditRecord, AuditVerificationResult
from phylax.audit.logger import GENESIS_HASH


class AuditVerifier:
    """Verifies the append-only cryptographic hash chain of an audit log."""

    def __init__(self, log_path: Path | str):
        self.log_path = Path(log_path)

    def verify(self) -> AuditVerificationResult:
        if not self.log_path.exists():
            return AuditVerificationResult(
                is_valid=True,
                total_records=0,
                head_hash=GENESIS_HASH,
                broken_record_id=None,
                error_message="Audit log does not exist yet",
            )

        expected_prev_hash = GENESIS_HASH
        expected_record_id = 1
        last_hash = GENESIS_HASH
        count = 0

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    data = json.loads(stripped)
                    record = AuditRecord.model_validate(data)
                except Exception as exc:
                    return AuditVerificationResult(
                        is_valid=False,
                        total_records=count,
                        head_hash=last_hash,
                        broken_record_id=expected_record_id,
                        error_message=f"Parse failure on line {line_no}: {exc}",
                    )

                # Check monotonic record ID
                if record.record_id != expected_record_id:
                    return AuditVerificationResult(
                        is_valid=False,
                        total_records=count,
                        head_hash=last_hash,
                        broken_record_id=record.record_id,
                        error_message=(
                            f"Non-monotonic record ID: expected {expected_record_id}, "
                            f"found {record.record_id}"
                        ),
                    )

                # Check prev_hash
                if record.prev_hash != expected_prev_hash:
                    return AuditVerificationResult(
                        is_valid=False,
                        total_records=count,
                        head_hash=last_hash,
                        broken_record_id=record.record_id,
                        error_message=(
                            f"Broken chain at record {record.record_id}: record.prev_hash "
                            f"'{record.prev_hash}' does not match expected '{expected_prev_hash}'"
                        ),
                    )

                # Recompute content hash
                canonical_content = json.dumps(
                    {
                        "record_id": record.record_id,
                        "timestamp_utc": record.timestamp_utc,
                        "prev_hash": record.prev_hash,
                        "subject": record.subject,
                        "verb": record.verb,
                        "resource_class": record.resource_class,
                        "canonical_resource": record.canonical_resource,
                        "effect": record.effect,
                        "matched_rule_id": record.matched_rule_id,
                        "policy_version": record.policy_version,
                        "trace_id": record.trace_id,
                        "execution_status": record.execution_status,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                recomputed_hash = hashlib.sha256(
                    f"{record.prev_hash}:{canonical_content}".encode("utf-8")
                ).hexdigest()

                if record.record_hash != recomputed_hash:
                    return AuditVerificationResult(
                        is_valid=False,
                        total_records=count,
                        head_hash=last_hash,
                        broken_record_id=record.record_id,
                        error_message=(
                            f"Tampered content at record {record.record_id}: stored hash "
                            f"'{record.record_hash}' != recomputed '{recomputed_hash}'"
                        ),
                    )

                expected_prev_hash = record.record_hash
                last_hash = record.record_hash
                expected_record_id += 1
                count += 1

        return AuditVerificationResult(
            is_valid=True,
            total_records=count,
            head_hash=last_hash,
            broken_record_id=None,
            error_message=None,
        )
