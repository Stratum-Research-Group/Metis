"""Tests for cryptographic audit logger, verification, and fail-closed semantics."""

from pathlib import Path
from unittest.mock import patch
import pytest

from phylax.audit.logger import AuditLogger, PhylaxAuditFailureError
from phylax.audit.verifier import AuditVerifier
from phylax.contracts.enums import Effect, ResourceClass, Verb
from phylax.contracts.decision import AuthorizationRecord, ExecutionReceipt
from phylax.contracts.policy import RedactionConfig


@pytest.fixture
def sample_auth() -> AuthorizationRecord:
    return AuthorizationRecord(
        decision_id="dec-audit-1",
        timestamp_utc="2026-09-30T12:00:00Z",
        subject="kybernetes",
        verb=Verb.READ,
        resource_class=ResourceClass.FILE,
        canonical_resource="/docs/manual.pdf",
        effect=Effect.ALLOW,
        matched_rule_id="allow_docs",
        policy_version="1.0.0",
        reason="Rule allowed docs read",
    )


def test_audit_hash_chain_and_verification(tmp_path: Path, sample_auth: AuthorizationRecord):
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_file)

    # Log 3 events
    rec1 = logger.log_event(sample_auth)
    assert rec1.record_id == 1
    assert rec1.prev_hash == "0" * 64

    rec2 = logger.log_event(sample_auth)
    assert rec2.record_id == 2
    assert rec2.prev_hash == rec1.record_hash

    receipt = ExecutionReceipt(
        receipt_id="rcpt-1",
        decision_id=sample_auth.decision_id,
        timestamp_utc="2026-09-30T12:00:01Z",
        success=True,
    )
    rec3 = logger.log_event(sample_auth, receipt=receipt)
    assert rec3.record_id == 3
    assert rec3.prev_hash == rec2.record_hash

    # Verify chain
    verifier = AuditVerifier(log_file)
    res = verifier.verify()
    assert res.is_valid is True
    assert res.total_records == 3
    assert res.head_hash == rec3.record_hash


def test_audit_tamper_detection(tmp_path: Path, sample_auth: AuthorizationRecord):
    log_file = tmp_path / "audit_tamper.jsonl"
    logger = AuditLogger(log_file)

    logger.log_event(sample_auth)
    logger.log_event(sample_auth)

    # Tamper with the first record in the file
    content = log_file.read_text()
    tampered_content = content.replace("allow_docs", "tampered_rule")
    log_file.write_text(tampered_content)

    verifier = AuditVerifier(log_file)
    res = verifier.verify()
    assert res.is_valid is False
    assert res.broken_record_id == 1
    assert "Tampered content" in (res.error_message or "")


def test_fail_closed_audit_failure_blocks_action(tmp_path: Path, sample_auth: AuthorizationRecord):
    log_file = tmp_path / "audit_fail.jsonl"
    logger = AuditLogger(log_file)

    # Simulate an OS write error / read-only filesystem
    with patch("builtins.open", side_effect=OSError("Disk write failed: No space left on device")):
        with pytest.raises(PhylaxAuditFailureError, match="Fail-Closed Triggered"):
            logger.log_event(sample_auth)


def test_audit_sensitive_data_redaction(tmp_path: Path, sample_auth: AuthorizationRecord):
    log_file = tmp_path / "audit_redacted.jsonl"
    config = RedactionConfig(
        redact_content=True,
        mask_patterns=["token", "password", "api_key"],
        mask_replacement="[REDACTED]",
    )
    logger = AuditLogger(log_file, redaction_config=config)

    sensitive_context = {
        "user": "alice",
        "api_key": "sk-1234567890",
        "nested": {"password": "supersecretpassword"},
        "payload": "very long secret document content",
    }

    rec = logger.log_event(sample_auth, context=sensitive_context)
    details = rec.redacted_details

    assert details["user"] == "alice"
    assert details["api_key"] == "[REDACTED]"
    assert details["nested"]["password"] == "[REDACTED]"
    assert "[REDACTED]" in details["payload"]
