"""Tests for approval lifecycle, argument digests, and atomic consumption."""

from pathlib import Path
import time
import pytest

from phylax.approvals.manager import ApprovalManager
from phylax.contracts.enums import ApprovalStatus, Effect, ResourceClass, Verb
from phylax.contracts.decision import AuthorizationRecord


@pytest.fixture
def sample_auth_record() -> AuthorizationRecord:
    return AuthorizationRecord(
        decision_id="dec-123456",
        timestamp_utc="2026-09-30T12:00:00Z",
        subject="kybernetes",
        verb=Verb.SEND,
        resource_class=ResourceClass.NETWORK,
        canonical_resource="https://external.api/v1/alert",
        effect=Effect.ASK,
        matched_rule_id="ask_send_rule",
        policy_version="1.0.0",
        reason="Owner approval required for external send",
        trace_id="trace-abc",
        approval_id="appr-test-1",
        argument_digest="abc123digest",
    )


def test_approval_creation_and_resolution(tmp_path: Path, sample_auth_record: AuthorizationRecord):
    storage = tmp_path / "approvals.json"
    mgr = ApprovalManager(storage)

    args = {"recipient": "admin", "msg": "cluster rebooting"}
    req = mgr.create_request(
        auth_record=sample_auth_record,
        action="send",
        target="https://external.api/v1/alert",
        arguments=args,
        ttl_seconds=10,
    )

    assert req.approval_id == "appr-test-1"
    assert req.status == ApprovalStatus.PENDING

    # Resolve approval
    mgr.resolve_request("appr-test-1", approved=True, reason="Approved by owner")
    resolved = mgr.get_request("appr-test-1")
    assert resolved.status == ApprovalStatus.APPROVED

    # Consume for execution
    ok, msg = mgr.consume_for_execution(
        approval_id="appr-test-1",
        subject="kybernetes",
        action="send",
        canonical_target="https://external.api/v1/alert",
        arguments=args,
        policy_version="1.0.0",
    )
    assert ok is True

    # Replay attempt must fail
    ok_replay, msg_replay = mgr.consume_for_execution(
        approval_id="appr-test-1",
        subject="kybernetes",
        action="send",
        canonical_target="https://external.api/v1/alert",
        arguments=args,
        policy_version="1.0.0",
    )
    assert ok_replay is False
    assert "already been consumed" in msg_replay


def test_argument_tampering_rejected(tmp_path: Path, sample_auth_record: AuthorizationRecord):
    mgr = ApprovalManager(tmp_path / "approvals.json")

    original_args = {"recipient": "safe_recipient", "data": "clean"}
    req = mgr.create_request(
        auth_record=sample_auth_record,
        action="send",
        target="https://external.api/v1/alert",
        arguments=original_args,
        ttl_seconds=30,
    )
    mgr.resolve_request(req.approval_id, approved=True)

    # Caller tries to execute with altered arguments
    tampered_args = {"recipient": "attacker", "data": "exfiltrated_keys"}
    ok, msg = mgr.consume_for_execution(
        approval_id=req.approval_id,
        subject="kybernetes",
        action="send",
        canonical_target="https://external.api/v1/alert",
        arguments=tampered_args,
        policy_version="1.0.0",
    )
    assert ok is False
    assert "Argument digest mismatch" in msg


def test_approval_ttl_expiration(tmp_path: Path, sample_auth_record: AuthorizationRecord):
    mgr = ApprovalManager(tmp_path / "approvals.json")

    args = {"msg": "quick"}
    req = mgr.create_request(
        auth_record=sample_auth_record,
        action="send",
        target="https://external.api/v1/alert",
        arguments=args,
        ttl_seconds=1,  # 1 second TTL
    )

    # Wait 1.2s for expiry
    time.sleep(1.2)

    # Fetching should mark it EXPIRED
    req_expired = mgr.get_request(req.approval_id)
    assert req_expired.status == ApprovalStatus.EXPIRED

    # Resolving expired request should raise TimeoutError
    with pytest.raises(TimeoutError, match="has expired"):
        mgr.resolve_request(req.approval_id, approved=True)


def test_persistence_across_restart(tmp_path: Path, sample_auth_record: AuthorizationRecord):
    storage = tmp_path / "persistent_approvals.json"
    mgr1 = ApprovalManager(storage)

    req = mgr1.create_request(
        auth_record=sample_auth_record,
        action="send",
        target="https://external.api/v1/alert",
        arguments={"k": "v"},
        ttl_seconds=100,
    )

    # Simulate restart by instantiating new manager from same storage
    mgr2 = ApprovalManager(storage)
    loaded_req = mgr2.get_request(req.approval_id)
    assert loaded_req is not None
    assert loaded_req.approval_id == req.approval_id
    assert loaded_req.status == ApprovalStatus.PENDING


def test_prosopon_approval_adapter(tmp_path: Path, sample_auth_record: AuthorizationRecord):
    from phylax.adapters.prosopon import ProsoponApprovalAdapter

    mgr = ApprovalManager(tmp_path / "approvals.json")
    adapter = ProsoponApprovalAdapter(mgr)

    req = mgr.create_request(
        auth_record=sample_auth_record,
        action="send",
        target="https://api.external/card",
        arguments={"title": "test"},
        ttl_seconds=60,
        turn_id="turn-999",
    )

    cards = adapter.get_pending_cards()
    assert len(cards) == 1
    assert cards[0]["approval_id"] == req.approval_id
    assert cards[0]["turn_id"] == "turn-999"
    assert cards[0]["action"] == "send"
    assert "expiry_utc" in cards[0]

    # Resolve via adapter
    resolved = adapter.resolve(req.approval_id, approved=True, reason="Owner approved on mobile")
    assert resolved["status"] == ApprovalStatus.APPROVED.value
    assert len(adapter.get_pending_cards()) == 0
