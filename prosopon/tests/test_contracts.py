"""Unit tests for Prosopon contracts and schemas."""

from datetime import datetime, timezone, timedelta
import pytest
from pydantic import ValidationError

from prosopon.contracts.kybernetes import Citation, TurnEvent
from prosopon.contracts.phylax import ApprovalRequest, ApprovalResponse
from prosopon.contracts.session import ChatMessage, SessionRecord


def test_citation_validation():
    c = Citation(
        citation_id="[1]",
        passage_id="pass_100",
        source_id="src_wiki",
        source_title="Wikipedia",
        text="Sample text",
        score=0.92,
    )
    assert c.citation_id == "[1]"
    assert c.score == 0.92


def test_turn_event_validation():
    ev = TurnEvent(
        turn_id="turn_1",
        seq_num=0,
        event_type="started",
        payload={"query": "hello"},
    )
    assert ev.turn_id == "turn_1"
    assert ev.event_type == "started"

    # Monotonic seq_num cannot be negative
    with pytest.raises(ValidationError):
        TurnEvent(turn_id="turn_1", seq_num=-1, event_type="started")


def test_approval_request_expiry():
    future_time = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    app = ApprovalRequest(
        approval_id="app-1",
        turn_id="t-1",
        action="WRITE_FILE",
        target="/tmp/test.txt",
        policy_rule="Rule 1",
        expiry_utc=future_time,
    )
    assert app.is_expired() is False

    past_time = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    expired_app = ApprovalRequest(
        approval_id="app-2",
        turn_id="t-1",
        action="WRITE_FILE",
        target="/tmp/test.txt",
        policy_rule="Rule 1",
        expiry_utc=past_time,
    )
    assert expired_app.is_expired() is True


def test_session_record_messages():
    sess = SessionRecord(
        session_id="s-1",
        name="Test Chat",
    )
    msg = ChatMessage(
        message_id="m-1",
        role="user",
        content="Hello Metis",
    )
    sess.messages.append(msg)
    assert len(sess.messages) == 1
    assert sess.messages[0].content == "Hello Metis"
