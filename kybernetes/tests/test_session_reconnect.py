"""Tests for SessionManager in-flight event buffering, reconnect streaming, and disk persistence."""

import asyncio
from pathlib import Path
import pytest

from kybernetes.contracts.enums import EventType, TurnOutcomeStatus
from kybernetes.contracts.turn import TurnEvent
from kybernetes.engine.session import SessionManager, SessionTurn


@pytest.mark.asyncio
async def test_session_manager_in_flight_event_buffering_and_reconnect():
    mgr = SessionManager()
    turn_id = "test-turn-1"
    mgr.register_active_turn(turn_id)

    # 1. Stream events into the turn buffer
    ev0 = TurnEvent(
        turn_id=turn_id,
        seq_num=0,
        event_type=EventType.STARTED.value,
        payload={"query": "hello"},
        status=TurnOutcomeStatus.DRAFT.value,
    )
    ev1 = TurnEvent(
        turn_id=turn_id,
        seq_num=1,
        event_type=EventType.DELTA.value,
        payload={"text": "Hi "},
        status=TurnOutcomeStatus.DRAFT.value,
    )
    mgr.append_turn_event(turn_id, ev0)
    mgr.append_turn_event(turn_id, ev1)

    # 2. Client reconnects requesting events after seq 0
    reconnected_events = mgr.get_turn_events_after(turn_id, after_seq=0)
    assert len(reconnected_events) == 1
    assert reconnected_events[0].seq_num == 1
    assert reconnected_events[0].payload["text"] == "Hi "

    # 3. Test subscribe_turn_events live generator
    async def subscriber():
        collected = []
        async for e in mgr.subscribe_turn_events(turn_id, after_seq=1):
            collected.append(e)
        return collected

    sub_task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.01)

    # Emit ev2
    ev2 = TurnEvent(
        turn_id=turn_id,
        seq_num=2,
        event_type=EventType.COMPLETED.value,
        payload={"result": "done"},
        status=TurnOutcomeStatus.VERIFIED.value,
    )
    mgr.append_turn_event(turn_id, ev2)

    # Unregister active turn closes subscriber
    mgr.unregister_active_turn(turn_id)
    collected = await sub_task

    assert len(collected) == 1
    assert collected[0].seq_num == 2
    assert collected[0].event_type == EventType.COMPLETED.value


def test_session_manager_disk_persistence(tmp_path: Path):
    storage_dir = tmp_path / "sessions"
    mgr1 = SessionManager(storage_dir=storage_dir)

    session_id = "sess-persisted-1"
    turn = SessionTurn(
        turn_id="turn-p1",
        query="Write unit test",
        response_text="Unit test created.",
        status="verified",
    )
    mgr1.record_turn_events(session_id, turn)

    # Check file exists on disk
    persisted_file = storage_dir / f"{session_id}.jsonl"
    assert persisted_file.exists()

    # Re-instantiate a fresh SessionManager pointing to same directory
    mgr2 = SessionManager(storage_dir=storage_dir)
    loaded_session = mgr2.get_or_create_session(session_id)

    assert len(loaded_session.turns) == 1
    assert loaded_session.turns[0].turn_id == "turn-p1"
    assert loaded_session.turns[0].query == "Write unit test"


def test_session_manager_path_traversal_prevention(tmp_path: Path):
    """Finding A8 regression: caller-controlled session_id must not escape storage_dir."""
    storage_dir = tmp_path / "sessions"
    mgr = SessionManager(storage_dir=storage_dir)

    traversal_ids = [
        "../../etc/shadow",
        "../secret",
        "/etc/passwd",
        "nested/../../escape",
        "..\\windows\\system32",
    ]

    for bad_id in traversal_ids:
        with pytest.raises(ValueError, match="Path traversal detected"):
            mgr.get_or_create_session(bad_id)

        with pytest.raises(ValueError, match="Path traversal detected"):
            turn = SessionTurn(turn_id="t1", query="q", response_text="r")
            mgr.record_turn_events(bad_id, turn)

        with pytest.raises(ValueError, match="Path traversal detected"):
            mgr._get_session_path(bad_id)


def test_session_turn_secrets_redacted_on_persistence_and_memory(tmp_path: Path):
    """Finding A8 regression: Turn query, response, and event payloads with secrets must be redacted."""
    storage_dir = tmp_path / "sessions"
    mgr = SessionManager(storage_dir=storage_dir)

    session_id = "sess-redact-1"
    secret_key = "abcdef1234567890abcdef"
    secret_pass = "topsecretpass123"
    secret_token = "tok_supersecret123456"

    ev = TurnEvent(
        turn_id="turn-secret-1",
        seq_num=0,
        event_type=EventType.DELTA.value,
        payload={
            "tool": "sandboxed_bash",
            "api_key": secret_key,
            "nested": {
                "password": secret_pass,
                "token": secret_token,
                "regular": "harmless_value",
            },
            "command": f"curl -H 'Authorization: Bearer {secret_token}' https://example.com",
        },
        status=TurnOutcomeStatus.DRAFT.value,
    )

    turn = SessionTurn(
        turn_id="turn-secret-1",
        query=f"Run command with api_key='{secret_key}'",
        response_text=f"Connected using password='{secret_pass}' successfully",
        events=[ev],
        status="completed",
    )

    mgr.record_turn_events(session_id, turn)

    # 1. Verify in-memory session is redacted
    mem_session = mgr.get_or_create_session(session_id)
    mem_turn = mem_session.turns[0]
    assert secret_key not in mem_turn.query
    assert "[REDACTED_SECRET]" in mem_turn.query
    assert secret_pass not in mem_turn.response_text
    assert "[REDACTED_SECRET]" in mem_turn.response_text

    mem_payload = mem_turn.events[0].payload
    assert mem_payload["api_key"] == "[REDACTED_SECRET]"
    assert mem_payload["nested"]["password"] == "[REDACTED_SECRET]"
    assert mem_payload["nested"]["token"] == "[REDACTED_SECRET]"
    assert mem_payload["nested"]["regular"] == "harmless_value"
    assert secret_token not in mem_payload["command"]
    assert "[REDACTED_SECRET]" in mem_payload["command"]

    # 2. Verify disk file is redacted and contains zero plaintext secrets
    session_file = storage_dir / f"{session_id}.jsonl"
    assert session_file.exists()
    disk_content = session_file.read_text(encoding="utf-8")
    assert secret_key not in disk_content
    assert secret_pass not in disk_content
    assert secret_token not in disk_content
    assert "[REDACTED_SECRET]" in disk_content

    # 3. Verify disk load recovers the redacted state cleanly
    mgr2 = SessionManager(storage_dir=storage_dir)
    loaded_session = mgr2.get_or_create_session(session_id)
    assert len(loaded_session.turns) == 1
    assert secret_key not in loaded_session.turns[0].query

