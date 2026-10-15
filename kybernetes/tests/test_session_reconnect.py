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
