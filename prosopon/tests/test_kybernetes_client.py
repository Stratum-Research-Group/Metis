"""Unit tests for Kybernetes mock client and event streaming."""

import asyncio
import pytest

from prosopon.client.mock_kybernetes import MockKybernetesClient


@pytest.mark.asyncio
async def test_mock_kybernetes_stream_standard_query():
    client = MockKybernetesClient()
    events = []

    async for event in client.submit_turn_stream(session_id="s-1", user_text="What is Metis?"):
        events.append(event)

    types = [e.event_type for e in events]
    assert "started" in types
    assert "plan_selected" in types
    assert "evidence" in types
    assert "delta" in types
    assert "completed" in types

    # Monotonic sequence numbers
    seqs = [e.seq_num for e in events]
    assert seqs == list(range(len(events)))

    # Final event status must be verified
    assert events[-1].status == "verified"
    assert "route_summary" in events[-1].payload


@pytest.mark.asyncio
async def test_mock_kybernetes_stream_phylax_approval():
    client = MockKybernetesClient()
    events = []

    async def consume_stream():
        async for event in client.submit_turn_stream(session_id="s-1", user_text="Please write file report.txt"):
            events.append(event)
            if event.event_type == "ask_approval":
                approval_id = event.payload["approval"]["approval_id"]
                # Simulate user clicking "Allow Once"
                await client.resolve_approval(approval_id, "allow_once")

    await consume_stream()

    types = [e.event_type for e in events]
    assert "ask_approval" in types
    assert "approval_resolved" in types
    assert "completed" in types
