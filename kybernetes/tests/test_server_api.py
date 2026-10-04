"""Integration tests for Kybernetes FastAPI endpoints and Prosopon SSE compatibility."""

import json
import pytest
from httpx import AsyncClient, ASGITransport

from kybernetes.contracts.turn import TurnEvent
from kybernetes.server.app import create_app


@pytest.fixture
def app():
    return create_app(db_path=":memory:")


@pytest.mark.asyncio
async def test_api_health(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "kybernetes"
        assert "dependencies" in data


@pytest.mark.asyncio
async def test_api_chat_turn_stream(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "session_id": "test-session",
            "query": "Hello Metis assistant",
        }
        async with client.stream("POST", "/v1/chat/turn", json=payload) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            events = []
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data:"):
                    raw = line[len("data:"):].strip()
                    if raw:
                        events.append(TurnEvent.model_validate_json(raw))

            assert len(events) >= 3
            assert events[0].event_type == "started"
            assert events[-1].event_type == "completed"


@pytest.mark.asyncio
async def test_api_cancel_turn(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/turns/turn-999/cancel")
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is True


@pytest.mark.asyncio
async def test_api_approval_not_found(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/approvals/nonexistent/resolve",
            json={"approval_id": "nonexistent", "decision": "deny"},
        )
        assert resp.status_code == 404
