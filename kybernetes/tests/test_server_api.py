"""Integration tests for Kybernetes FastAPI endpoints and Prosopon SSE compatibility."""

import json
import pytest
from httpx import AsyncClient, ASGITransport

from kybernetes.contracts.turn import TurnEvent
from kybernetes.server.app import create_app


@pytest.fixture
def app():
    return create_app(db_path=":memory:", service_token="test-service-token")


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
        async with client.stream(
            "POST",
            "/v1/chat/turn",
            json=payload,
            headers={"Authorization": "Bearer test-service-token"},
        ) as resp:
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
    # Register active turn
    app.state.sessions.register_active_turn("turn-999")
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Active turn cancellation succeeds
        resp = await client.post(
            "/v1/chat/turns/turn-999/cancel",
            headers={"Authorization": "Bearer test-service-token"},
        )
        assert resp.status_code == 200
        assert resp.json()["cancelled"] is True

        # Unknown / already finished turn returns 404
        resp_unknown = await client.post(
            "/v1/chat/turns/turn-nonexistent/cancel",
            headers={"Authorization": "Bearer test-service-token"},
        )
        assert resp_unknown.status_code == 404


@pytest.mark.asyncio
async def test_api_approval_not_found(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/approvals/nonexistent/resolve",
            json={"approval_id": "nonexistent", "decision": "deny"},
            headers={"Authorization": "Bearer test-service-token"},
        )
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_service_token_auth():
    """Item 3 & Item 9: Authenticate the control boundary on Kybernetes internal routes."""
    secured_app = create_app(db_path=":memory:", service_token="secret-service-token-123")
    secured_app.state.sessions.register_active_turn("turn-1")
    transport = ASGITransport(app=secured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Health is public
        h_resp = await client.get("/v1/health")
        assert h_resp.status_code == 200

        # Unauthenticated request to /v1/chat/turns/{turn_id}/cancel rejected
        unauth_resp = await client.post("/v1/chat/turns/turn-1/cancel")
        assert unauth_resp.status_code == 401
        assert "Valid Kybernetes service token required" in unauth_resp.json()["detail"]

        # Authenticated request on active turn succeeds
        auth_resp = await client.post(
            "/v1/chat/turns/turn-1/cancel",
            headers={"Authorization": "Bearer secret-service-token-123"},
        )
        assert auth_resp.status_code == 200
        assert auth_resp.json()["cancelled"] is True


@pytest.mark.asyncio
async def test_cors_restriction():
    """Item 3: Restrict CORS origins from wildcard '*'."""
    app_with_origins = create_app(
        db_path=":memory:",
        allowed_origins=["http://127.0.0.1:8000"],
    )
    transport = ASGITransport(app=app_with_origins)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Allowed origin
        resp_allowed = await client.options(
            "/v1/health",
            headers={
                "Origin": "http://127.0.0.1:8000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp_allowed.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"

        # Disallowed origin
        resp_disallowed = await client.options(
            "/v1/health",
            headers={
                "Origin": "http://evil-attacker.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp_disallowed.headers.get("access-control-allow-origin") != "http://evil-attacker.com"

