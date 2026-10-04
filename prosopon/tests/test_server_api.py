"""Integration tests for Prosopon FastAPI backend."""

from pathlib import Path
import pytest
from httpx import AsyncClient, ASGITransport

from prosopon.config import ProsoponConfig
from prosopon.server import create_app


@pytest.fixture
def app_instance(tmp_path):
    cfg = ProsoponConfig(
        data_dir=tmp_path,
        mock_kybernetes=True,
    )
    return create_app(cfg)


@pytest.mark.asyncio
async def test_health_check(app_instance):
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_session_crud(app_instance):
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create
        create_resp = await client.post("/api/sessions", json={"name": "Test Session"})
        assert create_resp.status_code == 200
        data = create_resp.json()
        session_id = data["session_id"]
        assert data["name"] == "Test Session"

        # List
        list_resp = await client.get("/api/sessions")
        assert list_resp.status_code == 200
        assert len(list_resp.json()) == 1

        # Get
        get_resp = await client.get(f"/api/sessions/{session_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["session_id"] == session_id

        # Delete
        del_resp = await client.delete(f"/api/sessions/{session_id}")
        assert del_resp.status_code == 200

        # Verify Deleted
        get_after = await client.get(f"/api/sessions/{session_id}")
        assert get_after.status_code == 404


@pytest.mark.asyncio
async def test_chat_turn_stream(app_instance):
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create session
        sess_resp = await client.post("/api/sessions", json={"name": "Stream Test"})
        session_id = sess_resp.json()["session_id"]

        # Stream
        stream_resp = await client.post(
            f"/api/chat/sessions/{session_id}/stream",
            json={"query": "What is the capital of France?"},
        )
        assert stream_resp.status_code == 200
        content = stream_resp.text
        assert "data:" in content
        assert "completed" in content

        # Check that session messages were saved
        sess_after = await client.get(f"/api/sessions/{session_id}")
        msgs = sess_after.json()["messages"]
        assert len(msgs) == 2  # 1 user + 1 assistant
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["status"] == "verified"


@pytest.mark.asyncio
async def test_pairing_auth_flow(app_instance):
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Generate PIN
        pin_resp = await client.post("/api/auth/pin")
        assert pin_resp.status_code == 200
        pin = pin_resp.json()["pin"]

        # Pair device
        pair_resp = await client.post("/api/auth/pair", json={"pin": pin, "device_name": "Phone"})
        assert pair_resp.status_code == 200
        token = pair_resp.json()["token"]
        assert token is not None

        # Revoke
        rev_resp = await client.post(
            "/api/auth/revoke",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert rev_resp.status_code == 200
        assert rev_resp.json()["revoked_count"] == 1
