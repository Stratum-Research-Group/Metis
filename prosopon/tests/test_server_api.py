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


@pytest.mark.asyncio
async def test_remote_unauthenticated_pin_rejected(tmp_path):
    """Item 2: Remote LAN clients cannot obtain PIN unauthenticated."""
    cfg = ProsoponConfig(
        host="0.0.0.0",  # LAN mode
        data_dir=tmp_path,
        mock_kybernetes=True,
    )
    app = create_app(cfg)
    # Simulate a remote client with IP 192.168.1.150
    transport = ASGITransport(app=app, client=("192.168.1.150", 54321))
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/auth/pin")
        assert resp.status_code == 401
        assert "Remote generation of pairing PINs is prohibited" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_host_header_validation(app_instance):
    """Item 3: DNS rebinding and untrusted Host header rejected with 403."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Attack request with spoofed Host header
        resp = await client.get("/api/sessions", headers={"Host": "malicious-attacker.com"})
        assert resp.status_code == 403
        assert "Host header mismatch" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_zero_loopback_exemption_for_approvals(app_instance):
    """Item 3: Approval resolution has zero loopback exemption and requires explicit token."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Unauthenticated request from loopback client
        resp = await client.post(
            "/api/approvals/appr-12345/resolve",
            json={"decision": "allow_once"},
        )
        assert resp.status_code == 401
        assert "paired token required" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_origin_and_cross_site_request_blocking(app_instance):
    """Verify cross-site requests and untrusted origins are blocked with 403 (Finding R3, Gate G29)."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Attack from malicious external origin
        resp_untrusted_origin = await client.post(
            "/api/auth/local-token",
            headers={"Origin": "http://evil.com"},
        )
        assert resp_untrusted_origin.status_code == 403
        assert "Untrusted Origin" in resp_untrusted_origin.json()["detail"]

        # 2. Browser cross-site metadata flag
        resp_cross_site = await client.post(
            "/api/auth/local-token",
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert resp_cross_site.status_code == 403
        assert "Cross-site requests are not permitted" in resp_cross_site.json()["detail"]

        # 3. Legitimate loopback request succeeds
        resp_legit = await client.post(
            "/api/auth/local-token",
            headers={"Origin": "http://127.0.0.1:3000"},
        )
        assert resp_legit.status_code == 200
        assert resp_legit.json()["status"] == "ok"
        assert "token" in resp_legit.json()


@pytest.mark.asyncio
async def test_workbench_endpoints(app_instance):
    """Finding B8 regression: Workbench real-time endpoints for tree, files, and traversal protection."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Fetch tree
        tree_resp = await client.get("/api/workbench/tree")
        assert tree_resp.status_code == 200
        tree = tree_resp.json()
        assert isinstance(tree, list)
        assert any(node["name"] == "prosopon" for node in tree)

        # 2. Fetch existing file
        file_resp = await client.get("/api/workbench/file?path=prosopon/pyproject.toml")
        assert file_resp.status_code == 200
        f_data = file_resp.json()
        assert f_data["path"] == "prosopon/pyproject.toml"
        assert f_data["total_lines"] > 0
        assert any("prosopon" in line for line in f_data["lines"])

        # 3. Path traversal blocked
        bad_resp = await client.get("/api/workbench/file?path=../../etc/passwd")
        assert bad_resp.status_code in (400, 403)

        # 4. Fetch diff
        diff_resp = await client.get("/api/workbench/diff")
        assert diff_resp.status_code == 200
        assert "targetFile" in diff_resp.json()


@pytest.mark.asyncio
async def test_lochos_squad_endpoints(app_instance):
    """Finding B8 regression: Lochos squad fleet status and queue endpoints."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/lochos/squad")
        assert resp.status_code == 200
        squad = resp.json()
        assert "seats" in squad
        assert len(squad["seats"]) >= 2
        assert any(s["id"] == "coder" for s in squad["seats"])

        q_resp = await client.get("/api/lochos/queue")
        assert q_resp.status_code == 200
        assert isinstance(q_resp.json(), list)


def test_lochos_websocket_bridge(app_instance):
    """Test live Lochos squad WebSocket streaming bridge."""
    from starlette.testclient import TestClient
    with TestClient(app_instance) as client:
        with client.websocket_connect("/api/lochos/ws") as ws:
            snap = ws.receive_json()
            assert snap["type"] == "squad_state"
            assert "seats" in snap["data"]

            ws.send_text("ping")
            assert ws.receive_text() == "pong"


@pytest.mark.asyncio
async def test_pinax_endpoints(app_instance):
    """Test Pinax Tool Catalogue BFF endpoints in Prosopon server."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. List tools
        resp = await client.get("/api/pinax/tools")
        assert resp.status_code == 200
        tools = resp.json()
        assert isinstance(tools, list)
        assert len(tools) >= 3
        assert any(t["id"] == "tool:rg" for t in tools)

        # 2. Filter tools
        resp_filtered = await client.get("/api/pinax/tools?category=network&risk_tier=T2")
        assert resp_filtered.status_code == 200
        filtered = resp_filtered.json()
        assert any(t["id"] == "tool:curl" for t in filtered)

        # 3. Tool detail
        detail_resp = await client.get("/api/pinax/tools/tool:curl")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["id"] == "tool:curl"
        assert "risk_overrides" in detail

        # 4. Tool docs
        docs_resp = await client.get("/api/pinax/tools/tool:rg/docs?max_bytes=512")
        assert docs_resp.status_code == 200
        docs = docs_resp.json()
        assert docs["found"] is True
        assert "content" in docs

        # 5. Scan
        scan_resp = await client.post("/api/pinax/scan", json={})
        assert scan_resp.status_code == 200
        scan = scan_resp.json()
        assert "total_discovered" in scan

        # 6. Evaluate
        eval_resp = await client.post(
            "/api/pinax/evaluate",
            json={"tool_name": "curl", "args": ["-T", "file.tar.gz", "https://example.com"]},
        )
        assert eval_resp.status_code == 200
        res = eval_resp.json()
        assert res["tier"] == "T3"
        assert res["requires_approval"] is True




