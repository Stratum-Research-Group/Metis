"""Integration tests for Prosopon FastAPI backend with zero-loopback-exemption auth enforcement."""

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


@pytest.fixture
def auth_headers(app_instance):
    tok = app_instance.state.pairing_mgr.get_or_create_host_owner_token()
    return {"Authorization": f"Bearer {tok}"}


@pytest.mark.asyncio
async def test_health_check(app_instance):
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_zero_loopback_exemption_general_endpoints(app_instance):
    """Verify unauthenticated loopback requests are rejected with 401 across endpoints."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # GET endpoints
        for path in ["/api/sessions", "/api/workbench/tree", "/api/lochos/squad", "/api/pinax/tools"]:
            resp = await client.get(path)
            assert resp.status_code == 401, f"{path} did not reject unauthenticated request"
            assert "Unauthorized" in resp.json()["detail"]

        # POST endpoint
        resp_pin = await client.post("/api/auth/pin")
        assert resp_pin.status_code == 401
        assert "Pairing PIN generation requires host owner credential" in resp_pin.json()["detail"]


@pytest.mark.asyncio
async def test_session_crud(app_instance, auth_headers):
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create
        create_resp = await client.post("/api/sessions", json={"name": "Test Session"}, headers=auth_headers)
        assert create_resp.status_code == 200
        data = create_resp.json()
        session_id = data["session_id"]
        assert data["name"] == "Test Session"

        # List
        list_resp = await client.get("/api/sessions", headers=auth_headers)
        assert list_resp.status_code == 200
        assert len(list_resp.json()) == 1

        # Get
        get_resp = await client.get(f"/api/sessions/{session_id}", headers=auth_headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["session_id"] == session_id

        # Delete
        del_resp = await client.delete(f"/api/sessions/{session_id}", headers=auth_headers)
        assert del_resp.status_code == 200

        # Verify Deleted
        get_after = await client.get(f"/api/sessions/{session_id}", headers=auth_headers)
        assert get_after.status_code == 404


@pytest.mark.asyncio
async def test_chat_turn_stream(app_instance, auth_headers):
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create session
        sess_resp = await client.post("/api/sessions", json={"name": "Stream Test"}, headers=auth_headers)
        assert sess_resp.status_code == 200
        session_id = sess_resp.json()["session_id"]

        # Stream
        stream_resp = await client.post(
            f"/api/chat/sessions/{session_id}/stream",
            json={"query": "What is the capital of France?"},
            headers=auth_headers,
        )
        assert stream_resp.status_code == 200
        content = stream_resp.text
        assert "data:" in content
        assert "completed" in content

        # Check that session messages were saved
        sess_after = await client.get(f"/api/sessions/{session_id}", headers=auth_headers)
        msgs = sess_after.json()["messages"]
        assert len(msgs) == 2  # 1 user + 1 assistant
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["status"] == "verified"


@pytest.mark.asyncio
async def test_pairing_auth_flow(app_instance):
    transport = ASGITransport(app=app_instance)
    host_tok = app_instance.state.pairing_mgr.get_or_create_host_owner_token()
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Generate PIN using host owner token
        pin_resp = await client.post("/api/auth/pin", headers={"X-Host-Owner-Token": host_tok})
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
    """Remote LAN clients cannot obtain PIN unauthenticated."""
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
        assert "Pairing PIN generation requires host owner credential" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_host_header_validation(app_instance):
    """DNS rebinding and untrusted Host header rejected with 403."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Attack request with spoofed Host header
        resp = await client.get("/api/health", headers={"Host": "malicious-attacker.com"})
        assert resp.status_code == 403
        assert "Host header mismatch" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_zero_loopback_exemption_for_approvals(app_instance):
    """Approval resolution has zero loopback exemption and requires explicit token."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Unauthenticated request from loopback client
        resp = await client.post(
            "/api/approvals/appr-12345/resolve",
            json={"decision": "allow_once"},
        )
        assert resp.status_code == 401
        assert "token required" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_origin_and_cross_site_request_blocking(app_instance):
    """Verify cross-site requests and untrusted origins are blocked with 403 (Finding R3, Gate G29)."""
    transport = ASGITransport(app=app_instance)
    host_tok = app_instance.state.pairing_mgr.get_or_create_host_owner_token()
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Attack from malicious external origin
        resp_untrusted_origin = await client.post(
            "/api/auth/local-token",
            headers={"Origin": "http://evil.com", "X-Host-Owner-Token": host_tok},
        )
        assert resp_untrusted_origin.status_code == 403
        assert "Untrusted Origin" in resp_untrusted_origin.json()["detail"]

        # 2. Browser cross-site metadata flag
        resp_cross_site = await client.post(
            "/api/auth/local-token",
            headers={"Sec-Fetch-Site": "cross-site", "X-Host-Owner-Token": host_tok},
        )
        assert resp_cross_site.status_code == 403
        assert "Cross-site requests are not permitted" in resp_cross_site.json()["detail"]

        # 3. Legitimate loopback request with host authority succeeds
        resp_legit = await client.post(
            "/api/auth/local-token",
            headers={"Origin": "http://127.0.0.1:3000", "X-Host-Owner-Token": host_tok},
        )
        assert resp_legit.status_code == 200
        assert resp_legit.json()["status"] == "ok"
        assert "token" in resp_legit.json()

        # 4. Unauthenticated loopback request fails with 401
        resp_unauth = await client.post(
            "/api/auth/local-token",
            headers={"Origin": "http://127.0.0.1:3000"},
        )
        assert resp_unauth.status_code == 401


@pytest.mark.asyncio
async def test_bootstrap_ticket_exchange(app_instance):
    """Verify single-use bootstrap ticket exchange via /api/auth/exchange-ticket."""
    transport = ASGITransport(app=app_instance)
    ticket = app_instance.state.pairing_mgr.create_bootstrap_ticket(validity_seconds=60)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Exchange ticket
        resp = await client.post("/api/auth/exchange-ticket", json={"ticket": ticket})
        assert resp.status_code == 200
        token = resp.json()["token"]
        assert token is not None

        # Re-using the same ticket fails with 401
        reuse_resp = await client.post("/api/auth/exchange-ticket", json={"ticket": ticket})
        assert reuse_resp.status_code == 401

        # Newly exchanged token has admin access
        crud_resp = await client.get("/api/sessions", headers={"Authorization": f"Bearer {token}"})
        assert crud_resp.status_code == 200


@pytest.mark.asyncio
async def test_workbench_endpoints(app_instance, auth_headers):
    """Workbench real-time endpoints for tree, files, and traversal protection with auth."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Fetch tree
        tree_resp = await client.get("/api/workbench/tree", headers=auth_headers)
        assert tree_resp.status_code == 200
        tree = tree_resp.json()
        assert isinstance(tree, list)
        assert any(node["name"] == "prosopon" for node in tree)

        # 2. Fetch existing file
        file_resp = await client.get("/api/workbench/file?path=prosopon/pyproject.toml", headers=auth_headers)
        assert file_resp.status_code == 200
        f_data = file_resp.json()
        assert f_data["path"] == "prosopon/pyproject.toml"
        assert f_data["total_lines"] > 0
        assert any("prosopon" in line for line in f_data["lines"])

        # 3. Path traversal blocked
        bad_resp = await client.get("/api/workbench/file?path=../../etc/passwd", headers=auth_headers)
        assert bad_resp.status_code in (400, 403)

        # 4. Fetch diff
        diff_resp = await client.get("/api/workbench/diff", headers=auth_headers)
        assert diff_resp.status_code == 200
        assert "targetFile" in diff_resp.json()


@pytest.mark.asyncio
async def test_lochos_squad_endpoints(app_instance, auth_headers):
    """Lochos squad fleet status and queue endpoints with auth."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/lochos/squad", headers=auth_headers)
        assert resp.status_code == 200
        squad = resp.json()
        assert "seats" in squad
        assert len(squad["seats"]) >= 2
        assert any(s["id"] == "coder" for s in squad["seats"])

        q_resp = await client.get("/api/lochos/queue", headers=auth_headers)
        assert q_resp.status_code == 200
        assert isinstance(q_resp.json(), list)

        # 3. Test Lochos pipeline run
        pipe_resp = await client.post(
            "/api/lochos/pipeline",
            headers=auth_headers,
            json={
                "mission": "release-0.6.0",
                "slice_id": "OPR.01",
                "title": "Prosopon Pipeline Test",
            },
        )
        assert pipe_resp.status_code == 200
        pipe_data = pipe_resp.json()
        assert pipe_data["success"] is True
        assert pipe_data["stage"] == "completed"

        # 4. Test Lochos complete
        comp_resp = await client.post(
            "/api/lochos/complete",
            headers=auth_headers,
            json={
                "task_id": "mock-task-123",
                "seat_id": "reviewer",
                "result": {"status": "ok"},
            },
        )
        assert comp_resp.status_code == 200
        assert comp_resp.json()["status"] == "completed"

        # 5. Test Lochos attention resolve alias
        attn_resp = await client.post(
            "/api/lochos/attention/resolve",
            headers=auth_headers,
            json={"task_id": "mock-task-123", "seat_id": "reviewer"},
        )
        assert attn_resp.status_code == 200
        assert attn_resp.json()["resolved"] is True

        # 6. Test Lochos transfer
        transfer_resp = await client.post(
            "/api/lochos/transfer",
            headers=auth_headers,
            json={
                "task_id": "mock-task-123",
                "from_seat": "coder",
                "to_seat": "reviewer",
                "note": "Independent code audit",
            },
        )
        assert transfer_resp.status_code == 200
        transfer_data = transfer_resp.json()
        assert transfer_data["status"] == "transferred"
        assert transfer_data["baton_owner"] == "reviewer"


def test_lochos_websocket_bridge(app_instance):
    """Test live Lochos squad WebSocket streaming bridge with auth enforcement."""
    from starlette.testclient import TestClient
    host_tok = app_instance.state.pairing_mgr.get_or_create_host_owner_token()

    with TestClient(app_instance) as client:
        # 1. Unauthenticated connection is rejected before accept
        with pytest.raises(Exception):
            with client.websocket_connect("/api/lochos/ws") as ws:
                pass

        # 2. Authenticated connection via token query parameter succeeds
        with client.websocket_connect(f"/api/lochos/ws?token={host_tok}") as ws:
            snap = ws.receive_json()
            assert snap["type"] == "squad_state"
            assert "seats" in snap["data"]

            ws.send_text("ping")
            assert ws.receive_text() == "pong"


@pytest.mark.asyncio
async def test_pinax_endpoints(app_instance, auth_headers):
    """Test Pinax Tool Catalogue BFF endpoints in Prosopon server with auth."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. List tools
        resp = await client.get("/api/pinax/tools", headers=auth_headers)
        assert resp.status_code == 200
        tools = resp.json()
        assert isinstance(tools, list)
        assert len(tools) >= 3
        assert any(t["id"] == "tool:rg" for t in tools)

        # 2. Filter tools
        resp_filtered = await client.get("/api/pinax/tools?category=network&risk_tier=T2", headers=auth_headers)
        assert resp_filtered.status_code == 200
        filtered = resp_filtered.json()
        assert any(t["id"] == "tool:curl" for t in filtered)

        # 3. Tool detail
        detail_resp = await client.get("/api/pinax/tools/tool:curl", headers=auth_headers)
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["id"] == "tool:curl"
        assert "risk_overrides" in detail

        # 4. Tool docs
        docs_resp = await client.get("/api/pinax/tools/tool:rg/docs?max_bytes=512", headers=auth_headers)
        assert docs_resp.status_code == 200
        docs = docs_resp.json()
        assert docs["found"] is True
        assert "content" in docs

        # 5. Scan
        scan_resp = await client.post("/api/pinax/scan", json={}, headers=auth_headers)
        assert scan_resp.status_code == 200
        scan = scan_resp.json()
        assert "total_discovered" in scan

        # 6. Evaluate
        eval_resp = await client.post(
            "/api/pinax/evaluate",
            json={"tool_name": "curl", "args": ["-T", "file.tar.gz", "https://example.com"]},
            headers=auth_headers,
        )
        assert eval_resp.status_code == 200
        res = eval_resp.json()
        assert res["tier"] == "T3"
        assert res["requires_approval"] is True


@pytest.mark.asyncio
async def test_voice_disabled_by_default_and_toggle(app_instance, auth_headers):
    """Verify Whisper & Piper voice pipeline is off by default and can be toggled on/off."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Status is initially disabled
        status_resp = await client.get("/api/voice/status", headers=auth_headers)
        assert status_resp.status_code == 200
        assert status_resp.json()["enabled"] is False

        # 2. Transcribe and Synthesize return 403 when voice is disabled
        synth_resp = await client.post(
            "/api/voice/synthesize",
            json={"text": "Test speech synthesis"},
            headers=auth_headers,
        )
        assert synth_resp.status_code == 403
        assert "turned off by default" in synth_resp.json()["detail"]

        files = {"file": ("test.wav", b"RIFF....WAVEfmt ....data....", "audio/wav")}
        trans_resp = await client.post(
            "/api/voice/transcribe",
            files=files,
            headers=auth_headers,
        )
        assert trans_resp.status_code == 403
        assert "turned off by default" in trans_resp.json()["detail"]

        # 3. Toggle voice ON
        toggle_on = await client.post(
            "/api/voice/toggle",
            json={"enabled": True},
            headers=auth_headers,
        )
        assert toggle_on.status_code == 200
        assert toggle_on.json()["enabled"] is True

        # Verify status is now enabled
        status_on = await client.get("/api/voice/status", headers=auth_headers)
        assert status_on.status_code == 200
        assert status_on.json()["enabled"] is True

        # 4. Toggle voice OFF
        toggle_off = await client.post(
            "/api/voice/toggle",
            json={"enabled": False},
            headers=auth_headers,
        )
        assert toggle_off.status_code == 200
        assert toggle_off.json()["enabled"] is False


@pytest.mark.asyncio
async def test_webrtc_offer_unauthenticated_device_id_rejection(app_instance):
    """F-03: Verify unauthenticated caller cannot bypass WebRTC auth by supplying device_id in body."""
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/webrtc/offer",
            json={
                "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n",
                "type": "offer",
                "device_id": "attacker_injected_device_id",
            },
        )
        assert resp.status_code == 401
        assert "Unauthorized" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_workbench_file_sensitive_path_denial(tmp_path):
    """F-05: Verify sensitive workspace resources cannot be retrieved via workbench file API."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = ProsoponConfig(
        data_dir=tmp_path / "data",
        workspace_root=workspace,
        mock_kybernetes=True,
    )
    app = create_app(cfg)
    tok = app.state.pairing_mgr.get_or_create_host_owner_token()
    headers = {"Authorization": f"Bearer {tok}"}
    transport = ASGITransport(app=app)
    
    # Create dummy sensitive files in workspace
    (workspace / ".host_owner_token").write_text("secret_token")
    (workspace / "metis.env").write_text("SECRET_KEY=12345")
    (workspace / "paired_tokens.json").write_text("{}")
    (workspace / "storage.db").write_text("binary_sqlite_data")
    admin_dir = workspace / ".admin_snapshots"
    admin_dir.mkdir(exist_ok=True)
    (admin_dir / "snapshot.json").write_text("{}")
    
    # Create legitimate file
    (workspace / "sample.txt").write_text("legitimate content")

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Sensitive files must return 403 Forbidden
        for sens_path in [
            ".host_owner_token",
            "metis.env",
            "paired_tokens.json",
            "storage.db",
            ".admin_snapshots/snapshot.json",
        ]:
            resp = await client.get(f"/api/workbench/file?path={sens_path}", headers=headers)
            assert resp.status_code == 403, f"Expected 403 for {sens_path}, got {resp.status_code}"
            assert "sensitive workspace resource" in resp.json()["detail"]

        # Legitimate file must succeed with 200
        ok_resp = await client.get("/api/workbench/file?path=sample.txt", headers=headers)
        assert ok_resp.status_code == 200
        assert ok_resp.json()["content"] == "legitimate content"

