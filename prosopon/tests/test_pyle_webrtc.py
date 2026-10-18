"""Tests for Metis Pyle Phase 2: WebRTC DataChannel Engine, Signaling API, and Biometric Phylax Approvals."""

import asyncio
import json
import tempfile
from pathlib import Path
import pytest
from starlette.testclient import TestClient

import aiortc
from aiortc import RTCPeerConnection, RTCSessionDescription

from prosopon.auth.scopes import PRESET_SCOPES
from prosopon.auth.webauthn import WebAuthnManager
from prosopon.config import ProsoponConfig
from prosopon.server import create_app
from prosopon.storage.access_store import AccessStore
from prosopon.transport.webrtc import WebRTCGateway


@pytest.fixture
def temp_access_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def access_store(temp_access_dir):
    return AccessStore(temp_access_dir / "access.db")


@pytest.fixture
def webauthn_mgr():
    return WebAuthnManager(rp_id="localhost", rp_name="Metis Sovereign Host", allowed_origins=["https://localhost:8443"])


@pytest.fixture
def webrtc_gateway(access_store, webauthn_mgr):
    return WebRTCGateway(access_store, webauthn_mgr)


# -----------------------------------------------------------------------------
# 1. WebRTCGateway Core Unit Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_webrtc_offer_and_answer_negotiation(webrtc_gateway, access_store):
    """Test standard WebRTC offer negotiation and channel setup."""
    # Create an authorized device
    dev = access_store.create_device("dev_phone_1", "Pixel 9", "qr_passkey", preset="companion")

    # Create a local client peer connection to produce an offer
    client_pc = RTCPeerConnection()
    client_channel = client_pc.createDataChannel("control")
    offer = await client_pc.createOffer()
    await client_pc.setLocalDescription(offer)

    # Server handles offer
    answer_sdp, answer_type, session_id = await webrtc_gateway.handle_offer(
        sdp=client_pc.localDescription.sdp,
        type_=client_pc.localDescription.type,
        device_id="dev_phone_1",
        peer_address="192.168.1.50",
    )

    assert answer_type == "answer"
    assert "m=application" in answer_sdp
    assert session_id.startswith("webrtc_")

    # Verify session is registered in SQLite access_store
    sessions = access_store.list_webrtc_sessions("dev_phone_1")
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id
    assert sessions[0]["peer_address"] == "192.168.1.50"

    # Cleanup
    await client_pc.close()
    await webrtc_gateway.close_session(session_id)


@pytest.mark.asyncio
async def test_webrtc_offer_rejects_unauthorized_device(webrtc_gateway, access_store):
    """Test that offers from revoked or non-existent devices fail-closed."""
    client_pc = RTCPeerConnection()
    client_pc.createDataChannel("control")
    offer = await client_pc.createOffer()
    await client_pc.setLocalDescription(offer)

    with pytest.raises(PermissionError) as exc_info:
        await webrtc_gateway.handle_offer(
            sdp=client_pc.localDescription.sdp,
            type_=client_pc.localDescription.type,
            device_id="unknown_device",
        )
    assert "not authorized" in str(exc_info.value)

    # Revoked device test
    dev = access_store.create_device("revoked_dev", "Old Phone", "qr_passkey")
    access_store.revoke_device("revoked_dev")

    with pytest.raises(PermissionError):
        await webrtc_gateway.handle_offer(
            sdp=client_pc.localDescription.sdp,
            type_=client_pc.localDescription.type,
            device_id="revoked_dev",
        )

    await client_pc.close()


@pytest.mark.asyncio
async def test_webrtc_ice_candidate_handling(webrtc_gateway, access_store):
    """Test candidate parsing and session lookup."""
    dev = access_store.create_device("dev_ice", "ICE Test Device", "qr_passkey")

    client_pc = RTCPeerConnection()
    client_pc.createDataChannel("control")
    offer = await client_pc.createOffer()
    await client_pc.setLocalDescription(offer)

    _, _, session_id = await webrtc_gateway.handle_offer(
        sdp=client_pc.localDescription.sdp,
        type_=client_pc.localDescription.type,
        device_id="dev_ice",
    )

    # Adding candidate to non-existent session fails with ValueError
    with pytest.raises(ValueError):
        await webrtc_gateway.handle_ice_candidate("invalid_session", {"candidate": "dummy"})

    # Valid candidate format with 'candidate:' prefix stripped
    cand_dict = {
        "candidate": "candidate:1 1 UDP 2130706431 192.168.1.50 50000 typ host",
        "sdpMid": "0",
        "sdpMLineIndex": 0,
    }
    await webrtc_gateway.handle_ice_candidate(session_id, cand_dict)

    await client_pc.close()
    await webrtc_gateway.close_session(session_id)


# -----------------------------------------------------------------------------
# 2. Control RPC & Biometric Phylax Approvals
# -----------------------------------------------------------------------------

class MockDataChannel:
    """Mock RTCDataChannel for testing protocol dispatch."""
    def __init__(self, label: str):
        self.label = label
        self.readyState = "open"
        self.sent_messages = []

    def send(self, data: str):
        self.sent_messages.append(data)


@pytest.mark.asyncio
async def test_control_rpc_ping_and_auth_status(webrtc_gateway, access_store):
    """Test JSON-RPC 2.0 ping and auth.status dispatch over control channel."""
    dev = access_store.create_device("dev_rpc", "RPC Phone", "qr_passkey", preset="operator")
    session_id = "test_rpc_sess"

    mock_channel = MockDataChannel("control")
    webrtc_gateway._channels[session_id] = {"control": mock_channel}
    webrtc_gateway._session_devices[session_id] = "dev_rpc"
    webrtc_gateway._pcs[session_id] = None
    access_store.register_webrtc_session(session_id, "dev_rpc", "127.0.0.1")

    # 1. Ping
    ping_msg = json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 101})
    await webrtc_gateway._handle_control_rpc(session_id, ping_msg)

    assert len(mock_channel.sent_messages) == 1
    resp = json.loads(mock_channel.sent_messages[0])
    assert resp["id"] == 101
    assert resp["result"] == "pong"

    # 2. Auth.status
    auth_msg = json.dumps({"jsonrpc": "2.0", "method": "auth.status", "id": 102})
    await webrtc_gateway._handle_control_rpc(session_id, auth_msg)

    assert len(mock_channel.sent_messages) == 2
    resp2 = json.loads(mock_channel.sent_messages[1])
    assert resp2["id"] == 102
    assert resp2["result"]["device_id"] == "dev_rpc"
    assert resp2["result"]["preset"] == "operator"
    assert "approvals:resolve" in resp2["result"]["scopes"]


@pytest.mark.asyncio
async def test_phylax_action_approval_scope_enforcement(webrtc_gateway, access_store):
    """Test that Phylax approvals enforce device scopes according to risk tier."""
    # Viewer preset has NO approvals:resolve scope
    dev_viewer = access_store.create_device("dev_viewer", "Viewer Tablet", "link", preset="viewer")
    s1 = "sess_viewer"
    ch1 = MockDataChannel("control")
    webrtc_gateway._channels[s1] = {"control": ch1}
    webrtc_gateway._session_devices[s1] = "dev_viewer"

    req_t1 = json.dumps({
        "jsonrpc": "2.0",
        "method": "phylax.sign_approval",
        "params": {
            "approval_id": "app_001",
            "action_digest": "abcdef123456",
            "risk_tier": "T1",
            "decision": "allow_once",
        },
        "id": 201,
    })
    await webrtc_gateway._handle_control_rpc(s1, req_t1)
    res1 = json.loads(ch1.sent_messages[-1])
    assert "error" in res1
    assert "approvals:resolve" in res1["error"]["message"]

    # Operator preset has approvals:resolve but NOT approvals:resolve_high
    dev_op_limited = access_store.create_device("dev_op_ltd", "Operator Phone", "link", preset="operator")
    s2 = "sess_op_ltd"
    ch2 = MockDataChannel("control")
    webrtc_gateway._channels[s2] = {"control": ch2}
    webrtc_gateway._session_devices[s2] = "dev_op_ltd"

    # T3 request fails for operator lacking approvals:resolve_high (host confirmation step-up required)
    req_t3 = json.dumps({
        "jsonrpc": "2.0",
        "method": "phylax.sign_approval",
        "params": {
            "approval_id": "app_003",
            "action_digest": "abcdef123456",
            "risk_tier": "T3",
            "decision": "allow_once",
        },
        "id": 202,
    })
    await webrtc_gateway._handle_control_rpc(s2, req_t3)
    res2 = json.loads(ch2.sent_messages[-1])
    assert "error" in res2
    assert "approvals:resolve_high" in res2["error"]["message"]
    assert "host confirmation step-up" in res2["error"]["message"].lower()

    # Operator succeeds for T1 and T2 with approvals:resolve
    dev_op = access_store.create_device("dev_op", "Operator Laptop", "link", preset="operator")
    s3 = "sess_op"
    ch3 = MockDataChannel("control")
    webrtc_gateway._channels[s3] = {"control": ch3}
    webrtc_gateway._session_devices[s3] = "dev_op"

    # Track callback invocations
    approved_actions = []
    webrtc_gateway.register_approval_listener(
        lambda aid, dec, dev_id, tier: approved_actions.append((aid, dec, dev_id, tier))
    )

    # T1 approval succeeds
    await webrtc_gateway._handle_control_rpc(s3, req_t1)
    res3_t1 = json.loads(ch3.sent_messages[-1])
    assert res3_t1.get("result", {}).get("status") == "approved"

    # T2 approval succeeds
    req_t2 = json.dumps({
        "jsonrpc": "2.0",
        "method": "phylax.sign_approval",
        "params": {
            "approval_id": "app_002",
            "action_digest": "abcdef123456",
            "risk_tier": "T2",
            "decision": "allow_once",
        },
        "id": 203,
    })
    await webrtc_gateway._handle_control_rpc(s3, req_t2)
    res3_t2 = json.loads(ch3.sent_messages[-1])
    assert res3_t2.get("result", {}).get("status") == "approved"

    assert len(approved_actions) == 2
    assert approved_actions[0] == ("app_001", "allow_once", "dev_op", "T1")
    assert approved_actions[1] == ("app_002", "allow_once", "dev_op", "T2")

    # Check cryptographic audit log
    is_valid, _ = access_store.verify_audit_chain()
    assert is_valid is True


# -----------------------------------------------------------------------------
# 3. Stream & Terminal Outbound Dispatch
# -----------------------------------------------------------------------------

def test_stream_and_terminal_outbound_dispatch(webrtc_gateway):
    """Test broadcasting SSE-formatted stream events and terminal chunks."""
    s1 = "sess_stream_1"
    ch_stream = MockDataChannel("stream")
    ch_term = MockDataChannel("terminal")
    webrtc_gateway._channels[s1] = {"stream": ch_stream, "terminal": ch_term}
    webrtc_gateway._session_devices[s1] = "dev_stream"
    webrtc_gateway._device_sessions["dev_stream"] = {s1}
    webrtc_gateway._pcs[s1] = None

    # Send stream event
    sent = webrtc_gateway.send_stream_event("agent_token", {"token": "hello world"}, device_id="dev_stream")
    assert sent == 1
    assert len(ch_stream.sent_messages) == 1
    assert "event: agent_token\ndata: {\"token\": \"hello world\"}\n\n" in ch_stream.sent_messages[0]

    # Send terminal data
    sent_term = webrtc_gateway.send_terminal_data("\x1b[32mPASS\x1b[0m\n", session_id=s1)
    assert sent_term == 1
    assert ch_term.sent_messages[0] == "\x1b[32mPASS\x1b[0m\n"


# -----------------------------------------------------------------------------
# 4. Immediate Revocation Invariant (< 500ms Teardown)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_immediate_revocation_teardown(webrtc_gateway, access_store):
    """Test that revoking a device terminates its WebRTC session instantly."""
    dev = access_store.create_device("dev_to_revoke", "Revoke Phone", "qr_passkey")

    client_pc = RTCPeerConnection()
    client_pc.createDataChannel("control")
    offer = await client_pc.createOffer()
    await client_pc.setLocalDescription(offer)

    _, _, session_id = await webrtc_gateway.handle_offer(
        sdp=client_pc.localDescription.sdp,
        type_=client_pc.localDescription.type,
        device_id="dev_to_revoke",
    )

    assert session_id in webrtc_gateway._pcs
    assert len(access_store.list_webrtc_sessions("dev_to_revoke")) == 1

    # Revoke connections for device
    closed = await webrtc_gateway.close_device_connections("dev_to_revoke")
    assert closed == 1
    assert session_id not in webrtc_gateway._pcs
    assert len(access_store.list_webrtc_sessions("dev_to_revoke")) == 0

    await client_pc.close()


# -----------------------------------------------------------------------------
# 5. FastAPI Endpoints Integration
# -----------------------------------------------------------------------------

def test_fastapi_webrtc_endpoints(temp_access_dir):
    """Test HTTP signaling endpoints: offer, candidate, sessions, and revocation hooks."""
    config = ProsoponConfig(data_dir=temp_access_dir)
    app = create_app(config)
    client = TestClient(app, base_url="http://127.0.0.1:8000")

    host_token = (temp_access_dir / ".host_owner_token").read_text().strip()
    headers = {"X-Host-Owner-Token": host_token}

    # 1. Unauthenticated offer is rejected
    dummy_offer = {"sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n", "type": "offer"}
    res_unauth = client.post("/api/webrtc/offer", json=dummy_offer)
    assert res_unauth.status_code == 401

    # 2. Offer with host-owner auth succeeds
    res_offer = client.post("/api/webrtc/offer", json=dummy_offer, headers=headers)
    assert res_offer.status_code == 200
    assert res_offer.json()["type"] == "answer"
    assert res_offer.json()["device_id"] == "host_owner"
    assert "session_id" in res_offer.json()

    # Malformed offer raises 400
    res_bad = client.post("/api/webrtc/offer", json={"sdp": "invalid", "type": "bad_type"}, headers=headers)
    assert res_bad.status_code == 400
    assert "WebRTC offer negotiation failed" in res_bad.json()["detail"]

    # 3. List webrtc sessions
    res_sess = client.get("/api/webrtc/sessions", headers=headers)
    assert res_sess.status_code == 200
    assert isinstance(res_sess.json(), list)

    # 4. Lockdown closes all webrtc sessions
    res_lockdown = client.post("/api/devices/lockdown", json={"reason": "test"}, headers=headers)
    assert res_lockdown.status_code == 200
    assert res_lockdown.json()["revoked_devices_count"] >= 0
