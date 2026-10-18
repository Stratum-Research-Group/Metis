"""Tests for Pyle Phase 1: SQLite AccessStore, WebAuthn RP, Device REST API, and CLI."""

import hashlib
import json
import tempfile
from pathlib import Path
import pytest
from click.testing import CliRunner
from starlette.testclient import TestClient

from prosopon.auth.scopes import VALID_SCOPES, PRESET_SCOPES, get_scopes_for_preset, has_scope
from prosopon.auth.webauthn import WebAuthnManager
from prosopon.cli import cli
from prosopon.config import ProsoponConfig
from prosopon.contracts.access import (
    DeviceRecord,
    PairDeviceRequest,
    PairDeviceResponse,
    RevokeDeviceRequest,
    LockdownResponse,
)
from prosopon.server import create_app
from prosopon.storage.access_store import AccessStore, GENESIS_HASH


@pytest.fixture
def temp_access_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def access_store(temp_access_dir):
    return AccessStore(temp_access_dir / "access.db")


# -----------------------------------------------------------------------------
# 1. AccessStore Unit Tests
# -----------------------------------------------------------------------------

def test_access_store_initialization(access_store):
    """Test default database initialization, host_owner creation, and genesis audit record."""
    host_dev = access_store.get_device("host_owner")
    assert host_dev is not None
    assert host_dev.device_id == "host_owner"
    assert host_dev.label == "This computer (host)"
    assert host_dev.preset == "owner"
    assert host_dev.is_active is True
    assert set(host_dev.scopes) == set(PRESET_SCOPES["owner"])

    # Audit chain should be verified
    is_valid, count = access_store.verify_audit_chain()
    assert is_valid is True
    assert count == 1


def test_device_lifecycle(access_store):
    """Test creating, listing, getting, revoking, and expiring devices."""
    dev = access_store.create_device(
        device_id="phone_001",
        label="Test Phone",
        paired_via="qr_passkey",
        preset="companion",
    )
    assert dev.device_id == "phone_001"
    assert dev.label == "Test Phone"
    assert dev.preset == "companion"
    assert dev.is_active is True

    # List active
    active_devs = access_store.list_devices(include_revoked=False)
    assert any(d.device_id == "phone_001" for d in active_devs)

    # Revoke
    revoked = access_store.revoke_device("phone_001", reason="user_revocation")
    assert revoked is True

    # Cannot revoke host_owner
    assert access_store.revoke_device("host_owner") is False

    # Check after revocation
    fetched = access_store.get_device("phone_001")
    assert fetched.is_active is False
    assert fetched.revoked_reason == "user_revocation"

    active_devs = access_store.list_devices(include_revoked=False)
    assert not any(d.device_id == "phone_001" for d in active_devs)

    all_devs = access_store.list_devices(include_revoked=True)
    assert any(d.device_id == "phone_001" for d in all_devs)


def test_panic_lockdown(access_store):
    """Test emergency lockdown revoking all remote devices and active sessions."""
    # Create two remote devices with active sessions
    d1 = access_store.create_device("dev_a", "Phone A", "link", preset="companion")
    t1, s1 = access_store.create_session("dev_a")

    d2 = access_store.create_device("dev_b", "Tablet B", "link", preset="operator")
    t2, s2 = access_store.create_session("dev_b")

    # Host session
    th, sh = access_store.create_session("host_owner")

    # Verify sessions before lockdown
    assert access_store.validate_session_token(t1) is not None
    assert access_store.validate_session_token(t2) is not None
    assert access_store.validate_session_token(th) is not None

    # Lockdown
    rev_devs, rev_sess = access_store.lockdown_all(reason="panic_test")
    assert rev_devs == 2
    assert rev_sess == 2

    # Remote sessions invalidated
    assert access_store.validate_session_token(t1) is None
    assert access_store.validate_session_token(t2) is None

    # Host owner remains valid
    host_val = access_store.validate_session_token(th)
    assert host_val is not None
    assert host_val[0].device_id == "host_owner"


def test_pairing_link_atomic_redemption(access_store):
    """Test pairing link generation, validity checks, and atomic single-use redemption."""
    pid, secret, rec = access_store.create_pairing_link(
        label="Pixel 9",
        preset="companion",
        validity_seconds=300,
    )
    assert pid.startswith("pair_")
    assert len(secret) > 20

    # Redeem link
    redeemed = access_store.redeem_pairing_link(pid, secret, device_id="pixel_device_1")
    assert redeemed is not None
    assert redeemed.consumed_by_device_id == "pixel_device_1"

    # Second redemption attempt fails (already consumed)
    second = access_store.redeem_pairing_link(pid, secret, device_id="pixel_device_2")
    assert second is None

    # Wrong secret fails
    pid2, sec2, _ = access_store.create_pairing_link()
    assert access_store.redeem_pairing_link(pid2, "wrong_secret", device_id="dev_x") is None


def test_passkey_management(access_store):
    """Test registering passkeys and monotonic sign counter updates."""
    access_store.create_device("passkey_phone", "Phone", "qr_passkey")
    pk = access_store.register_passkey(
        credential_id="cred_12345",
        device_id="passkey_phone",
        public_key_cbor="cbor_public_key_bytes_base64",
        sign_count=0,
        aaguid="aaguid_123",
        transports=["internal"],
    )
    assert pk.credential_id == "cred_12345"
    assert pk.sign_count == 0

    fetched = access_store.get_passkey("cred_12345")
    assert fetched is not None
    assert fetched.device_id == "passkey_phone"

    # Update sign count
    ok = access_store.update_passkey_sign_count("cred_12345", 5)
    assert ok is True
    updated = access_store.get_passkey("cred_12345")
    assert updated.sign_count == 5


def test_audit_chain_tamper_detection(access_store):
    """Test cryptographic SHA-256 forward-linked audit chain and tamper detection."""
    access_store.log_audit(action="test_action_1", detail={"k": "v1"})
    access_store.log_audit(action="test_action_2", detail={"k": "v2"})
    access_store.log_audit(action="test_action_3", detail={"k": "v3"})

    is_valid, count = access_store.verify_audit_chain()
    assert is_valid is True
    assert count >= 4  # including genesis

    # Tamper with an audit entry in SQLite
    conn = access_store._get_connection()
    with conn:
        conn.execute("UPDATE access_audit SET action = 'tampered_action' WHERE id = 2")

    is_valid, tampered_id = access_store.verify_audit_chain()
    assert is_valid is False
    assert tampered_id == 2


# -----------------------------------------------------------------------------
# 2. WebAuthnManager Unit Tests
# -----------------------------------------------------------------------------

def test_webauthn_manager_options():
    wm = WebAuthnManager(rp_id="localhost")
    opts, chal_b64 = wm.create_registration_options(
        user_id=b"test_user_id",
        user_name="tester",
        user_display_name="Tester Device",
    )
    assert "challenge" in opts
    assert opts["rp"]["id"] == "localhost"

    # Challenge cache
    wm.record_challenge("chal_key_1", b"raw_chal_bytes", metadata={"user": "tester"})
    popped = wm.pop_challenge("chal_key_1")
    assert popped is not None
    assert popped[0] == b"raw_chal_bytes"
    assert popped[1]["user"] == "tester"

    # Popped challenge cannot be reused
    assert wm.pop_challenge("chal_key_1") is None


# -----------------------------------------------------------------------------
# 3. Server REST API Integration Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def test_app(temp_access_dir):
    cfg = ProsoponConfig(
        data_dir=temp_access_dir,
        mock_kybernetes=True,
    )
    return create_app(cfg)


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


def test_device_api_endpoints(test_app, client, temp_access_dir):
    # Retrieve host owner token
    host_token = (temp_access_dir / ".host_owner_token").read_text(encoding="utf-8").strip()
    headers = {"X-Host-Owner-Token": host_token}

    # 1. GET /api/devices (initially contains host_owner)
    res = client.get("/api/devices", headers=headers)
    assert res.status_code == 200
    devs = res.json()
    assert len(devs) >= 1
    assert devs[0]["device_id"] == "host_owner"

    # 2. POST /api/devices/pair (create pairing invitation)
    pair_res = client.post(
        "/api/devices/pair",
        headers=headers,
        json={"label": "Pixel Phone", "preset": "companion", "validity_seconds": 300},
    )
    assert pair_res.status_code == 200
    pair_data = pair_res.json()
    assert "pairing_id" in pair_data
    assert "pairing_secret" in pair_data
    assert "pairing_url" in pair_data

    # 3. Create a device directly in access_store to test revocation
    access_store = test_app.state.access_store
    access_store.create_device("remote_device_99", "Remote Tab", "cli", preset="companion")

    # 4. GET /api/devices/me using new device session
    tok, sess = access_store.create_session("remote_device_99")
    me_res = client.get("/api/devices/me", headers={"Authorization": f"Bearer {tok}"})
    assert me_res.status_code == 200
    assert me_res.json()["device_id"] == "remote_device_99"

    # 5. Access protected resource (/api/sessions) using new device session token
    sess_res = client.get("/api/sessions", headers={"Authorization": f"Bearer {tok}"})
    assert sess_res.status_code == 200

    # 6. POST /api/devices/{id}/revoke
    rev_res = client.post(
        "/api/devices/remote_device_99/revoke",
        headers=headers,
        json={"reason": "admin_audit"},
    )
    assert rev_res.status_code == 200
    assert rev_res.json()["revoked"] is True

    # Subsequent access with revoked session token fails with 401
    unauth_res = client.get("/api/sessions", headers={"Authorization": f"Bearer {tok}"})
    assert unauth_res.status_code == 401

    # 7. GET /api/devices/audit
    audit_res = client.get("/api/devices/audit", headers=headers)
    assert audit_res.status_code == 200
    audit_data = audit_res.json()
    assert audit_data["chain_valid"] is True
    assert audit_data["total_records"] > 0

    # 8. POST /api/devices/lockdown
    lock_res = client.post("/api/devices/lockdown", headers=headers)
    assert lock_res.status_code == 200
    assert lock_res.json()["message"].startswith("Panic lockdown executed")


# -----------------------------------------------------------------------------
# 4. CLI Command Tests
# -----------------------------------------------------------------------------

def test_cli_device_commands(temp_access_dir):
    runner = CliRunner()

    # 1. prosopon device list
    res = runner.invoke(cli, ["device", "list", "--data-dir", str(temp_access_dir)])
    assert res.exit_code == 0
    assert "host_owner" in res.output

    # 2. prosopon device pair
    pair_res = runner.invoke(
        cli,
        [
            "device",
            "pair",
            "--data-dir",
            str(temp_access_dir),
            "--label",
            "Companion Tablet",
            "--preset",
            "operator",
        ],
    )
    assert pair_res.exit_code == 0
    assert "Pyle Device Pairing Invitation" in pair_res.output
    assert "Companion Tablet" in pair_res.output
    assert "operator" in pair_res.output

    # 3. Create device and test prosopon device revoke
    store = AccessStore(temp_access_dir / "access.db")
    store.create_device("cli_test_device", "CLI Phone", "cli")

    rev_res = runner.invoke(
        cli,
        ["device", "revoke", "cli_test_device", "--data-dir", str(temp_access_dir)],
    )
    assert rev_res.exit_code == 0
    assert "successfully revoked" in rev_res.output

    # 4. prosopon device lockdown
    lock_res = runner.invoke(
        cli,
        ["device", "lockdown", "--data-dir", str(temp_access_dir), "--yes"],
    )
    assert lock_res.exit_code == 0
    assert "Panic Lockdown Executed" in lock_res.output
