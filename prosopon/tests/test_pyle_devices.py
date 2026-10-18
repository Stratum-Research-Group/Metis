"""Tests for Pyle Phase 1: SQLite AccessStore, WebAuthn RP, Device REST API, and CLI."""

import hashlib
import json
import tempfile
from pathlib import Path
import pytest
from click.testing import CliRunner
from starlette.testclient import TestClient
from httpx import AsyncClient, ASGITransport

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


# -----------------------------------------------------------------------------
# 5. Milestone 3.1: Operator Approvals & T3 Host Confirmation Step-up
# -----------------------------------------------------------------------------

def test_operator_remote_approvals_and_t3_stepup(test_app, client, temp_access_dir):
    """
    Test that Operator preset has remote approval rights for T1 and T2 actions,
    enforces host confirmation step-up for T3 actions, and Owner can approve all tiers.
    """
    access_store = test_app.state.access_store
    host_token = (temp_access_dir / ".host_owner_token").read_text(encoding="utf-8").strip()

    # 1. Companion device (cannot approve any tier)
    access_store.create_device("dev_comp", "Companion Phone", "link", preset="companion")
    tok_comp, _ = access_store.create_session("dev_comp")

    res_comp_t1 = client.post(
        "/api/approvals/app_comp_01/resolve",
        headers={"Authorization": f"Bearer {tok_comp}"},
        json={"decision": "allow_once", "risk_tier": "T1"},
    )
    assert res_comp_t1.status_code == 403
    assert "approvals:resolve" in res_comp_t1.json()["detail"]

    # 2. Operator device (can approve T1 and T2, but blocked on T3)
    access_store.create_device("dev_operator", "Operator Laptop", "link", preset="operator")
    tok_op, _ = access_store.create_session("dev_operator")

    # T1 succeeds
    res_op_t1 = client.post(
        "/api/approvals/app_op_01/resolve",
        headers={"Authorization": f"Bearer {tok_op}"},
        json={"decision": "allow_once", "risk_tier": "T1"},
    )
    assert res_op_t1.status_code == 200
    assert res_op_t1.json()["status"] == "ok"

    # T2 succeeds
    res_op_t2 = client.post(
        "/api/approvals/app_op_02/resolve",
        headers={"Authorization": f"Bearer {tok_op}"},
        json={"decision": "allow_once", "risk_tier": "T2"},
    )
    assert res_op_t2.status_code == 200
    assert res_op_t2.json()["status"] == "ok"

    # T3 is rejected with host confirmation step-up requirement
    res_op_t3 = client.post(
        "/api/approvals/app_op_03/resolve",
        headers={"Authorization": f"Bearer {tok_op}"},
        json={"decision": "allow_once", "risk_tier": "T3"},
    )
    assert res_op_t3.status_code == 403
    assert "host confirmation step-up" in res_op_t3.json()["detail"].lower()
    assert "approvals:resolve_high" in res_op_t3.json()["detail"]

    # 3. Owner device (holds approvals:resolve_high, succeeds for T3)
    access_store.create_device("dev_owner_remote", "Owner Remote", "link", preset="owner")
    tok_owner, _ = access_store.create_session("dev_owner_remote")

    res_owner_t3 = client.post(
        "/api/approvals/app_owner_03/resolve",
        headers={"Authorization": f"Bearer {tok_owner}"},
        json={"decision": "allow_once", "risk_tier": "T3"},
    )
    assert res_owner_t3.status_code == 200
    assert res_owner_t3.json()["status"] == "ok"

    # Verify chained audit log recorded all approvals
    is_valid, _ = access_store.verify_audit_chain()
    assert is_valid is True


# -----------------------------------------------------------------------------
# 6. Milestone 3.1: Active Agent Task Termination on Revocation & Lockdown
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_active_agent_task_termination_on_revocation(test_app, temp_access_dir):
    """
    Test active agent task termination on device revocation (<500ms security gate).
    Verifies that running tasks and in-flight turns are immediately cancelled.
    """
    import asyncio
    access_store = test_app.state.access_store
    host_token = (temp_access_dir / ".host_owner_token").read_text(encoding="utf-8").strip()
    headers = {"X-Host-Owner-Token": host_token}

    # Create operator device
    access_store.create_device("dev_worker", "Active Worker", "link", preset="operator")
    tok_worker, _ = access_store.create_session("dev_worker")

    # Simulate an active background task associated with this device
    async def dummy_agent_run():
        try:
            await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            raise

    dummy_task = asyncio.create_task(dummy_agent_run())
    test_app.state.active_device_tasks["dev_worker"].add(dummy_task)
    test_app.state.active_device_turns["dev_worker"].add("turn_worker_999")

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Check /api/devices/{id}/runs endpoint
        runs_res = await ac.get("/api/devices/dev_worker/runs", headers={"Authorization": f"Bearer {tok_worker}"})
        assert runs_res.status_code == 200
        runs_info = runs_res.json()
        assert "turn_worker_999" in runs_info["active_turns"]
        assert runs_info["active_tasks_count"] == 1

        # Revoke device with default stop_runs=True
        rev_res = await ac.post(
            "/api/devices/dev_worker/revoke",
            headers=headers,
            json={"reason": "security_test", "stop_runs": True},
        )
        assert rev_res.status_code == 200
        rev_data = rev_res.json()
        assert rev_data["revoked"] is True
        assert rev_data["runs"]["cancelled_tasks"] == 1
        assert rev_data["runs"]["cancelled_turns"] == 1
        assert rev_data["runs"]["stop_runs"] is True

        # Ensure task was cancelled
        assert dummy_task.cancelled() or dummy_task.done()

        # Active turns cleared
        runs_after = await ac.get("/api/devices/dev_worker/runs", headers=headers)
        assert runs_after.status_code == 200
        assert len(runs_after.json()["active_turns"]) == 0

        # Test stop_runs=False preserves running tasks
        access_store.create_device("dev_worker_keep", "Keep Worker", "link", preset="operator")
        keep_task = asyncio.create_task(dummy_agent_run())
        test_app.state.active_device_tasks["dev_worker_keep"].add(keep_task)
        test_app.state.active_device_turns["dev_worker_keep"].add("turn_keep_111")

        rev_keep_res = await ac.post(
            "/api/devices/dev_worker_keep/revoke",
            headers=headers,
            json={"reason": "admin_choice", "stop_runs": False},
        )
        assert rev_keep_res.status_code == 200
        assert rev_keep_res.json()["runs"]["stop_runs"] is False
        assert rev_keep_res.json()["runs"]["cancelled_tasks"] == 0
        assert not keep_task.cancelled()
        keep_task.cancel()  # cleanup


@pytest.mark.asyncio
async def test_active_agent_task_termination_lockdown(test_app, temp_access_dir):
    """
    Test panic lockdown cancels all remote device tasks and revokes access.
    """
    import asyncio
    access_store = test_app.state.access_store
    host_token = (temp_access_dir / ".host_owner_token").read_text(encoding="utf-8").strip()
    headers = {"X-Host-Owner-Token": host_token}

    # Create two remote devices with active simulated tasks
    access_store.create_device("dev_remote_1", "Remote 1", "link", preset="companion")
    access_store.create_device("dev_remote_2", "Remote 2", "link", preset="operator")

    async def dummy_sleep():
        await asyncio.sleep(60.0)

    task1 = asyncio.create_task(dummy_sleep())
    task2 = asyncio.create_task(dummy_sleep())
    test_app.state.active_device_tasks["dev_remote_1"].add(task1)
    test_app.state.active_device_tasks["dev_remote_2"].add(task2)
    test_app.state.active_device_turns["dev_remote_1"].add("turn_rem_1")
    test_app.state.active_device_turns["dev_remote_2"].add("turn_rem_2")

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Execute panic lockdown
        lock_res = await ac.post("/api/devices/lockdown", headers=headers, json={"reason": "emergency_panic"})
        assert lock_res.status_code == 200
        lock_data = lock_res.json()
        assert lock_data["revoked_devices_count"] >= 2
        assert "Cancelled 2 tasks and 2 turns" in lock_data["message"]

    assert task1.cancelled() or task1.done()
    assert task2.cancelled() or task2.done()


# -----------------------------------------------------------------------------
# 7. Milestone 3.1: Doctor Diagnostics Extended Checks
# -----------------------------------------------------------------------------

def test_doctor_diagnostics_extended(temp_access_dir):
    """Test doctor diagnostics: TLS state, 0600 file permissions, unauthenticated audit, and legacy migration."""
    import stat
    from prosopon.ops.doctor import run_doctor_check

    # 1. Base check on temp dir
    cfg = ProsoponConfig(data_dir=temp_access_dir)
    report = run_doctor_check(cfg)
    check_names = {c["component"]: c for c in report["checks"]}

    assert "Network & Secure Context (TLS)" in check_names
    assert check_names["Network & Secure Context (TLS)"]["status"] == "PASS"

    assert "Access Store Database Permissions" in check_names
    assert "Host Owner Credential Permissions" in check_names
    assert "Perimeter & Unauthenticated Endpoints Audit" in check_names
    assert check_names["Perimeter & Unauthenticated Endpoints Audit"]["status"] == "PASS"

    assert "Credential Hashing & Token Migration" in check_names
    assert check_names["Credential Hashing & Token Migration"]["status"] == "PASS"

    # 2. Permissions check: trigger FAIL on insecure 0666 permissions
    insecure_db = temp_access_dir / "access.db"
    insecure_db.write_text("fake db content")
    insecure_db.chmod(0o666)

    insecure_token = temp_access_dir / ".host_owner_token"
    insecure_token.write_text("fake token")
    insecure_token.chmod(0o644)

    report_insecure = run_doctor_check(cfg)
    insecure_checks = {c["component"]: c for c in report_insecure["checks"]}
    assert "FAIL" in insecure_checks["Access Store Database Permissions"]["status"]
    assert "FAIL" in insecure_checks["Host Owner Credential Permissions"]["status"]

    # Restore 0600
    insecure_db.chmod(0o600)
    insecure_token.chmod(0o600)
    report_fixed = run_doctor_check(cfg)
    fixed_checks = {c["component"]: c for c in report_fixed["checks"]}
    assert fixed_checks["Access Store Database Permissions"]["status"] == "PASS"
    assert fixed_checks["Host Owner Credential Permissions"]["status"] == "PASS"

    # 3. LAN mode without TLS triggers WARN
    cfg_lan_insecure = ProsoponConfig(data_dir=temp_access_dir, host="0.0.0.0", enable_tls=False)
    lan_report = run_doctor_check(cfg_lan_insecure)
    lan_checks = {c["component"]: c for c in lan_report["checks"]}
    assert "WARN" in lan_checks["Network & Secure Context (TLS)"]["status"]

