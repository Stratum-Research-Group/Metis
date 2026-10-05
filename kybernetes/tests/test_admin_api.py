"""Tests for Kybernetes Admin Control Plane and Diagnostic Probes."""

import pytest
from httpx import ASGITransport, AsyncClient

from kybernetes.admin.status import ComponentState
from kybernetes.config import KybernetesConfig
from kybernetes.server.app import create_app


@pytest.mark.asyncio
async def test_health_redacts_paths_and_reports_honest_states(tmp_path):
    """Verify unauthenticated /v1/health never leaks absolute paths and is honest about simulation."""
    memory_dir = tmp_path / "secret_corpus"
    memory_dir.mkdir()
    (memory_dir / "passages.json").write_text('[{"passage_id": "p1", "text": "secret passage content"}]')
    cfg = KybernetesConfig(memory_root=str(memory_dir))
    app = create_app(config=cfg)
    transport = ASGITransport(app=app)
    
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        resp = await client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        
        # Verify no raw filesystem path leakage
        json_str = resp.text
        assert str(memory_dir) not in json_str
        
        deps = data["dependencies"]
        assert deps["lthcm"]["state"] in ("simulated", "unknown")
        assert deps["dianoia"]["state"] == "simulated"
        assert deps["phylax"]["state"] == "simulated"
        assert deps["krisis"]["state"] == "simulated"
        assert deps["hestia"]["state"] in ("live", "offline", "unknown")


@pytest.mark.asyncio
async def test_false_live_guard_unprobed_injected_adapter():
    """Verify an injected adapter is not falsely marked live without a recorded functional probe."""
    from kybernetes.adapters.dianoia import DianoiaAdapter
    
    # Even if dianoia_module is provided, without a functional probe it must not report live
    dummy_module = object()
    adapter = DianoiaAdapter(dianoia_module=dummy_module)
    app = create_app(dianoia_adapter=adapter)
    transport = ASGITransport(app=app)
    
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        resp = await client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        # Should be simulated or unknown, not live
        assert data["dependencies"]["dianoia"]["state"] != "live"


@pytest.mark.asyncio
async def test_admin_auth_matrix():
    """Verify admin endpoints enforce strict token authentication even from loopback."""
    # 1. Admin token unconfigured -> 403 Forbidden
    cfg_no_token = KybernetesConfig(admin_token=None)
    app_no_token = create_app(config=cfg_no_token)
    transport_no_token = ASGITransport(app=app_no_token)
    
    async with AsyncClient(transport=transport_no_token, base_url="http://127.0.0.1:8081") as client:
        resp = await client.get("/v1/admin/overview")
        assert resp.status_code == 403
        assert "KYBERNETES_ADMIN_TOKEN must be configured" in resp.json()["detail"]

    # 2. Admin token configured, but request missing / invalid token -> 401 Unauthorized
    cfg_with_token = KybernetesConfig(admin_token="secret-admin-12345")
    app_with_token = create_app(config=cfg_with_token)
    transport_with_token = ASGITransport(app=app_with_token)
    
    async with AsyncClient(transport=transport_with_token, base_url="http://127.0.0.1:8081") as client:
        # Missing auth header
        resp_missing = await client.get("/v1/admin/overview")
        assert resp_missing.status_code == 401

        # Wrong token
        resp_wrong = await client.get(
            "/v1/admin/overview",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp_wrong.status_code == 401

        # Valid token -> 200 OK
        resp_valid = await client.get(
            "/v1/admin/overview",
            headers={"Authorization": "Bearer secret-admin-12345"},
        )
        assert resp_valid.status_code == 200
        overview = resp_valid.json()
        assert overview["service"] == "kybernetes"
        assert "components" in overview
        assert "kybernetes" in overview["components"]


@pytest.mark.asyncio
async def test_admin_modules_and_config_inspection():
    """Verify /v1/admin/modules and /v1/admin/modules/{id}/config return truthful descriptors and redacted config."""
    cfg = KybernetesConfig(admin_token="valid-admin-token")
    app = create_app(config=cfg)
    transport = ASGITransport(app=app)
    headers = {"Authorization": "Bearer valid-admin-token"}

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        # List all modules
        resp_modules = await client.get("/v1/admin/modules", headers=headers)
        assert resp_modules.status_code == 200
        modules = resp_modules.json()
        module_ids = [m["id"] for m in modules]
        assert "kybernetes" in module_ids
        assert "hestia" in module_ids
        assert "prosopon" in module_ids
        assert "lthcm" in module_ids
        assert "dianoia" in module_ids
        assert "krisis" in module_ids
        assert "phylax" in module_ids
        assert "dokime" in module_ids
        assert "stoma" in module_ids
        assert "paideia" in module_ids

        # Inspect Kybernetes config
        resp_cfg = await client.get("/v1/admin/modules/kybernetes/config", headers=headers)
        assert resp_cfg.status_code == 200
        cfg_view = resp_cfg.json()
        assert cfg_view["module_id"] == "kybernetes"
        assert "admin_token" in cfg_view["effective"]
        assert cfg_view["effective"]["admin_token"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_admin_probes_and_audit_logging():
    """Verify functional canary probes can be triggered and admin actions are logged."""
    cfg = KybernetesConfig(admin_token="valid-admin-token")
    app = create_app(config=cfg)
    transport = ASGITransport(app=app)
    headers = {"Authorization": "Bearer valid-admin-token"}

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        # Run functional canary probe on Krisis
        resp_probe = await client.post(
            "/v1/admin/probes/run",
            headers=headers,
            json={"component_id": "krisis"},
        )
        assert resp_probe.status_code == 200
        status_data = resp_probe.json()
        assert status_data["state"] in ("simulated", "live")
        assert "Canary verification" in status_data["detail"]

        # Check audit log
        resp_audit = await client.get("/v1/admin/audit-log", headers=headers)
        assert resp_audit.status_code == 200
        audit_data = resp_audit.json()
        assert audit_data["total"] > 0
        actions = [entry["action"] for entry in audit_data["entries"]]
        assert "trigger_probe" in actions


@pytest.mark.asyncio
async def test_host_header_validation():
    """Verify requests with suspicious Host headers are rejected."""
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        # Valid loopback Host
        resp_good = await client.get("/v1/health", headers={"Host": "127.0.0.1:8081"})
        assert resp_good.status_code == 200

        # Invalid external/rebinding Host
        resp_bad = await client.get("/v1/health", headers={"Host": "attacker.com"})
        assert resp_bad.status_code == 400
        assert "Invalid Host header" in resp_bad.text
