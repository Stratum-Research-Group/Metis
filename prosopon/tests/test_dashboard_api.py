"""Tests for Prosopon Control Dashboard proxy endpoints, auth guards, and offline tolerance."""

import pytest
from httpx import ASGITransport, AsyncClient

from prosopon.config import ProsoponConfig
from prosopon.server import create_app


@pytest.mark.asyncio
async def test_dashboard_zero_loopback_exemption(tmp_path):
    """Verify /api/dashboard/* endpoints strictly reject unauthenticated callers even from loopback."""
    cfg = ProsoponConfig(data_dir=str(tmp_path), mock_kybernetes=True)
    app = create_app(cfg)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        # Loopback caller without authorization token
        resp = await client.get("/api/dashboard/overview")
        assert resp.status_code == 401
        assert "paired token required" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_dashboard_authenticated_overview_and_gpu(tmp_path):
    """Verify paired bearer token permits access to overview and GPU telemetry."""
    cfg = ProsoponConfig(data_dir=str(tmp_path), mock_kybernetes=True)
    app = create_app(cfg)
    pairing_mgr = app.state.pairing_mgr

    # Generate and verify token with admin role
    pin = pairing_mgr.generate_pairing_pin()
    token = pairing_mgr.verify_pin_and_pair(pin, device_name="Test Browser", role="admin")
    assert token is not None
    headers = {"Authorization": f"Bearer {token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        resp = await client.get("/api/dashboard/overview", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert "gpu_telemetry" in data
        assert data["mock_banner"] is True

        # Test modules endpoint
        resp_mods = await client.get("/api/dashboard/modules", headers=headers)
        assert resp_mods.status_code == 200
        mods = resp_mods.json()
        assert any(m["id"] == "prosopon" for m in mods)

        # Test prosopon config endpoint
        resp_cfg = await client.get("/api/dashboard/modules/prosopon/config", headers=headers)
        assert resp_cfg.status_code == 200
        cfg_data = resp_cfg.json()
        assert cfg_data["module_id"] == "prosopon"
        assert "host" in cfg_data["saved"]
        assert cfg_data["saved"]["kybernetes_service_token"] in (None, "[REDACTED]")


@pytest.mark.asyncio
async def test_dashboard_graceful_kybernetes_offline_tolerance(tmp_path):
    """Verify that if Kybernetes orchestrator is offline, dashboard returns clean offline status instead of crashing."""
    # Point to a dead port
    cfg = ProsoponConfig(
        data_dir=str(tmp_path),
        mock_kybernetes=False,
        kybernetes_api_url="http://127.0.0.1:59999/v1",
    )
    app = create_app(cfg)
    pairing_mgr = app.state.pairing_mgr
    pin = pairing_mgr.generate_pairing_pin()
    token = pairing_mgr.verify_pin_and_pair(pin, device_name="Test Browser", role="admin")
    assert token is not None
    headers = {"Authorization": f"Bearer {token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        resp = await client.get("/api/dashboard/overview", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["orchestrator_mode"] == "offline"
        assert data["orchestrator"]["state"] == "offline"


@pytest.mark.asyncio
async def test_csp_header_present(tmp_path):
    """Verify Content-Security-Policy headers are sent on responses."""
    cfg = ProsoponConfig(data_dir=str(tmp_path))
    app = create_app(cfg)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert "Content-Security-Policy" in resp.headers
        csp = resp.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp


@pytest.mark.asyncio
async def test_dashboard_config_validate(tmp_path):
    """Verify /api/dashboard/modules/{id}/config/validate enforces auth and validates properly."""
    cfg = ProsoponConfig(data_dir=str(tmp_path), mock_kybernetes=True)
    app = create_app(cfg)
    pairing_mgr = app.state.pairing_mgr
    pin = pairing_mgr.generate_pairing_pin()
    token = pairing_mgr.verify_pin_and_pair(pin, device_name="Admin Console", role="admin")
    assert token is not None
    headers = {"Authorization": f"Bearer {token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        # 1. Unauthenticated -> 401
        resp_unauth = await client.post(
            "/api/dashboard/modules/prosopon/config/validate",
            json={"proposed": {"host": "127.0.0.1"}},
        )
        assert resp_unauth.status_code == 401

        # 2. Valid Prosopon config
        resp_ok = await client.post(
            "/api/dashboard/modules/prosopon/config/validate",
            headers=headers,
            json={"proposed": {"host": "127.0.0.1", "port": 8000}},
        )
        assert resp_ok.status_code == 200
        data_ok = resp_ok.json()
        assert data_ok["valid"] is True
        assert data_ok["module_id"] == "prosopon"

        # 3. Invalid Prosopon config
        resp_bad = await client.post(
            "/api/dashboard/modules/prosopon/config/validate",
            headers=headers,
            json={"proposed": {"port": "invalid-port"}},
        )
        assert resp_bad.status_code == 200
        data_bad = resp_bad.json()
        assert data_bad["valid"] is False
        assert len(data_bad["errors"]) > 0

        # 4. Proxied validation for Kybernetes
        resp_proxy = await client.post(
            "/api/dashboard/modules/kybernetes/config/validate",
            headers=headers,
            json={"proposed": {"hestia_base_url": "http://127.0.0.1:8080/v1"}},
        )
        assert resp_proxy.status_code == 200
        data_proxy = resp_proxy.json()
        assert data_proxy["module_id"] == "kybernetes"
        assert data_proxy["valid"] is True


@pytest.mark.asyncio
async def test_dashboard_apply_history_and_rollback(tmp_path):
    """Verify Prosopon proxies apply, history, and rollback behind dedicated authentication."""
    cfg = ProsoponConfig(data_dir=str(tmp_path), mock_kybernetes=True)
    app = create_app(cfg)
    pairing_mgr = app.state.pairing_mgr
    pin = pairing_mgr.generate_pairing_pin()
    token = pairing_mgr.verify_pin_and_pair(pin, device_name="Admin Console", role="admin")
    assert token is not None
    headers = {"Authorization": f"Bearer {token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        # 1. Unauthenticated apply -> 401
        resp_unauth = await client.post(
            "/api/dashboard/modules/hestia/config/apply",
            json={"base_hash": "mock", "patch": {}},
        )
        assert resp_unauth.status_code == 401

        # 2. Authenticated apply -> 200 ChangeSet returned
        resp_apply = await client.post(
            "/api/dashboard/modules/hestia/config/apply",
            headers=headers,
            json={"base_hash": "mock-base", "patch": {"listen_port": 8080}},
        )
        assert resp_apply.status_code == 200
        cs = resp_apply.json()
        assert cs["module"] == "hestia"
        assert cs["status"] in ("written", "restart_pending")
        cs_id = cs["id"]

        # 3. Retrieve history
        resp_hist = await client.get(
            "/api/dashboard/modules/hestia/history",
            headers=headers,
        )
        assert resp_hist.status_code == 200
        hist = resp_hist.json()
        assert len(hist) > 0

        # 4. Rollback
        resp_rb = await client.post(
            f"/api/dashboard/modules/hestia/config/rollback/{cs_id}",
            headers=headers,
        )
        assert resp_rb.status_code == 200
        rb_cs = resp_rb.json()
        assert rb_cs["status"] == "rolled_back_on_disk"


@pytest.mark.asyncio
async def test_dashboard_models_load_unload(tmp_path):
    """Verify Prosopon proxies model load and unload actions with auth enforcement."""
    cfg = ProsoponConfig(data_dir=str(tmp_path), mock_kybernetes=True)
    app = create_app(cfg)
    pairing_mgr = app.state.pairing_mgr
    pin = pairing_mgr.generate_pairing_pin()
    token = pairing_mgr.verify_pin_and_pair(pin, device_name="Admin Console", role="admin")
    assert token is not None
    headers = {"Authorization": f"Bearer {token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        # 1. Unauthenticated load -> 401
        resp_unauth = await client.post(
            "/api/dashboard/models/load",
            json={"model": "llama-3.1-8b-instruct"},
        )
        assert resp_unauth.status_code == 401

        # 2. Authenticated load -> 200
        resp_load = await client.post(
            "/api/dashboard/models/load",
            headers=headers,
            json={"model": "llama-3.1-8b-instruct", "confirm": True},
        )
        assert resp_load.status_code == 200
        assert resp_load.json()["status"] == "loaded"

        # 3. Authenticated unload -> 200
        resp_unload = await client.post(
            "/api/dashboard/models/unload",
            headers=headers,
            json={"model": "llama-3.1-8b-instruct", "confirm": True},
        )
        assert resp_unload.status_code == 200
        assert resp_unload.json()["status"] == "unloaded"


@pytest.mark.asyncio
async def test_dashboard_batch_dokime_and_paideia(tmp_path):
    """Verify Prosopon proxies Dokime and Paideia batch run triggers and histories."""
    cfg = ProsoponConfig(data_dir=str(tmp_path), mock_kybernetes=True)
    app = create_app(cfg)
    pairing_mgr = app.state.pairing_mgr
    pin = pairing_mgr.generate_pairing_pin()
    token = pairing_mgr.verify_pin_and_pair(pin, device_name="Admin Console", role="admin")
    assert token is not None
    headers = {"Authorization": f"Bearer {token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as client:
        # 1. Unauthenticated -> 401
        resp_unauth = await client.get("/api/dashboard/batch/dokime/runs")
        assert resp_unauth.status_code == 401

        resp_unauth_post = await client.post("/api/dashboard/batch/dokime/run", json={"confirm_token": "CONFIRM"})
        assert resp_unauth_post.status_code == 401

        # 2. Dokime runs history
        resp_dokime_hist = await client.get("/api/dashboard/batch/dokime/runs", headers=headers)
        assert resp_dokime_hist.status_code == 200
        assert resp_dokime_hist.json()["tool"] == "dokime"

        # 3. Trigger Dokime run
        resp_dokime_run = await client.post(
            "/api/dashboard/batch/dokime/run",
            headers=headers,
            json={"confirm_token": "CONFIRM", "task_id": "arc_easy_25"},
        )
        assert resp_dokime_run.status_code == 200
        assert resp_dokime_run.json()["status"] == "success"

        # 4. Paideia runs history
        resp_paideia_hist = await client.get("/api/dashboard/batch/paideia/runs", headers=headers)
        assert resp_paideia_hist.status_code == 200
        assert resp_paideia_hist.json()["tool"] == "paideia"

        # 5. Trigger Paideia run
        resp_paideia_run = await client.post(
            "/api/dashboard/batch/paideia/run",
            headers=headers,
            json={"confirm_token": "CONFIRM", "task": "router"},
        )
        assert resp_paideia_run.status_code == 200
        assert resp_paideia_run.json()["status"] == "success"


