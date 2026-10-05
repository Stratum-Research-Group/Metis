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
