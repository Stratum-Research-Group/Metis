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
        assert deps["lthcm"]["state"] in ("simulated", "unknown", "live")
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


@pytest.mark.asyncio
async def test_admin_config_validate_endpoints():
    """Verify POST /v1/admin/modules/{id}/config/validate runs module native validators without side effects."""
    admin_token = "secret-token-validation-check"
    cfg = KybernetesConfig(admin_token=admin_token)
    app = create_app(config=cfg)
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        # 1. Unauthenticated request -> 401
        resp_unauth = await client.post(
            "/v1/admin/modules/kybernetes/config/validate",
            json={"proposed": {"port": 8081}},
        )
        assert resp_unauth.status_code == 401

        # 2. Unknown module -> 404
        resp_404 = await client.post(
            "/v1/admin/modules/nonexistent_mod/config/validate",
            headers=headers,
            json={"proposed": {}},
        )
        assert resp_404.status_code == 404

        # 3. Valid Kybernetes config
        resp_kyb_ok = await client.post(
            "/v1/admin/modules/kybernetes/config/validate",
            headers=headers,
            json={"proposed": {"hestia_base_url": "http://127.0.0.1:8080/v1", "db_path": "test.db"}},
        )
        assert resp_kyb_ok.status_code == 200
        res = resp_kyb_ok.json()
        assert res["valid"] is True
        assert res["module_id"] == "kybernetes"
        assert res["restart_required"] is True

        # 4. Invalid Kybernetes config (unknown field or invalid type)
        resp_kyb_bad = await client.post(
            "/v1/admin/modules/kybernetes/config/validate",
            headers=headers,
            json={"proposed": {"allowed_origins": 12345}},
        )
        assert resp_kyb_bad.status_code == 200
        res = resp_kyb_bad.json()
        assert res["valid"] is False
        assert len(res["errors"]) > 0

        # 5. Hestia valid config
        resp_hestia_ok = await client.post(
            "/v1/admin/modules/hestia/config/validate",
            headers=headers,
            json={"proposed": {"listen_host": "127.0.0.1", "listen_port": 8080}},
        )
        assert resp_hestia_ok.status_code == 200
        assert resp_hestia_ok.json()["valid"] is True

        # 6. Hestia invalid config: listen_host must be loopback!
        resp_hestia_bad = await client.post(
            "/v1/admin/modules/hestia/config/validate",
            headers=headers,
            json={"proposed": {"listen_host": "192.168.1.50"}},
        )
        assert resp_hestia_bad.status_code == 200
        res_h_bad = resp_hestia_bad.json()
        assert res_h_bad["valid"] is False
        assert any("listen_host must be a loopback" in e for e in res_h_bad["errors"])

        # 7. Stoma invalid config: chunk_overlap >= max_chunk
        resp_stoma_bad = await client.post(
            "/v1/admin/modules/stoma/config/validate",
            headers=headers,
            json={"proposed": {"max_chunk_tokens": 100, "chunk_overlap_tokens": 150}},
        )
        assert resp_stoma_bad.status_code == 200
        res_s_bad = resp_stoma_bad.json()
        assert res_s_bad["valid"] is False
        assert any("chunk_overlap_tokens" in e for e in res_s_bad["errors"])

        # 8. Library module without config (lthcm)
        resp_lthcm = await client.post(
            "/v1/admin/modules/lthcm/config/validate",
            headers=headers,
            json={"proposed": {"some_key": "val"}},
        )
        assert resp_lthcm.status_code == 200
        res_lib = resp_lthcm.json()
        assert res_lib["valid"] is False
        assert any("library without an editable configuration" in e for e in res_lib["errors"])


@pytest.mark.asyncio
async def test_admin_apply_optimistic_concurrency_rejected():
    """Verify concurrent edits with stale/mismatched base_hash are rejected with 409 Conflict."""
    admin_token = "secret-token-lock-check"
    cfg = KybernetesConfig(admin_token=admin_token)
    app = create_app(config=cfg)
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        resp = await client.post(
            "/v1/admin/modules/hestia/config/apply",
            headers=headers,
            json={
                "base_hash": "stale_hash_from_old_read",
                "patch": {"listen_host": "127.0.0.1", "listen_port": 8080},
            },
        )
        assert resp.status_code == 409
        assert "Concurrent modification detected" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_admin_apply_invalid_patch_never_touches_disk(tmp_path, monkeypatch):
    """Verify that an invalid patch fails validation and disk files remain completely untouched."""
    admin_token = "secret-token-invalid-patch"
    cfg = KybernetesConfig(admin_token=admin_token)
    app = create_app(config=cfg)
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    test_file = tmp_path / "hestia.yaml"
    initial_content = "listen_host: 127.0.0.1\nlisten_port: 8080\n"
    test_file.write_text(initial_content)

    from kybernetes.admin.descriptors import MODULE_REGISTRY
    monkeypatch.setattr(MODULE_REGISTRY["hestia"], "config_path", str(test_file))

    # Retrieve current config and hash
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        cfg_resp = await client.get("/v1/admin/modules/hestia/config", headers=headers)
        base_hash = cfg_resp.json()["config_hash"]

        # Submit invalid patch (listen_host must be loopback)
        resp_bad = await client.post(
            "/v1/admin/modules/hestia/config/apply",
            headers=headers,
            json={
                "base_hash": base_hash,
                "patch": {"listen_host": "192.168.1.100", "listen_port": 8080},
                "confirm_token": "CONFIRM",
            },
        )
        assert resp_bad.status_code == 422
        assert "Validation failed" in resp_bad.json()["detail"]

        # Assert disk file was NOT modified
        assert test_file.read_text() == initial_content


@pytest.mark.asyncio
async def test_admin_apply_sensitive_confirmation_gating(tmp_path, monkeypatch):
    """Verify sensitive configuration modifications require explicit confirmation and provide diff."""
    admin_token = "secret-token-confirm-check"
    cfg = KybernetesConfig(admin_token=admin_token)
    app = create_app(config=cfg)
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    test_file = tmp_path / "hestia.yaml"
    initial_content = "listen_host: 127.0.0.1\nlisten_port: 8080\n"
    test_file.write_text(initial_content)

    from kybernetes.admin.descriptors import MODULE_REGISTRY
    monkeypatch.setattr(MODULE_REGISTRY["hestia"], "config_path", str(test_file))

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        cfg_resp = await client.get("/v1/admin/modules/hestia/config", headers=headers)
        base_hash = cfg_resp.json()["config_hash"]

        # 1. Sensitive change without confirmation -> 400 with diff requirement
        resp_unconfirmed = await client.post(
            "/v1/admin/modules/hestia/config/apply",
            headers=headers,
            json={
                "base_hash": base_hash,
                "patch": {"listen_host": "127.0.0.1", "listen_port": 8090},
            },
        )
        assert resp_unconfirmed.status_code == 400
        assert "Explicit confirmation required" in resp_unconfirmed.json()["detail"]

        # 2. Resubmit with confirm_token='CONFIRM' -> succeeds
        resp_confirmed = await client.post(
            "/v1/admin/modules/hestia/config/apply",
            headers=headers,
            json={
                "base_hash": base_hash,
                "patch": {"listen_host": "127.0.0.1", "listen_port": 8090},
                "confirm_token": "CONFIRM",
            },
        )
        assert resp_confirmed.status_code == 200
        cs = resp_confirmed.json()
        assert cs["status"] == "restart_pending"
        assert cs["restart_required"] is True
        assert cs["diff"] is not None
        assert "+listen_port: 8090" in cs["diff"]


@pytest.mark.asyncio
async def test_admin_apply_and_rollback_restores_hash(tmp_path, monkeypatch):
    """Verify applying a valid ChangeSet and rolling back accurately restores original on-disk content and hash."""
    admin_token = "secret-token-rollback-check"
    cfg = KybernetesConfig(admin_token=admin_token)
    app = create_app(config=cfg)
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    test_file = tmp_path / "kybernetes.yaml"
    test_file.write_text("hestia_base_url: http://127.0.0.1:8080/v1\ndb_path: ':memory:'\n")

    from kybernetes.admin.descriptors import MODULE_REGISTRY
    monkeypatch.setattr(MODULE_REGISTRY["kybernetes"], "config_path", str(test_file))

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        # Step 1: Read initial state and hash
        resp_init = await client.get("/v1/admin/modules/kybernetes/config", headers=headers)
        h0 = resp_init.json()["config_hash"]

        # Step 2: Apply valid update
        resp_apply = await client.post(
            "/v1/admin/modules/kybernetes/config/apply",
            headers=headers,
            json={
                "base_hash": h0,
                "patch": {
                    "hestia_base_url": "http://127.0.0.1:8080/v1",
                    "db_path": "metis_test.db",
                },
            },
        )
        assert resp_apply.status_code == 200
        applied_cs = resp_apply.json()
        assert applied_cs["status"] == "restart_pending"
        cs_id = applied_cs["id"]

        # Verify hash changed on disk
        resp_updated = await client.get("/v1/admin/modules/kybernetes/config", headers=headers)
        h1 = resp_updated.json()["config_hash"]
        assert h1 != h0

        # Step 3: Check history endpoint contains the changeset
        resp_history = await client.get("/v1/admin/modules/kybernetes/history", headers=headers)
        assert resp_history.status_code == 200
        history = resp_history.json()
        assert any(entry["id"] == cs_id for entry in history)

        # Step 4: Execute rollback
        resp_rollback = await client.post(
            f"/v1/admin/modules/kybernetes/config/rollback/{cs_id}",
            headers=headers,
        )
        assert resp_rollback.status_code == 200
        rb_cs = resp_rollback.json()
        assert rb_cs["status"] == "rolled_back_on_disk"

        # Step 5: Verify restored hash matches h0 exactly
        resp_restored = await client.get("/v1/admin/modules/kybernetes/config", headers=headers)
        h_restored = resp_restored.json()["config_hash"]
        assert h_restored == h0


@pytest.mark.asyncio
async def test_admin_model_load_unload_audited():
    """Verify model load and unload actions are recorded in the append-only audit log."""
    admin_token = "secret-token-model-audit"
    cfg = KybernetesConfig(admin_token=admin_token)
    app = create_app(config=cfg)
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        # Trigger model load (Hestia may be offline in unit test, but action MUST be audited)
        await client.post(
            "/v1/admin/models/load",
            headers=headers,
            json={"model": "llama-3.1-8b-instruct", "confirm": True},
        )

        # Check audit log
        audit_resp = await client.get("/v1/admin/audit-log", headers=headers)
        assert audit_resp.status_code == 200
        entries = audit_resp.json()["entries"]
        assert any(e["action"] == "model_load" and "llama-3.1-8b-instruct" in e["target"] for e in entries)


@pytest.mark.asyncio
async def test_overview_polling_triggers_zero_model_loads(monkeypatch):
    """Verify that repeated polling of the overview endpoint triggers zero model loads or completions."""
    admin_token = "secret-token-polling-safety"
    cfg = KybernetesConfig(admin_token=admin_token)
    app = create_app(config=cfg)
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    # Instrument httpx.AsyncClient.post to intercept any outgoing network calls
    call_counts = {"load": 0, "completions": 0}
    import httpx

    original_post = httpx.AsyncClient.post

    async def fake_post(self, url, *args, **kwargs):
        url_str = str(url)
        if "/manage/load" in url_str:
            call_counts["load"] += 1
        elif "/v1/chat/completions" in url_str:
            call_counts["completions"] += 1
        return await original_post(self, url, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        for _ in range(100):
            resp = await client.get("/v1/admin/overview", headers=headers)
            assert resp.status_code == 200

    # Invariant: Polling is strictly read-only against cached telemetry; never triggers model loading
    assert call_counts["load"] == 0
    assert call_counts["completions"] == 0


@pytest.mark.asyncio
async def test_admin_batch_dokime_run_and_history(monkeypatch, tmp_path):
    """Verify Dokime batch run controls: confirmation gating, execution, audit, and history."""
    admin_token = "secret-token-dokime-test"
    cfg = KybernetesConfig(admin_token=admin_token)
    app = create_app(config=cfg)
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        # 1. Unauthenticated request rejected
        resp_unauth = await client.post(
            "/v1/admin/batch/dokime/run",
            json={"confirm_token": "CONFIRM"},
        )
        assert resp_unauth.status_code == 401

        # 2. Unconfirmed request rejected (Resource warning check)
        resp_unconfirmed = await client.post(
            "/v1/admin/batch/dokime/run",
            headers=headers,
            json={"confirm_token": "NO_CONFIRM"},
        )
        assert resp_unconfirmed.status_code == 400
        assert "Explicit confirmation required" in resp_unconfirmed.json()["detail"]

        # 3. Successful run with confirm_token='CONFIRM'
        resp_run = await client.post(
            "/v1/admin/batch/dokime/run",
            headers=headers,
            json={
                "confirm_token": "CONFIRM",
                "task_id": "arc_easy_25",
                "model_id": "llama-3.1-8b-instruct",
                "smoke_test": True,
            },
        )
        assert resp_run.status_code == 200
        run_data = resp_run.json()
        assert run_data["status"] == "success"
        assert run_data["tool"] == "dokime"
        assert run_data["run"]["is_simulated"] is True
        run_id = run_data["run"]["run_id"]

        # 4. Check history listing endpoint
        resp_history = await client.get("/v1/admin/batch/dokime/runs?limit=10", headers=headers)
        assert resp_history.status_code == 200
        hist_data = resp_history.json()
        assert hist_data["tool"] == "dokime"
        assert any(r["run_id"] == run_id for r in hist_data["runs"])

        # 5. Check audit log records the batch run
        resp_audit = await client.get("/v1/admin/audit-log", headers=headers)
        assert resp_audit.status_code == 200
        entries = resp_audit.json()["entries"]
        assert any(e["action"] == "batch_run_dokime" and e["status"] == "completed" for e in entries)


@pytest.mark.asyncio
async def test_admin_batch_paideia_run_and_history(monkeypatch, tmp_path):
    """Verify Paideia batch run controls: confirmation gating, preflight check, and manifest catalog."""
    admin_token = "secret-token-paideia-test"
    cfg = KybernetesConfig(admin_token=admin_token)
    app = create_app(config=cfg)
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        # 1. Unconfirmed request rejected
        resp_unconf = await client.post(
            "/v1/admin/batch/paideia/run",
            headers=headers,
            json={"confirm_token": "DENY"},
        )
        assert resp_unconf.status_code == 400

        # 2. Successful run with confirm_token='CONFIRM'
        resp_run = await client.post(
            "/v1/admin/batch/paideia/run",
            headers=headers,
            json={
                "confirm_token": "CONFIRM",
                "task": "router",
                "allow_simulated": True,
            },
        )
        assert resp_run.status_code == 200
        run_data = resp_run.json()
        assert run_data["status"] == "success"
        assert run_data["tool"] == "paideia"
        assert run_data["run"]["status"] in ("SIMULATED", "COMPLETED")
        assert run_data["run"]["is_simulated"] is True
        run_id = run_data["run"]["run_id"]

        # 3. Check history listing endpoint
        resp_history = await client.get("/v1/admin/batch/paideia/runs?limit=10", headers=headers)
        assert resp_history.status_code == 200
        hist_data = resp_history.json()
        assert hist_data["tool"] == "paideia"
        assert any(r["run_id"] == run_id for r in hist_data["runs"])

        # 4. Check audit log records the batch run
        resp_audit = await client.get("/v1/admin/audit-log", headers=headers)
        assert resp_audit.status_code == 200
        entries = resp_audit.json()["entries"]
        assert any(e["action"] == "batch_run_paideia" and e["status"] == "completed" for e in entries)


@pytest.mark.asyncio
async def test_admin_apply_preserves_secrets_when_redacted_placeholders_sent(tmp_path, monkeypatch):
    """Verify applying a patch with [REDACTED] placeholders preserves original secrets on disk (Finding R1)."""
    admin_token = "secret-token-redact-preservation"
    cfg = KybernetesConfig(admin_token=admin_token)
    app = create_app(config=cfg)
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    test_file = tmp_path / "kybernetes.yaml"
    real_secret = "real_ultra_secure_token_xyz987"
    test_file.write_text(f"admin_token: {real_secret}\nhestia_base_url: http://127.0.0.1:8080/v1\ndb_path: ':memory:'\n")

    from kybernetes.admin.descriptors import MODULE_REGISTRY
    monkeypatch.setattr(MODULE_REGISTRY["kybernetes"], "config_path", str(test_file))

    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8081") as client:
        # Step 1: Read config view - verify secret is redacted in view
        resp_view = await client.get("/v1/admin/modules/kybernetes/config", headers=headers)
        assert resp_view.status_code == 200
        data = resp_view.json()
        assert data["saved"]["admin_token"] == "[REDACTED]"
        base_hash = data["config_hash"]

        # Step 2: Apply patch round-tripping the redacted placeholder and updating hestia_base_url
        resp_apply = await client.post(
            "/v1/admin/modules/kybernetes/config/apply",
            headers=headers,
            json={
                "base_hash": base_hash,
                "patch": {
                    "admin_token": "[REDACTED]",
                    "hestia_base_url": "http://127.0.0.1:9090/v1",
                },
            },
        )
        assert resp_apply.status_code == 200

        # Step 3: Verify on-disk file still contains real secret, NOT the literal '[REDACTED]'
        saved_text = test_file.read_text()
        assert real_secret in saved_text
        assert "[REDACTED]" not in saved_text
        assert "http://127.0.0.1:9090/v1" in saved_text


@pytest.mark.asyncio
async def test_admin_snapshot_durability_and_permissions(tmp_path, monkeypatch):
    """Verify snapshot file permissions (0700/0600) and metadata durability across server restarts (Finding R2)."""
    from kybernetes.admin.changesets import ChangeSetManager
    import os

    snap_dir = tmp_path / ".admin_snapshots"
    mgr1 = ChangeSetManager(snapshot_dir=str(snap_dir))

    # Verify snapshot dir permissions are 0700
    dir_mode = os.stat(snap_dir).st_mode & 0o777
    assert dir_mode == 0o700

    test_file = tmp_path / "kybernetes.yaml"
    test_file.write_text("hestia_base_url: http://127.0.0.1:8080/v1\ndb_path: ':memory:'\n")

    from kybernetes.admin.descriptors import MODULE_REGISTRY
    monkeypatch.setattr(MODULE_REGISTRY["kybernetes"], "config_path", str(test_file))

    _, _, base_hash = mgr1.get_current_saved_content_and_hash("kybernetes")
    cs = mgr1.apply(
        module_id="kybernetes",
        base_hash=base_hash,
        patch={"hestia_base_url": "http://127.0.0.1:9999/v1", "db_path": "test.db"},
        requested_by="admin_test",
    )
    assert cs.prev_snapshot_id is not None

    # Check snapshot backup and metadata files exist and have 0600 mode
    bak_file = snap_dir / f"{cs.prev_snapshot_id}.bak"
    meta_file = snap_dir / f"{cs.prev_snapshot_id}.meta.json"
    assert bak_file.exists()
    assert meta_file.exists()
    assert (os.stat(bak_file).st_mode & 0o777) == 0o600
    assert (os.stat(meta_file).st_mode & 0o777) == 0o600

    # Simulate server restart by instantiating a fresh ChangeSetManager
    mgr2 = ChangeSetManager(snapshot_dir=str(snap_dir))
    assert cs.prev_snapshot_id in mgr2._snapshots

    # Execute rollback on the new manager instance
    rb_cs = mgr2.rollback(module_id="kybernetes", changeset_id=cs.id, requested_by="admin_test")
    assert rb_cs.status.value == "rolled_back_on_disk"

    # Verify file content is restored
    restored_text = test_file.read_text()
    assert "http://127.0.0.1:8080/v1" in restored_text


