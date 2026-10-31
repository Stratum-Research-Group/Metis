import pytest
from httpx import AsyncClient, ASGITransport
from kybernetes.config import KybernetesConfig
from kybernetes.lochos import LochosSquadManager, create_starter_rig
from kybernetes.server.app import create_app


@pytest.mark.asyncio
async def test_lochos_squad_api_endpoints(tmp_path):
    admin_token = "secret-service-token-lochos"
    cfg = KybernetesConfig(admin_token=admin_token)
    
    # Create isolated squad manager
    rig = create_starter_rig("api-test-squad")
    db_path = tmp_path / "lochos.sqlite"
    squad_manager = LochosSquadManager(
        rig_spec=rig,
        workspace_root=tmp_path,
        queue=None,
    )
    squad_manager.queue.db_path = db_path
    squad_manager.queue._init_db()

    app = create_app(
        config=cfg,
        service_token=admin_token,
        lochos_squad=squad_manager,
    )
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 0. Unauthenticated request must return 401 (Zero-Loopback security, Finding F-04)
        unauth_resp = await client.get("/v1/lochos/squad")
        assert unauth_resp.status_code == 401

        # 1. Get squad status with valid Bearer token
        resp = await client.get("/v1/lochos/squad", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["rig_name"] == "api-test-squad"
        assert len(data["seats"]) == 2
        assert data["seats"][0]["id"] == "coder"

        # 2. Dispatch work
        resp = await client.post(
            "/v1/lochos/dispatch",
            headers=headers,
            json={
                "mission": "release-0.5.8",
                "slice_id": "OPR.01",
                "title": "API Test Mission",
                "target_seat": "coder",
            },
        )
        assert resp.status_code == 200
        task_data = resp.json()
        task_id = task_data["task_id"]
        assert task_data["status"] == "in_progress"
        assert task_data["baton_owner"] == "coder@api-test-squad"

        # 3. Check queue
        resp = await client.get("/v1/lochos/queue?status=in_progress", headers=headers)
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["task_id"] == task_id

        # 4. Transfer baton to reviewer
        resp = await client.post(
            "/v1/lochos/transfer",
            headers=headers,
            json={
                "task_id": task_id,
                "from_seat": "coder",
                "to_seat": "reviewer",
                "note": "Ready for verification",
            },
        )
        assert resp.status_code == 200
        transfer_data = resp.json()
        assert transfer_data["baton_owner"] == "reviewer@api-test-squad"
        assert transfer_data["status"] == "in_progress"

        # 5. Raise attention on task
        squad_manager.queue.raise_attention(
            task_id=task_id,
            seat_id="reviewer@api-test-squad",
            reason="phylax_approval_required",
            details={"action": "edit_block"},
        )

        resp = await client.get("/v1/lochos/attention", headers=headers)
        assert resp.status_code == 200
        attn_list = resp.json()
        assert len(attn_list) == 1
        assert attn_list[0]["reason"] == "phylax_approval_required"

        # 6. Resolve attention
        resp = await client.post(
            "/v1/lochos/attention/resolve",
            headers=headers,
            json={
                "task_id": task_id,
                "seat_id": "reviewer@api-test-squad",
                "resolution": "approved_by_operator",
            },
        )
        assert resp.status_code == 200
        resolved_task = resp.json()
        assert resolved_task["status"] == "in_progress"

        # Verify attention cleared
        resp = await client.get("/v1/lochos/attention", headers=headers)
        assert resp.status_code == 200
        assert len(resp.json()) == 0

        # 7. Check seat brief endpoint
        resp = await client.get("/v1/lochos/seats/coder/brief", headers=headers)
        assert resp.status_code == 200
        brief = resp.json()
        assert brief["id"] == "coder"
        assert "python" in brief["skills"]

        # 8. Check transfer brief endpoint
        resp = await client.get(
            f"/v1/lochos/transfer/brief?task_id={task_id}&from_seat=coder&to_seat=reviewer&note=Check+this",
            headers=headers,
        )
        assert resp.status_code == 200
        transfer_brief = resp.json()
        assert transfer_brief["task_id"] == task_id
        assert transfer_brief["source_seat"]["id"] == "coder"
        assert transfer_brief["destination_seat"]["id"] == "reviewer"
        assert transfer_brief["handoff_note"] == "Check this"


def test_lochos_websocket_streaming(tmp_path):
    from starlette.testclient import TestClient

    admin_token = "secret-service-token-lochos"
    cfg = KybernetesConfig(admin_token=admin_token)
    rig = create_starter_rig("ws-test-squad")
    db_path = tmp_path / "lochos.sqlite"
    squad_manager = LochosSquadManager(
        rig_spec=rig,
        workspace_root=tmp_path,
        queue=None,
    )
    squad_manager.queue.db_path = db_path
    squad_manager.queue._init_db()

    app = create_app(
        config=cfg,
        service_token=admin_token,
        lochos_squad=squad_manager,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/lochos/ws") as ws:
            # 1. Initial snapshot received
            snap = ws.receive_json()
            assert snap["type"] == "squad_state"
            assert snap["data"]["rig_name"] == "ws-test-squad"

            # 2. Dispatch work triggers broadcast event
            squad_manager.dispatch_work(
                mission="m-ws",
                slice_id="s1",
                title="WS Streamed Task",
                target_seat="coder",
            )
            event = ws.receive_json()
            assert event["type"] == "task_dispatched"
            assert event["data"]["title"] == "WS Streamed Task"

            # 3. Ping-pong
            ws.send_text("ping")
            pong = ws.receive_text()
            assert pong == "pong"


@pytest.mark.asyncio
async def test_lochos_pipeline_api_endpoint(tmp_path):
    admin_token = "secret-service-token-lochos"
    cfg = KybernetesConfig(admin_token=admin_token)

    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(tmp_path), check=True)
    (tmp_path / "baseline.txt").write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "baseline.txt"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(tmp_path), check=True)

    rig = create_starter_rig("pipeline-api-squad")
    db_path = tmp_path / "lochos.sqlite"
    squad_manager = LochosSquadManager(
        rig_spec=rig,
        workspace_root=tmp_path,
        queue=None,
    )
    squad_manager.queue.db_path = db_path
    squad_manager.queue._init_db()

    app = create_app(
        config=cfg,
        service_token=admin_token,
        lochos_squad=squad_manager,
    )
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Unauthenticated request must return 401
        unauth_resp = await client.post("/v1/lochos/pipeline", json={"mission": "m", "slice_id": "s", "title": "t"})
        assert unauth_resp.status_code == 401

        # 2. Missing fields must return 400
        bad_resp = await client.post("/v1/lochos/pipeline", headers=headers, json={"mission": "m"})
        assert bad_resp.status_code == 400

        # 3. Successful pipeline execution via /v1/lochos/pipeline
        ok_resp = await client.post(
            "/v1/lochos/pipeline",
            headers=headers,
            json={
                "mission": "release-0.6.0",
                "slice_id": "OPR.10",
                "title": "API Multi-Agent Pipeline",
                "test_command": "python3 -c 'assert True'",
            },
        )
        assert ok_resp.status_code == 200
        ok_data = ok_resp.json()
        assert ok_data["success"] is True
        assert ok_data["stage"] == "completed"
        assert ok_data["verification_passed"] is True
        assert "task-" in ok_data["task_id"]

        # 4. Alias /api/lochos/pipeline also functions identically
        alias_resp = await client.post(
            "/api/lochos/pipeline",
            headers=headers,
            json={
                "mission": "release-0.6.0",
                "slice_id": "OPR.11",
                "title": "Alias API Pipeline",
                "test_command": "python3 -c 'assert True'",
            },
        )
        assert alias_resp.status_code == 200
        alias_data = alias_resp.json()
        assert alias_data["success"] is True
        assert alias_data["stage"] == "completed"

        # 5. Pipeline failure with rollback
        fail_resp = await client.post(
            "/v1/lochos/pipeline",
            headers=headers,
            json={
                "mission": "release-0.6.0",
                "slice_id": "OPR.12",
                "title": "Failing Pipeline",
                "test_command": "python3 -c 'raise SystemExit(1)'",
            },
        )
        assert fail_resp.status_code == 200
        fail_data = fail_resp.json()
        assert fail_data["success"] is False
        assert fail_data["stage"] == "rolled_back"
        assert fail_data["rolled_back"] is True


