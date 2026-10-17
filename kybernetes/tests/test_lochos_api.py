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
        # 1. Get squad status
        resp = await client.get("/v1/lochos/squad")
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
        resp = await client.get("/v1/lochos/queue?status=in_progress")
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

        resp = await client.get("/v1/lochos/attention")
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
        resp = await client.get("/v1/lochos/attention")
        assert resp.status_code == 200
        assert len(resp.json()) == 0
