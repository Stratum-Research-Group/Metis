"""Tests for /v1/models, admission queue, strict schemas, and graceful shutdown."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hestia.api.app import create_app
from hestia.backend import BackendProcess
from hestia.config import ServiceConfig
from hestia.registry import ModelEntry, ModelRegistry, ModelType, SamplingParams
from hestia.server import (
    QueueTimeoutError,
    ServerManager,
    ServerOverloadedError,
)


@pytest.fixture
def mock_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(
        ModelEntry(
            name="llama-3.1-8b-instruct",
            model_type=ModelType.CHAT,
            gguf_path="/models/fake_chat.gguf",
            quantization="Q4_K_M",
            context_size=8192,
            vram_estimate_mb=5000,
            gpu_target="rtx-5060-ti",
            sampling=SamplingParams(temperature=0.7, top_p=0.9),
        )
    )
    registry.register(
        ModelEntry(
            name="nomic-embed-text",
            model_type=ModelType.EMBEDDING,
            gguf_path="/models/fake_embed.gguf",
            quantization="Q4_K_M",
            context_size=2048,
            vram_estimate_mb=1000,
            gpu_target="rtx-5060-ti",
        )
    )
    return registry


@pytest.mark.asyncio
async def test_get_v1_models_list(mock_registry: ModelRegistry) -> None:
    """Test GET /v1/models returns OpenAI-compatible list of models."""
    app = create_app(config_path="hestia.yaml", models_path="models.yaml")

    with patch("hestia.api.app._get_registry", return_value=mock_registry):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "list"
            assert isinstance(data["data"], list)
            model_ids = [m["id"] for m in data["data"]]
            assert "llama-3.1-8b-instruct" in model_ids
            assert "nomic-embed-text" in model_ids

            item = next(m for m in data["data"] if m["id"] == "llama-3.1-8b-instruct")
            assert item["object"] == "model"
            assert item["owned_by"] == "hestia"
            assert item["gpu_target"] == "rtx-5060-ti"
            assert item["state"] == "unloaded"
            assert "artifact_digest" in item


@pytest.mark.asyncio
async def test_get_v1_model_by_id(mock_registry: ModelRegistry) -> None:
    """Test GET /v1/models/{model_name} returns OpenAI-compatible single model object."""
    app = create_app(config_path="hestia.yaml", models_path="models.yaml")

    with patch("hestia.api.app._get_registry", return_value=mock_registry):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/v1/models/llama-3.1-8b-instruct")
            assert resp.status_code == 200
            data = resp.json()
            assert data["id"] == "llama-3.1-8b-instruct"
            assert data["object"] == "model"
            assert data["owned_by"] == "hestia"
            assert data["gpu_target"] == "rtx-5060-ti"
            assert data["state"] == "unloaded"
            assert "artifact_digest" in data

            # 404 for unknown model
            resp_404 = await client.get("/v1/models/non-existent-model")
            assert resp_404.status_code == 404


@pytest.mark.asyncio
async def test_chat_completions_strict_validation(mock_registry: ModelRegistry) -> None:
    """Test strict schema validation on /v1/chat/completions."""
    app = create_app(config_path="hestia.yaml", models_path="models.yaml")

    mock_manager = AsyncMock()
    mock_manager.acquire_admission = AsyncMock()
    mock_manager.release_admission = MagicMock()
    mock_manager.ensure_loaded = AsyncMock(
        return_value=BackendProcess(model_name="llama-3.1-8b-instruct", pid=1, port=8000)
    )

    with (
        patch("hestia.api.app._get_manager", return_value=mock_manager),
        patch("hestia.api.app._get_registry", return_value=mock_registry),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Missing model field
            resp = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 400
            assert "model" in resp.text

            # Invalid temperature > 2.0
            resp_temp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b-instruct",
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 3.5,
                },
            )
            assert resp_temp.status_code == 400

            # Invalid top_p < 0.0
            resp_topp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b-instruct",
                    "messages": [{"role": "user", "content": "hi"}],
                    "top_p": -0.5,
                },
            )
            assert resp_topp.status_code == 400


@pytest.mark.asyncio
async def test_admission_queue_overloaded_429(mock_registry: ModelRegistry) -> None:
    """Test that when queue depth is exceeded, HTTP 429 Too Many Requests is returned."""
    app = create_app(config_path="hestia.yaml", models_path="models.yaml")

    mock_manager = AsyncMock()
    mock_manager.acquire_admission = AsyncMock(
        side_effect=ServerOverloadedError("llama-3.1-8b-instruct", 16)
    )
    mock_manager.release_admission = MagicMock()

    with (
        patch("hestia.api.app._get_manager", return_value=mock_manager),
        patch("hestia.api.app._get_registry", return_value=mock_registry),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b-instruct",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert resp.status_code == 429
            assert resp.headers.get("retry-after") == "5"
            assert "overloaded" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_admission_queue_timeout_504(mock_registry: ModelRegistry) -> None:
    """Test that when queue wait times out, HTTP 504 Gateway Timeout is returned."""
    app = create_app(config_path="hestia.yaml", models_path="models.yaml")

    mock_manager = AsyncMock()
    mock_manager.acquire_admission = AsyncMock(
        side_effect=QueueTimeoutError("llama-3.1-8b-instruct", 30.0)
    )
    mock_manager.release_admission = MagicMock()

    with (
        patch("hestia.api.app._get_manager", return_value=mock_manager),
        patch("hestia.api.app._get_registry", return_value=mock_registry),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b-instruct",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert resp.status_code == 504
            assert resp.headers.get("retry-after") == "5"
            assert "timed out" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_manager_admission_control_logic() -> None:
    """Direct unit test of ServerManager acquire_admission and release_admission."""
    config = ServiceConfig(
        max_concurrent_requests_per_model=1,
        max_queue_depth_per_model=2,
        queue_timeout_seconds=0.1,
    )
    registry = ModelRegistry()
    registry.register(
        ModelEntry(
            name="test-model",
            model_type=ModelType.CHAT,
            gguf_path="/models/fake.gguf",
            quantization="Q4_K_M",
            context_size=2048,
            vram_estimate_mb=1000,
            gpu_target="rtx-5060-ti",
        )
    )

    manager = ServerManager(config=config, registry=registry, backend=MagicMock())

    # Slot 1: should acquire immediately
    await manager.acquire_admission("test-model")

    # Queue slot 1
    task1 = asyncio.create_task(manager.acquire_admission("test-model"))
    await asyncio.sleep(0.01)

    # Queue slot 2
    task2 = asyncio.create_task(manager.acquire_admission("test-model"))
    await asyncio.sleep(0.01)

    # Queue slot 3: exceeds max_queue_depth_per_model=2 -> raises ServerOverloadedError
    with pytest.raises(ServerOverloadedError):
        await manager.acquire_admission("test-model")

    # Release slot 1 -> task1 should get admission
    manager.release_admission("test-model")
    await task1

    # Task2 will timeout after 0.1s
    with pytest.raises(QueueTimeoutError):
        await task2

    # Clean up task1 admission
    manager.release_admission("test-model")


@pytest.mark.asyncio
async def test_server_manager_graceful_shutdown() -> None:
    """Test that shutdown() cancels and awaits background tasks cleanly."""
    config = ServiceConfig()
    registry = ModelRegistry()
    mock_backend = MagicMock()
    mock_backend.stop = AsyncMock()

    manager = ServerManager(config=config, registry=registry, backend=mock_backend)

    # Fake active process
    process = BackendProcess(model_name="fake-model", pid=1234, port=8001)
    manager._processes["fake-model"] = process

    # Start manager background tasks
    await manager.start()
    assert manager._idle_checker_task is not None
    assert manager._health_checker_task is not None

    # Shutdown
    await manager.shutdown()

    assert manager._idle_checker_task.done()
    assert manager._health_checker_task.done()
    mock_backend.stop.assert_awaited_once_with(process)
    assert len(manager._processes) == 0


@pytest.mark.asyncio
async def test_hestia_security_auth_host_and_content_type(mock_registry: ModelRegistry) -> None:
    """Test Host header check, Content-Type 415, and Bearer token auth enforcement."""
    app = create_app(config_path="hestia.yaml", models_path="models.yaml")

    mock_manager = AsyncMock()
    mock_manager.acquire_admission = AsyncMock()
    mock_manager.release_admission = MagicMock()
    mock_manager.get_all_status = MagicMock(return_value=[])

    with (
        patch("hestia.api.app._get_manager", return_value=mock_manager),
        patch("hestia.api.app._get_registry", return_value=mock_registry),
    ):
        # 1. Invalid Host header returns 400
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://evil.example",
            headers={"Host": "evil.example:8080"},
        ) as client:
            resp = await client.get("/v1/models")
            assert resp.status_code == 400

        # 2. Invalid Content-Type on POST returns 415
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1",
            headers={"Host": "127.0.0.1:8080"},
        ) as client:
            resp = await client.post(
                "/manage/load",
                content=b"model=llama-3.1-8b-instruct",
                headers={"Content-Type": "text/plain"},
            )
            assert resp.status_code == 415

        # 3. Remote client without auth token returns 401, with token returns 200
        with patch.dict(os.environ, {"HESTIA_SERVICE_TOKEN": "secret-123", "METIS_INSECURE_DEV": "0"}):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, client=("192.168.1.100", 12345)),
                base_url="http://127.0.0.1",
                headers={"Host": "127.0.0.1:8080"},
            ) as client:
                # Without token
                resp = await client.get("/models")
                assert resp.status_code == 401

                # With wrong token
                resp = await client.get("/models", headers={"Authorization": "Bearer wrong-token"})
                assert resp.status_code == 401

                # With correct token
                resp = await client.get("/models", headers={"Authorization": "Bearer secret-123"})
                assert resp.status_code == 200

            # 4. Local loopback client (MSS Void) trusts local connection without token
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 12345)),
                base_url="http://127.0.0.1",
                headers={"Host": "127.0.0.1:8080"},
            ) as client:
                resp = await client.get("/models")
                assert resp.status_code == 200

