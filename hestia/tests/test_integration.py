"""Integration tests for the Hestia demo scenario.

Covers:
1. Explicit load of chat model
2. /v1/chat/completions streaming (stream: true)
3. /v1/embeddings lazy loading
4. Status showing both models loaded with GPU info
5. Idle auto-unload
6. Reload on next request
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import yaml

from hestia.api.app import create_app
from hestia.backend import BackendProcess
from hestia.config import ServiceConfig
from hestia.registry import ModelRegistry
from hestia.server import ServerManager


def _integration_ready() -> bool:
    """Check if configured binary and model files exist for hardware integration testing."""
    config_file = Path("hestia.yaml")
    models_file = Path("models.yaml")
    if not config_file.exists() or not models_file.exists():
        return False
    try:
        cfg = ServiceConfig.from_yaml(config_file)
        binary_path = Path(cfg.llamacpp_binary)
        if not binary_path.exists() and not shutil.which(cfg.llamacpp_binary):
            return False

        reg = ModelRegistry()
        reg.load_from_yaml(models_file)
        chat = reg.get("llama-3.1-8b-instruct")
        embed = reg.get("nomic-embed-text")
        if not chat or not embed:
            return False
        return Path(chat.gguf_path).exists() and Path(embed.gguf_path).exists()
    except Exception:
        return False


async def _wait_for_server_ready(
    server: Any, server_task: asyncio.Task[Any], max_wait_sec: float = 5.0
) -> None:
    """Poll for server readiness within max_wait_sec; surface exit exceptions if task terminates."""
    steps = max(1, int(max_wait_sec / 0.1))
    for _ in range(steps):
        if getattr(server, "started", False) or server_task.done():
            break
        await asyncio.sleep(0.1)

    if server_task.done():
        exc = server_task.exception()
        if exc is not None:
            raise RuntimeError(
                f"uvicorn.Server exited with error before becoming ready: {exc}"
            ) from exc
        raise RuntimeError("uvicorn.Server exited unexpectedly before becoming ready")

    assert getattr(server, "started", False), (
        f"uvicorn.Server failed to start within {max_wait_sec}s timeout"
    )


async def _stop_server_task(
    server: Any, server_task: asyncio.Task[Any], timeout: float = 5.0
) -> None:
    """Signal server exit and await completion, cancelling if timeout exceeded."""
    server.should_exit = True
    if not server_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(server_task), timeout=timeout)
        except TimeoutError:
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await server_task
    else:
        with contextlib.suppress(Exception):
            await server_task


@pytest.mark.integration
@pytest.mark.skipif(
    not _integration_ready(),
    reason="Configured llama-server binary and model GGUF files required for live hardware test",
)
@pytest.mark.asyncio
async def test_demo_scenario_live() -> None:
    """Physical hardware live demo: real llama-server processes on configured GPU."""
    import uvicorn
    from click.testing import CliRunner

    from hestia.api.app import _get_manager
    from hestia.cli import cli

    cfg = ServiceConfig.from_yaml(Path("hestia.yaml"))
    app = create_app(config_path="hestia.yaml", models_path="models.yaml")

    server_config = uvicorn.Config(
        app,
        host=cfg.listen_host,
        port=cfg.listen_port,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)
    server_task = asyncio.create_task(server.serve())

    base_url = f"http://{cfg.listen_host}:{cfg.listen_port}"

    try:
        await _wait_for_server_ready(server, server_task, max_wait_sec=5.0)
        async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
            # 1. Load chat model
            resp = await client.post("/manage/load", json={"model": "llama-3.1-8b-instruct"})
            assert resp.status_code == 200

            # 2. Chat completion streaming
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b-instruct",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            ) as stream_resp:
                assert stream_resp.status_code == 200
                chunks = [line async for line in stream_resp.aiter_lines()]
                assert len(chunks) > 0

            # 3. Lazy-load embedding model
            resp = await client.post(
                "/v1/embeddings",
                json={"model": "nomic-embed-text", "input": "Hello world"},
            )
            assert resp.status_code == 200

            # 4. Check status shows both models loaded with GPU info via live CLI
            runner = CliRunner()
            cli_res = await asyncio.to_thread(runner.invoke, cli, ["--url", base_url, "status"])
            assert cli_res.exit_code == 0, (
                f"CLI status command failed (exit code {cli_res.exit_code}): "
                f"{cli_res.output} (exception: {cli_res.exception})"
            )
            assert "llama-3.1-8b-instruct" in cli_res.output
            assert "nomic-embed-text" in cli_res.output

            # 5. Idle auto-unload
            manager = _get_manager()
            manager._last_activity["llama-3.1-8b-instruct"] = 0
            unloaded = await manager.check_idle_models()
            assert "llama-3.1-8b-instruct" in unloaded

            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            chat_info = next(m for m in data["models"] if m["name"] == "llama-3.1-8b-instruct")
            assert chat_info["state"] == "unloaded"

            # 6. Reload on next request
            resp = await client.post("/manage/load", json={"model": "llama-3.1-8b-instruct"})
            assert resp.status_code == 200
            data = (await client.get("/health")).json()
            chat_info = next(m for m in data["models"] if m["name"] == "llama-3.1-8b-instruct")
            assert chat_info["state"] == "loaded"
    finally:
        await _stop_server_task(server, server_task, timeout=5.0)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_demo_scenario_end_to_end() -> None:
    """End-to-end integration test of the full demo workflow with simulated backends."""
    models_yaml = {
        "models": [
            {
                "name": "chat-model",
                "type": "chat",
                "gguf_path": "/fake/chat.gguf",
                "quantization": "Q4_K_M",
                "context_size": 4096,
                "vram_estimate_mb": 5000,
                "gpu_target": "rtx-5060-ti",
            },
            {
                "name": "embed-model",
                "type": "embedding",
                "gguf_path": "/fake/embed.gguf",
                "quantization": "Q8_0",
                "context_size": 2048,
                "vram_estimate_mb": 1000,
                "gpu_target": "rtx-5060-ti",
            },
        ]
    }
    hestia_yaml = {
        "listen_host": "127.0.0.1",
        "listen_port": 8080,
        "llamacpp_binary": "/usr/local/bin/llama-server",
        "backend_port_range": {"start": 8081, "end": 8085},
        "idle_timeout_seconds": 1,
        "log_level": "INFO",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        mpath = Path(tmpdir) / "models.yaml"
        hpath = Path(tmpdir) / "hestia.yaml"
        mpath.write_text(yaml.dump(models_yaml))
        hpath.write_text(yaml.dump(hestia_yaml))

        backend = AsyncMock()
        processes: dict[str, BackendProcess] = {}

        async def fake_start(model, port):
            bp = BackendProcess(model_name=model.name, pid=9999, port=port)
            processes[model.name] = bp
            return bp

        backend.start = AsyncMock(side_effect=fake_start)
        backend.stop = AsyncMock()
        backend.health_check = AsyncMock(return_value=True)

        reg = ModelRegistry()
        reg.load_from_yaml(mpath)
        cfg = ServiceConfig.from_yaml(hpath)
        manager = ServerManager(reg, backend, cfg)

        app = create_app(config_path=str(hpath), models_path=str(mpath))

        mock_stream_client = AsyncMock()
        fake_stream_resp = AsyncMock()
        fake_stream_resp.status_code = 200

        async def fake_lines():
            yield 'data: {"choices": [{"delta": {"content": "Hi"}}]}'
            yield "data: [DONE]"

        fake_stream_resp.aiter_lines = fake_lines
        fake_stream_resp.aclose = AsyncMock()
        fake_stream_resp.aread = AsyncMock(return_value=b"")

        mock_proxy_resp = httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

        with (
            patch("hestia.api.app._get_manager", return_value=manager),
            patch("hestia.api.app._get_registry", return_value=reg),
            patch("hestia.server.get_free_vram_mb", return_value=14000),
            patch("hestia.server.get_total_vram_mb", return_value=16384),
            patch(
                "hestia.api.app._proxy_request",
                new_callable=AsyncMock,
                return_value=mock_proxy_resp,
            ),
            patch(
                "hestia.api.app._open_stream_request",
                new_callable=AsyncMock,
                return_value=(mock_stream_client, fake_stream_resp),
            ),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                # 1. Load chat model explicitly
                resp = await client.post("/manage/load", json={"model": "chat-model"})
                assert resp.status_code == 200
                assert resp.json() == {"status": "loaded", "model": "chat-model"}
                chat_entry = reg.get("chat-model")
                assert chat_entry is not None and chat_entry.state.value == "loaded"

                # 2. Chat completion with streaming
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "chat-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]

                # 3. Lazy-load embedding model via /v1/embeddings
                resp = await client.post(
                    "/v1/embeddings",
                    json={"model": "embed-model", "input": "test"},
                )
                assert resp.status_code == 200
                embed_entry = reg.get("embed-model")
                assert embed_entry is not None and embed_entry.state.value == "loaded"

                # 4. Check status shows both models loaded with GPU info
                resp = await client.get("/health")
                assert resp.status_code == 200
                health_data = resp.json()
                loaded_names = [m["name"] for m in health_data["models"] if m["state"] == "loaded"]
                assert "chat-model" in loaded_names
                assert "embed-model" in loaded_names
                assert "gpu" in health_data
                assert "vram_free_mb" in health_data["gpu"]

                # 5. Idle auto-unload: simulate timeout
                manager._last_activity["chat-model"] = 0
                await manager.check_idle_models()
                assert chat_entry.state.value == "unloaded"

                # 6. Reload on next request
                resp = await client.post("/manage/load", json={"model": "chat-model"})
                assert resp.status_code == 200
                assert chat_entry.state.value == "loaded"


@pytest.mark.asyncio
async def test_stream_error_formatting_with_quotes() -> None:
    """Test that mid-stream error events are serialized safely as JSON with quotes/backslashes."""
    import json
    from unittest.mock import MagicMock

    from hestia.registry import ModelEntry, ModelType, SamplingParams

    app = create_app(config_path="hestia.yaml", models_path="models.yaml")

    mock_registry = MagicMock()
    mock_registry.get.return_value = ModelEntry(
        name="llama-3.1-8b-instruct",
        model_type=ModelType.CHAT,
        gguf_path="/models/fake.gguf",
        quantization="Q4_K_M",
        context_size=8192,
        vram_estimate_mb=5000,
        gpu_target="rtx-5060-ti",
        sampling=SamplingParams(),
    )

    mock_manager = AsyncMock()
    mock_manager.request_finished = MagicMock()
    mock_process = BackendProcess(model_name="llama-3.1-8b-instruct", pid=9999, port=8081)
    mock_manager.ensure_loaded = AsyncMock(return_value=mock_process)

    mock_client = AsyncMock()
    fake_stream_resp = AsyncMock()
    fake_stream_resp.status_code = 200

    error_message = 'Failed: "quote" and \\ backslash in input'

    async def faulty_lines():
        yield 'data: {"choices": [{"delta": {"content": "ok"}}]}'
        raise RuntimeError(error_message)

    fake_stream_resp.aiter_lines = faulty_lines
    fake_stream_resp.aclose = AsyncMock()

    with (
        patch("hestia.api.app._get_manager", return_value=mock_manager),
        patch("hestia.api.app._get_registry", return_value=mock_registry),
        patch(
            "hestia.api.app._open_stream_request",
            new_callable=AsyncMock,
            return_value=(mock_client, fake_stream_resp),
        ),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "llama-3.1-8b-instruct", "messages": [], "stream": True},
            )
            assert resp.status_code == 200
            lines = [line.strip() for line in resp.text.split("\n") if line.strip()]
            error_line = [line for line in lines if line.startswith('data: {"error":')][0]
            payload = json.loads(error_line.removeprefix("data: "))
            assert payload["error"] == error_message


@pytest.mark.asyncio
async def test_open_stream_request_closes_client_on_connect_failure() -> None:
    """Test that _open_stream_request closes its AsyncClient if client.send raises."""
    from hestia.api.app import _open_stream_request

    with patch(
        "httpx.AsyncClient.send",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        with patch.object(httpx.AsyncClient, "aclose", new_callable=AsyncMock) as mock_aclose:
            with pytest.raises(httpx.ConnectError):
                await _open_stream_request("http://127.0.0.1:8081", {"prompt": "hi"})
            mock_aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_api_explicit_unload_concurrent_with_load() -> None:
    """Test concurrent /manage/load and /manage/unload via API results in clean unloaded state."""
    from hestia.api.app import _get_manager, lifespan

    models_yaml = {
        "models": [
            {
                "name": "race-model",
                "type": "chat",
                "gguf_path": "/fake/race.gguf",
                "quantization": "Q4_K_M",
                "context_size": 2048,
                "vram_estimate_mb": 2000,
                "gpu_target": "rtx-5060-ti",
            }
        ]
    }
    hestia_yaml = {
        "listen_host": "127.0.0.1",
        "listen_port": 8080,
        "llamacpp_binary": "/usr/local/bin/llama-server",
        "backend_port_range": {"start": 8081, "end": 8085},
        "idle_timeout_seconds": 300,
        "log_level": "INFO",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        mpath = Path(tmpdir) / "models.yaml"
        hpath = Path(tmpdir) / "hestia.yaml"
        mpath.write_text(yaml.dump(models_yaml))
        hpath.write_text(yaml.dump(hestia_yaml))

        app = create_app(config_path=str(hpath), models_path=str(mpath))
        async with lifespan(app):
            manager = _get_manager()
            backend = manager._backend

            bp = BackendProcess(model_name="race-model", pid=6001, port=8081)

            async def slow_start(*args, **kwargs) -> BackendProcess:
                await asyncio.sleep(0.05)
                return bp

            backend.start = AsyncMock(side_effect=slow_start)
            backend.stop = AsyncMock()

            with (
                patch("hestia.server.get_free_vram_mb", return_value=16000),
                patch("hestia.server.get_total_vram_mb", return_value=16384),
            ):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    load_task = asyncio.create_task(
                        client.post("/manage/load", json={"model": "race-model"})
                    )
                    await asyncio.sleep(0.01)
                    unload_task = asyncio.create_task(
                        client.post("/manage/unload", json={"model": "race-model"})
                    )

                    load_res, unload_res = await asyncio.gather(load_task, unload_task)
                    assert load_res.status_code == 200
                    assert unload_res.status_code == 200

                    health_res = await client.get("/health")
                    assert health_res.status_code == 200
                    data = health_res.json()
                    model_data = next(m for m in data["models"] if m["name"] == "race-model")
                    assert model_data["state"] == "unloaded"


@pytest.mark.asyncio
async def test_live_server_readiness_timeout_cleans_up_server_task() -> None:
    """Test that a server task that fails to become ready is cleaned up in finally."""

    class StalledServer:
        def __init__(self) -> None:
            self.started = False
            self.should_exit = False

        async def serve(self) -> None:
            while not self.should_exit:
                await asyncio.sleep(0.01)

    server = StalledServer()
    server_task = asyncio.create_task(server.serve())

    with pytest.raises(AssertionError, match="failed to start within"):
        try:
            await _wait_for_server_ready(server, server_task, max_wait_sec=0.2)
        finally:
            await _stop_server_task(server, server_task, timeout=1.0)

    assert server.should_exit is True
    assert server_task.done() is True


@pytest.mark.asyncio
async def test_live_server_early_crash_surfaces_exception_and_cleans_up() -> None:
    """Test that an early server crash surfaces the specific exception and cleans up."""

    class CrashingServer:
        def __init__(self) -> None:
            self.started = False
            self.should_exit = False

        async def serve(self) -> None:
            await asyncio.sleep(0.01)
            raise OSError("Address already in use")

    server = CrashingServer()
    server_task = asyncio.create_task(server.serve())

    with pytest.raises(RuntimeError, match="Address already in use"):
        try:
            await _wait_for_server_ready(server, server_task, max_wait_sec=1.0)
        finally:
            await _stop_server_task(server, server_task, timeout=1.0)

    assert server.should_exit is True
    assert server_task.done() is True
