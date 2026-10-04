"""Tests for server manager load locking and placement logic."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, call, patch

import pytest

from hestia.backend import BackendProcess, BackendStartError
from hestia.config import PortRange, ServiceConfig
from hestia.registry import ModelEntry, ModelRegistry, ModelState, ModelType, SamplingParams
from hestia.server import GpuQueryError, ModelNotFoundError, NoFreePortError, ServerManager


def _make_model(name: str, vram_mb: int, model_type: str = "chat") -> ModelEntry:
    return ModelEntry(
        name=name,
        model_type=ModelType(model_type),
        gguf_path=f"/models/{name}.gguf",
        quantization="Q4_K_M",
        context_size=4096,
        vram_estimate_mb=vram_mb,
        gpu_target="rtx-5060-ti",
        sampling=SamplingParams(),
    )


def _make_config(
    port_start: int = 8081, port_end: int = 8083, idle_timeout_seconds: int = 600
) -> ServiceConfig:
    return ServiceConfig(
        listen_port=8080,
        llamacpp_binary="/usr/local/bin/llama-server",
        backend_port_range=PortRange(start=port_start, end=port_end),
        idle_timeout_seconds=idle_timeout_seconds,
        log_level="WARNING",
    )


def _make_backend_mock() -> AsyncMock:
    backend = AsyncMock()
    backend.start = AsyncMock(
        side_effect=lambda model, port: BackendProcess(model_name=model.name, pid=12345, port=port)
    )
    backend.stop = AsyncMock()
    backend.health_check = AsyncMock(return_value=True)
    return backend


class TestLoadLocking:
    """Test that concurrent load requests don't create duplicate processes."""

    @pytest.mark.asyncio
    async def test_concurrent_loads_only_start_once(self) -> None:
        registry = ModelRegistry()
        model = _make_model("test-model", 3000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config()

        manager = ServerManager(registry, backend, config)

        # Patch VRAM to always have enough
        with (
            patch("hestia.server.get_free_vram_mb", return_value=16000),
            patch("hestia.server.get_total_vram_mb", return_value=16384),
        ):
            # Fire 5 concurrent load requests
            results = await asyncio.gather(*[manager.ensure_loaded("test-model") for _ in range(5)])

        # Backend should have been started exactly once
        assert backend.start.call_count == 1
        # All results should be the same process
        assert all(r.port == results[0].port for r in results)

    @pytest.mark.asyncio
    async def test_load_nonexistent_model_raises(self) -> None:
        registry = ModelRegistry()
        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        with pytest.raises(ModelNotFoundError):
            await manager.ensure_loaded("nonexistent")

    @pytest.mark.asyncio
    async def test_concurrent_unload_and_load_race(self) -> None:
        registry = ModelRegistry()
        model = _make_model("race-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        # Initially loaded
        bp1 = BackendProcess(model_name="race-model", pid=3001, port=8081)
        manager._processes["race-model"] = bp1
        manager._port_allocations[8081] = "race-model"
        model.state = ModelState.LOADED
        model.backend_port = 8081
        model.pid = 3001

        bp2 = BackendProcess(model_name="race-model", pid=3002, port=8081)
        backend.start = AsyncMock(return_value=bp2)

        # Slow down backend.stop to simulate realistic teardown delay
        async def slow_stop(proc: BackendProcess) -> None:
            await asyncio.sleep(0.05)

        backend.stop = AsyncMock(side_effect=slow_stop)

        with (
            patch("hestia.server.get_free_vram_mb", return_value=16000),
            patch("hestia.server.get_total_vram_mb", return_value=16384),
        ):
            # Concurrently unload and load
            unload_task = asyncio.create_task(manager._unload_model("race-model"))
            load_task = asyncio.create_task(
                manager.ensure_loaded("race-model", track_inflight=True)
            )

            await asyncio.gather(unload_task, load_task)

        # The model must end up LOADED with the new process and with in-flight tracked
        assert model.state == ModelState.LOADED
        assert manager._processes["race-model"] == bp2
        assert manager.has_inflight("race-model") is True

    @pytest.mark.asyncio
    async def test_explicit_unload_waits_for_concurrent_load(self) -> None:
        registry = ModelRegistry()
        model = _make_model("slow-load-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        bp = BackendProcess(model_name="slow-load-model", pid=4001, port=8081)

        async def slow_start(*args, **kwargs) -> BackendProcess:
            await asyncio.sleep(0.05)
            return bp

        backend.start = AsyncMock(side_effect=slow_start)

        with (
            patch("hestia.server.get_free_vram_mb", return_value=16000),
            patch("hestia.server.get_total_vram_mb", return_value=16384),
        ):
            load_task = asyncio.create_task(manager.ensure_loaded("slow-load-model"))
            await asyncio.sleep(0.01)

            unload_task = asyncio.create_task(manager.unload_model_explicit("slow-load-model"))

            await asyncio.gather(load_task, unload_task)

        assert model.state == ModelState.UNLOADED
        assert "slow-load-model" not in manager._processes
        assert manager._port_allocations[8081] is None
        backend.stop.assert_awaited_once_with(bp)


class TestPortAllocation:
    """Test port allocation from the configured range."""

    @pytest.mark.asyncio
    async def test_allocates_sequential_ports(self) -> None:
        registry = ModelRegistry()
        m1 = _make_model("model-a", 2000)
        m2 = _make_model("model-b", 2000)
        registry._models = {m.name: m for m in [m1, m2]}

        backend = _make_backend_mock()
        config = _make_config(8081, 8083)
        manager = ServerManager(registry, backend, config)

        with (
            patch("hestia.server.get_free_vram_mb", return_value=16000),
            patch("hestia.server.get_total_vram_mb", return_value=16384),
        ):
            p1 = await manager.ensure_loaded("model-a")
            p2 = await manager.ensure_loaded("model-b")

        assert p1.port == 8081
        assert p2.port == 8082

    @pytest.mark.asyncio
    async def test_no_free_port_raises(self) -> None:
        registry = ModelRegistry()
        models = [_make_model(f"m{i}", 1000) for i in range(4)]
        registry._models = {m.name: m for m in models}

        backend = _make_backend_mock()
        # Only 3 ports available (8081, 8082, 8083)
        config = _make_config(8081, 8083)
        manager = ServerManager(registry, backend, config)

        with (
            patch("hestia.server.get_free_vram_mb", return_value=16000),
            patch("hestia.server.get_total_vram_mb", return_value=16384),
        ):
            await manager.ensure_loaded("m0")
            await manager.ensure_loaded("m1")
            await manager.ensure_loaded("m2")

            with pytest.raises(NoFreePortError):
                await manager.ensure_loaded("m3")


class TestInFlightTracking:
    """Test in-flight request tracking."""

    @pytest.mark.asyncio
    async def test_inflight_count(self) -> None:
        registry = ModelRegistry()
        model = _make_model("test", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        with (
            patch("hestia.server.get_free_vram_mb", return_value=16000),
            patch("hestia.server.get_total_vram_mb", return_value=16384),
        ):
            await manager.ensure_loaded("test")

        manager.request_started("test")
        manager.request_started("test")
        assert manager.has_inflight("test") is True

        manager.request_finished("test")
        assert manager.has_inflight("test") is True

        manager.request_finished("test")
        assert manager.has_inflight("test") is False

    @pytest.mark.asyncio
    async def test_idle_unload_aborted_if_model_becomes_active_while_waiting_for_lock(self) -> None:
        registry = ModelRegistry()
        model = _make_model("idle-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config(idle_timeout_seconds=10)
        manager = ServerManager(registry, backend, config)

        bp = BackendProcess(model_name="idle-model", pid=5001, port=8081)
        manager._processes["idle-model"] = bp
        manager._port_allocations[8081] = "idle-model"
        model.state = ModelState.LOADED
        model.backend_port = 8081
        model.pid = 5001

        manager._last_activity["idle-model"] = time.monotonic() - 100

        lock = manager._load_locks.setdefault("idle-model", asyncio.Lock())
        await lock.acquire()

        idle_task = asyncio.create_task(manager.check_idle_models())
        await asyncio.sleep(0.01)

        manager._touch("idle-model")
        lock.release()

        unloaded = await idle_task

        assert "idle-model" not in unloaded
        assert model.state == ModelState.LOADED
        assert manager._processes["idle-model"] is bp
        backend.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_idle_unload_aborted_if_inflight_finishes_during_drain(self) -> None:
        registry = ModelRegistry()
        model = _make_model("idle-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config(idle_timeout_seconds=10)
        manager = ServerManager(registry, backend, config)

        bp = BackendProcess(model_name="idle-model", pid=5001, port=8081)
        manager._processes["idle-model"] = bp
        manager._port_allocations[8081] = "idle-model"
        model.state = ModelState.LOADED
        model.backend_port = 8081
        model.pid = 5001

        manager._last_activity["idle-model"] = time.monotonic() - 100
        manager._inflight["idle-model"] = 1

        async def fake_sleep(duration: float) -> None:
            manager.request_finished("idle-model")

        with patch("asyncio.sleep", side_effect=fake_sleep):
            did_unload = await manager._unload_model("idle-model", is_idle_check=True)

        assert did_unload is False
        assert model.state == ModelState.LOADED
        assert manager._processes["idle-model"] is bp
        backend.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_unload_drains_inflight_and_unloads(self) -> None:
        registry = ModelRegistry()
        model = _make_model("busy-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config(idle_timeout_seconds=10)
        manager = ServerManager(registry, backend, config)

        bp = BackendProcess(model_name="busy-model", pid=5002, port=8081)
        manager._processes["busy-model"] = bp
        manager._port_allocations[8081] = "busy-model"
        model.state = ModelState.LOADED
        model.backend_port = 8081
        model.pid = 5002

        manager._inflight["busy-model"] = 1

        async def fake_sleep(duration: float) -> None:
            manager.request_finished("busy-model")

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await manager.unload_model_explicit("busy-model")

        assert model.state == ModelState.UNLOADED
        assert "busy-model" not in manager._processes
        assert manager._port_allocations[8081] is None
        backend.stop.assert_awaited_once_with(bp)


class TestEnsureLoadedInFlight:
    """Test atomic in-flight increment in ensure_loaded."""

    @pytest.mark.asyncio
    async def test_ensure_loaded_with_track_inflight(self) -> None:
        registry = ModelRegistry()
        model = _make_model("test-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        with (
            patch("hestia.server.get_free_vram_mb", return_value=16000),
            patch("hestia.server.get_total_vram_mb", return_value=16384),
        ):
            await manager.ensure_loaded("test-model", track_inflight=True)

        assert manager.has_inflight("test-model") is True
        manager.request_finished("test-model")
        assert manager.has_inflight("test-model") is False


class TestCrashHandling:
    """Test process cleanup and retry backoff behavior on crashes."""

    @pytest.mark.asyncio
    async def test_wedged_process_stopped_at_max_retries(self) -> None:
        registry = ModelRegistry()
        model = _make_model("crashed-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        # Set retries to 3 (max)
        manager._restart_counts["crashed-model"] = 3
        # Register a wedged process
        bp = BackendProcess(model_name="crashed-model", pid=8888, port=8081)
        manager._processes["crashed-model"] = bp
        manager._port_allocations[8081] = "crashed-model"

        await manager._handle_crash(model)

        # Process should be stopped and port released
        backend.stop.assert_awaited_once_with(bp)
        assert "crashed-model" not in manager._processes
        assert manager._port_allocations[8081] is None
        assert model.state == ModelState.FAILED

    @pytest.mark.asyncio
    async def test_restart_counter_resets_on_success(self) -> None:
        registry = ModelRegistry()
        model = _make_model("flaky-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        # Previous 2 failures
        manager._restart_counts["flaky-model"] = 2
        bp = BackendProcess(model_name="flaky-model", pid=7777, port=8081)
        manager._processes["flaky-model"] = bp
        manager._port_allocations[8081] = "flaky-model"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await manager._handle_crash(model)

        # Should be reset to 0 after successful restart
        assert manager._restart_counts["flaky-model"] == 0
        assert model.state == ModelState.LOADED

    @pytest.mark.asyncio
    async def test_crash_retry_fails_twice_then_succeeds(self) -> None:
        registry = ModelRegistry()
        model = _make_model("flaky-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        bp_crashed = BackendProcess(model_name="flaky-model", pid=1001, port=8081)
        manager._processes["flaky-model"] = bp_crashed
        manager._port_allocations[8081] = "flaky-model"
        model.state = ModelState.LOADED
        model.backend_port = 8081
        model.pid = 1001

        bp_restarted = BackendProcess(model_name="flaky-model", pid=1002, port=8081)
        backend.start = AsyncMock(
            side_effect=[
                BackendStartError("flaky-model", "fail attempt 1"),
                BackendStartError("flaky-model", "fail attempt 2"),
                bp_restarted,
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await manager._handle_crash(model)

        assert backend.start.call_count == 3
        # Backoffs were 1s, 2s, 4s (2**0, 2**1, 2**2)
        assert mock_sleep.await_count == 3
        assert mock_sleep.await_args_list == [call(1), call(2), call(4)]
        assert model.state == ModelState.LOADED
        assert manager._processes["flaky-model"] == bp_restarted
        assert manager._restart_counts["flaky-model"] == 0
        assert manager._port_allocations[8081] == "flaky-model"

    @pytest.mark.asyncio
    async def test_crash_retry_fails_all_three_attempts(self) -> None:
        registry = ModelRegistry()
        model = _make_model("dead-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        bp_crashed = BackendProcess(model_name="dead-model", pid=2001, port=8081)
        manager._processes["dead-model"] = bp_crashed
        manager._port_allocations[8081] = "dead-model"
        model.state = ModelState.LOADED
        model.backend_port = 8081
        model.pid = 2001

        backend.start = AsyncMock(side_effect=BackendStartError("dead-model", "persistent failure"))

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await manager._handle_crash(model)

        assert backend.start.call_count == 3
        assert mock_sleep.await_count == 3
        assert mock_sleep.await_args_list == [call(1), call(2), call(4)]
        assert model.state == ModelState.FAILED
        assert model.backend_port is None
        assert model.pid is None
        assert "dead-model" not in manager._processes
        assert manager._port_allocations[8081] is None
        assert manager._restart_counts["dead-model"] == 3

    @pytest.mark.asyncio
    async def test_stale_health_check_does_not_restart_newer_process(self) -> None:
        registry = ModelRegistry()
        model = _make_model("test-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        stale_proc = BackendProcess(model_name="test-model", pid=1001, port=8081)
        newer_proc = BackendProcess(model_name="test-model", pid=1002, port=8081)

        manager._processes["test-model"] = newer_proc
        manager._port_allocations[8081] = "test-model"
        model.state = ModelState.LOADED
        model.backend_port = 8081
        model.pid = 1002

        await manager._handle_crash(model, failed_process=stale_proc)

        backend.stop.assert_not_called()
        assert manager._processes["test-model"] is newer_proc
        assert model.state == ModelState.LOADED
        assert model.pid == 1002


class TestVramBudgetFit:
    """Test aggregate VRAM budget checking against total card capacity."""

    @pytest.mark.asyncio
    async def test_aggregate_reservation_exceeds_total(self) -> None:
        registry = ModelRegistry()
        m1 = _make_model("m1", 10000)
        m2 = _make_model("m2", 8000)
        registry._models = {m.name: m for m in [m1, m2]}

        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        with (
            patch("hestia.server.get_free_vram_mb", return_value=16000),
            patch("hestia.server.get_total_vram_mb", return_value=16384),
        ):
            # Load m1 (10000 MB reserved out of 16384 MB total)
            await manager.ensure_loaded("m1")
            # Mark m1 as in-flight so it cannot be evicted
            manager.request_started("m1")

            # Try to fit m2 (needs 8000 MB, free is 10000 MB;
            # 10000 + 8000 > 16384 total and m1 is in-flight so cannot evict)
            with patch("hestia.server.get_free_vram_mb", return_value=10000):
                fits = await manager._try_fit(m2)
                assert fits is False

    @pytest.mark.asyncio
    async def test_gpu_query_error_raised_when_vram_query_fails(self) -> None:
        registry = ModelRegistry()
        model = _make_model("test-model", 2000)
        registry._models = {model.name: model}

        backend = _make_backend_mock()
        config = _make_config()
        manager = ServerManager(registry, backend, config)

        with patch("hestia.server.get_total_vram_mb", return_value=0):
            with pytest.raises(GpuQueryError, match="failed to query GPU VRAM capacity"):
                await manager.ensure_loaded("test-model")
