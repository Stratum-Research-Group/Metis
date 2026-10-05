"""Server manager — orchestrates model loading, unloading, and VRAM budgets."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from hestia.backend import BackendProcess, BackendStartError
from hestia.device_lock import GpuDeviceLease, acquire_device_lease
from hestia.registry import ModelEntry, ModelState
from hestia.vram import get_free_vram_mb, get_total_vram_mb

if TYPE_CHECKING:
    from hestia.backend import Backend
    from hestia.config import ServiceConfig
    from hestia.registry import ModelRegistry

logger = logging.getLogger(__name__)

# Maximum restart attempts before marking a model as failed
_MAX_RESTART_RETRIES = 3


class ServerManager:
    """Manages backend processes for registered models.

    Responsibilities:
    - Allocate ports from the configured range
    - Enforce VRAM budget (aggregate reservations + live free check)
    - Lazy-load on first request with per-model load locks
    - Idle timeout auto-unload
    - Crash detection with exponential backoff restarts
    """

    def __init__(
        self,
        registry: ModelRegistry,
        backend: Backend,
        config: ServiceConfig,
    ) -> None:
        self._registry = registry
        self._backend = backend
        self._config = config

        # port -> model_name (None = free)
        self._port_allocations: dict[int, str | None] = {
            p: None for p in config.backend_port_range.ports
        }

        # model_name -> BackendProcess
        self._processes: dict[str, BackendProcess] = {}

        # model_name -> last request timestamp
        self._last_activity: dict[str, float] = {}

        # model_name -> asyncio.Lock (prevent duplicate loads)
        self._load_locks: dict[str, asyncio.Lock] = {}

        # model_name -> restart count
        self._restart_counts: dict[str, int] = {}

        # model_name -> in-flight request count
        self._inflight: dict[str, int] = {}

        # Concurrency and admission queue tracking
        self._model_semaphores: dict[str, asyncio.Semaphore] = {}
        self._model_queue_depths: dict[str, int] = {}

        # device -> active GPU device lease
        self._device_leases: dict[str, GpuDeviceLease] = {}
        # device -> count of loaded models using this device
        self._device_model_counts: dict[str, int] = {}

        # Global admission lock for atomic VRAM fit checks, evictions, and startups
        self._admission_lock: asyncio.Lock = asyncio.Lock()

        # Background task handles
        self._idle_checker_task: asyncio.Task[None] | None = None
        self._health_checker_task: asyncio.Task[None] | None = None

    # ── Lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        """Start background tasks (idle checker, health checker)."""
        self._idle_checker_task = asyncio.create_task(self._idle_check_loop())
        self._health_checker_task = asyncio.create_task(self._health_check_loop())
        logger.info("Server manager started")

    async def shutdown(self) -> None:
        """Stop all loaded models and background tasks cleanly."""
        tasks = []
        if self._idle_checker_task:
            self._idle_checker_task.cancel()
            tasks.append(self._idle_checker_task)
        if self._health_checker_task:
            self._health_checker_task.cancel()
            tasks.append(self._health_checker_task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        for model_name in list(self._processes.keys()):
            try:
                await asyncio.wait_for(self._unload_model(model_name, force=True), timeout=35.0)
            except Exception as e:
                logger.warning("Error unloading model %s during shutdown: %s", model_name, e)

        if self._processes:
            unresolved = list(self._processes.keys())
            logger.error("Unresolved live model processes after shutdown: %s", unresolved)
        else:
            logger.info("Server manager shut down cleanly")

    # ── Model Loading ─────────────────────────────────────────

    async def ensure_loaded(self, model_name: str, track_inflight: bool = False) -> BackendProcess:
        """Ensure a model is loaded, lazy-loading if needed.

        This is the primary entry point for the API layer.
        Thread-safe via per-model locks.
        """
        model = self._registry.get(model_name)
        if model is None:
            raise ModelNotFoundError(model_name)

        if model.state == ModelState.FAILED:
            raise ModelFailedError(model_name)

        # Acquire per-model lock to serialize load and unload operations
        lock = self._load_locks.setdefault(model_name, asyncio.Lock())
        async with lock:
            # Recheck state after acquiring lock
            if model.state == ModelState.FAILED:
                raise ModelFailedError(model_name)

            if model.state == ModelState.LOADED and model_name in self._processes:
                self._touch(model_name)
                if track_inflight:
                    self.request_started(model_name)
                return self._processes[model_name]

            process = await self._load_model(model)
            if track_inflight:
                self.request_started(model_name)
            return process

    async def load_model_explicit(self, model_name: str) -> BackendProcess:
        """Explicitly load a model (CLI command)."""
        return await self.ensure_loaded(model_name, track_inflight=False)

    async def unload_model_explicit(self, model_name: str) -> None:
        """Explicitly unload a model (CLI command)."""
        model = self._registry.get(model_name)
        if model is None:
            raise ModelNotFoundError(model_name)
        await self._unload_model(model_name)

    # ── In-flight tracking ────────────────────────────────────

    def request_started(self, model_name: str) -> None:
        """Mark a request as in-flight for a model."""
        self._inflight[model_name] = self._inflight.get(model_name, 0) + 1
        self._touch(model_name)

    def request_finished(self, model_name: str) -> None:
        """Mark a request as completed for a model."""
        count = self._inflight.get(model_name, 0)
        self._inflight[model_name] = max(0, count - 1)
        self._touch(model_name)

    def has_inflight(self, model_name: str) -> bool:
        """Check if a model has in-flight requests."""
        return self._inflight.get(model_name, 0) > 0

    # ── Admission & Concurrency Queue Control ─────────────────

    async def acquire_admission(self, model_name: str) -> None:
        """Acquire a concurrency execution slot with bounded queue depth and timeout."""
        max_queue = self._config.max_queue_depth_per_model
        timeout = self._config.queue_timeout_seconds

        queue_count = self._model_queue_depths.get(model_name, 0)
        if queue_count >= max_queue:
            raise ServerOverloadedError(model_name, queue_count)

        self._model_queue_depths[model_name] = queue_count + 1

        sem = self._model_semaphores.setdefault(
            model_name,
            asyncio.Semaphore(self._config.max_concurrent_requests_per_model),
        )

        slot_acquired = False
        try:
            await asyncio.wait_for(sem.acquire(), timeout=timeout)
            slot_acquired = True
        except TimeoutError as exc:
            raise QueueTimeoutError(model_name, timeout) from exc
        finally:
            prev_depth = self._model_queue_depths.get(model_name, 1)
            self._model_queue_depths[model_name] = max(0, prev_depth - 1)
            if not slot_acquired and sem.locked():
                pass

    def release_admission(self, model_name: str) -> None:
        """Release a concurrency execution slot."""
        sem = self._model_semaphores.get(model_name)
        if sem:
            sem.release()

    # ── Internal ──────────────────────────────────────────────

    async def _load_model(self, model: ModelEntry) -> BackendProcess:
        """Load a model, evicting idle models if necessary to fit."""
        async with self._admission_lock:
            # Check VRAM budget
            if not await self._try_fit(model):
                raise InsufficientVramError(model.name, model.vram_estimate_mb)

            # Acquire or reference exclusive GPU device lease for this device target
            device = model.gpu_target
            if device not in self._device_leases:
                lease = acquire_device_lease(device=device, owner="hestia")
                if lease is None:
                    raise GpuDeviceConflictError(model.name, device)
                self._device_leases[device] = lease
            self._device_model_counts[device] = self._device_model_counts.get(device, 0) + 1

            port = self._allocate_port(model.name)
            if port is None:
                cnt = self._device_model_counts.get(device, 1) - 1
                if cnt <= 0:
                    self._device_model_counts.pop(device, None)
                    l = self._device_leases.pop(device, None)
                    if l:
                        l.release()
                else:
                    self._device_model_counts[device] = cnt
                raise NoFreePortError(model.name)

            model.state = ModelState.LOADING
            try:
                process = await self._backend.start(model, port)
            except BaseException:
                model.state = ModelState.UNLOADED
                self._release_port(port)
                cnt = self._device_model_counts.get(device, 1) - 1
                if cnt <= 0:
                    self._device_model_counts.pop(device, None)
                    l = self._device_leases.pop(device, None)
                    if l:
                        l.release()
                else:
                    self._device_model_counts[device] = cnt
                raise
            model.state = ModelState.LOADED
            model.backend_port = port
            model.pid = process.pid
            self._processes[model.name] = process
            self._restart_counts[model.name] = 0
            self._touch(model.name)

            logger.info(
                "Model %s loaded on port %d (VRAM: ~%d MB, device: %s)",
                model.name,
                port,
                model.vram_estimate_mb,
                model.gpu_target,
            )
            return process

    async def _unload_model(
        self,
        model_name: str,
        force: bool = False,
        is_idle_check: bool = False,
    ) -> bool:
        """Unload a model, draining in-flight requests first (unless force)."""
        lock = self._load_locks.setdefault(model_name, asyncio.Lock())
        async with lock:
            model = self._registry.get(model_name)
            is_loaded = model is not None and model.state == ModelState.LOADED
            if not is_loaded and model_name not in self._processes:
                logger.info("Model %s is not loaded", model_name)
                return False

            if is_idle_check:
                now = time.monotonic()
                last = self._last_activity.get(model_name, now)
                timeout = self._config.idle_timeout_seconds
                if (now - last) < timeout:
                    logger.info(
                        "Model %s became active while waiting for lock (idle %ds < %ds); "
                        "aborting auto-unload",
                        model_name,
                        int(now - last),
                        timeout,
                    )
                    return False

            if not force and self.has_inflight(model_name):
                logger.info("Waiting for in-flight requests to drain for %s", model_name)
                for _ in range(60):  # Wait up to 60 seconds
                    if not self.has_inflight(model_name):
                        break
                    await asyncio.sleep(1)
                if self.has_inflight(model_name):
                    logger.warning(
                        "Model %s still has in-flight requests; aborting unload",
                        model_name,
                    )
                    return False

            if is_idle_check:
                now = time.monotonic()
                last = self._last_activity.get(model_name, now)
                timeout = self._config.idle_timeout_seconds
                if (now - last) < timeout:
                    logger.info(
                        "Model %s became active during in-flight drain (idle %ds < %ds); "
                        "aborting auto-unload",
                        model_name,
                        int(now - last),
                        timeout,
                    )
                    return False

            process = self._processes.get(model_name)
            if process:
                try:
                    await self._backend.stop(process)
                finally:
                    self._processes.pop(model_name, None)
                    self._release_port(process.port)

            if model:
                model.state = ModelState.UNLOADED
                model.backend_port = None
                model.pid = None
                device = model.gpu_target
                cnt = self._device_model_counts.get(device, 1) - 1
                if cnt <= 0:
                    self._device_model_counts.pop(device, None)
                    l = self._device_leases.pop(device, None)
                    if l:
                        l.release()
                else:
                    self._device_model_counts[device] = cnt

            self._last_activity.pop(model_name, None)
            self._inflight.pop(model_name, None)
            logger.info("Model %s unloaded", model_name)
            return True

    async def _try_fit(self, model: ModelEntry) -> bool:
        """Check if the model fits in VRAM, evicting idle models if needed."""
        reserved = self._registry.total_vram_reserved_mb
        free = get_free_vram_mb()
        needed = model.vram_estimate_mb
        total = get_total_vram_mb()
        if total <= 0:
            raise GpuQueryError(f"Cannot load '{model.name}': failed to query GPU VRAM capacity")

        # Fast path: it fits without eviction and aggregate reservation stays within total
        if needed <= free and (reserved + needed) <= total:
            return True

        # Try evicting longest-idle models that have no in-flight requests
        idle_models = self._get_eviction_candidates()
        for candidate in idle_models:
            fits = (
                needed <= get_free_vram_mb()
                and (self._registry.total_vram_reserved_mb + needed) <= total
            )
            if fits:
                return True
            logger.info("Evicting idle model %s to make room for %s", candidate.name, model.name)
            await self._unload_model(candidate.name)

        # Final check after evictions
        return (
            needed <= get_free_vram_mb()
            and (self._registry.total_vram_reserved_mb + needed) <= total
        )

    def _get_eviction_candidates(self) -> list[ModelEntry]:
        """Get loaded models sorted by oldest last activity, excluding those with in-flight."""
        candidates = []
        for model in self._registry.loaded_models():
            if not self.has_inflight(model.name):
                candidates.append(model)

        # Sort by last activity (oldest first)
        candidates.sort(key=lambda m: self._last_activity.get(m.name, 0))
        return candidates

    def _touch(self, model_name: str) -> None:
        """Update last activity timestamp."""
        self._last_activity[model_name] = time.monotonic()

    def _is_port_available(self, port: int) -> bool:
        """Probe if a port is actually available to bind on the host OS."""
        import socket
        host = self._config.listen_host
        if host == "localhost":
            host = "127.0.0.1"
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                return True
        except OSError:
            return False

    def _allocate_port(self, model_name: str) -> int | None:
        """Allocate the next free port from the configured range."""
        for port, owner in self._port_allocations.items():
            if owner is None:
                if not self._is_port_available(port):
                    logger.warning(
                        "Port %d is free in Hestia map but occupied on host OS, skipping",
                        port,
                    )
                    continue
                self._port_allocations[port] = model_name
                return port
        return None

    def _release_port(self, port: int) -> None:
        """Release a port back to the pool."""
        if port in self._port_allocations:
            self._port_allocations[port] = None

    # ── Background Tasks ──────────────────────────────────────

    async def check_idle_models(self) -> list[str]:
        """Check and unload models that have exceeded the idle timeout."""
        now = time.monotonic()
        timeout = self._config.idle_timeout_seconds
        unloaded: list[str] = []

        for model in self._registry.loaded_models():
            last = self._last_activity.get(model.name, now)
            if (now - last) >= timeout and not self.has_inflight(model.name):
                logger.info(
                    "Auto-unloading %s after %ds idle",
                    model.name,
                    int(now - last),
                )
                did_unload = await self._unload_model(model.name, is_idle_check=True)
                if did_unload:
                    unloaded.append(model.name)

        return unloaded

    async def _idle_check_loop(self) -> None:
        """Periodically unload models that have been idle too long."""
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            await self.check_idle_models()

    async def _health_check_loop(self) -> None:
        """Periodically health-check loaded models, restarting on failure."""
        while True:
            await asyncio.sleep(15)  # Check every 15 seconds

            for model in self._registry.loaded_models():
                process = self._processes.get(model.name)
                if process is None:
                    continue

                healthy = await self._backend.health_check(process)
                if not healthy:
                    logger.warning("Health check failed for %s", model.name)
                    await self._handle_crash(model, failed_process=process)

    async def _handle_crash(
        self,
        model: ModelEntry,
        failed_process: BackendProcess | None = None,
    ) -> None:
        """Handle a crashed backend: restart with exponential backoff."""
        lock = self._load_locks.setdefault(model.name, asyncio.Lock())
        async with lock:
            current_proc = self._processes.get(model.name)
            # Stale check: if a specific process was checked, make sure it is still current
            if failed_process is not None and (current_proc is not failed_process):
                logger.info(
                    "Ignoring stale health check failure for %s (current: %s, checked: %s)",
                    model.name,
                    current_proc,
                    failed_process,
                )
                return

            if model.state != ModelState.LOADED and current_proc is None:
                logger.info("Model %s is no longer loaded, skipping crash handling", model.name)
                return

            if self._restart_counts.get(model.name, 0) >= _MAX_RESTART_RETRIES:
                logger.error(
                    "Model %s exceeded max restart attempts (%d), marking as FAILED",
                    model.name,
                    _MAX_RESTART_RETRIES,
                )
                model.state = ModelState.FAILED
                model.backend_port = None
                model.pid = None
                process = self._processes.pop(model.name, None)
                if process:
                    await self._backend.stop(process)
                    self._release_port(process.port)
                return

            # Clean up old/wedged process if present
            old_process = self._processes.pop(model.name, None)
            if old_process:
                await self._backend.stop(old_process)
                self._release_port(old_process.port)

            model.state = ModelState.LOADING
            model.backend_port = None
            model.pid = None

            start_attempt = self._restart_counts.get(model.name, 0)
            for attempt in range(start_attempt, _MAX_RESTART_RETRIES):
                self._restart_counts[model.name] = attempt + 1
                backoff = 2**attempt  # 1s, 2s, 4s
                logger.info(
                    "Restarting %s (attempt %d/%d) after %ds backoff",
                    model.name,
                    attempt + 1,
                    _MAX_RESTART_RETRIES,
                    backoff,
                )
                await asyncio.sleep(backoff)

                port = self._allocate_port(model.name)
                if port is None:
                    logger.error("No free port available to restart %s", model.name)
                    break

                try:
                    process = await self._backend.start(model, port)
                    self._processes[model.name] = process
                    model.state = ModelState.LOADED
                    model.backend_port = port
                    model.pid = process.pid
                    self._touch(model.name)
                    self._restart_counts[model.name] = 0
                    logger.info("Successfully restarted %s on port %d", model.name, port)
                    return
                except BackendStartError as exc:
                    logger.warning(
                        "Restart attempt %d/%d failed for %s: %s",
                        attempt + 1,
                        _MAX_RESTART_RETRIES,
                        model.name,
                        exc,
                    )
                    self._release_port(port)
                    failed_proc = self._processes.pop(model.name, None)
                    if failed_proc:
                        await self._backend.stop(failed_proc)
                        self._release_port(failed_proc.port)

            logger.error(
                "Model %s exceeded max restart attempts (%d), marking as FAILED",
                model.name,
                _MAX_RESTART_RETRIES,
            )
            model.state = ModelState.FAILED
            model.backend_port = None
            model.pid = None
            process = self._processes.pop(model.name, None)
            if process:
                await self._backend.stop(process)
                self._release_port(process.port)

    # ── Status ────────────────────────────────────────────────

    def get_model_status(self, model_name: str) -> dict:
        """Get status info for a single model."""
        model = self._registry.get(model_name)
        if model is None:
            raise ModelNotFoundError(model_name)

        return {
            "name": model.name,
            "type": model.model_type.value,
            "state": model.state.value,
            "vram_estimate_mb": model.vram_estimate_mb,
            "port": model.backend_port,
            "pid": model.pid,
            "inflight_requests": self._inflight.get(model_name, 0),
        }

    def get_all_status(self) -> list[dict]:
        """Get status info for all models."""
        return [self.get_model_status(m.name) for m in self._registry.list_all()]


# ── Exceptions ────────────────────────────────────────────────


class ModelNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Model '{name}' not found in registry")
        self.model_name = name


class ModelFailedError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Model '{name}' is in FAILED state and cannot be loaded")
        self.model_name = name


class InsufficientVramError(Exception):
    def __init__(self, name: str, needed_mb: int) -> None:
        super().__init__(f"Not enough VRAM to load '{name}' (needs ~{needed_mb} MB)")
        self.model_name = name
        self.needed_mb = needed_mb


class NoFreePortError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"No free backend port available to load '{name}'")
        self.model_name = name


class GpuQueryError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GpuDeviceConflictError(Exception):
    def __init__(self, name: str, device: str) -> None:
        super().__init__(
            f"Cannot load '{name}': GPU device '{device}' is locked by another process (e.g. training)"
        )
        self.model_name = name
        self.device = device


class ServerOverloadedError(Exception):
    def __init__(self, name: str, queue_depth: int) -> None:
        msg = f"Model '{name}' queue is full ({queue_depth} requests queued). Server overloaded."
        super().__init__(msg)
        self.model_name = name
        self.queue_depth = queue_depth


class QueueTimeoutError(Exception):
    def __init__(self, name: str, timeout_seconds: float) -> None:
        msg = f"Timed out waiting for execution slot for '{name}' after {timeout_seconds}s"
        super().__init__(msg)
        self.model_name = name
        self.timeout_seconds = timeout_seconds
