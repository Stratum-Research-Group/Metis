"""llama.cpp server backend implementation."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shutil
import subprocess
from pathlib import Path

import httpx

from hestia.backend import Backend, BackendProcess, BackendStartError
from hestia.registry import ModelEntry, ModelType

logger = logging.getLogger(__name__)

# How long to wait for the backend to become healthy after launch
_STARTUP_TIMEOUT_SECONDS = 120
_HEALTH_POLL_INTERVAL = 1.0


class LlamaCppBackend(Backend):
    """Manages llama.cpp server (llama-server) processes."""

    @classmethod
    def discover_binary(cls, binary_path: str | None = None) -> Path:
        """Resolve the llama-server binary path using configured and fallback search locations."""
        if binary_path:
            resolved = Path(binary_path)
            if resolved.is_file():
                return resolved

        env_bin = os.environ.get("HESTIA_LLAMACPP_BINARY")
        if env_bin and Path(env_bin).is_file():
            return Path(env_bin)

        which_bin = shutil.which("llama-server")
        if which_bin and Path(which_bin).is_file():
            return Path(which_bin)

        for candidate in (
            Path("/usr/local/bin/llama-server"),
            Path.home() / ".unsloth/llama.cpp/llama-server",
            Path("/usr/bin/llama-server"),
            Path("/usr/lib/ollama/llama-server"),
        ):
            if candidate.is_file():
                return candidate

        return Path(binary_path or "/usr/local/bin/llama-server")

    @staticmethod
    def resolve_cuda_visible_devices(
        gpu_target: str,
        gpu_count: int | None = None,
    ) -> str:
        """Resolve a model's gpu_target into a valid CUDA_VISIBLE_DEVICES string.

        Supports dual-GPU default assignment (V100 on 0, RTX 5060 Ti on 1) with
        automatic dynamic fallback to GPU 0 for RTX 5060 Ti when only a single
        GPU is installed.
        """
        target_clean = gpu_target.strip().lower()
        known_gpu_map = {
            "v100": "0",
            "tesla-v100": "0",
            "tesla_v100": "0",
            "rtx-5060-ti": "1",
            "rtx-5060ti": "1",
            "rtx5060ti": "1",
            "0": "0",
            "1": "1",
            "cuda:0": "0",
            "cuda:1": "1",
        }
        if target_clean in ("rtx-5060-ti", "rtx-5060ti", "rtx5060ti"):
            if gpu_count is None:
                from hestia.vram import get_gpu_count

                gpu_count = get_gpu_count()
            if gpu_count == 1:
                return "0"
            return "1"

        if target_clean in known_gpu_map:
            return known_gpu_map[target_clean]

        if (
            target_clean.startswith("gpu-")
            or target_clean.isdigit()
            or target_clean.startswith("cuda:")
        ):
            return gpu_target.strip()

        raise ValueError(
            f"Unknown or invalid gpu_target '{gpu_target}'. "
            f"Expected one of {sorted(known_gpu_map.keys())} or numeric GPU index / GPU UUID."
        )

    def __init__(
        self,
        binary_path: str = "/usr/local/bin/llama-server",
        log_dir: str = "logs",
    ) -> None:
        self._binary = self.discover_binary(binary_path)
        self._log_dir = Path(log_dir)
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._log_files: dict[str, io.BufferedWriter] = {}

    def build_launch_args(self, model: ModelEntry, port: int) -> list[str]:
        """Derive llama-server CLI flags from the model's registry fields."""
        args = [
            str(self._binary),
            "--model",
            str(model.gguf_path),
            "--ctx-size",
            str(model.context_size),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            # Full GPU offload — all layers to GPU
            "-ngl",
            "999",
        ]

        # Enable embedding endpoint for embedding-type models
        if model.model_type == ModelType.EMBEDDING:
            args.append("--embedding")
        elif model.model_type == ModelType.CHAT:
            args.append("--skip-chat-parsing")

        return args

    async def start(self, model: ModelEntry, port: int) -> BackendProcess:
        """Launch a llama-server process and wait for it to become healthy."""
        if not Path(model.gguf_path).is_file():
            raise BackendStartError(
                model.name,
                f"Model weights not found at '{model.gguf_path}'. "
                "Please configure a valid GGUF path in models.yaml.",
            )
        if not self._binary.is_file():
            raise BackendStartError(
                model.name,
                f"llama-server binary not found at '{self._binary}'. "
                "Please install llama.cpp or set HESTIA_LLAMACPP_BINARY.",
            )

        args = self.build_launch_args(model, port)
        logger.info("Starting llama.cpp for %s: %s", model.name, " ".join(args))

        self._log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self._log_dir / f"{model.name}.log"
        log_file = open(log_path, "ab")

        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if model.gpu_target:
            try:
                env["CUDA_VISIBLE_DEVICES"] = self.resolve_cuda_visible_devices(model.gpu_target)
            except ValueError as exc:
                log_file.close()
                raise BackendStartError(model.name, str(exc)) from exc

        try:
            proc = subprocess.Popen(
                args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                # Don't let the child inherit our signals
                start_new_session=True,
            )
        except FileNotFoundError:
            log_file.close()
            raise BackendStartError(
                model.name,
                f"llama-server binary not found at {self._binary}",
            )
        except OSError as exc:
            log_file.close()
            raise BackendStartError(model.name, str(exc))

        self._processes[model.name] = proc
        self._log_files[model.name] = log_file
        bp = BackendProcess(model_name=model.name, pid=proc.pid, port=port)

        # Wait for the health endpoint to respond
        if not await self._wait_healthy(bp):
            # Clean up on failure
            await self.stop(bp)
            raise BackendStartError(
                model.name,
                f"Backend did not become healthy within {_STARTUP_TIMEOUT_SECONDS}s",
            )

        logger.info("llama.cpp healthy for %s on port %d (pid %d)", model.name, port, proc.pid)
        return bp

    async def stop(self, process: BackendProcess) -> None:
        """Stop a llama-server process. Sends SIGTERM, waits, then SIGKILL."""
        proc = self._processes.pop(process.model_name, None)
        log_file = self._log_files.pop(process.model_name, None)
        if proc is None:
            if log_file:
                log_file.close()
            return

        logger.info("Stopping llama.cpp for %s (pid %d)", process.model_name, process.pid)

        proc.terminate()
        try:
            # Give it time to drain in-flight requests
            await asyncio.to_thread(proc.wait, timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("Force-killing llama.cpp for %s (pid %d)", process.model_name, proc.pid)
            proc.kill()
            await asyncio.to_thread(proc.wait, timeout=5)
        finally:
            if log_file:
                log_file.close()

    async def health_check(self, process: BackendProcess) -> bool:
        """Check the /health endpoint of a running llama-server."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{process.base_url}/health", timeout=5.0)
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def _wait_healthy(self, process: BackendProcess) -> bool:
        """Poll the health endpoint until it responds or timeout expires."""
        elapsed = 0.0
        while elapsed < _STARTUP_TIMEOUT_SECONDS:
            # Check if the process died
            proc = self._processes.get(process.model_name)
            if proc and proc.poll() is not None:
                logger.error(
                    "llama.cpp for %s exited with code %d during startup",
                    process.model_name,
                    proc.returncode,
                )
                return False

            if await self.health_check(process):
                return True

            await asyncio.sleep(_HEALTH_POLL_INTERVAL)
            elapsed += _HEALTH_POLL_INTERVAL

        return False

    def get_process(self, model_name: str) -> subprocess.Popen[bytes] | None:
        """Get the raw subprocess handle for log capture, etc."""
        return self._processes.get(model_name)
