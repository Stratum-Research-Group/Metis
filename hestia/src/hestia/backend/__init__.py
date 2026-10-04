"""Backend interface — abstract base for model serving backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from hestia.registry import ModelEntry


@dataclass
class BackendProcess:
    """Represents a running backend process."""

    model_name: str
    pid: int
    port: int
    host: str = "127.0.0.1"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class Backend(ABC):
    """Abstract backend interface.

    Implementations manage the lifecycle of a model serving process
    (e.g. llama.cpp server, vLLM). This abstraction allows swapping
    backends without changing the server manager or API layer.
    """

    @abstractmethod
    async def start(self, model: ModelEntry, port: int) -> BackendProcess:
        """Start a backend process for the given model on the specified port.

        Returns a BackendProcess handle on success.
        Raises BackendStartError on failure.
        """

    @abstractmethod
    async def stop(self, process: BackendProcess) -> None:
        """Gracefully stop a backend process, draining in-flight requests."""

    @abstractmethod
    async def health_check(self, process: BackendProcess) -> bool:
        """Return True if the backend process is healthy and ready to serve."""

    @abstractmethod
    def build_launch_args(self, model: ModelEntry, port: int) -> list[str]:
        """Build the command-line arguments for launching the backend.

        Derives flags from registry fields (context size, GPU offload,
        embedding mode, port, etc.).
        """


class BackendStartError(Exception):
    """Raised when a backend process fails to start."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        self.reason = reason
        super().__init__(f"Failed to start backend for {model_name}: {reason}")
