"""Service configuration loader for hestia.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PortRange:
    """Inclusive port range for backend processes."""

    start: int = 8081
    end: int = 8099

    def __post_init__(self) -> None:
        if not (1024 <= self.start <= self.end <= 65535):
            raise ValueError(f"Invalid port range: {self.start}-{self.end}")

    @property
    def ports(self) -> range:
        return range(self.start, self.end + 1)


@dataclass(frozen=True)
class ServiceConfig:
    """Top-level Hestia service configuration."""

    listen_host: str = "127.0.0.1"
    listen_port: int = 8080
    llamacpp_binary: str = "/usr/local/bin/llama-server"
    backend_port_range: PortRange = field(default_factory=PortRange)
    idle_timeout_seconds: int = 600
    log_level: str = "INFO"
    max_concurrent_requests_per_model: int = 8
    max_queue_depth_per_model: int = 16
    queue_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        import ipaddress

        host = self.listen_host.strip()
        if host == "localhost":
            return

        err_msg = (
            f"listen_host must be a loopback address (127.0.0.1, ::1, or localhost), "
            f"got: {self.listen_host}"
        )
        try:
            ip = ipaddress.ip_address(host)
            if not ip.is_loopback:
                raise ValueError(err_msg)
        except ValueError as exc:
            raise ValueError(err_msg) from exc

    @classmethod
    def from_yaml(cls, path: Path) -> ServiceConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        port_range_raw = raw.get("backend_port_range", {})
        port_range = PortRange(
            start=port_range_raw.get("start", 8081),
            end=port_range_raw.get("end", 8099),
        )

        return cls(
            listen_host=raw.get("listen_host", "127.0.0.1"),
            listen_port=raw.get("listen_port", 8080),
            llamacpp_binary=raw.get("llamacpp_binary", "/usr/local/bin/llama-server"),
            backend_port_range=port_range,
            idle_timeout_seconds=raw.get("idle_timeout_seconds", 600),
            log_level=raw.get("log_level", "INFO"),
            max_concurrent_requests_per_model=raw.get("max_concurrent_requests_per_model", 8),
            max_queue_depth_per_model=raw.get("max_queue_depth_per_model", 16),
            queue_timeout_seconds=float(raw.get("queue_timeout_seconds", 30.0)),
        )


# Default config search path
DEFAULT_CONFIG_PATH = Path("hestia.yaml")
