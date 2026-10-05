"""Configuration schema and loader for Kybernetes orchestrator."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import yaml


class KybernetesConfig(BaseModel):
    """
    Configuration model for Kybernetes orchestrator server.
    Supports file, environment, and CLI override layers.
    """
    hestia_base_url: str = Field(
        default="http://127.0.0.1:8080/v1",
        description="Base URL for local Hestia model server API",
    )
    memory_root: Optional[str] = Field(
        default=None,
        description="Filesystem root path for indexed LTHCM memory storage",
    )
    db_path: str = Field(
        default=":memory:",
        description="SQLite database path for task budget reconciliation (:memory: or file path)",
    )
    allowed_origins: List[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:8081",
            "http://localhost:8081",
        ],
        description="Explicit allowed CORS origins",
    )
    service_token: Optional[str] = Field(
        default=None,
        description="Shared secret bearer token for upstream chat and turn service callers",
    )
    admin_token: Optional[str] = Field(
        default=None,
        description="Bearer token required for all /v1/admin/* management endpoints",
    )
    admin_token_file: Optional[str] = Field(
        default=None,
        description="Path to secure token file (0600) containing the admin token",
    )
    admin_enabled: bool = Field(
        default=True,
        description="Whether to mount /v1/admin/* control plane endpoints",
    )
    config_path: Optional[str] = Field(
        default=None,
        description="Path to YAML configuration file loaded",
    )

    def resolve_admin_token(self) -> Optional[str]:
        """Resolve effective admin token from direct value, file, or environment."""
        if self.admin_token:
            return self.admin_token
        
        env_token = os.environ.get("KYBERNETES_ADMIN_TOKEN")
        if env_token:
            return env_token.strip()

        token_file = self.admin_token_file or os.environ.get("KYBERNETES_ADMIN_TOKEN_FILE")
        if token_file and Path(token_file).exists():
            try:
                return Path(token_file).read_text(encoding="utf-8").strip()
            except Exception:
                return None

        return None

    def resolve_service_token(self) -> Optional[str]:
        """Resolve effective service token from value or environment."""
        if self.service_token:
            return self.service_token
        return os.environ.get("KYBERNETES_SERVICE_TOKEN")


def load_config(config_path: Optional[str] = None) -> KybernetesConfig:
    """
    Load Kybernetes configuration from YAML file and environment variables.
    Precedence: Env vars > YAML config file > Defaults.
    """
    resolved_path = config_path or os.environ.get("KYBERNETES_CONFIG")
    raw_data: Dict[str, Any] = {}

    if resolved_path and Path(resolved_path).exists():
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    raw_data = loaded
        except Exception as e:
            raise ValueError(f"Failed to parse Kybernetes configuration file at {resolved_path}: {e}")

    # Environment variable overrides
    if "KYBERNETES_HESTIA_BASE_URL" in os.environ:
        raw_data["hestia_base_url"] = os.environ["KYBERNETES_HESTIA_BASE_URL"]
    elif "HESTIA_BASE_URL" in os.environ:
        raw_data["hestia_base_url"] = os.environ["HESTIA_BASE_URL"]

    if "KYBERNETES_MEMORY_ROOT" in os.environ:
        raw_data["memory_root"] = os.environ["KYBERNETES_MEMORY_ROOT"]

    if "KYBERNETES_DB_PATH" in os.environ:
        raw_data["db_path"] = os.environ["KYBERNETES_DB_PATH"]

    if "KYBERNETES_ALLOWED_ORIGINS" in os.environ:
        origins_str = os.environ["KYBERNETES_ALLOWED_ORIGINS"]
        raw_data["allowed_origins"] = [o.strip() for o in origins_str.split(",") if o.strip()]

    if "KYBERNETES_SERVICE_TOKEN" in os.environ:
        raw_data["service_token"] = os.environ["KYBERNETES_SERVICE_TOKEN"]

    if "KYBERNETES_ADMIN_TOKEN" in os.environ:
        raw_data["admin_token"] = os.environ["KYBERNETES_ADMIN_TOKEN"]

    if "KYBERNETES_ADMIN_TOKEN_FILE" in os.environ:
        raw_data["admin_token_file"] = os.environ["KYBERNETES_ADMIN_TOKEN_FILE"]

    if resolved_path:
        raw_data["config_path"] = str(Path(resolved_path).resolve())

    return KybernetesConfig(**raw_data)
