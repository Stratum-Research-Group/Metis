"""Prosopon configuration settings."""

from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class ProsoponConfig(BaseModel):
    """Runtime configuration for Prosopon server and client."""
    host: str = Field(default="127.0.0.1", description="Server bind address: 127.0.0.1 (loopback) or 0.0.0.0 (LAN)")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    enable_tls: bool = Field(default=False, description="Enable TLS (recommended for phone LAN push-to-talk)")
    cert_file: Optional[str] = Field(default=None, description="Path to TLS certificate file")
    key_file: Optional[str] = Field(default=None, description="Path to TLS private key file")
    
    # Kybernetes upstream connection
    kybernetes_api_url: str = Field(
        default="http://127.0.0.1:8081/v1",
        description="Base URL for versioned Kybernetes API",
    )
    mock_kybernetes: bool = Field(
        default=True,
        description="Use realistic local mock if real Kybernetes server is offline",
    )
    
    # Security & Pairing
    data_dir: Path = Field(
        default_factory=lambda: Path("./data"),
        description="Directory for sessions, pairing state, and temporary audio",
    )
    require_pairing_in_lan: bool = Field(
        default=True,
        description="Enforce 6-digit device pairing token when accessed over non-loopback addresses",
    )
    session_expiry_hours: int = Field(default=72, ge=1, description="Paired session expiry time")
    allowed_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "https://localhost:8000", "http://127.0.0.1:8000", "https://127.0.0.1:8000"],
        description="Allowed CORS / CSRF origins",
    )

    # Local voice paths
    whisper_binary: str = Field(default="whisper.cpp", description="Path to whisper.cpp binary or alias")
    whisper_model: str = Field(default="base.en", description="Whisper model name or GGML file path")
    piper_binary: str = Field(default="piper", description="Path to piper TTS binary or alias")
    piper_model: str = Field(default="en_US-lessac-medium", description="Piper voice model file")

    def is_lan(self) -> bool:
        return self.host not in ("127.0.0.1", "localhost")
