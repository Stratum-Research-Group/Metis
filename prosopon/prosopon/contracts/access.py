"""Data models for device registry, sessions, passkeys, pairing links, and audit."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DeviceRecord(BaseModel):
    """Registered device hardware identity."""
    device_id: str
    label: str
    created_at_utc: str
    paired_via: str  # 'qr_passkey' | 'cli' | 'link' | 'code' | 'local'
    preset: str      # 'viewer' | 'companion' | 'operator' | 'owner' | 'custom'
    scopes: List[str]
    expires_at_utc: Optional[str] = None
    revoked_at_utc: Optional[str] = None
    revoked_reason: Optional[str] = None
    pinned_ip_range: Optional[str] = None
    passkey_count: int = 0
    is_active: bool = True


class PasskeyRecord(BaseModel):
    """WebAuthn credential registered to a device."""
    credential_id: str  # base64url encoded
    device_id: str
    public_key_cbor: str  # base64url encoded
    sign_count: int = 0
    aaguid: Optional[str] = None
    created_at_utc: str
    last_used_at_utc: str
    transports: List[str] = Field(default_factory=list)


class DeviceSessionRecord(BaseModel):
    """Active device session."""
    session_id: str
    device_id: str
    token_hash: str
    method: str  # 'bearer' | 'cookie' | 'webrtc'
    issued_at_utc: str
    expires_at_utc: str
    revoked_at_utc: Optional[str] = None
    route_id: Optional[str] = None


class PairingLinkRecord(BaseModel):
    """Single-use invitation / pairing link."""
    id: str
    credential_hash: str
    label: str
    preset: str
    scopes: List[str]
    created_at_utc: str
    expires_at_utc: str
    consumed_at_utc: Optional[str] = None
    consumed_by_device_id: Optional[str] = None
    revoked_at_utc: Optional[str] = None
    created_by_device_id: Optional[str] = None


class RouteRecord(BaseModel):
    """Reachable network route for Prosopon."""
    route_id: str
    kind: str  # 'lan' | 'tailnet' | 'ssh' | 'custom' | 'loopback'
    address: str
    state: str  # 'active' | 'unreachable' | 'disabled'
    learned: bool = False
    priority: int = 0
    last_verified_at_utc: Optional[str] = None
    tls: str = "none"  # 'none' | 'self-signed' | 'trusted'


class AccessAuditRecord(BaseModel):
    """Cryptographically chained access audit entry."""
    id: int
    ts_utc: str
    actor_device_id: Optional[str] = None
    action: str
    target_device_id: Optional[str] = None
    ip: Optional[str] = None
    route_id: Optional[str] = None
    detail_json: str
    prev_row_hash: str
    row_hash: str


# REST API Request/Response schemas

class PairDeviceRequest(BaseModel):
    label: str = Field(default="Phone Client", description="Friendly device label")
    preset: str = Field(default="companion", description="Preset role (viewer|companion|operator|owner)")
    scopes: Optional[List[str]] = Field(default=None, description="Custom scopes if preset is custom")
    validity_seconds: int = Field(default=300, description="Validity in seconds (default 5 min)")


class PairDeviceResponse(BaseModel):
    pairing_id: str
    pairing_secret: str
    pairing_url: str
    expires_at_utc: str
    preset: str
    scopes: List[str]


class RevokeDeviceRequest(BaseModel):
    reason: str = Field(default="owner", description="Revocation reason (owner|panic|policy)")
    stop_runs: bool = Field(default=True, description="Whether to stop active agent runs / turns started by this device")


class LockdownResponse(BaseModel):
    revoked_devices_count: int
    revoked_sessions_count: int
    message: str
