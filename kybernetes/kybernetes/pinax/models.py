"""Pinax Tool Catalogue data models and schema definitions.
Ancient Greek 'pinax': tablet, register, catalogue.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class RiskTier(str, Enum):
    """
    Risk tiers consumed by Phylax policy gating.
    T0: Read-only, local, no secrets. Default allow in scoped roots.
    T1: Local writes scoped to allowed roots. Path validation mandatory.
    T2: Network read, allowlisted destinations, size/redirect caps.
    T3: Irreversible, egress-capable, privileged, system-mutating, or arbitrary exec.
    """
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


class ToolKind(str, Enum):
    BINARY = "binary"
    SCRIPT = "script"
    BUILTIN = "builtin"
    STRUCTURED = "structured"


class EntryKind(str, Enum):
    INSTALLED = "installed"
    CANDIDATE = "candidate"


class InstallState(str, Enum):
    INSTALLED = "installed"
    NOT_INSTALLED = "not-installed"
    BROKEN = "broken"
    UNKNOWN = "unknown"


class ProbeStatus(str, Enum):
    """
    Status of an active, executed health probe (Phase 1 / Agent-Reach model).
    OK: Binary launched successfully and returned exit code 0.
    MISSING: File not found or unresolvable on PATH.
    BROKEN: File exists but cannot launch (e.g. stale venv shim with missing interpreter).
    TIMEOUT: Probe command exceeded strict timeout budget.
    ERROR: Binary launched but exited non-zero or crashed.
    UNKNOWN: Tool has not yet been probed.
    """
    OK = "ok"
    MISSING = "missing"
    BROKEN = "broken"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ProbeResult:
    """Outcome of executing a curated, side-effect-free health probe on a tool."""
    tool_id: str
    status: ProbeStatus
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    detail: str = ""
    is_stale_shim: bool = False
    probed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "status": self.status.value if isinstance(self.status, ProbeStatus) else str(self.status),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
            "is_stale_shim": self.is_stale_shim,
            "probed_at": self.probed_at,
        }


@dataclass
class ToolRecord:
    """Canonical representation of an inventoried tool in Pinax."""
    id: str
    host_id: str = "localhost"
    name: str = ""
    kind: ToolKind = ToolKind.BINARY
    entry_kind: EntryKind = EntryKind.CANDIDATE
    summary: str = ""
    category: str = "general"
    invocation_path: Optional[str] = None
    resolved_path: Optional[str] = None
    env_identity: Optional[str] = None
    interpreter: Optional[str] = None
    sha256: Optional[str] = None
    version: Optional[str] = None
    package: Optional[str] = "unknown"
    package_manager: Optional[str] = "unknown"
    install_state: InstallState = InstallState.UNKNOWN
    risk_tier: RiskTier = RiskTier.T3
    discovered_by: str = "curated"
    first_seen: float = field(default_factory=time.time)
    last_verified: float = field(default_factory=time.time)
    notes: Optional[str] = None
    probe_status: ProbeStatus = ProbeStatus.UNKNOWN
    last_probed: float = 0.0
    probe_detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "host_id": self.host_id,
            "name": self.name,
            "kind": self.kind.value if isinstance(self.kind, ToolKind) else str(self.kind),
            "entry_kind": self.entry_kind.value if isinstance(self.entry_kind, EntryKind) else str(self.entry_kind),
            "summary": self.summary,
            "category": self.category,
            "invocation_path": self.invocation_path,
            "resolved_path": self.resolved_path,
            "env_identity": self.env_identity,
            "interpreter": self.interpreter,
            "sha256": self.sha256,
            "version": self.version,
            "package": self.package,
            "package_manager": self.package_manager,
            "install_state": self.install_state.value if isinstance(self.install_state, InstallState) else str(self.install_state),
            "risk_tier": self.risk_tier.value if isinstance(self.risk_tier, RiskTier) else str(self.risk_tier),
            "discovered_by": self.discovered_by,
            "first_seen": self.first_seen,
            "last_verified": self.last_verified,
            "notes": self.notes,
            "probe_status": self.probe_status.value if isinstance(self.probe_status, ProbeStatus) else str(self.probe_status),
            "last_probed": self.last_probed,
            "probe_detail": self.probe_detail,
        }


@dataclass
class ToolDoc:
    """Document locator and reference for an inventoried tool."""
    tool_id: str
    doc_kind: str  # man | help | info | upstream-url | example
    locator: str   # man file path+section, file path, or URL
    version_scope: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: int = 0
    fetched_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "doc_kind": self.doc_kind,
            "locator": self.locator,
            "version_scope": self.version_scope,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "fetched_at": self.fetched_at,
        }


@dataclass
class ToolCapability:
    """Capability attribute of a tool."""
    tool_id: str
    capability: str  # file-read | file-write | network-read | network-write | process-exec | destructive | privilege | install | interactive
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "capability": self.capability,
            "detail": self.detail,
        }


@dataclass
class ToolRiskOverride:
    """Curated pattern-level risk escalation."""
    tool_id: str
    pattern_kind: str  # subcommand | flag | regex
    pattern: str       # e.g. "push", "--upload-file", "--delete"
    tier: RiskTier = RiskTier.T3
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "pattern_kind": self.pattern_kind,
            "pattern": self.pattern,
            "tier": self.tier.value if isinstance(self.tier, RiskTier) else str(self.tier),
            "note": self.note,
        }


@dataclass
class ToolSignature:
    """Structured JSON schema for curated tool invocation."""
    tool_id: str
    call_name: str
    arg_schema_json: str
    curated_by: str = "pinax-curator"
    curated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "call_name": self.call_name,
            "arg_schema_json": self.arg_schema_json,
            "curated_by": self.curated_by,
            "curated_at": self.curated_at,
        }
