"""Module descriptors and configuration inspection views for all 11 Metis modules."""

from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import yaml


class ModuleKind(str, Enum):
    SERVICE = "service"
    LIBRARY = "library"
    BATCH_TOOL = "batch_tool"


class ModuleDescriptor(BaseModel):
    id: str = Field(..., description="Unique module identifier")
    display_name: str = Field(..., description="Human-readable module title")
    section: str = Field(..., description="Dashboard tab/section")
    kind: ModuleKind = Field(..., description="Classification: service | library | batch_tool")
    config_path: Optional[str] = Field(default=None, description="Primary relative configuration path")
    config_format: str = Field(default="none", description="Configuration format: yaml | toml | json | pydantic | none")
    schema_ref: Optional[str] = Field(default=None, description="Reference schema identifier")
    restart_required: bool = Field(default=True, description="Whether config modification requires process restart")
    mutable: bool = Field(default=False, description="Whether runtime/saved configuration is mutable in v1")
    description: str = Field(default="", description="Functional description of the module")


class ConfigView(BaseModel):
    module_id: str
    saved: Dict[str, Any] = Field(default_factory=dict, description="Configuration values parsed from disk")
    effective: Optional[Dict[str, Any]] = Field(default=None, description="Configuration values currently active in memory")
    sources: Dict[str, str] = Field(default_factory=dict, description="Source origin per key: default | file | env | cli")
    restart_required_keys: List[str] = Field(default_factory=list, description="Keys that require restart to take effect")
    schema_ref: Optional[str] = None
    config_hash: Optional[str] = None
    loaded_at: Optional[str] = None
    drift: Optional[bool] = Field(default=None, description="Whether saved on-disk config differs from effective runtime config")


MODULE_REGISTRY: Dict[str, ModuleDescriptor] = {
    "kybernetes": ModuleDescriptor(
        id="kybernetes",
        display_name="Kybernetes Orchestrator",
        section="Overview / Sessions",
        kind=ModuleKind.SERVICE,
        config_path="kybernetes.yaml",
        config_format="pydantic",
        schema_ref="KybernetesConfig",
        restart_required=True,
        mutable=True,
        description="Versioned central orchestrator and control plane coordinating inference, memory, deliberation, and budget.",
    ),
    "hestia": ModuleDescriptor(
        id="hestia",
        display_name="Hestia Inference Server",
        section="Models & Resources",
        kind=ModuleKind.SERVICE,
        config_path="hestia/hestia.yaml",
        config_format="yaml",
        schema_ref="ServiceConfig",
        restart_required=True,
        mutable=True,
        description="Local OpenAI-compatible inference server with multi-model pool and VRAM admission control.",
    ),
    "prosopon": ModuleDescriptor(
        id="prosopon",
        display_name="Prosopon Frontend & BFF",
        section="Overview",
        kind=ModuleKind.SERVICE,
        config_path="prosopon/prosopon.yaml",
        config_format="pydantic",
        schema_ref="ProsoponConfig",
        restart_required=True,
        mutable=True,
        description="User client interface, voice synthesis/transcription gateway, and dashboard presentation layer.",
    ),
    "lthcm": ModuleDescriptor(
        id="lthcm",
        display_name="LTHCM Vector Memory",
        section="Memory & Ingestion",
        kind=ModuleKind.LIBRARY,
        config_path=None,
        config_format="none",
        schema_ref=None,
        restart_required=True,
        mutable=False,
        description="Long-Term Hierarchical Context Memory for dense indexing, lexical search, and passage retrieval.",
    ),
    "dianoia": ModuleDescriptor(
        id="dianoia",
        display_name="Dianoia Deliberation Engine",
        section="Reasoning",
        kind=ModuleKind.LIBRARY,
        config_path=None,
        config_format="none",
        schema_ref=None,
        restart_required=True,
        mutable=False,
        description="Structured thought-graph deliberation engine with bounded graph walks and conflict resolution.",
    ),
    "krisis": ModuleDescriptor(
        id="krisis",
        display_name="Krisis Response Verifier",
        section="Reasoning",
        kind=ModuleKind.LIBRARY,
        config_path="krisis/krisis-policy-0.1.json",
        config_format="json",
        schema_ref="krisis-policy-0.1.schema.json",
        restart_required=False,
        mutable=True,
        description="Independent factuality, citation grounding, and constraint verification engine.",
    ),
    "phylax": ModuleDescriptor(
        id="phylax",
        display_name="Phylax Capability Guard",
        section="Guardrails",
        kind=ModuleKind.LIBRARY,
        config_path="phylax/policies/demo_policy.yaml",
        config_format="yaml",
        schema_ref="PhylaxPolicy",
        restart_required=False,
        mutable=True,
        description="Zero-trust capability firewall, policy enforcement gateway, and cryptographic audit log.",
    ),
    "kleroterion": ModuleDescriptor(
        id="kleroterion",
        display_name="Kleroterion Ensemble Consensus",
        section="Reasoning",
        kind=ModuleKind.LIBRARY,
        config_path=None,
        config_format="none",
        schema_ref=None,
        restart_required=False,
        mutable=False,
        description="Deterministic multi-sample consensus and ensemble scoring library.",
    ),
    "stoma": ModuleDescriptor(
        id="stoma",
        display_name="Stoma Ingestion Pipeline",
        section="Memory & Ingestion",
        kind=ModuleKind.BATCH_TOOL,
        config_path=".stoma/config.yaml",
        config_format="yaml",
        schema_ref="StomaConfig",
        restart_required=False,
        mutable=True,
        description="Document extraction, chunking, sanitization, and provenance-tracked ingestion tool.",
    ),
    "dokime": ModuleDescriptor(
        id="dokime",
        display_name="Dokime Evaluation Suite",
        section="Evaluation",
        kind=ModuleKind.BATCH_TOOL,
        config_path="dokime/examples/config.example.toml",
        config_format="toml",
        schema_ref="DokimeConfig",
        restart_required=False,
        mutable=False,
        description="Continuous evaluation harness, benchmarking, and automated regression testing suite.",
    ),
    "paideia": ModuleDescriptor(
        id="paideia",
        display_name="Paideia Adapter Trainer",
        section="Training",
        kind=ModuleKind.BATCH_TOOL,
        config_path=None,
        config_format="none",
        schema_ref="PaideiaManifest",
        restart_required=False,
        mutable=False,
        description="Offline LoRA adapter fine-tuning, training dataset curation, and adapter registry compiler.",
    ),
}


def redact_sensitive_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Redact passwords, secret tokens, and sensitive internal paths."""
    redacted: Dict[str, Any] = {}
    SENSITIVE_EXACT_KEYS = {
        "token", "secret", "password", "api_key", "service_token", "admin_token",
        "access_token", "private_key", "auth_token", "pin", "pairing_pin", "device_token"
    }
    
    for k, v in data.items():
        k_lower = str(k).lower()
        is_sensitive = k_lower in SENSITIVE_EXACT_KEYS or any(
            k_lower.endswith(f"_{sk}") or k_lower.startswith(f"{sk}_")
            for sk in ("token", "secret", "password", "key", "pin", "auth")
        )
        if is_sensitive:
            redacted[k] = "[REDACTED]" if v else None
        elif isinstance(v, dict):
            redacted[k] = redact_sensitive_dict(v)
        elif isinstance(v, list):
            redacted[k] = [
                redact_sensitive_dict(item) if isinstance(item, dict) else item
                for item in v
            ]
        elif isinstance(v, str) and (v.startswith("/") or "\\" in v) and not v.startswith("http"):
            # Sanitize full absolute local path to basename representation without disk probe
            redacted[k] = Path(v).name or v
        else:
            redacted[k] = v
    return redacted


def compute_dict_hash(data: Dict[str, Any]) -> str:
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
