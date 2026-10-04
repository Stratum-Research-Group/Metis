"""Immutable run manifest schema for Paideia experiments and training."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class RunManifest(BaseModel):
    """
    Immutable run manifest.
    Written atomically before execution starts, updated atomically upon completion or failure.
    Never silently overwritten.
    """
    run_id: str = Field(..., description="Unique UUID for this execution run")
    task_name: str = Field(..., description="Task identifier, e.g. 'router_baseline', 'walk_qlora'")
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "PARTIAL"] = Field(
        default="PENDING",
        description="Run execution status",
    )
    started_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of start",
    )
    completed_at_utc: Optional[str] = Field(default=None, description="ISO 8601 UTC timestamp of finish")
    duration_seconds: Optional[float] = Field(default=None, description="Total execution time")
    
    # Provenance
    git_commit: Optional[str] = Field(default=None, description="Git HEAD commit hash")
    dirty_tree: bool = Field(default=False, description="Whether uncommitted changes existed")
    hardware_line: str = Field(..., description="GPU model, VRAM, CPU, and host info")
    package_versions: Dict[str, str] = Field(default_factory=dict, description="Python package versions (torch, peft, etc.)")
    cuda_version: Optional[str] = Field(default=None, description="CUDA driver/runtime version")
    
    # Data & Splits
    dataset_checksum: str = Field(..., description="SHA-256 of primary input dataset")
    split_checksums: Dict[str, str] = Field(default_factory=dict, description="SHA-256 digests for train/dev/test splits")
    split_counts: Dict[str, int] = Field(default_factory=dict, description="Number of items in each split")
    
    # Config & Hyperparameters
    seed: int = Field(default=42, description="Random seed")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters or baseline configuration")
    
    # Artifacts & Outputs
    adapter_digest: Optional[str] = Field(default=None, description="SHA-256 digest of trained weights if applicable")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Primary metrics, bootstrap CI, guardrails")
    failure_reason: Optional[str] = Field(default=None, description="Error message / stack trace if failed or OOM")
