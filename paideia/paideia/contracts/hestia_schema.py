"""Hestia adapter registry contract."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class HestiaAdapterRegistryEntry(BaseModel):
    """
    Metadata specification for registering a trained adapter with Hestia serving layer.
    Allows Hestia to verify weight digests, tokenizer compatibility, and rollback targets.
    """
    schema_version: str = Field(default="1.0.0", description="Contract version")
    adapter_id: str = Field(..., description="Unique semantic adapter identifier, e.g. 'router-qlora-v1'")
    base_model_name: str = Field(..., description="Base model name or repo ID, e.g. 'meta-llama/Llama-3.1-8B-Instruct'")
    base_model_digest: str = Field(..., description="SHA-256 digest of base model weights/directory")
    adapter_digest: str = Field(..., description="SHA-256 digest of adapter safetensors/bin")
    tokenizer_digest: str = Field(..., description="SHA-256 digest of tokenizer files")
    template_digest: str = Field(..., description="SHA-256 digest of chat/prompt template")
    task: str = Field(..., description="Task type: 'question_router', 'walk_policy', etc.")
    compatible_backend: str = Field(default="vllm_or_hf", description="Target serving backend: 'vllm', 'sglang', 'hf'")
    training_data_fingerprint: str = Field(..., description="Checksum of the dataset used to train this adapter")
    evaluation_run_id: str = Field(..., description="Dokime/Paideia run ID that certified this release")
    resource_footprint: Dict[str, Any] = Field(
        default_factory=dict,
        description="Measured runtime footprint: e.g. {'peak_vram_mb': 14200, 'p95_latency_ms': 12.4}",
    )
    rollback_target: Optional[str] = Field(
        default=None,
        description="Previous approved adapter ID to revert to upon regression",
    )
