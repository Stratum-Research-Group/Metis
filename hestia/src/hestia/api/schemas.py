"""OpenAI-compatible request and response schemas with strict validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """A single chat message in a conversational thread."""

    model_config = ConfigDict(extra="allow")

    role: str = Field(..., description="Role of the message author (system, user, assistant)")
    content: str = Field(..., description="Contents of the message")
    name: str | None = Field(default=None, description="Optional author name")


class ChatCompletionRequest(BaseModel):
    """Validated chat completion request payload."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="ID of the model to use for completion")
    messages: list[ChatMessage] = Field(
        default_factory=list, description="List of messages comprising the conversation"
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling probability"
    )
    top_k: int | None = Field(default=None, ge=1, description="Top-k sampling cutoff")
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum number of tokens to generate"
    )
    stream: bool = Field(default=False, description="Whether to stream back partial progress")
    stream_options: dict[str, Any] | None = Field(
        default=None, description="Options for streaming response"
    )
    seed: int | None = Field(default=None, description="Random seed for deterministic generation")


class EmbeddingRequest(BaseModel):
    """Validated text embedding request payload."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="ID of the model to use for embeddings")
    input: str | list[str] = Field(..., description="Input text or list of texts to embed")
    encoding_format: str | None = Field(
        default="float", description="Format to return the embeddings in"
    )


class ManageModelRequest(BaseModel):
    """Validated request to explicitly load or unload a model."""

    model: str = Field(..., description="Name of the model to load or unload")


class ModelObject(BaseModel):
    """OpenAI-compatible model descriptor object."""

    id: str
    object: Literal["model"] = "model"
    created: int = 1700000000
    owned_by: str = "hestia"
    permission: list[dict[str, Any]] = Field(default_factory=list)
    root: str
    parent: str | None = None
    type: str | None = None
    context_size: int | None = None
    quantization: str | None = None
    artifact_digest: str | None = Field(default=None, description="SHA-256 digest of served model artifact")
    gpu_target: str | None = Field(default=None, description="Target GPU backend / device tier")
    state: str | None = Field(default="registered", description="Model runtime state e.g. loaded, registered, unloading")


class ModelListResponse(BaseModel):
    """OpenAI-compatible list of available models."""

    object: Literal["list"] = "list"
    data: list[ModelObject]


class ModelSwapRequest(BaseModel):
    """Validated model swap request payload."""

    model_config = ConfigDict(extra="allow")

    slot: str = Field(default="primary", description="Target slot identifier, e.g. 'primary'")
    candidate_bundle_id: str = Field(..., description="ID of registered ModelBundle to swap in")
    expected_generation: int = Field(..., ge=0, description="Expected active generation for CAS verification")
    idempotency_key: str = Field(..., min_length=1, description="Idempotency key to prevent duplicated execution")
    staging_mode: str = Field(default="MODE_B_MAINTENANCE", description="Staging mode: MODE_A_SIDE_BY_SIDE or MODE_B_MAINTENANCE")
    authorization_reference: str = Field(default="", description="Optional authorization or governance reference")
    auto_commit: bool = Field(default=True, description="Whether to automatically commit if activation succeeds")
    drain_timeout: float = Field(default=10.0, ge=0.0, description="Maximum seconds to wait for active stream drainage")


class ModelRollbackRequest(BaseModel):
    """Validated model rollback request payload."""

    model_config = ConfigDict(extra="allow")

    change_id: str = Field(..., description="Change ID of the ACTIVE_PENDING or active change to roll back")
    expected_generation: int = Field(..., ge=1, description="Expected active generation for CAS verification")
    reason: str = Field(default="Manual rollback requested", description="Reason for initiating rollback")


class RegisterBundleRequest(BaseModel):
    """Validated request to register a model bundle or Paideia export entry."""

    model_config = ConfigDict(extra="allow")

    bundle_id: str | None = Field(default=None, description="Bundle ID if registering directly")
    model_name: str | None = Field(default=None, description="Model name in registry")
    paideia_entry: dict[str, Any] | None = Field(default=None, description="Paideia hestia_registry_entry.json dictionary")
    adapter_path: str | None = Field(default=None, description="Path to LoRA adapter weights on disk")
    base_model_name: str | None = Field(default=None, description="Base model name if overriding Paideia manifest")

