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
