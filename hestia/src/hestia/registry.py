"""Model registry — loads and validates models.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


class ModelType(StrEnum):
    """Supported model types."""

    CHAT = "chat"
    EMBEDDING = "embedding"


class ModelState(StrEnum):
    """Runtime state of a registered model."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


@dataclass
class SamplingParams:
    """Default sampling parameters for a model."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SamplingParams:
        if not raw:
            return cls()
        return cls(
            temperature=float(raw.get("temperature", 0.7)),
            top_p=float(raw.get("top_p", 0.9)),
            top_k=int(raw.get("top_k", 40)),
            repeat_penalty=float(raw.get("repeat_penalty", 1.1)),
        )


@dataclass
class ModelEntry:
    """A single model in the registry."""

    name: str
    model_type: ModelType
    gguf_path: Path
    quantization: str
    context_size: int
    vram_estimate_mb: int
    gpu_target: str = "rtx-5060-ti"
    sampling: SamplingParams = field(default_factory=SamplingParams)
    artifact_digest: str | None = None

    # Runtime state — not persisted
    state: ModelState = field(default=ModelState.UNLOADED, repr=False)
    backend_port: int | None = field(default=None, repr=False)
    pid: int | None = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelEntry:
        return cls(
            name=raw["name"],
            model_type=ModelType(raw["type"]),
            gguf_path=Path(raw["gguf_path"]),
            quantization=raw["quantization"],
            context_size=int(raw["context_size"]),
            vram_estimate_mb=int(raw["vram_estimate_mb"]),
            gpu_target=raw.get("gpu_target", "rtx-5060-ti"),
            sampling=SamplingParams.from_dict(raw.get("sampling", {})),
            artifact_digest=raw.get("artifact_digest"),
        )


class ModelRegistry:
    """In-memory registry of all configured models."""

    def __init__(self) -> None:
        self._models: dict[str, ModelEntry] = {}

    def load_from_yaml(self, path: Path) -> None:
        """Load model definitions from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        for entry in raw.get("models", []):
            model = ModelEntry.from_dict(entry)
            self._models[model.name] = model

    def register(self, model: ModelEntry) -> None:
        """Register a model entry."""
        self._models[model.name] = model

    def get(self, name: str) -> ModelEntry | None:
        """Get a model by name, or None if not registered."""
        return self._models.get(name)

    def list_all(self) -> list[ModelEntry]:
        """Return all registered models."""
        return list(self._models.values())

    def loaded_models(self) -> list[ModelEntry]:
        """Return only models currently loaded."""
        return [m for m in self._models.values() if m.state == ModelState.LOADED]

    @property
    def total_vram_reserved_mb(self) -> int:
        """Sum of VRAM estimates for all loaded/loading models."""
        return sum(
            m.vram_estimate_mb
            for m in self._models.values()
            if m.state in (ModelState.LOADED, ModelState.LOADING)
        )
