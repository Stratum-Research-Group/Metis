"""
Stoma Provenance Manifest.
Provides durable tracking of passages, source files, SHA-256 checksums, and model metadata.
Guarantees passage ID resolution to source ID, root ID, path, content hash, location, and timestamp.
Enforces strict model identity and fail-closed validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class ModelIdentityMismatchError(ValueError):
    """Raised when model identity (weights checksum, revision, name, or dimension) mismatches."""
    pass


class LegacyManifestError(ValueError):
    """Raised when loading a legacy manifest with ambiguous IDs or missing root_id."""
    pass


class CorruptedManifestError(ValueError):
    """Raised when manifest integrity check fails."""
    pass


@dataclass
class PassageProvenance:
    """Provenance metadata for an indexed passage."""
    passage_id: str
    source_id: str
    root_id: str
    root_path: str
    rel_path: str
    file_content_hash: str
    chunk_index: int
    chunk_hash: str
    location: Dict[str, Any]
    ingested_at: float
    token_count: int
    char_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PassageProvenance:
        if "root_id" not in data or "source_id" not in data:
            raise LegacyManifestError(
                f"Passage '{data.get('passage_id')}' lacks root_id or source_id. "
                "Safe rejection: legacy ambiguous format is not permitted."
            )
        return cls(**data)


@dataclass
class SourceFileRecord:
    """Metadata for an ingested source file."""
    source_id: str
    root_id: str
    root_path: str
    rel_path: str
    content_hash: str
    size_bytes: int
    mtime: float
    last_ingested_at: float
    passage_ids: List[str] = field(default_factory=list)
    plugin_name: str = "text"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SourceFileRecord:
        if "root_id" not in data or "source_id" not in data:
            raise LegacyManifestError(
                f"Source '{data.get('rel_path')}' lacks root_id or source_id. "
                "Safe rejection: legacy ambiguous format is not permitted."
            )
        return cls(**data)


@dataclass
class ModelMetadata:
    """Encoder model identity and revision metadata for strict reproducibility."""
    model_name: str
    d_model: int
    max_seq_length: int
    tokenizer_name: str
    revision: Optional[str] = None
    weights_checksum: Optional[str] = None
    special_tokens_overhead: int = 2
    effective_max_tokens: int = 254

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelMetadata:
        if "special_tokens_overhead" not in data or data["special_tokens_overhead"] is None:
            raise LegacyManifestError("ModelMetadata lacks mandatory 'special_tokens_overhead'.")
        if "effective_max_tokens" not in data or data["effective_max_tokens"] is None:
            raise LegacyManifestError("ModelMetadata lacks mandatory 'effective_max_tokens'.")
        if "weights_checksum" not in data or not data["weights_checksum"]:
            raise ModelIdentityMismatchError("ModelMetadata lacks mandatory 'weights_checksum'.")

        return cls(
            model_name=data["model_name"],
            d_model=int(data["d_model"]),
            max_seq_length=int(data["max_seq_length"]),
            tokenizer_name=data["tokenizer_name"],
            revision=data.get("revision"),
            weights_checksum=data["weights_checksum"],
            special_tokens_overhead=int(data["special_tokens_overhead"]),
            effective_max_tokens=int(data["effective_max_tokens"]),
        )


class SourcesDict(dict):
    """
    Dictionary of source records keyed by source_id: '{root_id}:{rel_path}'.
    Provides backwards compatibility for rel_path lookups ('rel_path' in sources).
    """

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            for src in self.values():
                if src.rel_path == key:
                    return True
        return False

    def __getitem__(self, key: str) -> SourceFileRecord:
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str):
            for src in self.values():
                if src.rel_path == key:
                    return src
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str):
            for src in self.values():
                if src.rel_path == key:
                    return src
        return default

    def pop(self, key: str, default: Any = None) -> Any:
        if super().__contains__(key):
            return super().pop(key)
        if isinstance(key, str):
            matched_key = None
            for sid, src in self.items():
                if src.rel_path == key:
                    matched_key = sid
                    break
            if matched_key is not None:
                return super().pop(matched_key)
        if default is not None:
            return default
        return None


class ProvenanceManifest:
    """
    Durable, serializable manifest storing provenance for every indexed passage.
    Enforces strict model validation and atomic persistence.
    """

    def __init__(
        self,
        model_meta: Optional[ModelMetadata] = None,
        d_model: int = 384,
        num_facets: int = 1,
        manifest_version: str = "2.0",
    ):
        self.manifest_version = manifest_version
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.model_meta = model_meta
        self.d_model = d_model
        self.num_facets = num_facets
        self.sources: SourcesDict = SourcesDict()            # Keyed by source_id: "{root_id}:{rel_path}"
        self.passages: Dict[str, PassageProvenance] = {}     # Keyed by passage_id: "{root_id}:{rel_path}#{chunk:04d}"

    def get_provenance(self, passage_id: str) -> Optional[PassageProvenance]:
        """Look up provenance by passage ID."""
        return self.passages.get(passage_id)

    def record_source_and_passages(
        self,
        source: SourceFileRecord,
        passages: List[PassageProvenance],
    ) -> None:
        """Atomically record or update a source file and its generated passages."""
        # Remove any old passages associated with this source_id if it previously existed
        old_record = self.sources.get(source.source_id)
        if old_record:
            for pid in old_record.passage_ids:
                self.passages.pop(pid, None)

        self.sources[source.source_id] = source
        for p in passages:
            self.passages[p.passage_id] = p
        self.updated_at = time.time()

    def register_passage(self, prov: PassageProvenance) -> None:
        """Register or update a single passage provenance."""
        self.passages[prov.passage_id] = prov
        self.updated_at = time.time()

    def remove_source(self, source_id: str) -> List[str]:
        """
        Remove a source file and all its associated passages by source_id.
        Returns the list of removed passage IDs.
        """
        record = self.sources.pop(source_id, None)
        if not record:
            return []
        removed_ids = []
        for pid in record.passage_ids:
            if pid in self.passages:
                del self.passages[pid]
                removed_ids.append(pid)
        self.updated_at = time.time()
        return removed_ids

    def validate_integrity(self) -> None:
        """
        Validate generation-wide internal integrity:
        1. Every passage referenced in source.passage_ids must exist in self.passages.
        2. Every passage in self.passages must belong to an existing source in self.sources.
        3. Passage IDs must exactly match their key.
        """
        for src_id, src in self.sources.items():
            for pid in src.passage_ids:
                if pid not in self.passages:
                    raise CorruptedManifestError(
                        f"Manifest integrity error: source '{src_id}' references passage '{pid}' which is missing."
                    )
                p = self.passages[pid]
                if p.source_id != src_id:
                    raise CorruptedManifestError(
                        f"Manifest integrity error: passage '{pid}' has source_id '{p.source_id}', expected '{src_id}'."
                    )

        for pid, p in self.passages.items():
            if pid != p.passage_id:
                raise CorruptedManifestError(
                    f"Manifest integrity error: passage key '{pid}' != passage.passage_id '{p.passage_id}'."
                )
            if p.source_id not in self.sources:
                raise CorruptedManifestError(
                    f"Manifest integrity error: passage '{pid}' references unknown source '{p.source_id}'."
                )
            src = self.sources[p.source_id]
            if p.root_id != src.root_id or p.root_path != src.root_path or p.rel_path != src.rel_path:
                raise CorruptedManifestError(
                    f"Manifest integrity error: passage '{pid}' path metadata ({p.root_id}:{p.rel_path}) "
                    f"does not match source '{src.source_id}' ({src.root_id}:{src.rel_path})."
                )
            if p.file_content_hash != src.content_hash:
                raise CorruptedManifestError(
                    f"Manifest integrity error: passage '{pid}' file_content_hash '{p.file_content_hash}' "
                    f"does not match source content_hash '{src.content_hash}'."
                )

    def validate_against_embedder(
        self,
        model_name: str,
        d_model: int,
        revision: Optional[str] = None,
        weights_checksum: Optional[str] = None,
        num_facets: int = 1,
        tokenizer_name: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        special_tokens_overhead: Optional[int] = None,
        effective_max_tokens: Optional[int] = None,
    ) -> None:
        """
        Validate that the loaded manifest matches the current embedder and engine config.
        Loudly rejects shape, dimension, model name, revision, weights checksum, tokenizer,
        or sequence limit mismatches.
        """
        if self.d_model != d_model:
            raise ModelIdentityMismatchError(
                f"Index/manifest d_model mismatch: manifest has {self.d_model}, "
                f"current embedder has {d_model}."
            )
        if self.num_facets != num_facets:
            raise ModelIdentityMismatchError(
                f"Index/manifest num_facets mismatch: manifest has {self.num_facets}, "
                f"current configuration has {num_facets}."
            )
        if not self.model_meta:
            raise ModelIdentityMismatchError("Manifest lacks model_meta record.")

        if self.model_meta.d_model != d_model:
            raise ModelIdentityMismatchError(
                f"Model metadata d_model mismatch: recorded {self.model_meta.d_model}, "
                f"got {d_model}."
            )
        if self.model_meta.model_name != model_name:
            raise ModelIdentityMismatchError(
                f"Model identity mismatch: manifest was indexed with '{self.model_meta.model_name}', "
                f"attempting to load with '{model_name}'."
            )

        # Strict weights checksum validation
        if not self.model_meta.weights_checksum:
            raise ModelIdentityMismatchError(
                "Manifest model_meta has no weights_checksum. Refusing to load unverifiable model identity."
            )
        if not weights_checksum:
            raise ModelIdentityMismatchError(
                "Current embedder has no verified weights_checksum. Refusing to load unverifiable model identity."
            )
        if self.model_meta.weights_checksum != weights_checksum:
            raise ModelIdentityMismatchError(
                f"Model weights checksum mismatch: manifest has '{self.model_meta.weights_checksum}', "
                f"current model has '{weights_checksum}'."
            )

        # Strict revision validation if present
        if revision and self.model_meta.revision and revision != self.model_meta.revision:
            raise ModelIdentityMismatchError(
                f"Model revision mismatch: manifest was indexed with revision '{self.model_meta.revision}', "
                f"current model revision is '{revision}'."
            )

        # Strict tokenizer and limit checks
        if tokenizer_name and self.model_meta.tokenizer_name != tokenizer_name:
            raise ModelIdentityMismatchError(
                f"Tokenizer mismatch: manifest used '{self.model_meta.tokenizer_name}', "
                f"current embedder has '{tokenizer_name}'."
            )
        if max_seq_length and self.model_meta.max_seq_length != max_seq_length:
            raise ModelIdentityMismatchError(
                f"Max sequence length mismatch: manifest recorded {self.model_meta.max_seq_length}, "
                f"current embedder has {max_seq_length}."
            )
        if special_tokens_overhead is not None and self.model_meta.special_tokens_overhead != special_tokens_overhead:
            raise ModelIdentityMismatchError(
                f"Special tokens overhead mismatch: manifest recorded {self.model_meta.special_tokens_overhead}, "
                f"current embedder has {special_tokens_overhead}."
            )
        if effective_max_tokens is not None and self.model_meta.effective_max_tokens != effective_max_tokens:
            raise ModelIdentityMismatchError(
                f"Effective max tokens mismatch: manifest recorded {self.model_meta.effective_max_tokens}, "
                f"current embedder has {effective_max_tokens}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "d_model": self.d_model,
            "num_facets": self.num_facets,
            "model_meta": self.model_meta.to_dict() if self.model_meta else None,
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
            "passages": {k: v.to_dict() for k, v in self.passages.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProvenanceManifest:
        manifest = cls(
            model_meta=ModelMetadata.from_dict(data["model_meta"]) if data.get("model_meta") else None,
            d_model=data.get("d_model", 384),
            num_facets=data.get("num_facets", 1),
            manifest_version=data.get("manifest_version", "2.0"),
        )
        manifest.created_at = data.get("created_at", time.time())
        manifest.updated_at = data.get("updated_at", manifest.created_at)
        manifest.sources = SourcesDict({
            k: SourceFileRecord.from_dict(v) for k, v in data.get("sources", {}).items()
        })
        manifest.passages = {
            k: PassageProvenance.from_dict(v) for k, v in data.get("passages", {}).items()
        }
        manifest.validate_integrity()
        return manifest

    def save(self, path: Path) -> None:
        """
        Atomically save manifest to disk via temporary file and replace.
        """
        self.validate_integrity()
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}_{int(time.time()*1000)}")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> ProvenanceManifest:
        """Load manifest from disk."""
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_content_sha256(content: str) -> str:
    """Compute SHA-256 checksum of a string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
