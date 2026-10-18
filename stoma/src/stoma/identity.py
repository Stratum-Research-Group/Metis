"""
Exact Embedder Identity & Representation Compatibility Architecture (RFC 8785 JCS).
Implements the 3-part decoupled identity:
  1. EmbeddingSpaceSpec (mathematical space representation)
  2. ProducerBuildSpec (concrete engine & weights execution)
  3. IndexBuildSpec (corpus snapshot & index configuration)

Enforces domain-separated hashing, runtime role-separated envelopes ("document" vs "query"),
NaN/infinity rejection, zero-vector defense, and normalization tolerance verification.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np


class IdentityValidationError(ValueError):
    """Raised when an identity spec, envelope, or vector payload fails validation."""
    pass


class IncompatibleSpaceError(IdentityValidationError):
    """Raised when query/document embedding space ID does not match index space ID."""
    pass


class InvalidVectorPayloadError(IdentityValidationError):
    """Raised when vectors contain NaN, Inf, zero norm for cosine, or dimension mismatch."""
    pass


def canonical_jcs_serialize(obj: Any) -> bytes:
    """
    Serialize an object to canonical JSON per RFC 8785 (JSON Canonicalization Scheme).
    - Keys are lexicographically sorted by UTF-16 code units.
    - No whitespace between tokens (separators=(',', ':')).
    - Non-finite numbers (NaN, Infinity) are rejected.
    - Numbers formatted without unnecessary exponents or trailing zeros.
    """
    def _normalize(val: Any) -> Any:
        if val is None:
            return None
        elif isinstance(val, bool):
            return val
        elif isinstance(val, int):
            return val
        elif isinstance(val, float):
            if not math.isfinite(val):
                raise IdentityValidationError(f"RFC 8785 non-finite number forbidden: {val}")
            # Format float cleanly per RFC 8785
            if val == 0.0:
                return 0
            if val.is_integer():
                return int(val)
            return val
        elif isinstance(val, str):
            return val
        elif isinstance(val, (list, tuple)):
            return [_normalize(item) for item in val]
        elif isinstance(val, dict):
            # Keys sorted lexicographically
            sorted_keys = sorted(val.keys())
            return {k: _normalize(val[k]) for k in sorted_keys}
        elif hasattr(val, "to_dict"):
            return _normalize(val.to_dict())
        else:
            raise IdentityValidationError(f"Cannot serialize unhandled type to canonical JCS: {type(val)}")

    normalized = _normalize(obj)
    # separators=(',', ':') removes whitespace
    # sort_keys=True ensures sorted keys in Python json encoder
    json_str = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return json_str.encode("utf-8")


def domain_separated_hash(domain: str, payload_bytes: bytes) -> str:
    """
    Compute domain-separated SHA-256 digest:
    SHA256(domain || payload_bytes)
    """
    hasher = hashlib.sha256()
    hasher.update(domain.encode("utf-8"))
    hasher.update(payload_bytes)
    return hasher.hexdigest()


@dataclass
class EmbeddingSpaceSpec:
    """
    Mathematical representation contract for queries and documents.
    Changes here invalidate representation compatibility.
    """
    schema_version: str = "1.0.0"
    model_assets: List[Tuple[str, str]] = field(default_factory=list)  # (relative_name, sha256)
    tokenizer_assets: List[Tuple[str, str]] = field(default_factory=list)
    tokenizer_configuration_hash: str = ""
    text_preprocessing_spec_hash: str = ""
    query_encoder_spec_hash: str = ""
    document_encoder_spec_hash: str = ""
    pooling_spec_hash: str = ""
    projection_or_adapter_assets_hash: str = "none"
    dimension: int = 384
    dtype: str = "float32"
    quantization_spec_hash: str = "none"
    normalization: str = "unit_l2"  # "none", "unit_l2", or named transform
    allowed_similarity: str = "cosine"  # "cosine", "dot", "euclidean"
    truncation_length: int = 256
    truncation_policy: str = "truncate_right"
    modality: str = "text"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure model_assets and tokenizer_assets are sorted lists of [str, str]
        d["model_assets"] = sorted([[k, v] for k, v in self.model_assets])
        d["tokenizer_assets"] = sorted([[k, v] for k, v in self.tokenizer_assets])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmbeddingSpaceSpec:
        d = dict(data)
        if "model_assets" in d:
            d["model_assets"] = [tuple(x) for x in d["model_assets"]]
        if "tokenizer_assets" in d:
            d["tokenizer_assets"] = [tuple(x) for x in d["tokenizer_assets"]]
        return cls(**d)

    def canonical_bytes(self) -> bytes:
        return canonical_jcs_serialize(self.to_dict())

    @property
    def space_id(self) -> str:
        return domain_separated_hash("embedding-space/v1", self.canonical_bytes())


@dataclass
class ProducerBuildSpec:
    """
    Concrete implementation and execution environment that produced the embeddings.
    """
    schema_version: str = "1.0.0"
    embedding_space_id: str = ""
    executable_digest: str = ""
    library_versions: Dict[str, str] = field(default_factory=dict)
    backend_precision: str = "fp32"
    reproducibility_profile: str = "strict_deterministic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProducerBuildSpec:
        return cls(**data)

    def canonical_bytes(self) -> bytes:
        return canonical_jcs_serialize(self.to_dict())

    @property
    def producer_build_id(self) -> str:
        return domain_separated_hash("embedding-producer/v1", self.canonical_bytes())


@dataclass
class IndexBuildSpec:
    """
    Corpus snapshot and index build configuration.
    """
    schema_version: str = "1.0.0"
    embedding_space_id: str = ""
    corpus_snapshot_id: str = ""
    chunking_spec_hash: str = ""
    vector_set_manifest_hash: str = ""
    index_algorithm_config_hash: str = ""
    build_seed: int = 42
    producer_build_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["producer_build_ids"] = sorted(self.producer_build_ids)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IndexBuildSpec:
        return cls(**data)

    def canonical_bytes(self) -> bytes:
        return canonical_jcs_serialize(self.to_dict())

    @property
    def index_build_id(self) -> str:
        return domain_separated_hash("retrieval-index/v1", self.canonical_bytes())


@dataclass
class VectorBatchEnvelope:
    """
    Runtime envelope for document or query vector batches.
    Separates roles ("document" vs "query") and enforces strict payload integrity.
    """
    space_id: str
    producer_build_id: str
    role: Literal["document", "query"]
    dimension: int
    dtype: str
    vector_count: int
    input_manifest_hash: str
    vector_payload_hash: str
    vectors: np.ndarray

    def validate_or_raise(
        self,
        expected_space_id: Optional[str] = None,
        expected_role: Optional[Literal["document", "query"]] = None,
        normalization_tolerance: float = 1e-4,
    ) -> None:
        """
        Executes strict 6-step validation pipeline:
        1. Bound lengths and counts.
        2. Check batch count, vector width, dtype, payload hash.
        3. Reject NaN, Infinity, and zero-vectors.
        4. Check normalization within tolerance if required.
        5. Match space ID.
        6. Verify role matches expectation.
        """
        # Step 1: Bound checks
        if self.vector_count <= 0:
            raise IdentityValidationError(f"vector_count must be > 0, got {self.vector_count}")
        if self.dimension <= 0:
            raise IdentityValidationError(f"dimension must be > 0, got {self.dimension}")

        # Step 2: Shape, dtype, payload hash
        if not isinstance(self.vectors, np.ndarray):
            raise InvalidVectorPayloadError("Vectors must be a numpy ndarray")
        if self.vectors.shape != (self.vector_count, self.dimension):
            raise InvalidVectorPayloadError(
                f"Array shape {self.vectors.shape} does not match declared count/dim: "
                f"({self.vector_count}, {self.dimension})"
            )

        # Compute payload hash
        computed_hash = hashlib.sha256(self.vectors.tobytes()).hexdigest()
        if computed_hash != self.vector_payload_hash:
            raise InvalidVectorPayloadError(
                f"Vector payload hash mismatch: computed {computed_hash} != declared {self.vector_payload_hash}"
            )

        # Step 3: Reject NaN/Inf and zero vectors
        if not np.all(np.isfinite(self.vectors)):
            raise InvalidVectorPayloadError("Vectors contain NaN or Infinity")

        norms = np.linalg.norm(self.vectors, axis=1)
        zero_indices = np.where(norms < 1e-9)[0]
        if len(zero_indices) > 0:
            raise InvalidVectorPayloadError(
                f"Zero-norm vector detected at index {zero_indices[0]}. Cosine similarity undefined."
            )

        # Step 4: Normalization check (for unit_l2)
        norm_diff = np.abs(norms - 1.0)
        unnormalized = np.where(norm_diff > normalization_tolerance)[0]
        if len(unnormalized) > 0:
            idx = unnormalized[0]
            raise InvalidVectorPayloadError(
                f"Vector at index {idx} violates unit_l2 normalization: norm={norms[idx]:.6f}, "
                f"diff={norm_diff[idx]:.6f} > tol={normalization_tolerance}"
            )

        # Step 5: Space ID matching
        if expected_space_id and self.space_id != expected_space_id:
            raise IncompatibleSpaceError(
                f"Embedding space mismatch: envelope {self.space_id} != expected {expected_space_id}"
            )

        # Step 6: Role matching
        if expected_role and self.role != expected_role:
            raise IdentityValidationError(
                f"Envelope role mismatch: declared '{self.role}' != expected '{expected_role}'"
            )


def create_vector_envelope(
    vectors: Union[np.ndarray, List[List[float]]],
    role: Literal["document", "query"],
    space_spec: EmbeddingSpaceSpec,
    producer_spec: ProducerBuildSpec,
    input_manifest_hash: str = "",
) -> VectorBatchEnvelope:
    """Helper to construct and sign a VectorBatchEnvelope from raw vectors."""
    vec_array = np.asarray(vectors, dtype=np.float32)
    if vec_array.ndim == 1:
        vec_array = vec_array.reshape(1, -1)

    payload_hash = hashlib.sha256(vec_array.tobytes()).hexdigest()
    envelope = VectorBatchEnvelope(
        space_id=space_spec.space_id,
        producer_build_id=producer_spec.producer_build_id,
        role=role,
        dimension=vec_array.shape[1],
        dtype="float32",
        vector_count=vec_array.shape[0],
        input_manifest_hash=input_manifest_hash or hashlib.sha256(b"empty_input").hexdigest(),
        vector_payload_hash=payload_hash,
        vectors=vec_array,
    )
    return envelope
