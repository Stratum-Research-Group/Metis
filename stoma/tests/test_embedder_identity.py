"""
Unit tests for Exact Embedder Identity & Representation Compatibility Architecture
per specification 01-embedder-identity-lthcm-metis-6cd9ff83.txt.
"""

import hashlib
import json
import math
import numpy as np
import pytest
import torch

from stoma.identity import (
    EmbeddingSpaceSpec,
    IdentityValidationError,
    IncompatibleSpaceError,
    IndexBuildSpec,
    InvalidVectorPayloadError,
    ProducerBuildSpec,
    VectorBatchEnvelope,
    canonical_jcs_serialize,
    create_vector_envelope,
    domain_separated_hash,
)
from stoma.adapter import StomaLTHCMAdapter
from stoma.embedder import StomaEmbedder


def test_canonical_jcs_serialization_key_ordering():
    """Verify that key ordering in dictionaries yields identical canonical byte sequences."""
    dict_a = {"b": 2, "a": 1, "c": {"y": "val_y", "x": "val_x"}}
    dict_b = {"a": 1, "c": {"x": "val_x", "y": "val_y"}, "b": 2}

    bytes_a = canonical_jcs_serialize(dict_a)
    bytes_b = canonical_jcs_serialize(dict_b)

    assert bytes_a == bytes_b
    assert bytes_a == b'{"a":1,"b":2,"c":{"x":"val_x","y":"val_y"}}'


def test_canonical_jcs_rejects_non_finite_numbers():
    """Verify that RFC 8785 canonical serializer rejects NaN and Infinity."""
    with pytest.raises(IdentityValidationError, match="non-finite number forbidden"):
        canonical_jcs_serialize({"num": float("nan")})

    with pytest.raises(IdentityValidationError, match="non-finite number forbidden"):
        canonical_jcs_serialize({"num": float("inf")})


def test_domain_separated_hashing():
    """Verify domain separation prevents cross-type digest collisions."""
    content = b'{"dimension":384}'
    hash_space = domain_separated_hash("embedding-space/v1", content)
    hash_producer = domain_separated_hash("embedding-producer/v1", content)
    hash_index = domain_separated_hash("retrieval-index/v1", content)

    assert len(hash_space) == 64
    assert hash_space != hash_producer
    assert hash_space != hash_index
    assert hash_producer != hash_index


def test_embedding_space_spec_changes_with_parameters():
    """Verify that altering model assets, normalization, or pooling changes space ID."""
    spec_base = EmbeddingSpaceSpec(
        dimension=384,
        model_assets=[("weights", "abc123hash")],
        normalization="unit_l2",
        pooling_spec_hash="mean_pooling",
    )
    base_id = spec_base.space_id

    # Altering model weights changes space ID
    spec_diff_weights = EmbeddingSpaceSpec(
        dimension=384,
        model_assets=[("weights", "different_hash")],
        normalization="unit_l2",
        pooling_spec_hash="mean_pooling",
    )
    assert spec_diff_weights.space_id != base_id

    # Altering pooling changes space ID
    spec_diff_pooling = EmbeddingSpaceSpec(
        dimension=384,
        model_assets=[("weights", "abc123hash")],
        normalization="unit_l2",
        pooling_spec_hash="cls_pooling",
    )
    assert spec_diff_pooling.space_id != base_id

    # Altering normalization changes space ID
    spec_diff_norm = EmbeddingSpaceSpec(
        dimension=384,
        model_assets=[("weights", "abc123hash")],
        normalization="none",
        pooling_spec_hash="mean_pooling",
    )
    assert spec_diff_norm.space_id != base_id


def test_decoupled_identities_independence():
    """Adding documents changes index ID without changing space ID."""
    space_spec = EmbeddingSpaceSpec(dimension=384)
    space_id = space_spec.space_id

    index_spec_1 = IndexBuildSpec(
        embedding_space_id=space_id,
        corpus_snapshot_id="snapshot_v1",
        chunking_spec_hash="chunk_512",
    )
    index_spec_2 = IndexBuildSpec(
        embedding_space_id=space_id,
        corpus_snapshot_id="snapshot_v2",  # new documents added
        chunking_spec_hash="chunk_512",
    )

    assert index_spec_1.embedding_space_id == space_id
    assert index_spec_2.embedding_space_id == space_id
    assert index_spec_1.index_build_id != index_spec_2.index_build_id


def test_runtime_envelope_role_separation():
    """Verify envelope role cannot be substituted (document != query)."""
    space_spec = EmbeddingSpaceSpec(dimension=4)
    producer_spec = ProducerBuildSpec(embedding_space_id=space_spec.space_id)

    # Normalized 4D vector
    v = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    envelope = create_vector_envelope(
        vectors=v,
        role="document",
        space_spec=space_spec,
        producer_spec=producer_spec,
    )

    # Valid when expecting "document"
    envelope.validate_or_raise(expected_space_id=space_spec.space_id, expected_role="document")

    # Fails when submitted as "query"
    with pytest.raises(IdentityValidationError, match="Envelope role mismatch"):
        envelope.validate_or_raise(expected_space_id=space_spec.space_id, expected_role="query")


def test_runtime_envelope_rejects_nan_and_inf():
    """Envelope validation rejects NaN and Infinity in vectors."""
    space_spec = EmbeddingSpaceSpec(dimension=2)
    producer_spec = ProducerBuildSpec(embedding_space_id=space_spec.space_id)

    v_nan = np.array([[float("nan"), 1.0]], dtype=np.float32)
    envelope = create_vector_envelope(
        vectors=v_nan,
        role="query",
        space_spec=space_spec,
        producer_spec=producer_spec,
    )
    with pytest.raises(InvalidVectorPayloadError, match="NaN or Infinity"):
        envelope.validate_or_raise()

    v_inf = np.array([[float("inf"), 1.0]], dtype=np.float32)
    envelope_inf = create_vector_envelope(
        vectors=v_inf,
        role="query",
        space_spec=space_spec,
        producer_spec=producer_spec,
    )
    with pytest.raises(InvalidVectorPayloadError, match="NaN or Infinity"):
        envelope_inf.validate_or_raise()


def test_runtime_envelope_rejects_zero_vector():
    """Envelope validation rejects zero-norm vector because cosine similarity is undefined."""
    space_spec = EmbeddingSpaceSpec(dimension=3)
    producer_spec = ProducerBuildSpec(embedding_space_id=space_spec.space_id)

    v_zero = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    envelope = create_vector_envelope(
        vectors=v_zero,
        role="query",
        space_spec=space_spec,
        producer_spec=producer_spec,
    )
    with pytest.raises(InvalidVectorPayloadError, match="Zero-norm vector detected"):
        envelope.validate_or_raise()


def test_runtime_envelope_normalization_check():
    """Envelope validation checks unit L2 normalization within tolerance."""
    space_spec = EmbeddingSpaceSpec(dimension=2)
    producer_spec = ProducerBuildSpec(embedding_space_id=space_spec.space_id)

    # Vector norm is sqrt(1^2 + 1^2) = 1.414 != 1.0
    v_unnorm = np.array([[1.0, 1.0]], dtype=np.float32)
    envelope = create_vector_envelope(
        vectors=v_unnorm,
        role="query",
        space_spec=space_spec,
        producer_spec=producer_spec,
    )
    with pytest.raises(InvalidVectorPayloadError, match="violates unit_l2 normalization"):
        envelope.validate_or_raise()


def test_adapter_save_persists_space_spec(tmp_path):
    """Adapter save persists .space.json sidecar and load verifies it."""
    class MockEmbedder:
        def __init__(self, d_model=384):
            self.d_model = d_model
            self.weights_checksum = "mock_sha256"

        def get_metadata(self):
            from stoma.provenance import ModelMetadata
            return ModelMetadata(
                model_name="mock_model",
                d_model=self.d_model,
                max_seq_length=256,
                tokenizer_name="mock_tok",
                weights_checksum=self.weights_checksum,
            )

        def get_embedding_space_spec(self):
            return EmbeddingSpaceSpec(
                dimension=self.d_model,
                model_assets=[("weights", self.weights_checksum)],
            )

        def get_producer_build_spec(self):
            return ProducerBuildSpec(embedding_space_id=self.get_embedding_space_spec().space_id)

        def create_facet_vector(self, t):
            if t.dim() == 1:
                return t.unsqueeze(0)
            return t

        def encode_single(self, text):
            v = torch.ones(self.d_model, dtype=torch.float32)
            return v / torch.norm(v)

    mock_emb = MockEmbedder(d_model=384)
    engine = StomaLTHCMAdapter.create_fresh_engine(mock_emb)
    adapter = StomaLTHCMAdapter(engine=engine, embedder=mock_emb)

    idx_file = tmp_path / "test_index.pkl"
    adapter.save(idx_file)

    # Check companion file
    space_file = idx_file.with_suffix(".space.json")
    assert space_file.is_file()
    with open(space_file, "r") as f:
        data = json.load(f)
        assert data["dimension"] == 384
        assert data["model_assets"] == [["weights", "mock_sha256"]]

    # Load successfully with matching embedder
    loaded = StomaLTHCMAdapter.load(
        index_path=idx_file,
        embedder=mock_emb,
        allow_trusted_pickle=True,
    )
    assert loaded.space_spec.space_id == adapter.space_spec.space_id

    # Load with incompatible embedder raises IncompatibleSpaceError
    incompatible_emb = MockEmbedder(d_model=384)
    incompatible_emb.weights_checksum = "completely_different_weights"
    with pytest.raises(IncompatibleSpaceError, match="Index embedding space ID mismatch"):
        StomaLTHCMAdapter.load(
            index_path=idx_file,
            embedder=incompatible_emb,
            allow_trusted_pickle=True,
        )


def test_adapter_query_with_envelope(tmp_path):
    """Adapter.query works with a valid query envelope and rejects mismatched space."""
    class MockEmbedder:
        def __init__(self, d_model=384):
            self.d_model = d_model
            self.weights_checksum = "mock_weights"

        def get_metadata(self):
            from stoma.provenance import ModelMetadata
            return ModelMetadata(
                model_name="mock_model",
                d_model=self.d_model,
                max_seq_length=256,
                tokenizer_name="mock_tok",
                weights_checksum=self.weights_checksum,
            )

        def get_embedding_space_spec(self):
            return EmbeddingSpaceSpec(
                dimension=self.d_model,
                model_assets=[("weights", self.weights_checksum)],
            )

        def get_producer_build_spec(self):
            return ProducerBuildSpec(embedding_space_id=self.get_embedding_space_spec().space_id)

        def create_facet_vector(self, t):
            if t.dim() == 1:
                return t.unsqueeze(0)
            return t

        def encode_single(self, text):
            v = torch.ones(self.d_model, dtype=torch.float32)
            return v / torch.norm(v)

    mock_emb = MockEmbedder(d_model=384)
    engine = StomaLTHCMAdapter.create_fresh_engine(mock_emb)
    adapter = StomaLTHCMAdapter(engine=engine, embedder=mock_emb)

    # Index a passage
    emb_doc = torch.ones(384, dtype=torch.float32)
    emb_doc = emb_doc / torch.norm(emb_doc)
    adapter.index_passage(doc_id="doc1", content="Hello world", embedding=emb_doc)

    # Create matching query envelope
    q_vec = np.ones((1, 384), dtype=np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)
    env_valid = create_vector_envelope(
        vectors=q_vec,
        role="query",
        space_spec=adapter.space_spec,
        producer_spec=adapter.producer_spec,
    )

    res = adapter.query(envelope=env_valid)
    assert res["candidates_count"] >= 0

    # Envelope with foreign space ID is rejected
    foreign_space = EmbeddingSpaceSpec(dimension=384, model_assets=[("weights", "foreign")])
    env_foreign = create_vector_envelope(
        vectors=q_vec,
        role="query",
        space_spec=foreign_space,
        producer_spec=adapter.producer_spec,
    )
    with pytest.raises(IncompatibleSpaceError, match="Embedding space mismatch"):
        adapter.query(envelope=env_foreign)
