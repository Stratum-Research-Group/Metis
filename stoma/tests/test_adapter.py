"""
Unit tests for Stoma LTHCM Adapter.
Verifies locked single-facet shape [1, d_model], loud failure on shape mismatch,
and pickle serialization.
"""

import pickle
import pytest
import torch
from lthcm.engine.pipeline import LTHCMEngine

from stoma.adapter import ShapeMismatchError, StomaLTHCMAdapter


def test_adapter_validates_matching_shapes(shared_embedder):
    engine = LTHCMEngine(d_model=shared_embedder.d_model, num_facets=1)
    adapter = StomaLTHCMAdapter(engine=engine, embedder=shared_embedder)
    # Should not raise
    adapter.validate_shapes()


def test_adapter_fails_loudly_on_d_model_mismatch(shared_embedder):
    """
    LTHCM's superposition store silently pads or truncates mismatched dimensions.
    Assert Stoma fails loudly with ShapeMismatchError when engine d_model != embedder.d_model.
    """
    wrong_dim = shared_embedder.d_model + 64
    engine = LTHCMEngine(d_model=wrong_dim, num_facets=1)

    with pytest.raises(ShapeMismatchError, match="LOUD MISMATCH: Engine d_model"):
        StomaLTHCMAdapter(engine=engine, embedder=shared_embedder)


def test_adapter_fails_loudly_on_num_facets_mismatch(shared_embedder):
    """
    Stoma locked decision mandates single-facet embeddings (num_facets=1).
    Assert Stoma fails loudly with ShapeMismatchError when engine num_facets != 1.
    """
    engine = LTHCMEngine(d_model=shared_embedder.d_model, num_facets=4)

    with pytest.raises(ShapeMismatchError, match="LOUD MISMATCH: Engine num_facets"):
        StomaLTHCMAdapter(engine=engine, embedder=shared_embedder)


def test_adapter_enforces_single_facet_tensor(shared_embedder):
    engine = LTHCMEngine(d_model=shared_embedder.d_model, num_facets=1)
    adapter = StomaLTHCMAdapter(engine=engine, embedder=shared_embedder)

    raw_emb = torch.randn(shared_embedder.d_model)
    facet_vec = shared_embedder.create_facet_vector(raw_emb)
    assert facet_vec.shape == (1, shared_embedder.d_model)

    # Wrong shape passed to create_facet_vector must raise ValueError
    wrong_emb = torch.randn(shared_embedder.d_model // 2)
    with pytest.raises(ValueError, match="Facet vector d_model mismatch"):
        shared_embedder.create_facet_vector(wrong_emb)


def test_adapter_pickle_serialization(shared_embedder, temp_workspace):
    engine = LTHCMEngine(d_model=shared_embedder.d_model, num_facets=1)
    adapter = StomaLTHCMAdapter(engine=engine, embedder=shared_embedder)
    save_path = temp_workspace / "engine.pkl"

    adapter.save(save_path)
    assert save_path.is_file()

    loaded_adapter = StomaLTHCMAdapter.load(save_path, embedder=shared_embedder)
    assert isinstance(loaded_adapter.engine, LTHCMEngine)
    assert loaded_adapter.engine.d_model == shared_embedder.d_model
    assert loaded_adapter.engine.num_facets == 1
