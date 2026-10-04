"""
Causal Filtering and Lorentzian Light-Cone Tests.
Mandated by Stoma Build Spec v4, Line 42:
- Eligible past document is present
- Future-timestamped top-ranked candidate is excluded
- Result metadata shows causal_fallback = False
- Also demonstrates that default query_timestamp (100.0) causes silent causal fallback
"""

import time
import pytest
import torch
from lthcm.engine.pipeline import LTHCMEngine

from stoma.adapter import StomaLTHCMAdapter


def test_causal_light_cone_filtering(shared_embedder):
    """
    Mandated test:
    Index an eligible past document and a future-timestamped top-ranked candidate.
    Assert that when querying at the current timestamp:
    - The future candidate is absent
    - The past document is present
    - Metadata shows causal_fallback is False
    """
    engine = LTHCMEngine(d_model=shared_embedder.d_model, num_facets=1)
    adapter = StomaLTHCMAdapter(engine=engine, embedder=shared_embedder)

    now = time.time()
    past_timestamp = now - 120.0     # 2 minutes in the past (dt = +120s > 0, inside past light cone)
    future_timestamp = now + 3600.0  # 1 hour in the future (dt = -3600s < 0, outside light cone)

    # Future candidate is a near-exact match to the query
    query_text = "Cellular sheaf cohomology and topological obstruction verification"
    future_content = f"{query_text} exact duplicate high-scoring match from the future."
    past_content = "Cellular sheaf cohomology and topological obstruction in memory manifolds."

    # Index past document
    adapter.index_passage(
        doc_id="doc_past_001",
        content=past_content,
        embedding=shared_embedder.encode_single(past_content),
        domain="general",
        timestamp=past_timestamp,
    )

    # Index future document
    adapter.index_passage(
        doc_id="doc_future_001",
        content=future_content,
        embedding=shared_embedder.encode_single(future_content),
        domain="general",
        timestamp=future_timestamp,
    )

    # Query at current epoch time 'now'
    res = adapter.query(
        query_text=query_text,
        top_k=5,
        query_timestamp=now,
        enforce_causality=True,
    )

    retrieved_ids = [item["doc_id"] for item in res["results"]]

    # Assertions mandated by spec line 42:
    assert "doc_future_001" not in retrieved_ids, (
        "Causal violation: Future candidate was returned even though it lies outside the past light cone!"
    )
    assert "doc_past_001" in retrieved_ids, (
        "Eligible past candidate was missing from causally verified results!"
    )
    assert res["causal_fallback"] is False, (
        f"Expected causal_fallback=False, but got {res['causal_fallback']}!"
    )
    assert res["causally_valid_count"] >= 1


def test_default_timestamp_silent_fallback_reproduction(shared_embedder):
    """
    Demonstrates the stack-wide flaw:
    When LTHCM's default query_timestamp (100.0) is used with epoch-second documents,
    dt = 100.0 - epoch_seconds << 0. All candidates fall outside the light cone,
    and the engine silently triggers causal_fallback = True.
    """
    engine = LTHCMEngine(d_model=shared_embedder.d_model, num_facets=1)
    adapter = StomaLTHCMAdapter(engine=engine, embedder=shared_embedder)

    epoch_now = time.time()
    content = "Standard document with real epoch timestamp."
    adapter.index_passage(
        doc_id="doc_epoch_001",
        content=content,
        embedding=shared_embedder.encode_single(content),
        timestamp=epoch_now,
    )

    # Calling retrieve_and_verify with LTHCM's default query_timestamp = 100.0
    raw_res = engine.retrieve_and_verify(
        query_vector=shared_embedder.encode_single(content),
        query_domain="general",
        query_timestamp=100.0,  # Default in LTHCM pipeline.py:168
        top_k=5,
        enforce_causality=True,
    )

    # All epoch-second candidates are excluded from light cone, forcing silent fallback
    assert raw_res["causally_valid_count"] == 0
    assert raw_res["causal_fallback"] is True, (
        "Expected silent causal fallback when query_timestamp=100.0 is used with epoch timestamps."
    )
