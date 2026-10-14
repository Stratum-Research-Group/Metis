"""Tests for LTHCM Damage-Trained Recovery Plan (Sections 5.1 - 5.4).

Verifies:
- Frozen corpus integrity, topic-family isolation, zero test leakage.
- Clean retrieval baseline run and exact candidate ID logging.
- Deterministic damage injection with seed reproducibility and in/out-of-budget modes.
- Wrapper and validator enforcement of immutable sources, external eligibility gates,
  rejection of prohibited writes, non-existent sources, and unsupported links.
"""

import copy
import math
import pytest

from recovery.benchmark import RecoveryBenchmarkHarness
from recovery.corpus import (
    FrozenSourceCorpus,
    TEST_FAMILIES,
    TRAIN_FAMILIES,
    build_frozen_recovery_dataset,
)
from recovery.damage import DamageInjector
from recovery.schema import (
    CanonicalDocument,
    CorruptedProvenanceError,
    DamageConfig,
    DerivedEdge,
    EligibilityViolationError,
    MissingSourceError,
    NonFiniteStateError,
    ProhibitedWriteError,
    ShadowGraphState,
    ShadowNode,
    UnsupportedLinkError,
)
from recovery.validator import IndependentRecoveryValidator
from recovery.wrapper import RecoveryShadowGraphWrapper


# ── 5.1: Frozen Corpus, Version, and Family Isolation ─────────


def test_frozen_corpus_hash_integrity_and_family_split():
    corpus, queries = build_frozen_recovery_dataset()

    # All canonical documents must have cryptographically valid hashes
    all_docs = corpus.get_all()
    assert len(all_docs) >= 16
    for doc in all_docs:
        assert doc.verify_hash() is True

    # Families must be disjoint with zero leakage between train and test
    assert TRAIN_FAMILIES.isdisjoint(TEST_FAMILIES)
    train_queries = [q for q in queries if not q["is_test_split"]]
    test_queries = [q for q in queries if q["is_test_split"]]

    assert len(train_queries) == 2
    assert len(test_queries) == 2

    train_families = {q["family"] for q in train_queries}
    test_families = {q["family"] for q in test_queries}

    assert train_families.issubset(TRAIN_FAMILIES)
    assert test_families.issubset(TEST_FAMILIES)
    assert train_families.isdisjoint(test_families)


def test_tampered_document_hash_fails_integrity():
    doc = CanonicalDocument.create(
        doc_id="test_doc",
        text="Original clean text",
        timestamp=100.0,
        family="crypto_quantum",
    )
    assert doc.verify_hash() is True

    # Mutate text without updating content_sha256 (simulating corrupted provenance)
    tampered = CanonicalDocument(
        doc_id=doc.doc_id,
        text="Corrupted tampered text",
        content_sha256=doc.content_sha256,
        timestamp=doc.timestamp,
        provenance=doc.provenance,
        family=doc.family,
    )
    assert tampered.verify_hash() is False


# ── 5.2: Clean Retrieval Baseline and Candidate ID Capture ────


def test_clean_baseline_retrieval_and_source_edge_derivation():
    harness = RecoveryBenchmarkHarness()
    clean_report = harness.evaluate_clean_baseline()

    assert clean_report["mode"] == "clean_baseline"
    assert clean_report["overall_full_support_rate"] == 1.0
    assert len(clean_report["query_results"]) == 4

    for q_res in clean_report["query_results"]:
        assert len(q_res["exact_candidate_ids"]) > 0
        assert q_res["state_hash"] != ""
        assert q_res["recall_at_4"] == 1.0


def test_derived_edges_rely_on_source_text_entities():
    corpus, _ = build_frozen_recovery_dataset()
    edges = corpus.derive_source_entity_edges()
    assert len(edges) > 0

    # Ensure every derived edge is backed by entity appearing in both endpoints
    for edge in edges:
        doc1 = corpus[edge.source_id]
        doc2 = corpus[edge.target_id]
        basis = edge.evidence_basis.lower()
        assert basis in doc1.text.lower()
        assert basis in doc2.text.lower()


# ── 5.3: Deterministic Damage Injector and Seed Reproducibility


def test_damage_injector_determinism_across_runs():
    corpus, queries = build_frozen_recovery_dataset()
    harness = RecoveryBenchmarkHarness(corpus=corpus, queries=queries)
    clean_state, _ = harness._build_clean_state_for_query(queries[0])
    all_doc_ids = [d.doc_id for d in corpus.get_all()]

    cfg = DamageConfig(seed=1337, edge_drop_prob=0.5, distractor_link_count=2)
    inj1 = DamageInjector(cfg)
    inj2 = DamageInjector(cfg)

    d1, diag1 = inj1.apply_damage(clean_state, all_doc_ids)
    d2, diag2 = inj2.apply_damage(clean_state, all_doc_ids)

    assert d1.state_hash == d2.state_hash
    assert diag1["dropped_edge_count"] == diag2["dropped_edge_count"]
    assert diag1["distractor_edge_count"] == diag2["distractor_edge_count"]


def test_damage_injector_applies_all_in_budget_modes():
    corpus, queries = build_frozen_recovery_dataset()
    harness = RecoveryBenchmarkHarness(corpus=corpus, queries=queries)
    clean_state, _ = harness._build_clean_state_for_query(queries[0])
    all_doc_ids = [d.doc_id for d in corpus.get_all()]

    cfg = DamageConfig(
        seed=42,
        edge_drop_prob=0.3,
        score_noise_std=0.2,
        distractor_link_count=3,
        stale_candidate_count=2,
        duplicate_node_count=2,
    )
    injector = DamageInjector(cfg)
    damaged_state, diag = injector.apply_damage(clean_state, all_doc_ids)

    assert diag["dropped_edge_count"] > 0
    assert diag["noisy_scores_applied"] > 0
    assert diag["distractor_edge_count"] == 3
    assert diag["stale_node_count"] == 2
    assert diag["duplicate_node_count"] == 2


# ── 5.4: Wrapper, Immutability Checks, and Rejection Gates ────


def test_prohibited_write_to_canonical_corpus_raises():
    corpus, _ = build_frozen_recovery_dataset()
    new_doc = CanonicalDocument.create(
        doc_id="illegal_doc",
        text="Prohibited injected text",
        timestamp=500.0,
        family="crypto_quantum",
    )
    with pytest.raises(ProhibitedWriteError):
        corpus.write_document(new_doc)


def test_validator_rejects_missing_canonical_source():
    corpus, _ = build_frozen_recovery_dataset()
    validator = IndependentRecoveryValidator(corpus=corpus, as_of=300.0, fail_closed=True)

    fake_candidates = [{"doc_id": "non_existent_hallucinated_doc", "score": 0.99}]
    with pytest.raises(MissingSourceError):
        validator.filter_and_validate_candidates(fake_candidates, strict_missing=True)


def test_validator_enforces_external_eligibility_over_pipeline_fallback():
    corpus, queries = build_frozen_recovery_dataset()
    # Query timestamp is 300.0; create a candidate with timestamp 400.0 (future violation)
    future_doc = CanonicalDocument.create(
        doc_id="future_leak_doc",
        text="Future information leaking",
        timestamp=400.0,
        family="crypto_quantum",
    )
    # Put in test corpus
    test_docs = corpus.get_all() + [future_doc]
    test_corpus = FrozenSourceCorpus(test_docs)

    validator = IndependentRecoveryValidator(corpus=test_corpus, as_of=300.0, fail_closed=True)
    candidates = [{"doc_id": future_doc.doc_id, "score": 0.95}]

    # Strict external eligibility gate must reject future doc
    with pytest.raises(EligibilityViolationError):
        validator.filter_and_validate_candidates(candidates, strict_temporal=True)


def test_validator_rejects_unsupported_link_without_source_evidence():
    corpus, _ = build_frozen_recovery_dataset()
    validator = IndependentRecoveryValidator(corpus=corpus, as_of=300.0)

    # Pick two existing docs that do not share "UnicornTheory"
    edge = DerivedEdge(
        source_id="cq_hop1_kyber",
        target_id="cb_hop1_lumicor",
        relation_type="hallucinated_relation",
        weight=0.9,
        evidence_basis="UnicornTheoryNonExistent",
    )
    with pytest.raises(UnsupportedLinkError):
        validator.validate_derived_edge(edge)


def test_validator_rejects_non_finite_scores():
    corpus, _ = build_frozen_recovery_dataset()
    validator = IndependentRecoveryValidator(corpus=corpus, as_of=300.0)

    bad_candidates = [{"doc_id": "cq_hop1_kyber", "score": float("nan")}]
    with pytest.raises(NonFiniteStateError):
        validator.filter_and_validate_candidates(bad_candidates)


def test_end_to_end_recovery_benchmark_execution():
    harness = RecoveryBenchmarkHarness()
    results = harness.evaluate_damaged_and_recovered(max_steps=3)

    assert results["mode"] == "damaged_and_recovered"
    assert "damaged_full_support_rate" in results
    assert "recovered_full_support_rate" in results
    assert results["recovered_full_support_rate"] >= results["damaged_full_support_rate"]
    assert results["p50_latency_ms"] < 500.0

    # Ensure all recovered details have audit trail steps
    for rec in results["recovered_details"]:
        assert rec["audit_step_count"] > 0
        assert rec["status"] in {"SUCCESS", "DEGRADED", "ABSTAIN"}
