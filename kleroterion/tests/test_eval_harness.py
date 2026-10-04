"""
Tests for evaluation metrics, manifest, bootstrap CI, and acceptance gates (Section 10).
"""

import numpy as np
import pytest

from eval.harness import (
    ArmEvaluationResult,
    bootstrap_ci,
    verify_acceptance_gates,
)
from eval.metrics import (
    compute_contradiction_rate,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_top_k_overlap,
)
from eval.snapshot_manifest import (
    AcceptanceGates,
    FrozenExperimentManifest,
    QueryItem,
)
from kleroterion.errors import InvalidConfigError


def test_metrics_computations():
    ranked = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc2", "doc4"}

    # Recall@4: 2 / 2 = 1.0
    assert compute_recall_at_k(ranked, relevant, k=4) == 1.0
    # Recall@2: 1 / 2 = 0.5
    assert compute_recall_at_k(ranked, relevant, k=2) == 0.5

    # MRR@4: first relevant is at rank 2 => 1/2 = 0.5
    assert compute_mrr_at_k(ranked, relevant, k=4) == 0.5

    # nDCG@4
    ndcg = compute_ndcg_at_k(ranked, relevant, k=4)
    assert 0.0 < ndcg <= 1.0

    # Overlap
    assert compute_top_k_overlap(["doc1", "doc2"], ["doc2", "doc3"], k=2) == 0.5

    # Contradiction rate
    contradictions = {"doc3"}
    assert compute_contradiction_rate(ranked, contradictions) == 0.25


def test_bootstrap_confidence_interval():
    deltas = [0.1, 0.2, 0.15, 0.05, 0.3, 0.25]
    mean, lower, upper = bootstrap_ci(deltas, n_bootstrap=1000)
    assert lower <= mean <= upper
    assert lower > 0.0


def test_acceptance_gates_blocked_when_unconfigured():
    """
    Section 10.3: Missing values mean evaluation acceptance is blocked, not automatically passed.
    """
    manifest = FrozenExperimentManifest(
        manifest_id="exp_001",
        lthcm_revision="rev_1",
        dianoia_revision="rev_1",
        corpus_snapshot_id="corpus_v1",
        graph_snapshot_id="graph_v1",
        index_snapshot_id="index_v1",
        scorer_version="scorer_v1",
        embedding_version="emb_v1",
        queries=[
            QueryItem("q1", "text", "exploratory", {"doc1"}, "held_out")
        ],
        acceptance_gates=AcceptanceGates(),  # Missing all values!
    )

    result = ArmEvaluationResult(
        arm_name="test_arm",
        recall_at_k=0.8,
        mrr_at_k=0.7,
        ndcg_at_k=0.75,
        contradiction_rate=0.0,
        empty_intersection_rate=0.0,
        p95_latency_sec=0.05,
        per_query_recall=[0.8],
    )

    gate_eval = verify_acceptance_gates(manifest, result)
    assert gate_eval["status"] == "BLOCKED"
    assert not gate_eval["gates_passed"]


def test_acceptance_gates_passed_when_configured():
    gates = AcceptanceGates(
        overlap_min=0.8,
        recall_loss_max=0.05,
        gain_min=0.02,
        exact_fact_loss_max=0.0,
        contradiction_max=0.01,
        p95_latency_budget_sec=0.2,
        empty_intersection_rate_max=0.05,
    )

    manifest = FrozenExperimentManifest(
        manifest_id="exp_001",
        lthcm_revision="rev_1",
        dianoia_revision="rev_1",
        corpus_snapshot_id="corpus_v1",
        graph_snapshot_id="graph_v1",
        index_snapshot_id="index_v1",
        scorer_version="scorer_v1",
        embedding_version="emb_v1",
        queries=[
            QueryItem("q1", "text", "exploratory", {"doc1"}, "held_out")
        ],
        acceptance_gates=gates,
    )

    baseline = ArmEvaluationResult(
        arm_name="baseline",
        recall_at_k=0.7,
        mrr_at_k=0.6,
        ndcg_at_k=0.65,
        contradiction_rate=0.0,
        empty_intersection_rate=0.0,
        p95_latency_sec=0.05,
        per_query_recall=[0.7, 0.7, 0.7],
    )

    evaluated = ArmEvaluationResult(
        arm_name="kleroterion",
        recall_at_k=0.85,
        mrr_at_k=0.75,
        ndcg_at_k=0.8,
        contradiction_rate=0.0,
        empty_intersection_rate=0.01,
        p95_latency_sec=0.08,
        per_query_recall=[0.85, 0.85, 0.85],
    )

    gate_eval = verify_acceptance_gates(manifest, evaluated, baseline)
    assert gate_eval["status"] == "PASSED"
    assert gate_eval["gates_passed"]
