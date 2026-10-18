"""Unit tests for AdmissionGatedPool multi-expert routing baseline (Phase 4)."""

from dataclasses import dataclass
import numpy as np
import pytest

from paideia.baselines.admission_pool import AdmissionGatedPool, ExpertProfile


@dataclass
class MockItem:
    question: str
    action: str


def test_admission_pool_fit_and_predict():
    items = [
        MockItem(question="Find previous files in database", action="retrieve"),
        MockItem(question="Search past logs in index", action="retrieve"),
        MockItem(question="Query memory storage for document", action="retrieve"),
        MockItem(question="Calculate matrix multiplication 2x2", action="compute"),
        MockItem(question="Solve integral and numerical equation", action="compute"),
        MockItem(question="Compute eigenvalue decomposition", action="compute"),
    ]

    pool = AdmissionGatedPool(n_features=128, top_k=1)
    pool.fit(items)

    assert pool.is_fitted
    assert len(pool.experts) == 2

    pred_ret = pool.predict_one("Look up archived documents in memory")
    assert pred_ret == "retrieve"

    pred_calc = pool.predict_one("Run matrix arithmetic formula")
    assert pred_calc == "compute"


def test_top_k_admission_gating():
    pool = AdmissionGatedPool(n_features=64, top_k=2)
    pool.experts["exp1"] = ExpertProfile("exp1", "alpha", np.array([1.0, 0.0] + [0.0] * 62))
    pool.experts["exp2"] = ExpertProfile("exp2", "beta", np.array([0.8, 0.2] + [0.0] * 62))
    pool.experts["exp3"] = ExpertProfile("exp3", "gamma", np.array([0.0, 1.0] + [0.0] * 62))

    query_vec = np.array([1.0, 0.0] + [0.0] * 62)
    admitted = pool.rank_and_admit(query_vec)

    assert len(admitted) == 2
    assert admitted[0][0].expert_id == "exp1"
    assert admitted[1][0].expert_id == "exp2"
    assert pool.experts["exp1"].admission_count == 1
    assert pool.experts["exp2"].admission_count == 1
    assert pool.experts["exp3"].admission_count == 0


def test_recombination_growth():
    pool = AdmissionGatedPool(n_features=32, top_k=2, recombination_threshold=0.8)
    pool.experts["e1"] = ExpertProfile("e1", "math", np.ones(32) / np.sqrt(32))
    pool.experts["e2"] = ExpertProfile("e2", "code", np.ones(32) / np.sqrt(32))

    # Query with low affinity
    query_vec = np.zeros(32)
    query_vec[0] = 1.0
    admitted = pool.rank_and_admit(query_vec)

    initial_count = len(pool.experts)
    new_exp = pool.recombine_if_uncertain(query_vec, admitted)

    assert new_exp is not None
    assert len(pool.experts) == initial_count + 1
    assert "recomb" in new_exp.expert_id
    assert new_exp.admission_count == 1


def test_admission_driven_pruning():
    pool = AdmissionGatedPool(n_features=16, survival_window=5)
    # Add two experts for the same action: e1 active, e2 stale
    pool.experts["e1"] = ExpertProfile("e1", "task_a", np.ones(16), admission_count=10, last_admitted_step=20)
    pool.experts["e2"] = ExpertProfile("e2", "task_a", np.ones(16), admission_count=1, last_admitted_step=2)
    # e3 is only expert for task_b (must not be pruned even if stale)
    pool.experts["e3"] = ExpertProfile("e3", "task_b", np.ones(16), admission_count=1, last_admitted_step=1)

    pool.step_counter = 22
    pruned = pool.prune_stale_experts()

    assert pruned == ["e2"]
    assert "e2" not in pool.experts
    assert "e1" in pool.experts
    assert "e3" in pool.experts  # Preserved as single representative for task_b
