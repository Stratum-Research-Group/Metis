"""Unit tests for paired statistical evaluation, promotion gates, and Dokime bundles."""

from pathlib import Path
import pytest

from paideia.evaluation.bootstrap import compute_paired_bootstrap_ci
from paideia.evaluation.null_test import paired_permutation_test, mcnemar_test
from paideia.evaluation.promotion import OperationalGuardrails, evaluate_promotion
from paideia.evaluation.dokime_bundle import DokimeRunBundle


def test_paired_bootstrap_ci():
    # Synthetic case: Candidate strictly better on most items
    baseline = [0.5, 0.6, 0.4, 0.5, 0.5, 0.6, 0.4, 0.5]
    candidate = [0.8, 0.9, 0.7, 0.8, 0.8, 0.9, 0.7, 0.8]

    res = compute_paired_bootstrap_ci(baseline, candidate, n_resamples=1000, seed=42)
    assert res.mean_delta > 0.0
    assert res.ci_lower > 0.0
    assert res.ci_excludes_zero is True

    # Case where there is no difference
    res_null = compute_paired_bootstrap_ci(baseline, baseline, n_resamples=1000, seed=42)
    assert res_null.mean_delta == 0.0
    assert res_null.ci_lower <= 0.0 <= res_null.ci_upper
    assert res_null.ci_excludes_zero is False


def test_paired_null_tests():
    # Permutation test
    baseline = [0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2, 0.1]
    candidate = [0.9, 0.8, 0.9, 0.7, 0.8, 0.9, 0.8, 0.9]

    perm_res = paired_permutation_test(baseline, candidate, n_permutations=1000, seed=42)
    assert perm_res.p_value < 0.01
    assert perm_res.is_significant is True

    # McNemar test
    b_correct = [True, False, False, True, False, False, False, False, False, False]
    c_correct = [True, True, True, True, True, True, True, True, False, True]
    mcnemar_res = mcnemar_test(b_correct, c_correct)
    assert mcnemar_res.is_significant is True


def test_promotion_gate_decision():
    # 1. Definite win
    baseline = [0.2] * 20
    candidate = [0.8] * 20
    latencies = [12.0] * 20

    verdict_win = evaluate_promotion(
        candidate_name="cand_v1",
        baseline_name="base_v1",
        baseline_scores=baseline,
        candidate_scores=candidate,
        candidate_latencies_ms=latencies,
        candidate_failures=0,
        candidate_peak_mem_mb=1000.0,
        n_resamples=1000,
        seed=42,
    )
    assert verdict_win.decision == "PROMOTED"
    assert verdict_win.is_positive_improvement is True
    assert len(verdict_win.rejection_reasons) == 0

    # 2. Rejection due to parity/null failure
    verdict_reject = evaluate_promotion(
        candidate_name="cand_v2",
        baseline_name="base_v1",
        baseline_scores=baseline,
        candidate_scores=baseline,  # Identical
        candidate_latencies_ms=latencies,
        candidate_failures=0,
        candidate_peak_mem_mb=1000.0,
        n_resamples=1000,
        seed=42,
    )
    assert verdict_reject.decision == "REJECTED"
    assert verdict_reject.is_positive_improvement is False
    assert len(verdict_reject.rejection_reasons) > 0


def test_dokime_bundle_generation(tmp_path):
    bundle = DokimeRunBundle(
        run_id="run-test-dokime",
        task_name="router_eval",
        candidate_name="adapter_v1",
        baseline_name="regex_v1",
        item_ids=["q1", "q2"],
        gold_labels=["comparison", "bridge"],
        baseline_predictions=["comparison", "comparison"],
        candidate_predictions=["comparison", "bridge"],
        baseline_scores=[1.0, 0.0],
        candidate_scores=[1.0, 1.0],
        candidate_latencies_ms=[12.0, 14.0],
        verdict={"decision": "PROMOTED", "p_value": 0.02},
    )

    out_file = tmp_path / "dokime_bundle.json"
    bundle.save(out_file)
    assert out_file.exists()
    assert bundle.bundle_hash is not None
    assert len(bundle.bundle_hash) == 64
