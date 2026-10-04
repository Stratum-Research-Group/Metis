"""Unit tests for dokime.stats with known-answer fixtures."""

import numpy as np
import pytest

from dokime.stats import (
    paired_bootstrap_test,
    per_query_diff_analysis,
    single_run_bootstrap,
)


def test_single_run_bootstrap_known_fixture():
    # 5-element known fixture vector: [1.0, 1.0, 0.0, 1.0, 0.0]
    scores = [1.0, 1.0, 0.0, 1.0, 0.0]
    res = single_run_bootstrap(scores, num_samples=10000, seed=42)

    assert res["count"] == 5
    assert res["mean"] == 0.6
    # Check that CI surrounds mean and is bounded by [0.0, 1.0]
    assert 0.0 <= res["ci_lower"] <= 0.6
    assert 0.6 <= res["ci_upper"] <= 1.0
    # Seed 42 determinism check
    res2 = single_run_bootstrap(scores, num_samples=10000, seed=42)
    assert res["ci_lower"] == res2["ci_lower"]
    assert res["ci_upper"] == res2["ci_upper"]


def test_single_run_bootstrap_edge_cases():
    # Empty scores (n=0: unmeasured / not-measured)
    empty_res = single_run_bootstrap([], seed=42)
    assert empty_res["mean"] is None
    assert empty_res["ci_lower"] is None
    assert empty_res["ci_upper"] is None
    assert empty_res["status"] == "not-measured"
    assert empty_res["count"] == 0

    # Measured zeros (n>0: actual score = 0.0)
    zero_res = single_run_bootstrap([0.0, 0.0, 0.0], seed=42)
    assert zero_res["mean"] == 0.0
    assert zero_res["ci_lower"] == 0.0
    assert zero_res["ci_upper"] == 0.0
    assert zero_res["count"] == 3

    # Constant scores (all 1.0)
    const_res = single_run_bootstrap([1.0, 1.0, 1.0, 1.0], seed=42)
    assert const_res["mean"] == 1.0
    assert const_res["ci_lower"] == 1.0
    assert const_res["ci_upper"] == 1.0
    assert const_res["std"] == 0.0


def test_paired_bootstrap_identical_runs():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0]
    res = paired_bootstrap_test(scores, scores, num_samples=10000, seed=42)

    assert res["delta"] == 0.0
    assert res["ci_lower"] == 0.0
    assert res["ci_upper"] == 0.0
    assert res["ci_excludes_zero"] is False
    assert res["verdict"] == "NO_CHANGE"


def test_paired_bootstrap_significant_improvement():
    # Baseline: all 0.0, Candidate: all 1.0
    baseline = [0.0] * 20
    candidate = [1.0] * 20
    res = paired_bootstrap_test(baseline, candidate, num_samples=10000, seed=42)

    assert res["delta"] == 1.0
    assert res["ci_lower"] == 1.0
    assert res["ci_upper"] == 1.0
    assert res["ci_excludes_zero"] is True
    assert res["verdict"] == "IMPROVEMENT"
    assert res["is_significant"] is True


def test_paired_bootstrap_significant_regression():
    # Baseline: all 1.0, Candidate: all 0.0
    baseline = [1.0] * 20
    candidate = [0.0] * 20
    res = paired_bootstrap_test(baseline, candidate, num_samples=10000, seed=42)

    assert res["delta"] == -1.0
    assert res["ci_lower"] == -1.0
    assert res["ci_upper"] == -1.0
    assert res["ci_excludes_zero"] is True
    assert res["verdict"] == "REGRESSION"
    assert res["is_significant"] is True


def test_paired_bootstrap_partial_diff():
    # 20 queries: baseline has 6/20 (0.3), candidate has 18/20 (0.9)
    base = [1.0] * 6 + [0.0] * 14
    cand = [1.0] * 18 + [0.0] * 2
    res = paired_bootstrap_test(base, cand, num_samples=10000, seed=42)

    assert res["delta"] == 0.6
    assert res["mean_baseline"] == 0.3
    assert res["mean_candidate"] == 0.9
    assert res["ci_lower"] > 0.0
    assert res["ci_excludes_zero"] is True
    assert res["verdict"] == "IMPROVEMENT"


def test_per_query_diff_analysis():
    base = {"q1": 1.0, "q2": 0.0, "q3": 1.0}
    cand = {"q1": 1.0, "q2": 1.0, "q3": 0.0}

    diff = per_query_diff_analysis(base, cand)
    assert diff["total_paired"] == 3
    assert diff["improved_count"] == 1
    assert diff["regressed_count"] == 1
    assert diff["unchanged_count"] == 1
    assert diff["improved"][0]["query_id"] == "q2"
    assert diff["regressed"][0]["query_id"] == "q3"
    assert diff["unchanged"][0]["query_id"] == "q1"


def test_stats_refuses_nonfinite_inputs():
    from dokime.stats import round_stat

    for bad_val in [float('nan'), float('inf'), -float('inf')]:
        with pytest.raises(ValueError, match="scores must contain only finite numbers"):
            single_run_bootstrap([bad_val, 1.0], seed=42)

        with pytest.raises(ValueError, match="scores_candidate must contain only finite numbers"):
            paired_bootstrap_test([0.0, 0.0], [bad_val, 1.0], seed=42)

        with pytest.raises(ValueError, match="scores_baseline must contain only finite numbers"):
            paired_bootstrap_test([bad_val, 1.0], [0.0, 0.0], seed=42)

        with pytest.raises(ValueError, match="Cannot round non-finite value"):
            round_stat(bad_val)

        with pytest.raises(ValueError, match="Non-finite score encountered"):
            per_query_diff_analysis({"q1": bad_val}, {"q1": 1.0})


def test_per_query_diff_analysis_small_effects_and_zero():
    base = {
        "q_pos_5": 0.0,
        "q_neg_5": 1e-5,
        "q_pos_9": 0.0,
        "q_neg_9": 1e-9,
        "q_zero": 0.0,
    }
    cand = {
        "q_pos_5": 1e-5,
        "q_neg_5": 0.0,
        "q_pos_9": 1e-9,
        "q_neg_9": 0.0,
        "q_zero": 0.0,
    }

    diff = per_query_diff_analysis(base, cand)
    assert diff["total_paired"] == 5
    assert diff["improved_count"] == 2
    assert diff["regressed_count"] == 2
    assert diff["unchanged_count"] == 1

    diffs = diff["diffs_by_query"]
    assert diffs["q_pos_5"] == 1e-5
    assert diffs["q_neg_5"] == -1e-5
    assert diffs["q_pos_9"] == 1e-9
    assert diffs["q_neg_9"] == -1e-9
    assert diffs["q_zero"] == 0.0

    improved_ids = {e["query_id"]: e["diff"] for e in diff["improved"]}
    assert improved_ids == {"q_pos_5": 1e-5, "q_pos_9": 1e-9}

    regressed_ids = {e["query_id"]: e["diff"] for e in diff["regressed"]}
    assert regressed_ids == {"q_neg_5": -1e-5, "q_neg_9": -1e-9}

    unchanged_ids = {e["query_id"]: e["diff"] for e in diff["unchanged"]}
    assert unchanged_ids == {"q_zero": 0.0}


