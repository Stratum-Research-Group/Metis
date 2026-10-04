"""Statistical analysis for Dokime evaluations.

Follows the LTHCM convention:
- Point estimates: sample mean
- Paired bootstrap resampling: 10,000 resamples, seed 42, 95% confidence intervals
- Preserves per-query diffs and evaluates zero-exclusion for regression / improvement.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def round_stat(val: float | None, default_precision: int = 4, max_precision: int = 8) -> float | None:
    """Round a float to default_precision, or more decimals if needed so nonzero values do not become 0.0."""
    if val is None:
        return None
    if not np.isfinite(val):
        raise ValueError(f"Cannot round non-finite value: {val}")
    if val == 0.0:
        return 0.0
    val_float = float(val)
    rounded = round(val_float, default_precision)
    if rounded != 0.0:
        return rounded
    for p in range(default_precision + 1, max_precision + 1):
        r = round(val_float, p)
        if r != 0.0:
            return r
    return float(f"{val_float:.{default_precision}e}")


def single_run_bootstrap(
    scores: list[float],
    num_samples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Calculate point estimate and bootstrap confidence interval for a single run.

    Args:
        scores: List of query-level scores (typically in [0.0, 1.0]).
        num_samples: Number of bootstrap resamples (default: 10,000).
        confidence_level: Confidence level (default: 0.95 for 95% CI).
        seed: Random seed for deterministic reproducibility.

    Returns:
        dict with mean, ci_lower, ci_upper, std, count.
    """
    n = len(scores)
    if n == 0:
        return {
            "mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "std": None,
            "count": 0,
            "num_samples": num_samples,
            "status": "not-measured",
        }

    arr = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must contain only finite numbers (found NaN or Inf)")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n == 1 or np.all(arr == arr[0]):
        return {
            "mean": round_stat(mean),
            "ci_lower": round_stat(mean),
            "ci_upper": round_stat(mean),
            "std": round_stat(std),
            "count": n,
            "num_samples": num_samples,
        }

    rng = np.random.default_rng(seed)
    # Generate resampled index matrix (num_samples, n)
    indices = rng.integers(0, n, size=(num_samples, n))
    bootstrap_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean": round_stat(mean),
        "ci_lower": round_stat(ci_lower),
        "ci_upper": round_stat(ci_upper),
        "std": round_stat(std),
        "count": n,
        "num_samples": num_samples,
    }


def paired_bootstrap_test(
    scores_baseline: list[float],
    scores_candidate: list[float],
    num_samples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Perform a paired bootstrap hypothesis test comparing two runs on the same query set.

    Matches LTHCM benchmark convention:
    delta = mean(candidate) - mean(baseline)
    Positive delta indicates candidate outperformed baseline.

    Args:
        scores_baseline: Scores from the baseline run (Run A).
        scores_candidate: Scores from the candidate run (Run B).
        num_samples: Bootstrap resamples (default 10,000).
        confidence_level: Nominal confidence level (default 0.95).
        seed: Random seed for exact reproducibility (default 42).

    Returns:
        Dict with mean_baseline, mean_candidate, delta, ci_lower, ci_upper,
        p_value, is_significant, ci_excludes_zero, and verdict.
    """
    assert len(scores_baseline) == len(scores_candidate), (
        f"Length mismatch: {len(scores_baseline)} vs {len(scores_candidate)}"
    )
    n = len(scores_baseline)
    if n == 0:
        return {
            "mean_baseline": 0.0,
            "mean_candidate": 0.0,
            "delta": 0.0,
            "p_value": 1.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "is_significant": False,
            "ci_excludes_zero": False,
            "verdict": "NO_CHANGE",
            "num_queries": 0,
            "num_samples": num_samples,
        }

    arr_base = np.asarray(scores_baseline, dtype=np.float64)
    arr_cand = np.asarray(scores_candidate, dtype=np.float64)
    if not np.all(np.isfinite(arr_base)):
        raise ValueError("scores_baseline must contain only finite numbers (found NaN or Inf)")
    if not np.all(np.isfinite(arr_cand)):
        raise ValueError("scores_candidate must contain only finite numbers (found NaN or Inf)")

    mean_base = float(np.mean(arr_base))
    mean_cand = float(np.mean(arr_cand))
    observed_delta = mean_cand - mean_base

    # Paired differences (candidate - baseline)
    diffs = arr_cand - arr_base

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(num_samples, n))
    bootstrap_deltas = np.mean(diffs[indices], axis=1)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_deltas, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_deltas, 100.0 * (1.0 - alpha / 2.0)))

    # Two-sided empirical p-value
    if observed_delta > 0:
        p_val = 2.0 * float(np.mean(bootstrap_deltas <= 0))
    elif observed_delta < 0:
        p_val = 2.0 * float(np.mean(bootstrap_deltas >= 0))
    else:
        p_val = 1.0

    p_val = min(1.0, max(0.0, p_val))

    # CI excludes zero if both lower and upper have same strictly non-zero sign
    ci_excludes_zero = bool((ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0))

    if ci_lower > 0:
        verdict = "IMPROVEMENT"
    elif ci_upper < 0:
        verdict = "REGRESSION"
    else:
        verdict = "NO_CHANGE"

    return {
        "mean_baseline": round_stat(mean_base),
        "mean_candidate": round_stat(mean_cand),
        "delta": round_stat(observed_delta),
        "ci_lower": round_stat(ci_lower),
        "ci_upper": round_stat(ci_upper),
        "p_value": round_stat(p_val),
        "is_significant": bool(p_val < (1.0 - confidence_level)),
        "ci_excludes_zero": ci_excludes_zero,
        "verdict": verdict,
        "num_queries": n,
        "num_samples": num_samples,
    }


def per_query_diff_analysis(
    baseline_scores: dict[str, float],
    candidate_scores: dict[str, float],
) -> dict[str, Any]:
    """Analyze per-query differences between two runs.

    Returns:
        dict containing paired diffs, improved queries, regressed queries, unchanged queries.
    """
    common_ids = sorted(set(baseline_scores.keys()) & set(candidate_scores.keys()))
    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    diffs_by_query: dict[str, float] = {}

    for qid in common_ids:
        s_base = baseline_scores[qid]
        s_cand = candidate_scores[qid]
        if not np.isfinite(s_base) or not np.isfinite(s_cand):
            raise ValueError(f"Non-finite score encountered for query '{qid}' in per_query_diff_analysis")
        raw_diff = float(s_cand) - float(s_base)
        d = round_stat(raw_diff)
        diffs_by_query[qid] = d

        entry = {
            "query_id": qid,
            "baseline": s_base,
            "candidate": s_cand,
            "diff": d,
        }
        if raw_diff > 0:
            improved.append(entry)
        elif raw_diff < 0:
            regressed.append(entry)
        else:
            unchanged.append(entry)

    return {
        "total_paired": len(common_ids),
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "unchanged_count": len(unchanged),
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "diffs_by_query": diffs_by_query,
    }
