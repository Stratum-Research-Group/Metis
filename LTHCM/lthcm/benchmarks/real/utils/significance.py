"""
Statistical Significance Testing for IR and NLP Benchmarks
==========================================================
Implements paired randomization (sign-flip permutation) testing for the null p-value
and paired bootstrap resampling (10,000 iterations) for 95% confidence intervals
per SIGIR/TREC/EMNLP standards (Smucker et al., 2007).
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def paired_randomization_test(
    scores_a: List[float],
    scores_b: List[float],
    num_permutations: int = 10000,
    seed: int = 42,
) -> Tuple[float, str]:
    """
    Computes exact or Monte Carlo paired sign-flip permutation p-value under H0: mean(a - b) = 0.
    Uses the standard unbiased Monte Carlo estimator (count + 1) / (B + 1),
    strictly bounding p >= 1 / (B + 1) and avoiding p = 0.0000.
    """
    diffs = np.asarray(scores_a, dtype=np.float64) - np.asarray(scores_b, dtype=np.float64)
    n = len(diffs)
    if n == 0 or np.all(diffs == 0):
        return 1.0, "1.0000"

    t_obs = abs(float(np.mean(diffs)))
    if t_obs < 1e-12:
        return 1.0, "1.0000"

    # For small n <= 12, compute exact full enumeration over 2^n sign flips
    if n <= 12:
        signs = np.array(
            [[1.0 if (i >> j) & 1 else -1.0 for j in range(n)] for i in range(1 << n)],
            dtype=np.float64,
        )
        perm_stats = np.abs(np.mean(signs * diffs, axis=1))
        p_val = float(np.mean(perm_stats >= t_obs))
        p_fmt = f"{p_val:.4f}"
        return p_val, p_fmt

    # Monte Carlo sign-flip permutation test
    rng = np.random.default_rng(seed)
    random_signs = rng.choice([-1.0, 1.0], size=(num_permutations, n))
    perm_stats = np.abs(np.mean(random_signs * diffs, axis=1))

    count_greater_or_equal = int(np.sum(perm_stats >= t_obs))
    p_val = float((count_greater_or_equal + 1) / (num_permutations + 1))

    if count_greater_or_equal == 0:
        p_fmt = f"< {1.0 / (num_permutations + 1):.4f}"
    else:
        p_fmt = f"{p_val:.4f}"

    return p_val, p_fmt


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    num_samples: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Performs paired statistical testing comparing two systems over the same query set.
    The null p-value is evaluated via a paired randomization / permutation test.
    The confidence interval for delta is evaluated via paired bootstrap resampling.

    Args:
        scores_a: List of per-query metric scores for System A (e.g. LTHCM).
        scores_b: List of per-query metric scores for System B (e.g. Baseline).
        num_samples: Number of bootstrap/permutation iterations (standard: 10,000).
        confidence_level: Nominal confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for exact reproducibility.

    Returns:
        Dict with:
          - mean_a, mean_b, delta (mean_a - mean_b)
          - p_value (two-sided paired permutation p-value, bounded >= 1/(B+1))
          - p_value_formatted: string representation (e.g. '< 0.0001' or '0.0123')
          - ci_lower, ci_upper (empirical bootstrap confidence interval of delta)
          - is_significant: bool (p_value < (1.0 - confidence_level))
          - test_type: description of the statistical protocol
    """
    assert len(scores_a) == len(scores_b), f"Length mismatch: {len(scores_a)} vs {len(scores_b)}"
    n = len(scores_a)
    if n == 0:
        return {
            "mean_a": 0.0,
            "mean_b": 0.0,
            "delta": 0.0,
            "p_value": 1.0,
            "p_value_formatted": "1.0000",
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "is_significant": False,
            "num_queries": 0,
            "num_samples": num_samples,
            "test_type": "paired_randomization_test_with_bootstrap_ci",
        }

    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)

    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    observed_delta = mean_a - mean_b

    # Paired differences
    diffs = arr_a - arr_b

    # 1. Null Hypothesis Significance Testing: Paired Randomization / Permutation Test
    p_val, p_fmt = paired_randomization_test(
        scores_a, scores_b, num_permutations=num_samples, seed=seed
    )

    # 2. Confidence Interval for Delta: Paired Bootstrap Resampling
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(num_samples, n))
    bootstrap_deltas = np.mean(diffs[indices], axis=1)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_deltas, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_deltas, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "delta": round(observed_delta, 4),
        "p_value": round(p_val, 4),
        "p_value_formatted": p_fmt,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "is_significant": bool(p_val < (1.0 - confidence_level)),
        "num_queries": n,
        "num_samples": num_samples,
        "test_type": "paired_randomization_test_with_bootstrap_ci",
    }

