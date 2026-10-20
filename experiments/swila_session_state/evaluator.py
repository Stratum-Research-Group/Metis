"""Statistical evaluation and Gate decision rules for session state experiment.

Section 9: Paired bootstrap 95% CI and 5 mutually exclusive classification classes.
"""

from __future__ import annotations

from typing import Literal
import numpy as np


DecisionClass = Literal["STRONG PASS", "PASS", "NO EFFECT", "BELOW TARGET", "INCONCLUSIVE"]


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_resamples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute paired bootstrap 95% confidence interval of mean differences.
    
    Returns:
        (point_estimate, lo, hi)
    """
    point_est = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    n = len(diffs)
    
    if n == 0:
        return 0.0, 0.0, 0.0

    # Resample with replacement
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(diffs[boot_indices], axis=1)

    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return point_est, lo, hi


def classify_decision(
    point_est: float,
    lo: float,
    hi: float,
    delta0: float = 0.05,
) -> DecisionClass:
    """Classify arm comparison against Section 9 decision rules.
    
    Classes are mutually exclusive and applied in this exact order:
    1. STRONG PASS: lo > delta0
    2. PASS: lo > 0 and point_est >= delta0 (and not class 1)
    3. NO EFFECT: hi <= 0
    4. BELOW TARGET: 0 < hi < delta0
    5. INCONCLUSIVE: everything else
    """
    if lo > delta0:
        return "STRONG PASS"
    if lo > 0.0 and point_est >= delta0:
        return "PASS"
    if hi <= 0.0:
        return "NO EFFECT"
    if 0.0 < hi < delta0:
        return "BELOW TARGET"
    return "INCONCLUSIVE"
