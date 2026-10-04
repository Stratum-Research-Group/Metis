"""Paired bootstrap confidence interval calculator conforming to LTHCM convention."""

from typing import List, Tuple
import numpy as np
from pydantic import BaseModel, Field


class BootstrapResult(BaseModel):
    """Statistical summary from paired bootstrap resampling."""
    mean_baseline: float = Field(..., description="Sample mean of baseline metric")
    mean_candidate: float = Field(..., description="Sample mean of candidate metric")
    mean_delta: float = Field(..., description="Observed mean effect size (candidate - baseline)")
    ci_lower: float = Field(..., description="Lower bound of 95% paired bootstrap CI")
    ci_upper: float = Field(..., description="Upper bound of 95% paired bootstrap CI")
    ci_excludes_zero: bool = Field(..., description="Whether CI strictly excludes zero")
    n_resamples: int = Field(default=10000, description="Resampling iterations")
    seed: int = Field(default=42, description="RNG seed")


def compute_paired_bootstrap_ci(
    baseline_scores: List[float],
    candidate_scores: List[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """
    Compute paired bootstrap confidence interval for the difference (candidate - baseline).
    Matches LTHCM standard: 10,000 resamples, seed 42.
    
    Args:
        baseline_scores: Per-item score vector for baseline.
        candidate_scores: Per-item score vector for candidate model.
        n_resamples: Number of bootstrap draws (default 10,000).
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for deterministic reproducibility.
        
    Returns:
        BootstrapResult object.
    """
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError(
            f"Paired evaluation requires identical lengths: "
            f"got {len(baseline_scores)} vs {len(candidate_scores)}"
        )
    if len(baseline_scores) == 0:
        raise ValueError("Score vectors cannot be empty")

    b_arr = np.array(baseline_scores, dtype=np.float64)
    c_arr = np.array(candidate_scores, dtype=np.float64)
    deltas = c_arr - b_arr

    n = len(deltas)
    rng = np.random.default_rng(seed)

    # Draw n_resamples with replacement
    indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_deltas = deltas[indices].mean(axis=1)

    ci_low = float(np.percentile(resampled_deltas, 100 * (alpha / 2.0)))
    ci_high = float(np.percentile(resampled_deltas, 100 * (1.0 - alpha / 2.0)))
    mean_d = float(deltas.mean())

    excludes_zero = (ci_low > 0.0) or (ci_high < 0.0)

    return BootstrapResult(
        mean_baseline=float(b_arr.mean()),
        mean_candidate=float(c_arr.mean()),
        mean_delta=mean_d,
        ci_lower=ci_low,
        ci_upper=ci_high,
        ci_excludes_zero=excludes_zero,
        n_resamples=n_resamples,
        seed=seed,
    )
