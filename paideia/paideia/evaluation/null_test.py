"""Valid paired null hypothesis testing (paired sign-permutation and McNemar tests)."""

from typing import List
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field


class NullTestResult(BaseModel):
    """Result of paired null hypothesis statistical test."""
    test_name: str = Field(..., description="Name of test: 'paired_permutation_test' or 'mcnemar'")
    statistic: float = Field(..., description="Observed test statistic (e.g. mean delta or chi2)")
    p_value: float = Field(..., description="Two-tailed p-value under null hypothesis")
    is_significant: bool = Field(..., description="Whether p-value < alpha (default 0.05)")
    alpha: float = Field(default=0.05, description="Significance threshold")


def paired_permutation_test(
    baseline_scores: List[float],
    candidate_scores: List[float],
    n_permutations: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> NullTestResult:
    """
    Perform a two-tailed paired sign-permutation test.
    Under the null hypothesis H0, the paired difference d_i has zero expectation and
    its sign is equally likely to be positive or negative.
    
    Args:
        baseline_scores: Per-item score vector for baseline.
        candidate_scores: Per-item score vector for candidate.
        n_permutations: Number of random sign flips.
        seed: Random seed.
        alpha: Significance threshold.
        
    Returns:
        NullTestResult.
    """
    if len(baseline_scores) != len(candidate_scores):
        raise ValueError("Score vectors must be equal length")

    b_arr = np.array(baseline_scores, dtype=np.float64)
    c_arr = np.array(candidate_scores, dtype=np.float64)
    deltas = c_arr - b_arr

    n = len(deltas)
    if n == 0:
        raise ValueError("Cannot test empty vectors")

    observed_stat = float(np.abs(np.mean(deltas)))

    rng = np.random.default_rng(seed)
    # Random signs +1 or -1 with p=0.5
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    permuted_means = np.abs(np.mean(deltas * signs, axis=1))

    # Two-tailed p-value with Laplace continuity correction
    p_val = float((np.sum(permuted_means >= observed_stat) + 1) / (n_permutations + 1))

    return NullTestResult(
        test_name="paired_permutation_test",
        statistic=float(np.mean(deltas)),
        p_value=p_val,
        is_significant=(p_val < alpha),
        alpha=alpha,
    )


def mcnemar_test(
    baseline_correct: List[bool],
    candidate_correct: List[bool],
    alpha: float = 0.05,
) -> NullTestResult:
    """
    Perform McNemar's exact test on paired binary correctness outcomes.
    
    b: Baseline wrong, Candidate correct (win for candidate)
    c: Baseline correct, Candidate wrong (loss for candidate)
    """
    if len(baseline_correct) != len(candidate_correct):
        raise ValueError("Vectors must be equal length")

    # Contingency table counts
    b = sum(1 for bc, cc in zip(baseline_correct, candidate_correct) if not bc and cc)
    c = sum(1 for bc, cc in zip(baseline_correct, candidate_correct) if bc and not cc)

    total_discordant = b + c
    if total_discordant == 0:
        return NullTestResult(
            test_name="mcnemar_exact",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            alpha=alpha,
        )

    # Exact binomial test for discordant pairs under p=0.5
    binom_res = stats.binomtest(b, total_discordant, p=0.5, alternative="two-sided")
    p_val = float(binom_res.pvalue)

    return NullTestResult(
        test_name="mcnemar_exact",
        statistic=float(b - c),
        p_value=p_val,
        is_significant=(p_val < alpha),
        alpha=alpha,
    )
