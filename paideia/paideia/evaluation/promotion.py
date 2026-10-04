"""Adapter promotion gatekeeper enforcing preregistered metrics and guardrails."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from paideia.evaluation.bootstrap import BootstrapResult, compute_paired_bootstrap_ci
from paideia.evaluation.null_test import NullTestResult, paired_permutation_test


class OperationalGuardrails(BaseModel):
    """Guardrails that must pass for release promotion."""
    max_p95_latency_ms: float = Field(default=50.0, description="Maximum permitted p95 latency in ms")
    max_peak_memory_mb: float = Field(default=16000.0, description="Maximum permitted peak memory in MB")
    max_failure_rate: float = Field(default=0.01, description="Maximum permitted prediction failure rate")
    min_baseline_parity: float = Field(default=0.0, description="Minimum acceptable delta")


class PromotionVerdict(BaseModel):
    """Definitive release promotion evaluation verdict."""
    decision: Literal["PROMOTED", "REJECTED"] = Field(..., description="Final promotion decision")
    candidate_name: str = Field(..., description="Name of candidate model")
    baseline_name: str = Field(..., description="Name of reference baseline model")
    bootstrap_result: BootstrapResult = Field(..., description="Paired bootstrap 95% CI")
    null_test_result: NullTestResult = Field(..., description="Paired null hypothesis test outcome")
    guardrail_results: Dict[str, bool] = Field(..., description="Status of each guardrail")
    rejection_reasons: List[str] = Field(default_factory=list, description="Explicit failure reasons if rejected")
    is_positive_improvement: bool = Field(..., description="Whether all criteria met")


def evaluate_promotion(
    candidate_name: str,
    baseline_name: str,
    baseline_scores: List[float],
    candidate_scores: List[float],
    candidate_latencies_ms: List[float],
    candidate_failures: int,
    candidate_peak_mem_mb: float,
    guardrails: Optional[OperationalGuardrails] = None,
    n_resamples: int = 10000,
    seed: int = 42,
) -> PromotionVerdict:
    """
    Enforce non-negotiable Metis promotion criteria:
    1. Primary held-out metric delta > 0.
    2. Paired 95% bootstrap CI strictly excludes zero (ci_lower > 0).
    3. Paired null test p-value < 0.05.
    4. Operational guardrails (p95 latency, memory, failures) pass.
    """
    if guardrails is None:
        guardrails = OperationalGuardrails()

    bootstrap = compute_paired_bootstrap_ci(
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
        n_resamples=n_resamples,
        seed=seed,
    )

    null_test = paired_permutation_test(
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
        n_permutations=n_resamples,
        seed=seed,
    )

    p95_lat = float(sorted(candidate_latencies_ms)[int(0.95 * len(candidate_latencies_ms))]) if candidate_latencies_ms else 0.0
    failure_rate = candidate_failures / max(len(candidate_scores), 1)

    guardrail_checks = {
        "p95_latency_ok": (p95_lat <= guardrails.max_p95_latency_ms),
        "peak_memory_ok": (candidate_peak_mem_mb <= guardrails.max_peak_memory_mb),
        "failure_rate_ok": (failure_rate <= guardrails.max_failure_rate),
    }

    reasons: List[str] = []

    if bootstrap.mean_delta <= guardrails.min_baseline_parity:
        reasons.append(
            f"Mean metric delta {bootstrap.mean_delta:+.4f} does not exceed "
            f"baseline parity threshold ({guardrails.min_baseline_parity:.4f})."
        )
    if not (bootstrap.ci_lower > 0.0):
        reasons.append(
            f"Paired 95% bootstrap CI [{bootstrap.ci_lower:+.4f}, {bootstrap.ci_upper:+.4f}] "
            f"does not strictly exceed zero."
        )
    if not null_test.is_significant:
        reasons.append(
            f"Paired permutation test p-value ({null_test.p_value:.4f}) is not statistically "
            f"significant at alpha={null_test.alpha}."
        )
    for g_name, passed in guardrail_checks.items():
        if not passed:
            reasons.append(f"Operational guardrail violation: {g_name} failed.")

    decision: Literal["PROMOTED", "REJECTED"] = "PROMOTED" if len(reasons) == 0 else "REJECTED"

    return PromotionVerdict(
        decision=decision,
        candidate_name=candidate_name,
        baseline_name=baseline_name,
        bootstrap_result=bootstrap,
        null_test_result=null_test,
        guardrail_results=guardrail_checks,
        rejection_reasons=reasons,
        is_positive_improvement=(decision == "PROMOTED"),
    )
