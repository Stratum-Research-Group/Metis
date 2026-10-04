"""
Evaluation harness and bootstrap comparison runner (Section 10).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from eval.metrics import (
    compute_contradiction_rate,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    compute_top_k_overlap,
)
from eval.snapshot_manifest import FrozenExperimentManifest, QueryItem


@dataclass
class ArmEvaluationResult:
    arm_name: str
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    contradiction_rate: float
    empty_intersection_rate: float
    p95_latency_sec: float
    per_query_recall: List[float]


def bootstrap_ci(
    deltas: List[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute mean and percentile bootstrap confidence interval for paired differences.
    Returns:
        (mean, ci_lower, ci_upper)
    """
    if not deltas:
        return 0.0, 0.0, 0.0

    arr = np.array(deltas, dtype=np.float64)
    mean_val = float(np.mean(arr))
    n = len(arr)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean_val, ci_lower, ci_upper


def verify_acceptance_gates(
    manifest: FrozenExperimentManifest,
    arm_result: ArmEvaluationResult,
    baseline_result: Optional[ArmEvaluationResult] = None,
) -> Dict[str, Any]:
    """
    Evaluate explicit acceptance gates according to Section 10.3.
    
    If gates are missing or unconfigured, acceptance is BLOCKED.
    """
    gates = manifest.acceptance_gates
    if not gates.is_fully_configured():
        return {
            "status": "BLOCKED",
            "reason": "Explicit acceptance gates contain unset values pending product requirements.",
            "gates_passed": False,
            "details": {},
        }

    details = {}
    passed = True

    # 1. Contradiction gate
    if arm_result.contradiction_rate > gates.contradiction_max:
        passed = False
        details["contradiction"] = f"FAILED: {arm_result.contradiction_rate:.4f} > max {gates.contradiction_max:.4f}"
    else:
        details["contradiction"] = "PASSED"

    # 2. Empty intersection gate
    if arm_result.empty_intersection_rate > gates.empty_intersection_rate_max:
        passed = False
        details["empty_intersection"] = f"FAILED: {arm_result.empty_intersection_rate:.4f} > max {gates.empty_intersection_rate_max:.4f}"
    else:
        details["empty_intersection"] = "PASSED"

    # 3. Latency gate
    if arm_result.p95_latency_sec > gates.p95_latency_budget_sec:
        passed = False
        details["latency"] = f"FAILED: {arm_result.p95_latency_sec:.4f}s > budget {gates.p95_latency_budget_sec:.4f}s"
    else:
        details["latency"] = "PASSED"

    # 4. Paired improvement against baseline if supplied
    if baseline_result is not None and gates.gain_min is not None:
        deltas = [
            a - b
            for a, b in zip(arm_result.per_query_recall, baseline_result.per_query_recall)
        ]
        mean_delta, ci_lower, ci_upper = bootstrap_ci(deltas)
        if ci_lower < gates.gain_min:
            passed = False
            details["gain"] = f"FAILED: bootstrap lower bound {ci_lower:.4f} < required gain_min {gates.gain_min:.4f}"
        else:
            details["gain"] = f"PASSED: CI [{ci_lower:.4f}, {ci_upper:.4f}] exceeds {gates.gain_min:.4f}"

    return {
        "status": "PASSED" if passed else "FAILED",
        "gates_passed": passed,
        "details": details,
    }
