"""Boundary-aware practice batch sampler (EoG Item 2, arXiv:2610.35598).

Implements statistical variance estimation over repeated rollout attempts and
selects learning frontier batches (highest variance) mixed with retention checks
and hard diagnostic cases.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from paideia.contracts.boundary_schema import (
    BoundarySelectionConfig,
    BoundarySelectionReport,
    CaseAttemptCategory,
    CaseAttemptRecord,
    CaseAttemptStats,
)


def compute_case_stats(
    case_id: str,
    attempts: Sequence[CaseAttemptRecord],
    min_variance: float = 1e-4,
) -> CaseAttemptStats:
    """Compute mean score, unbiased sample variance, and category for a case.

    Args:
        case_id: Unique case identifier.
        attempts: List of repeated attempt records for this case.
        min_variance: Threshold above which a case is classified as BOUNDARY.

    Returns:
        CaseAttemptStats record.
    """
    if not attempts:
        raise ValueError(f"Cannot compute statistics for empty attempt list on case '{case_id}'")

    n = len(attempts)
    scores = [float(a.score) for a in attempts]
    mean_s = sum(scores) / float(n)

    if n < 2:
        variance = 0.0
        std = 0.0
    else:
        variance = sum((s - mean_s) ** 2 for s in scores) / float(n - 1)
        # Numerical guard against small negative floats
        variance = max(0.0, variance)
        std = math.sqrt(variance)

    # Classify category
    if variance > min_variance:
        category = CaseAttemptCategory.BOUNDARY
    elif abs(mean_s - 1.0) < 1e-6:
        category = CaseAttemptCategory.RETENTION_CHECK
    elif abs(mean_s - 0.0) < 1e-6:
        category = CaseAttemptCategory.HARD_DIAGNOSTIC
    else:
        category = CaseAttemptCategory.CONSISTENT_PARTIAL

    # Failure mode tally
    failure_counts: Dict[str, int] = {}
    for a in attempts:
        if a.failure_label:
            failure_counts[a.failure_label] = failure_counts.get(a.failure_label, 0) + 1

    return CaseAttemptStats(
        case_id=case_id,
        num_attempts=n,
        mean_score=round(mean_s, 6),
        sample_variance=round(variance, 6),
        sample_std=round(std, 6),
        category=category,
        min_score=min(scores),
        max_score=max(scores),
        failure_counts=failure_counts,
        scores=scores,
    )


def compute_attempt_stats_by_case(
    records: Sequence[CaseAttemptRecord],
    min_variance: float = 1e-4,
) -> Dict[str, CaseAttemptStats]:
    """Group attempt records by case_id and compute summary statistics.

    Returns:
        Dictionary mapping case_id -> CaseAttemptStats.
    """
    grouped: Dict[str, List[CaseAttemptRecord]] = {}
    for r in records:
        grouped.setdefault(r.case_id, []).append(r)

    stats_by_case: Dict[str, CaseAttemptStats] = {}
    for cid, attempts in grouped.items():
        stats_by_case[cid] = compute_case_stats(cid, attempts, min_variance=min_variance)

    return stats_by_case


def select_boundary_practice_batch(
    stats_by_case: Dict[str, CaseAttemptStats],
    config: Optional[BoundarySelectionConfig] = None,
) -> Tuple[List[str], BoundarySelectionReport]:
    """Select a boundary-aware practice batch balancing variance and retention.

    Stratifies available cases into:
      1. Boundary (sigma^2 > min_variance): highest contrastive learning gradient
      2. Retention checks (mean == 1.0, sigma^2 == 0): guard against catastrophic forgetting
      3. Hard diagnostics (mean == 0.0, sigma^2 == 0): explore stubborn failure modes
      4. Consistent partials (0 < mean < 1.0, sigma^2 == 0): invariant partial credit

    Args:
        stats_by_case: Mapping of case_id to evaluated attempt statistics.
        config: Selection ratios and budget configuration.

    Returns:
        Tuple of (List of selected case IDs, Detailed selection audit report).
    """
    cfg = config or BoundarySelectionConfig()
    rng = random.Random(cfg.seed)

    # 1. Stratify cases
    boundary_pool: List[str] = []
    retention_pool: List[str] = []
    hard_pool: List[str] = []
    partial_pool: List[str] = []

    for cid, stat in stats_by_case.items():
        if stat.category == CaseAttemptCategory.BOUNDARY:
            boundary_pool.append(cid)
        elif stat.category == CaseAttemptCategory.RETENTION_CHECK:
            retention_pool.append(cid)
        elif stat.category == CaseAttemptCategory.HARD_DIAGNOSTIC:
            hard_pool.append(cid)
        else:
            partial_pool.append(cid)

    # Sort boundary cases by descending variance to prioritize steepest gradients
    if cfg.prioritize_highest_variance:
        boundary_pool.sort(key=lambda c: (-stats_by_case[c].sample_variance, c))
    else:
        rng.shuffle(boundary_pool)

    rng.shuffle(retention_pool)
    rng.shuffle(hard_pool)
    rng.shuffle(partial_pool)

    # 2. Compute quotas
    budget = min(cfg.budget, len(stats_by_case))
    target_boundary = int(round(budget * cfg.boundary_ratio))
    target_retention = int(round(budget * cfg.retention_ratio))
    target_hard = int(round(budget * cfg.hard_ratio))

    # Adjust sum to not exceed budget
    while (target_boundary + target_retention + target_hard) > budget:
        if target_boundary > 0:
            target_boundary -= 1
        elif target_hard > 0:
            target_hard -= 1
        elif target_retention > 0:
            target_retention -= 1

    selected_boundary = boundary_pool[:target_boundary]
    selected_retention = retention_pool[:target_retention]
    selected_hard = hard_pool[:target_hard]

    selected_set = set(selected_boundary + selected_retention + selected_hard)

    # 3. Fill remaining quota if any stratum had insufficient candidates
    remaining_quota = budget - len(selected_set)
    if remaining_quota > 0:
        fill_candidates = (
            [c for c in boundary_pool if c not in selected_set]
            + [c for c in retention_pool if c not in selected_set]
            + [c for c in hard_pool if c not in selected_set]
            + [c for c in partial_pool if c not in selected_set]
        )
        for cand in fill_candidates:
            if len(selected_set) >= budget:
                break
            selected_set.add(cand)

    final_selected = sorted(list(selected_set))

    # 4. Metrics
    selected_boundary_subset = [c for c in final_selected if stats_by_case[c].category == CaseAttemptCategory.BOUNDARY]
    selected_retention_subset = [c for c in final_selected if stats_by_case[c].category == CaseAttemptCategory.RETENTION_CHECK]
    selected_hard_subset = [c for c in final_selected if stats_by_case[c].category == CaseAttemptCategory.HARD_DIAGNOSTIC]

    mean_b_var = 0.0
    if selected_boundary_subset:
        mean_b_var = sum(stats_by_case[c].sample_variance for c in selected_boundary_subset) / float(len(selected_boundary_subset))

    num_attempts = next(iter(stats_by_case.values())).num_attempts if stats_by_case else 0

    report = BoundarySelectionReport(
        total_cases_analyzed=len(stats_by_case),
        num_attempts_per_case=num_attempts,
        boundary_pool_size=len(boundary_pool),
        retention_pool_size=len(retention_pool),
        hard_pool_size=len(hard_pool),
        consistent_partial_pool_size=len(partial_pool),
        selected_count=len(final_selected),
        selected_boundary_count=len(selected_boundary_subset),
        selected_retention_count=len(selected_retention_subset),
        selected_hard_count=len(selected_hard_subset),
        selected_cases=final_selected,
        mean_boundary_variance=round(mean_b_var, 6),
        selection_strategy="boundary_aware_mixture",
    )

    return final_selected, report


def select_uniform_practice_batch(
    stats_by_case: Dict[str, CaseAttemptStats],
    budget: int = 20,
    seed: int = 42,
) -> Tuple[List[str], BoundarySelectionReport]:
    """Baseline uniform random sampling at matched budget for comparative evaluation."""
    all_cases = sorted(list(stats_by_case.keys()))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(all_cases, min(budget, len(all_cases))))

    b_count = sum(1 for c in chosen if stats_by_case[c].category == CaseAttemptCategory.BOUNDARY)
    r_count = sum(1 for c in chosen if stats_by_case[c].category == CaseAttemptCategory.RETENTION_CHECK)
    h_count = sum(1 for c in chosen if stats_by_case[c].category == CaseAttemptCategory.HARD_DIAGNOSTIC)
    cp_count = sum(1 for c in chosen if stats_by_case[c].category == CaseAttemptCategory.CONSISTENT_PARTIAL)

    b_cases = [c for c in chosen if stats_by_case[c].category == CaseAttemptCategory.BOUNDARY]
    mean_b_var = 0.0
    if b_cases:
        mean_b_var = sum(stats_by_case[c].sample_variance for c in b_cases) / float(len(b_cases))

    num_attempts = next(iter(stats_by_case.values())).num_attempts if stats_by_case else 0

    report = BoundarySelectionReport(
        total_cases_analyzed=len(stats_by_case),
        num_attempts_per_case=num_attempts,
        boundary_pool_size=sum(1 for s in stats_by_case.values() if s.category == CaseAttemptCategory.BOUNDARY),
        retention_pool_size=sum(1 for s in stats_by_case.values() if s.category == CaseAttemptCategory.RETENTION_CHECK),
        hard_pool_size=sum(1 for s in stats_by_case.values() if s.category == CaseAttemptCategory.HARD_DIAGNOSTIC),
        consistent_partial_pool_size=sum(1 for s in stats_by_case.values() if s.category == CaseAttemptCategory.CONSISTENT_PARTIAL),
        selected_count=len(chosen),
        selected_boundary_count=b_count,
        selected_retention_count=r_count,
        selected_hard_count=h_count,
        selected_cases=chosen,
        mean_boundary_variance=round(mean_b_var, 6),
        selection_strategy="uniform_baseline",
    )

    return chosen, report


def load_case_attempts(path: Path) -> List[CaseAttemptRecord]:
    """Load attempt records from JSON or JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Attempt records file not found at {path}")

    records: List[CaseAttemptRecord] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        # JSON array
        data = json.loads(text)
        for d in data:
            records.append(CaseAttemptRecord.model_validate(d))
    else:
        # JSON Lines
        for line in text.splitlines():
            line_str = line.strip()
            if line_str:
                records.append(CaseAttemptRecord.model_validate_json(line_str))

    return records


def save_case_attempts(records: Sequence[CaseAttemptRecord], path: Path) -> None:
    """Save attempt records to JSON or JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in records], f, indent=2)
