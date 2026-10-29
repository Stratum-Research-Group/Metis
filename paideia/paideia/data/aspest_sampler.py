"""ASPEST active selective prediction sampler (arXiv:2304.03870).

Implements uncertainty-aware diverse sampling over Krisis abstentions and
conformal prediction ambiguities to curate high-yield training batches for Paideia.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from paideia.contracts.krisis_schema import (
    ActiveSamplingConfig,
    ASPESTAuditReport,
    KrisisActiveRecord,
)


def _tokenize(text: str) -> Set[str]:
    """Simple lowercase alphanumeric word tokenizer."""
    return set(re.findall(r"\w+", text.lower()))


def compute_distance(
    rec_a: KrisisActiveRecord,
    rec_b: KrisisActiveRecord,
    metric: str = "token_jaccard",
) -> float:
    """Computes diversity distance in [0, 1] between two query records.
    
    1.0 means completely distinct / orthogonal; 0.0 means identical.
    """
    if metric == "cosine" and rec_a.embedding and rec_b.embedding:
        v_a = rec_a.embedding
        v_b = rec_b.embedding
        if len(v_a) == len(v_b) and len(v_a) > 0:
            dot = sum(x * y for x, y in zip(v_a, v_b))
            norm_a = math.sqrt(sum(x * x for x in v_a))
            norm_b = math.sqrt(sum(y * y for y in v_b))
            if norm_a > 1e-9 and norm_b > 1e-9:
                sim = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
                return max(0.0, min(1.0, 1.0 - ((sim + 1.0) / 2.0)))

    # Default fallback: Token Jaccard distance over question text
    tokens_a = _tokenize(rec_a.question)
    tokens_b = _tokenize(rec_b.question)

    if not tokens_a and not tokens_b:
        return 0.0
    if not tokens_a or not tokens_b:
        return 1.0

    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    jaccard_sim = intersection / float(union) if union > 0 else 0.0
    return 1.0 - jaccard_sim


class ASPESTSampler:
    """Active Selective Prediction Sampler combining uncertainty and representativeness."""

    @staticmethod
    def select_active_batch(
        records: List[KrisisActiveRecord],
        config: Optional[ActiveSamplingConfig] = None,
    ) -> Tuple[List[KrisisActiveRecord], ASPESTAuditReport]:
        """Selects a subset of abstained/uncertain records balancing uncertainty and diversity.
        
        Args:
            records: Pool of candidate Krisis evaluation records.
            config: Sampling configuration (budget, beta, min_rejection_score, metric).
            
        Returns:
            Tuple of (selected records, audit report).
        """
        cfg = config or ActiveSamplingConfig()

        # Step 1: Filter eligible records based on rejection score
        eligible = [
            r for r in records
            if r.rejection_score >= cfg.min_rejection_score
            and r.decision_status != "selected"
        ]

        # Deduplicate on query_id
        seen_ids: Set[str] = set()
        unique_eligible: List[KrisisActiveRecord] = []
        for r in eligible:
            if r.query_id not in seen_ids:
                seen_ids.add(r.query_id)
                unique_eligible.append(r)

        total_pool = len(records)
        eligible_count = len(unique_eligible)

        if cfg.budget <= 0 or not unique_eligible:
            report = ASPESTAuditReport(
                total_pool_size=total_pool,
                eligible_count=eligible_count,
                budget=cfg.budget,
                selected_count=0,
                mean_rejection_score=0.0,
                mean_diversity_distance=0.0,
                reason_distribution={},
            )
            return [], report

        # If eligible count is within budget, take all
        if eligible_count <= cfg.budget:
            selected = list(unique_eligible)
            mean_rej = sum(r.rejection_score for r in selected) / len(selected)
            mean_dist = _calculate_mean_pairwise_distance(selected, cfg.diversity_metric)
            reasons: Dict[str, int] = {}
            for r in selected:
                reas = r.decision_reason or "unknown"
                reasons[reas] = reasons.get(reas, 0) + 1

            report = ASPESTAuditReport(
                total_pool_size=total_pool,
                eligible_count=eligible_count,
                budget=cfg.budget,
                selected_count=len(selected),
                mean_rejection_score=round(mean_rej, 4),
                mean_diversity_distance=round(mean_dist, 4),
                reason_distribution=reasons,
            )
            return selected, report

        # Step 2: Greedy iterative ASPEST selection
        # Initial seed: record with highest uncertainty (tie broken deterministically by query_id)
        unique_eligible.sort(key=lambda r: (-r.rejection_score, r.query_id))
        selected = [unique_eligible[0]]
        remaining = unique_eligible[1:]

        while len(selected) < cfg.budget and remaining:
            best_idx = -1
            best_score = -float("inf")

            for idx, cand in enumerate(remaining):
                u_score = cand.rejection_score  # In [0, 1]
                # Diversity is distance to nearest already-selected item
                d_score = min(
                    compute_distance(cand, s, cfg.diversity_metric)
                    for s in selected
                )
                # Combined acquisition score
                score = (cfg.beta * u_score) + ((1.0 - cfg.beta) * d_score)

                if score > best_score:
                    best_score = score
                    best_idx = idx
                elif math.isclose(score, best_score, rel_tol=1e-6):
                    # Deterministic tie-breaking on query_id
                    if cand.query_id < remaining[best_idx].query_id:
                        best_score = score
                        best_idx = idx

            assert best_idx >= 0
            selected.append(remaining.pop(best_idx))

        # Step 3: Compute audit telemetry
        mean_rej = sum(r.rejection_score for r in selected) / len(selected)
        mean_dist = _calculate_mean_pairwise_distance(selected, cfg.diversity_metric)
        reasons: Dict[str, int] = {}
        for r in selected:
            reas = r.decision_reason or "unknown"
            reasons[reas] = reasons.get(reas, 0) + 1

        report = ASPESTAuditReport(
            total_pool_size=total_pool,
            eligible_count=eligible_count,
            budget=cfg.budget,
            selected_count=len(selected),
            mean_rejection_score=round(mean_rej, 4),
            mean_diversity_distance=round(mean_dist, 4),
            reason_distribution=reasons,
        )
        return selected, report


def _calculate_mean_pairwise_distance(
    records: List[KrisisActiveRecord],
    metric: str,
) -> float:
    """Computes average pairwise distance across a set of records."""
    if len(records) <= 1:
        return 1.0

    total_dist = 0.0
    pairs = 0
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            total_dist += compute_distance(records[i], records[j], metric)
            pairs += 1

    return total_dist / float(pairs) if pairs > 0 else 1.0


def load_krisis_active_records(path: Path) -> List[KrisisActiveRecord]:
    """Loads KrisisActiveRecord instances from JSON or JSONL file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Active records file not found: {p}")

    records: List[KrisisActiveRecord] = []
    with open(p, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            raw_items = json.load(f)
        else:
            raw_items = [json.loads(line) for line in f if line.strip()]

    for item in raw_items:
        records.append(KrisisActiveRecord.model_validate(item))
    return records


def save_krisis_active_records(records: List[KrisisActiveRecord], path: Path) -> None:
    """Saves records to JSONL format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
