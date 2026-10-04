"""
Evaluation metrics computation (Section 10.2).
"""

from typing import Dict, List, Sequence, Set
import numpy as np


def compute_recall_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Set[str],
    k: int = 10,
) -> float:
    """
    Macro-averaged per-query Recall@k.
    If relevant_doc_ids is empty, returns 1.0 if no documents returned else 0.0 (abstention).
    """
    if not relevant_doc_ids:
        return 1.0 if len(ranked_doc_ids) == 0 else 0.0

    top_k = ranked_doc_ids[:k]
    hits = sum(1 for doc in top_k if doc in relevant_doc_ids)
    return float(hits / len(relevant_doc_ids))


def compute_mrr_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Set[str],
    k: int = 10,
) -> float:
    """
    Mean Reciprocal Rank @ k.
    """
    if not relevant_doc_ids:
        return 0.0
    for rank, doc in enumerate(ranked_doc_ids[:k], start=1):
        if doc in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Set[str],
    k: int = 10,
) -> float:
    """
    Normalized Discounted Cumulative Gain @ k (binary relevance).
    """
    if not relevant_doc_ids:
        return 0.0

    top_k = ranked_doc_ids[:k]
    dcg = 0.0
    for rank, doc in enumerate(top_k, start=1):
        if doc in relevant_doc_ids:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def compute_top_k_overlap(
    baseline_ranked_ids: Sequence[str],
    evaluated_ranked_ids: Sequence[str],
    k: int = 10,
) -> float:
    """
    Jaccard overlap or intersection fraction of top-k ranked items.
    """
    set_a = set(baseline_ranked_ids[:k])
    set_b = set(evaluated_ranked_ids[:k])
    if not set_a and not set_b:
        return 1.0
    intersection = set_a.intersection(set_b)
    return float(len(intersection) / max(len(set_a), 1))


def compute_contradiction_rate(
    returned_doc_ids: Sequence[str],
    contradictory_doc_ids: Set[str],
) -> float:
    """
    Fraction of returned documents that violate contradiction constraints.
    """
    if not returned_doc_ids:
        return 0.0
    contradictions = sum(1 for doc in returned_doc_ids if doc in contradictory_doc_ids)
    return float(contradictions / len(returned_doc_ids))
