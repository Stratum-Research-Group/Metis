"""
In-Context Batch Isolation & Permutation Invariance Guard (Item 115).

Inspired by 'Judged in Company: In-Context Batching Makes LLM Verdicts Depend on Other Items'
(arXiv:2610.40241). Audits multi-item judge evaluations for neighbor contamination,
ordering hysteresis, and batch context leakage by comparing batched evaluations
against isolated single-item evaluations across forward, reverse, and shuffled permutations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class BatchItem:
    """An individual evaluation item or claim submitted in a batch."""
    item_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemPermutationResult:
    """Audit comparison for a single item under isolated and batched order permutations."""
    item_id: str
    isolated_verdict: Any
    forward_batch_verdict: Any
    reverse_batch_verdict: Any
    shuffled_batch_verdict: Any
    is_order_invariant: bool
    is_batch_isolated: bool
    diagnostic: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "isolated_verdict": str(self.isolated_verdict),
            "forward_batch_verdict": str(self.forward_batch_verdict),
            "reverse_batch_verdict": str(self.reverse_batch_verdict),
            "shuffled_batch_verdict": str(self.shuffled_batch_verdict),
            "is_order_invariant": self.is_order_invariant,
            "is_batch_isolated": self.is_batch_isolated,
            "diagnostic": self.diagnostic,
        }


@dataclass
class BatchIsolationReport:
    """Consolidated report on batching stability, neighbor contamination, and order invariance."""
    total_items: int
    isolated_agreement_ratio: float
    permutation_invariance_ratio: float
    is_batching_safe: bool
    contaminated_item_ids: List[str] = field(default_factory=list)
    item_results: List[ItemPermutationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "isolated_agreement_ratio": round(self.isolated_agreement_ratio, 4),
            "permutation_invariance_ratio": round(self.permutation_invariance_ratio, 4),
            "is_batching_safe": self.is_batching_safe,
            "contaminated_item_ids": self.contaminated_item_ids,
            "item_results": [r.to_dict() for r in self.item_results],
        }


class BatchIsolationAuditor:
    """Audits batch evaluations against isolated single-item evaluations."""

    @classmethod
    def audit_batch(
        cls,
        items: Sequence[BatchItem],
        isolated_eval_fn: Callable[[BatchItem], Any],
        batch_eval_fn: Callable[[Sequence[BatchItem]], Dict[str, Any]],
        seed: int = 42,
    ) -> BatchIsolationReport:
        """
        Audit a batch of evaluation items.

        Args:
            items: Sequence of BatchItem objects.
            isolated_eval_fn: Evaluates a single item in total isolation.
            batch_eval_fn: Evaluates a sequence of items in a single batch prompt and returns {item_id: verdict}.
        """
        num_items = len(items)
        if num_items == 0:
            return BatchIsolationReport(
                total_items=0,
                isolated_agreement_ratio=1.0,
                permutation_invariance_ratio=1.0,
                is_batching_safe=True,
            )

        # 1. Isolated ground-truth evaluation per item
        isolated_verdicts: Dict[str, Any] = {}
        for it in items:
            isolated_verdicts[it.item_id] = isolated_eval_fn(it)

        # 2. Forward batch evaluation
        forward_verdicts = batch_eval_fn(items)

        # 3. Reverse batch evaluation
        reverse_items = list(reversed(items))
        reverse_verdicts = batch_eval_fn(reverse_items)

        # 4. Deterministically shuffled batch evaluation
        rng = random.Random(seed)
        shuffled_items = list(items)
        rng.shuffle(shuffled_items)
        shuffled_verdicts = batch_eval_fn(shuffled_items)

        # 5. Compare stability and detect contamination
        isolated_matches = 0
        order_invariant_matches = 0
        contaminated_ids: List[str] = []
        item_results: List[ItemPermutationResult] = []

        for it in items:
            i_id = it.item_id
            iso_v = isolated_verdicts.get(i_id)
            fwd_v = forward_verdicts.get(i_id)
            rev_v = reverse_verdicts.get(i_id)
            shf_v = shuffled_verdicts.get(i_id)

            is_order_inv = (fwd_v == rev_v == shf_v)
            is_isolated_match = (fwd_v == iso_v) and is_order_inv

            if is_order_inv:
                order_invariant_matches += 1
            if is_isolated_match:
                isolated_matches += 1
            else:
                contaminated_ids.append(i_id)

            diag = None
            if not is_isolated_match:
                diag = (
                    f"Contamination on item '{i_id}': isolated={iso_v!r}, "
                    f"fwd_batch={fwd_v!r}, rev_batch={rev_v!r}, shuffled={shf_v!r}"
                )

            item_results.append(
                ItemPermutationResult(
                    item_id=i_id,
                    isolated_verdict=iso_v,
                    forward_batch_verdict=fwd_v,
                    reverse_batch_verdict=rev_v,
                    shuffled_batch_verdict=shf_v,
                    is_order_invariant=is_order_inv,
                    is_batch_isolated=is_isolated_match,
                    diagnostic=diag,
                )
            )

        iso_ratio = isolated_matches / num_items
        perm_ratio = order_invariant_matches / num_items
        is_safe = (iso_ratio == 1.0 and perm_ratio == 1.0)

        return BatchIsolationReport(
            total_items=num_items,
            isolated_agreement_ratio=round(iso_ratio, 4),
            permutation_invariance_ratio=round(perm_ratio, 4),
            is_batching_safe=is_safe,
            contaminated_item_ids=contaminated_ids,
            item_results=item_results,
        )
