"""Sufficient Input Subsets (SIS) Explainer.

Implements black-box backward selection for identifying minimal sufficient subsets
of inputs that satisfy a model decision threshold tau (Carter et al., Google Research,
arXiv:2107.09752).

Key properties:
- Greedy backward selection: at each step evaluates f on every input with one more
  position masked, removing the position whose exclusion retains the highest score.
- Ties break deterministically to the lowest index.
- Minimal along the greedy path: nothing further can be removed without dropping f below tau.
- Disjoint collection: masks previously found rationales to uncover alternative/redundant pathways.
- Model-agnostic: requires only f(x), needing no gradients, model weights, or internal hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class SISResult:
    """The result of a single Sufficient Input Subset search."""
    subset_indices: List[int]
    subset_items: List[str]
    initial_score: float
    final_score: float
    threshold: float
    total_items: int
    steps_taken: int
    is_sufficient: bool
    reduction_ratio: float = field(init=False)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_items > 0:
            removed = self.total_items - len(self.subset_indices)
            self.reduction_ratio = round(removed / self.total_items, 4)
        else:
            self.reduction_ratio = 0.0

    @property
    def reconstructed_text(self) -> str:
        """Joins the subset items back into readable text."""
        return " ".join(self.subset_items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subset_indices": self.subset_indices,
            "subset_items": self.subset_items,
            "reconstructed_text": self.reconstructed_text,
            "initial_score": round(self.initial_score, 4),
            "final_score": round(self.final_score, 4),
            "threshold": round(self.threshold, 4),
            "total_items": self.total_items,
            "subset_size": len(self.subset_indices),
            "steps_taken": self.steps_taken,
            "is_sufficient": self.is_sufficient,
            "reduction_ratio": self.reduction_ratio,
            "trajectory": self.trajectory,
        }


@dataclass
class SISCollectionResult:
    """Collection of disjoint Sufficient Input Subsets uncovered from the same input."""
    subsets: List[SISResult]
    total_items: int
    residual_score: float
    threshold: float

    @property
    def total_subsets(self) -> int:
        return len(self.subsets)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_subsets": self.total_subsets,
            "total_items": self.total_items,
            "residual_score": round(self.residual_score, 4),
            "threshold": round(self.threshold, 4),
            "subsets": [s.to_dict() for s in self.subsets],
        }


class SufficientInputSubsets:
    """Black-box greedy backward selection explainer."""

    def __init__(
        self,
        score_fn: Callable[[List[str]], float],
        tau: float,
        mask_token: Optional[str] = None,
    ) -> None:
        """
        score_fn: function taking a list of string items (active prompt components) and returning float score.
        tau: minimum threshold of score_fn for a subset to be considered sufficient.
        mask_token: if provided, masked elements are replaced by mask_token instead of being deleted from sequence.
        """
        self.score_fn = score_fn
        self.tau = float(tau)
        self.mask_token = mask_token

    def _render_input(self, full_items: Sequence[str], active_indices: Sequence[int]) -> List[str]:
        """Renders either by subset extraction or mask_token substitution."""
        if self.mask_token is None:
            # Drop omitted elements completely
            return [full_items[idx] for idx in sorted(active_indices)]
        else:
            # Substitute omitted elements with mask_token
            active_set = set(active_indices)
            return [
                full_items[i] if i in active_set else self.mask_token
                for i in range(len(full_items))
            ]

    def find_sis(self, items: List[str]) -> SISResult:
        """Finds a minimal sufficient input subset via greedy backward selection."""
        n = len(items)
        if n == 0:
            return SISResult(
                subset_indices=[],
                subset_items=[],
                initial_score=0.0,
                final_score=0.0,
                threshold=self.tau,
                total_items=0,
                steps_taken=0,
                is_sufficient=False,
            )

        active = list(range(n))
        initial_input = self._render_input(items, active)
        initial_score = float(self.score_fn(initial_input))

        # Check if full input even satisfies threshold
        if initial_score < self.tau:
            return SISResult(
                subset_indices=active,
                subset_items=[items[i] for i in active],
                initial_score=initial_score,
                final_score=initial_score,
                threshold=self.tau,
                total_items=n,
                steps_taken=0,
                is_sufficient=False,
            )

        current_score = initial_score
        trajectory = [{
            "step": 0,
            "active_count": len(active),
            "score": round(current_score, 4),
            "removed_index": None,
        }]

        steps = 0
        while len(active) > 1:
            best_candidate_removal: Optional[int] = None
            best_score = -float("inf")

            # Evaluate each possible one-position removal
            for idx in active:
                candidate_active = [i for i in active if i != idx]
                candidate_input = self._render_input(items, candidate_active)
                score = float(self.score_fn(candidate_input))

                # Greedy choice: removal that retains highest score
                # Deterministic tie-breaking: prefer lowest index (earlier position)
                if score > best_score:
                    best_score = score
                    best_candidate_removal = idx

            # If the best removal drops below threshold tau, we cannot reduce further
            if best_score < self.tau or best_candidate_removal is None:
                break

            # Apply greedy removal
            active.remove(best_candidate_removal)
            current_score = best_score
            steps += 1
            trajectory.append({
                "step": steps,
                "active_count": len(active),
                "score": round(current_score, 4),
                "removed_index": best_candidate_removal,
            })

        active.sort()
        return SISResult(
            subset_indices=active,
            subset_items=[items[i] for i in active],
            initial_score=initial_score,
            final_score=current_score,
            threshold=self.tau,
            total_items=n,
            steps_taken=steps,
            is_sufficient=True,
            trajectory=trajectory,
        )

    def find_sis_collection(self, items: List[str], max_subsets: int = 5) -> SISCollectionResult:
        """Finds multiple disjoint rationales by masking earlier selected subsets."""
        n = len(items)
        subsets: List[SISResult] = []
        available_indices = set(range(n))

        for _ in range(max_subsets):
            if not available_indices:
                break

            # Re-map available indices to local sub-list
            local_mapping = sorted(list(available_indices))
            local_items = [items[idx] for idx in local_mapping]

            # Wrapper score function mapping local indices back to global context
            def sub_score_fn(sub_local: List[str]) -> float:
                # Reconstruct full sequence with only currently active elements
                # sub_local corresponds to a subset of local_items
                active_local_set = set(sub_local)
                active_global = [
                    global_idx for global_idx in local_mapping
                    if items[global_idx] in active_local_set
                ]
                rendered = self._render_input(items, active_global)
                return float(self.score_fn(rendered))

            sub_explainer = SufficientInputSubsets(
                score_fn=sub_score_fn,
                tau=self.tau,
                mask_token=self.mask_token,
            )
            sis = sub_explainer.find_sis(local_items)

            if not sis.is_sufficient or not sis.subset_indices:
                break

            # Map local subset indices back to global indices
            global_subset_indices = sorted([local_mapping[li] for li in sis.subset_indices])
            global_subset_items = [items[gi] for gi in global_subset_indices]

            global_sis = SISResult(
                subset_indices=global_subset_indices,
                subset_items=global_subset_items,
                initial_score=sis.initial_score,
                final_score=sis.final_score,
                threshold=self.tau,
                total_items=n,
                steps_taken=sis.steps_taken,
                is_sufficient=True,
                trajectory=sis.trajectory,
            )
            subsets.append(global_sis)

            # Mask out the uncovered rationale to find disjoint alternatives
            for gi in global_subset_indices:
                available_indices.discard(gi)

        # Residual score of remaining unselected items
        remaining_indices = sorted(list(available_indices))
        if remaining_indices:
            residual_input = self._render_input(items, remaining_indices)
            residual_score = float(self.score_fn(residual_input))
        else:
            residual_score = 0.0

        return SISCollectionResult(
            subsets=subsets,
            total_items=n,
            residual_score=residual_score,
            threshold=self.tau,
        )


def split_text_into_units(text: str, unit: str = "sentence") -> List[str]:
    """Splits input text into discrete explanation units."""
    if not text.strip():
        return []
    if unit == "sentence":
        # Split on sentence boundaries (.!?) preserving punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]
    elif unit == "paragraph":
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]
    elif unit == "word":
        words = text.split()
        return [w.strip() for w in words if w.strip()]
    else:
        raise ValueError(f"Unsupported unit '{unit}', expected 'sentence', 'paragraph', or 'word'.")


def explain_text_sufficiency(
    text: str,
    score_fn: Callable[[str], float],
    tau: float,
    unit: str = "sentence",
    find_disjoint: bool = False,
    max_subsets: int = 3,
) -> SISResult | SISCollectionResult:
    """Convenience helper explaining text sufficiency at sentence, paragraph, or word granularity."""
    units = split_text_into_units(text, unit=unit)

    def sequence_score_fn(sub_units: List[str]) -> float:
        joined_text = " ".join(sub_units)
        return float(score_fn(joined_text))

    explainer = SufficientInputSubsets(score_fn=sequence_score_fn, tau=tau)
    if find_disjoint:
        return explainer.find_sis_collection(units, max_subsets=max_subsets)
    return explainer.find_sis(units)
