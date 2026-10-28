"""KnowNo Conformal Uncertainty Engine for Krisis.

Implements finite-sample split conformal prediction sets with statistical coverage
guarantees (Ren et al., arXiv 2307.01928, CoRL 2023).

Replaces heuristic scalar thresholds (tau) with prediction sets C(X) = {c | s(c) >= 1 - q_hat}.
- |C(X)| == 1: Singleton set -> statistically validated single answer -> SELECT.
- |C(X)| > 1:  Ambiguous set -> multiple plausible candidates -> ABSTAIN / ASK_USER.
- |C(X)| == 0: Empty set -> no candidate meets calibration threshold -> ABSTAIN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import math
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ConformalPredictionSet:
    """The conformal prediction set and uncertainty diagnostics emitted for a query."""
    candidate_ids: List[str]
    scores: Dict[str, float]
    q_hat: float
    alpha: float
    threshold: float  # 1.0 - q_hat
    set_size: int = field(init=False)
    is_singleton: bool = field(init=False)
    is_empty: bool = field(init=False)
    status: str = field(init=False)  # "singleton", "ambiguous", "empty"
    selected_id: Optional[str] = field(init=False)

    def __post_init__(self) -> None:
        self.set_size = len(self.candidate_ids)
        self.is_singleton = (self.set_size == 1)
        self.is_empty = (self.set_size == 0)
        if self.is_singleton:
            self.status = "singleton"
            self.selected_id = self.candidate_ids[0]
        elif self.set_size > 1:
            self.status = "ambiguous"
            self.selected_id = None
        else:
            self.status = "empty"
            self.selected_id = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_ids": self.candidate_ids,
            "set_size": self.set_size,
            "q_hat": round(self.q_hat, 4),
            "alpha": self.alpha,
            "coverage_guarantee": round(1.0 - self.alpha, 4),
            "threshold": round(self.threshold, 4),
            "status": self.status,
            "selected_id": self.selected_id,
        }


class ConformalCalibrator:
    """Computes conformal quantiles and constructs prediction sets over candidate answers."""

    @staticmethod
    def compute_q_hat(calibration_scores: List[float], alpha: float = 0.05) -> float:
        """
        Computes the conformal non-conformity threshold q_hat at confidence level 1 - alpha.
        calibration_scores: scores s_i in [0, 1] of ground-truth correct answers on held-out data.
        Non-conformity score: E_i = 1 - s_i.
        q_hat is the ceil((n + 1)(1 - alpha)) / n quantile of nonconformity scores.
        """
        if not calibration_scores:
            raise ValueError("Calibration set cannot be empty for conformal calibration.")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")

        n = len(calibration_scores)
        # Nonconformity: lower score = more nonconforming
        nonconformities = [1.0 - float(s) for s in calibration_scores]
        nonconformities.sort()

        # Finite-sample correction index: ceil((n + 1) * (1 - alpha))
        rank = math.ceil((n + 1) * (1.0 - alpha))
        index = min(rank, n) - 1  # 0-indexed
        index = max(0, index)

        return float(nonconformities[index])

    @classmethod
    def evaluate(
        cls,
        scores: Dict[str, float | Decimal],
        q_hat: float,
        alpha: float = 0.05,
    ) -> ConformalPredictionSet:
        """
        Constructs the prediction set C(X) = {cand_id | score(cand_id) >= 1.0 - q_hat}.
        """
        threshold = 1.0 - float(q_hat)
        # Normalize all scores to float for comparison
        float_scores: Dict[str, float] = {
            cid: float(s) for cid, s in scores.items() if s is not None
        }

        # Select candidates meeting threshold: score >= 1 - q_hat
        # Tie-break: highest score first, then lexicographical ID
        survivors = [
            cid for cid, s in float_scores.items()
            if s >= threshold
        ]
        survivors.sort(key=lambda cid: (-float_scores[cid], cid))

        return ConformalPredictionSet(
            candidate_ids=survivors,
            scores=float_scores,
            q_hat=float(q_hat),
            alpha=float(alpha),
            threshold=threshold,
        )

    @classmethod
    def calibrate_and_evaluate(
        cls,
        scores: Dict[str, float | Decimal],
        calibration_scores: List[float],
        alpha: float = 0.05,
    ) -> ConformalPredictionSet:
        """Calibrates q_hat from held-out data and evaluates prediction set for current scores."""
        q_hat = cls.compute_q_hat(calibration_scores, alpha=alpha)
        return cls.evaluate(scores, q_hat=q_hat, alpha=alpha)
