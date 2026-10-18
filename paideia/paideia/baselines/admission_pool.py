"""Admission-Gated Multi-Expert Pool Baseline (Phase 4 / mini-AGI model).

Implements:
- Admission Gating: Texts/queries rank the pool; only the top-K voted experts are admitted.
- Dynamic Recombination: Creates new expert profiles from parents when routing uncertainty is high.
- Admission-Driven Pruning: Safely prunes stale experts based on zero admission activity over a survival window.
- Isolated Moments: Preserves per-expert state and moments independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from paideia.baselines.base import BasePolicy


@dataclass
class ExpertProfile:
    """An individual expert inside the dynamic pool with isolated moments and centroids."""

    expert_id: str
    target_action: str
    centroid: np.ndarray
    admission_count: int = 0
    last_admitted_step: int = 0
    moments: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "target_action": self.target_action,
            "admission_count": self.admission_count,
            "last_admitted_step": self.last_admitted_step,
            "centroid_norm": float(np.linalg.norm(self.centroid)),
        }


class AdmissionGatedPool(BasePolicy):
    """
    Multi-expert routing policy using admission voting, top-K gating,
    recombination growth, and admission-based pruning.
    """

    def __init__(
        self,
        name: str = "admission_gated_pool",
        n_features: int = 256,
        top_k: int = 2,
        recombination_threshold: float = 0.25,
        survival_window: int = 20,
    ):
        super().__init__(name=name)
        self.n_features = n_features
        self.top_k = top_k
        self.recombination_threshold = recombination_threshold
        self.survival_window = survival_window
        self.vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False)
        self.experts: Dict[str, ExpertProfile] = {}
        self.step_counter: int = 0
        self._fallback_class: str = "bridge"

    def _embed(self, text: str) -> np.ndarray:
        vec = self.vectorizer.transform([text]).toarray()[0]
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def fit(
        self,
        items: List[Any],
        target_extractor: Optional[Callable[[Any], Optional[str]]] = None,
    ) -> "AdmissionGatedPool":
        """Initialize expert centroids per class from training items."""
        self.experts.clear()
        self.step_counter = 0

        class_vectors: Dict[str, List[np.ndarray]] = {}
        for item in items:
            q = getattr(item, "question", str(item))
            y = target_extractor(item) if target_extractor else getattr(item, "gold_type", getattr(item, "action", None))
            if y is not None:
                vec = self._embed(q)
                class_vectors.setdefault(y, []).append(vec)

        if not class_vectors:
            self.is_fitted = True
            return self

        for action, vecs in class_vectors.items():
            mean_vec = np.mean(vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-9:
                mean_vec = mean_vec / norm
            exp_id = f"exp_{action}_{hashlib.sha256(action.encode()).hexdigest()[:6]}"
            self.experts[exp_id] = ExpertProfile(
                expert_id=exp_id,
                target_action=action,
                centroid=mean_vec,
            )

        self._fallback_class = list(class_vectors.keys())[0]
        self.is_fitted = True
        return self

    def rank_and_admit(self, query_vec: np.ndarray) -> List[Tuple[ExpertProfile, float]]:
        """
        Rank all active experts by cosine affinity to the query vector.
        Admits the top-k most voted experts.
        """
        if not self.experts:
            return []

        scores: List[Tuple[ExpertProfile, float]] = []
        for exp in self.experts.values():
            affinity = float(np.dot(query_vec, exp.centroid))
            scores.append((exp, affinity))

        # Sort descending by affinity
        scores.sort(key=lambda x: x[1], reverse=True)
        admitted = scores[: self.top_k]

        # Record admission activity
        self.step_counter += 1
        for exp, _ in admitted:
            exp.admission_count += 1
            exp.last_admitted_step = self.step_counter

        return admitted

    def recombine_if_uncertain(
        self,
        query_vec: np.ndarray,
        admitted: List[Tuple[ExpertProfile, float]],
    ) -> Optional[ExpertProfile]:
        """
        If top admitted affinity is below the recombination threshold, assemble
        a new expert by recombining parent expert centroids.
        """
        if not admitted or admitted[0][1] >= self.recombination_threshold:
            return None

        # Recombine top parent experts
        parents = [p for p, _ in admitted if p is not None]
        if not parents:
            return None

        new_centroid = np.mean([p.centroid for p in parents], axis=0)
        norm = np.linalg.norm(new_centroid)
        if norm > 1e-9:
            new_centroid = new_centroid / norm

        new_id = f"exp_recomb_{self.step_counter}_{hashlib.sha256(new_centroid.tobytes()).hexdigest()[:6]}"
        new_expert = ExpertProfile(
            expert_id=new_id,
            target_action=parents[0].target_action,
            centroid=new_centroid,
            admission_count=1,
            last_admitted_step=self.step_counter,
        )
        self.experts[new_id] = new_expert
        return new_expert

    def prune_stale_experts(self) -> List[str]:
        """
        Prune experts that have not been admitted within the survival window.
        Guarantees at least one expert per unique target_action remains.
        """
        pruned_ids: List[str] = []
        # Group experts by action
        by_action: Dict[str, List[ExpertProfile]] = {}
        for exp in self.experts.values():
            by_action.setdefault(exp.target_action, []).append(exp)

        for action, exps in by_action.items():
            if len(exps) <= 1:
                continue  # Keep at least one expert per action
            for exp in exps:
                if (self.step_counter - exp.last_admitted_step) > self.survival_window and len(by_action[action]) > 1:
                    pruned_ids.append(exp.expert_id)
                    by_action[action].remove(exp)

        for pid in pruned_ids:
            self.experts.pop(pid, None)

        return pruned_ids

    def predict_one(self, item: Any) -> str:
        if not self.experts:
            return self._fallback_class

        q = getattr(item, "question", str(item))
        query_vec = self._embed(q)

        admitted = self.rank_and_admit(query_vec)
        if not admitted:
            return self._fallback_class

        # Recombination growth check
        self.recombine_if_uncertain(query_vec, admitted)

        # Trigger admission pruning if pool has grown
        if len(self.experts) > len(admitted) + 2 and self.step_counter % 10 == 0:
            self.prune_stale_experts()

        # Decision is determined by the top admitted expert
        return admitted[0][0].target_action
