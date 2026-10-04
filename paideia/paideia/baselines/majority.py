"""Majority class baseline predictor."""

from collections import Counter
from typing import Any, Callable, List, Optional
from paideia.baselines.base import BasePolicy


class MajorityBaseline(BasePolicy):
    """Predicts the most frequent class in the training split."""
    def __init__(self, fallback_class: str = "bridge"):
        super().__init__(name="majority_baseline")
        self.majority_class: str = fallback_class

    def fit(self, items: List[Any], target_extractor: Optional[Callable[[Any], Optional[str]]] = None) -> "MajorityBaseline":
        if not items:
            return self
        targets: List[str] = []
        for item in items:
            t = target_extractor(item) if target_extractor else getattr(item, "gold_type", getattr(item, "action", None))
            if t:
                targets.append(t)
        if targets:
            counts = Counter(targets)
            self.majority_class = counts.most_common(1)[0][0]
        self.is_fitted = True
        return self

    def predict_one(self, item: Any) -> str:
        return self.majority_class
