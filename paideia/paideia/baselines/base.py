"""Base interface for all routing and policy models."""

import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple


class BasePolicy(ABC):
    """Abstract base class for routing and action prediction policies."""
    def __init__(self, name: str):
        self.name = name
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, items: List[Any], target_extractor: Optional[Any] = None) -> "BasePolicy":
        """Fit model on training data."""
        pass

    @abstractmethod
    def predict_one(self, item: Any) -> str:
        """Predict target action/class for a single item."""
        pass

    def predict(self, items: List[Any]) -> List[str]:
        """Predict for a batch of items."""
        return [self.predict_one(item) for item in items]

    def predict_with_timing(self, items: List[Any]) -> Tuple[List[str], List[float]]:
        """
        Predict for a batch and return per-item latency in milliseconds.
        """
        predictions: List[str] = []
        latencies_ms: List[float] = []
        for item in items:
            t0 = time.perf_counter()
            pred = self.predict_one(item)
            t1 = time.perf_counter()
            predictions.append(pred)
            latencies_ms.append((t1 - t0) * 1000.0)
        return predictions, latencies_ms
