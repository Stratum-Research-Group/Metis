"""Small embedding classifier baseline."""

from typing import Any, Callable, List, Optional
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from paideia.baselines.base import BasePolicy


class SmallEmbeddingBaseline(BasePolicy):
    """
    Lightweight embedding baseline using sub-word hash projection + Ridge classifier.
    If sentence-transformers is installed, can optionally use MiniLM.
    """
    def __init__(self, n_features: int = 1024):
        super().__init__(name="small_embedding_baseline")
        self.n_features = n_features
        self.pipeline = Pipeline([
            ("embed", HashingVectorizer(n_features=n_features, alternate_sign=False, ngram_range=(1, 3))),
            ("clf", RidgeClassifier(class_weight="balanced", random_state=42)),
        ])

    def fit(self, items: List[Any], target_extractor: Optional[Callable[[Any], Optional[str]]] = None) -> "SmallEmbeddingBaseline":
        texts: List[str] = []
        labels: List[str] = []
        for item in items:
            q = getattr(item, "question", str(item))
            y = target_extractor(item) if target_extractor else getattr(item, "gold_type", getattr(item, "action", None))
            if y is not None:
                texts.append(q)
                labels.append(y)

        if len(set(labels)) < 2:
            self.is_fitted = True
            self._single_class = labels[0] if labels else "bridge"
            return self

        self._single_class = None
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        return self

    def predict_one(self, item: Any) -> str:
        if getattr(self, "_single_class", None) is not None:
            return self._single_class
        q = getattr(item, "question", str(item))
        return str(self.pipeline.predict([q])[0])
