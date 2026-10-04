"""Lexical and logistic regression baseline classifier."""

from typing import Any, Callable, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from paideia.baselines.base import BasePolicy


class LexicalLogisticBaseline(BasePolicy):
    """
    TF-IDF n-gram vectorizer paired with L2-regularized Logistic Regression.
    Lightweight, fast CPU inference, highly competitive baseline for short text classification.
    """
    def __init__(self, c_param: float = 1.0, max_features: int = 5000):
        super().__init__(name="lexical_logistic_baseline")
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, lowercase=True)),
            ("clf", LogisticRegression(C=c_param, max_iter=500, class_weight="balanced", random_state=42)),
        ])

    def fit(self, items: List[Any], target_extractor: Optional[Callable[[Any], Optional[str]]] = None) -> "LexicalLogisticBaseline":
        texts: List[str] = []
        labels: List[str] = []
        for item in items:
            q = getattr(item, "question", str(item))
            y = target_extractor(item) if target_extractor else getattr(item, "gold_type", getattr(item, "action", None))
            if y is not None:
                texts.append(q)
                labels.append(y)

        if len(set(labels)) < 2:
            # Degenerate case: single class
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
