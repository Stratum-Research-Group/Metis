"""Existing LTHCM regex rule baseline router."""

import re
from typing import Any, Callable, List, Optional
from paideia.baselines.base import BasePolicy


class RegexRouterBaseline(BasePolicy):
    """
    LTHCM heuristic regex router.
    Detects comparison queries vs bridge multi-hop queries.
    """
    def __init__(self):
        super().__init__(name="lthcm_regex_router")
        self.comp_patterns = [
            re.compile(r"\b(?:are|were|do|did|have|had|can|could|will|would)\s+both\b", re.IGNORECASE),
            re.compile(r"\bboth\b", re.IGNORECASE),
            re.compile(r"\beither\b", re.IGNORECASE),
            re.compile(r"\bneither\b", re.IGNORECASE),
            re.compile(r"\bwhich\s+(?:one|of\s+these|of\s+the|film|book|album|novel|city|country|person|river|lake|mountain|university|station|song)\b", re.IGNORECASE),
            re.compile(r"\bwho\s+(?:was|is)\s+(?:born\s+first|older|younger|first)\b", re.IGNORECASE),
            re.compile(r"\bwho\s+lived\s+longer\b", re.IGNORECASE),
            re.compile(r"\bwho\s+(?:has|had|won|earned)\s+more\b", re.IGNORECASE),
            re.compile(r"\b\w+\s+or\s+\w+\b", re.IGNORECASE),
            re.compile(r"\bmore\b", re.IGNORECASE),
            re.compile(r"\bolder\b", re.IGNORECASE),
            re.compile(r"\byounger\b", re.IGNORECASE),
            re.compile(r"\bearlier\b", re.IGNORECASE),
            re.compile(r"\blater\b", re.IGNORECASE),
            re.compile(r"\bfirst\b", re.IGNORECASE),
            re.compile(r"\bsame\b", re.IGNORECASE),
            re.compile(r"\bdifferent\b", re.IGNORECASE),
        ]
        self.is_fitted = True

    def fit(self, items: List[Any], target_extractor: Optional[Callable[[Any], Optional[str]]] = None) -> "RegexRouterBaseline":
        # Rule router is deterministic and has no parameters to fit
        self.is_fitted = True
        return self

    def predict_one(self, item: Any) -> str:
        question = getattr(item, "question", str(item))
        q_lower = question.strip().lower()
        for pat in self.comp_patterns:
            if pat.search(q_lower):
                return "comparison"
        return "bridge"
