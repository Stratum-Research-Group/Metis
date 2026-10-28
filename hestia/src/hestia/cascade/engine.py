"""Core Speculative Cascades engine for Hestia serving.

Implements early-exit drafter/verifier serving with deferral rules,
acceptance tracking, and speedup estimation (ICLR 2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass
class CascadeConfig:
    """Configuration for a speculative drafter-verifier cascade pair."""
    draft_model: str
    verifier_model: str
    deferral_threshold: float = 0.80
    criterion: str = "mean_prob"  # "mean_prob", "min_prob", "margin"
    draft_max_tokens: Optional[int] = None
    cost_ratio: float = 0.25  # Relative compute/FLOP ratio of draft to verifier model

    def __post_init__(self) -> None:
        if not (0.0 <= self.deferral_threshold <= 1.0):
            raise ValueError(f"deferral_threshold must be in [0, 1], got {self.deferral_threshold}")
        if self.criterion not in ("mean_prob", "min_prob", "margin"):
            raise ValueError(f"Unsupported criterion: {self.criterion}")


@dataclass
class CascadeDecision:
    """Evaluation outcome and telemetry for a cascaded request."""
    status: str  # "ACCEPTED_DRAFT" | "DEFERRED_TO_VERIFIER"
    chosen_model: str
    deferral_score: float
    threshold: float
    criterion: str
    draft_tokens: int
    verifier_tokens: int
    total_tokens: int
    speedup_estimate: float
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "chosen_model": self.chosen_model,
            "deferral_score": round(self.deferral_score, 4),
            "threshold": round(self.threshold, 4),
            "criterion": self.criterion,
            "draft_tokens": self.draft_tokens,
            "verifier_tokens": self.verifier_tokens,
            "total_tokens": self.total_tokens,
            "speedup_estimate": round(self.speedup_estimate, 3),
            "latency_ms": round(self.latency_ms, 2),
        }


class SpeculativeCascadeEngine:
    """Manages deferral rule evaluation and execution across model cascades."""

    def __init__(self, config: CascadeConfig) -> None:
        self.config = config

    @staticmethod
    def extract_token_probs(logprobs_data: Optional[Dict[str, Any]]) -> List[float]:
        """Extracts linear token probabilities from OpenAI / llama.cpp logprobs structure."""
        if not logprobs_data:
            return []

        # Standard OpenAI format: logprobs.content -> list of items with logprob float
        content_items = logprobs_data.get("content")
        if isinstance(content_items, list):
            probs: List[float] = []
            for item in content_items:
                if isinstance(item, dict) and "logprob" in item:
                    lp = float(item["logprob"])
                    probs.append(math.exp(lp))
            return probs

        # llama.cpp raw format: token_logprobs list
        token_logprobs = logprobs_data.get("token_logprobs")
        if isinstance(token_logprobs, list):
            return [math.exp(float(lp)) for lp in token_logprobs if lp is not None]

        return []

    def compute_deferral_score(self, probs: List[float]) -> float:
        """Computes deferral score against the configured criterion."""
        if not probs:
            return 0.0

        if self.config.criterion == "mean_prob":
            return sum(probs) / len(probs)
        elif self.config.criterion == "min_prob":
            return min(probs)
        elif self.config.criterion == "margin":
            # Margin proxy: proportion of confident tokens >= 0.8
            high_conf = sum(1 for p in probs if p >= 0.8)
            return high_conf / len(probs)
        return 0.0

    async def execute(
        self,
        request_body: Dict[str, Any],
        forward_func: Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]],
    ) -> tuple[Dict[str, Any], CascadeDecision]:
        """
        Executes speculative cascade:
        1. Queries draft model with logprobs enabled.
        2. Evaluates token confidence / deferral metric.
        3. If >= threshold: accepts draft response early.
        4. If < threshold: cascades to full verifier model.
        """
        start_time = time.perf_counter()

        # Step 1: Draft model generation
        draft_body = dict(request_body)
        draft_body["model"] = self.config.draft_model
        draft_body["logprobs"] = True
        if self.config.draft_max_tokens:
            draft_body["max_tokens"] = self.config.draft_max_tokens

        draft_resp = await forward_func(self.config.draft_model, draft_body)

        draft_usage = draft_resp.get("usage", {})
        draft_tokens = int(draft_usage.get("completion_tokens", 0))

        # Extract logprobs from first choice
        choices = draft_resp.get("choices", [])
        logprobs_data = choices[0].get("logprobs") if choices else None
        probs = self.extract_token_probs(logprobs_data)

        score = self.compute_deferral_score(probs)

        # Step 2: Evaluate acceptance rule
        if score >= self.config.deferral_threshold and draft_tokens > 0:
            # Accepted draft path (early exit)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            # Speedup relative to full verifier run (1 / cost_ratio)
            speedup = 1.0 / max(self.config.cost_ratio, 0.05)

            decision = CascadeDecision(
                status="ACCEPTED_DRAFT",
                chosen_model=self.config.draft_model,
                deferral_score=score,
                threshold=self.config.deferral_threshold,
                criterion=self.config.criterion,
                draft_tokens=draft_tokens,
                verifier_tokens=0,
                total_tokens=draft_tokens,
                speedup_estimate=speedup,
                latency_ms=elapsed_ms,
            )

            # Annotate response with cascade metadata
            if "usage" not in draft_resp:
                draft_resp["usage"] = {}
            draft_resp["usage"]["cascade"] = decision.to_dict()
            return draft_resp, decision

        # Step 3: Deferral path to verifier model
        verifier_body = dict(request_body)
        verifier_body["model"] = self.config.verifier_model
        # Verifier does not require logprobs unless requested by original caller
        if not request_body.get("logprobs"):
            verifier_body.pop("logprobs", None)

        verifier_resp = await forward_func(self.config.verifier_model, verifier_body)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        verifier_usage = verifier_resp.get("usage", {})
        verifier_tokens = int(verifier_usage.get("completion_tokens", 0))
        total_tokens = draft_tokens + verifier_tokens

        # Speedup when deferred is slightly below 1.0 due to draft overhead
        speedup = 1.0 / (1.0 + self.config.cost_ratio)

        decision = CascadeDecision(
            status="DEFERRED_TO_VERIFIER",
            chosen_model=self.config.verifier_model,
            deferral_score=score,
            threshold=self.config.deferral_threshold,
            criterion=self.config.criterion,
            draft_tokens=draft_tokens,
            verifier_tokens=verifier_tokens,
            total_tokens=total_tokens,
            speedup_estimate=speedup,
            latency_ms=elapsed_ms,
        )

        if "usage" not in verifier_resp:
            verifier_resp["usage"] = {}
        verifier_resp["usage"]["cascade"] = decision.to_dict()
        return verifier_resp, decision
