"""LTHCM benchmark outcomes and routing counterfactual schema."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class RouteOutcome(BaseModel):
    """Outcome of executing a specific retrieval route on a query."""
    route_id: str = Field(..., description="Route name, e.g. 'flat', 'mined', 'routed_rerank'")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Measured retrieval quality (e.g. recall@k, MAP, or exact match)")
    latency_ms: float = Field(..., ge=0.0, description="Execution duration in milliseconds")
    compute_cost: float = Field(default=1.0, ge=0.0, description="Normalized compute cost (e.g. FLOPs, relative GPU cycles)")
    status: str = Field(default="completed", description="Execution status: completed, timeout, failed, rejected")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of passages or entities retrieved")

    @field_validator("route_id")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("route_id cannot be empty")
        return v


class RouteUtility(BaseModel):
    """Utility function parameters for evaluating best-action routing."""
    lambda_latency: float = Field(default=0.001, ge=0.0, description="Penalty coefficient per millisecond of latency")
    lambda_cost: float = Field(default=0.1, ge=0.0, description="Penalty coefficient per unit of compute cost")
    tie_tolerance: float = Field(default=0.005, ge=0.0, description="Score delta threshold below which outcomes are considered tied")

    def compute(self, outcome: RouteOutcome) -> float:
        """Calculate net utility for an outcome."""
        if outcome.status != "completed":
            return -1000.0  # Large penalty for failures
        return (
            outcome.quality_score
            - (self.lambda_latency * outcome.latency_ms)
            - (self.lambda_cost * outcome.compute_cost)
        )


class LTHCMRoutingRecord(BaseModel):
    """
    LTHCM item-level benchmark record.
    Supports both:
    1. Task-type classification: question text -> question type ('bridge' | 'comparison' | ...)
    2. Best-route routing: question text -> optimal route under declared utility function.
    """
    query_id: str = Field(..., description="Immutable unique query identifier")
    question: str = Field(..., description="Query prompt text")
    gold_type: Optional[str] = Field(default=None, description="HotpotQA or benchmark question type label (e.g. bridge, comparison)")
    route_outcomes: Dict[str, RouteOutcome] = Field(default_factory=dict, description="Paired counterfactual outcomes for eligible routes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Benchmark metadata, corpus, source IDs")

    def get_best_route(self, utility: RouteUtility, eligible_routes: Optional[List[str]] = None) -> Optional[str]:
        """
        Derive optimal route label from counterfactual paired outcomes.
        Returns None if missing required counterfactuals or in case of an unresolved tie.
        """
        targets = eligible_routes or list(self.route_outcomes.keys())
        if not targets:
            return None
        
        # Verify all eligible routes are present
        for r in targets:
            if r not in self.route_outcomes:
                return None  # Missing counterfactual
        
        utilities = {r: utility.compute(self.route_outcomes[r]) for r in targets}
        sorted_routes = sorted(utilities.items(), key=lambda kv: kv[1], reverse=True)
        best_r, best_u = sorted_routes[0]
        
        if len(sorted_routes) > 1:
            second_r, second_u = sorted_routes[1]
            if abs(best_u - second_u) <= utility.tie_tolerance:
                # Mark as tie - ambiguous target
                return None
                
        return best_r
