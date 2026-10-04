"""Execution plan and stage outcome contracts."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from kybernetes.contracts.enums import RouteTag, StageName


class StagePlan(BaseModel):
    """Specification for an individual pipeline execution stage."""
    stage: StageName
    description: str
    timeout_ms: int = Field(default=15000, description="Stage timeout in milliseconds")
    required_capabilities: List[str] = Field(default_factory=list)
    enabled: bool = Field(default=True)
    params: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    """
    Ordered, typed, inspectable multi-stage execution plan for a turn.
    Replaces brittle siloed dispatching with an inspectable sequence.
    """
    plan_id: str
    turn_id: str
    route_label: str = Field(..., description="High-level plan strategy, e.g., 'direct', 'retrieval_augmented', 'deliberate_graph', 'guarded_tool'")
    route_tags: List[RouteTag] = Field(default_factory=list, description="Observability tags activated by this plan")
    stages: List[StagePlan] = Field(default_factory=list, description="Ordered pipeline stages to execute")
    budget_limits: Dict[str, float] = Field(default_factory=dict, description="Resource limits (tokens, seconds)")


class StageOutcome(BaseModel):
    """Recorded result of an individual stage execution."""
    stage: StageName
    success: bool
    latency_ms: float
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
