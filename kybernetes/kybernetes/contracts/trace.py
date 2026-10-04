"""Trace records and observability audit contracts."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from kybernetes.contracts.enums import RouteTag, TurnOutcomeStatus
from kybernetes.contracts.plan import StageOutcome


class TraceRecord(BaseModel):
    """
    Comprehensive execution trace for a conversational turn.
    Guarantees that every route, component invocation, dependency version,
    and outcome status is logged for Dokime evaluation and audit.
    """
    trace_id: str
    turn_id: str
    session_id: str
    route_label: str
    route_tags: List[RouteTag] = Field(default_factory=list)
    components_used: List[str] = Field(default_factory=list)
    dependency_versions: Dict[str, str] = Field(default_factory=dict)
    stage_outcomes: List[StageOutcome] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    final_status: TurnOutcomeStatus = TurnOutcomeStatus.DRAFT
    zero_edge_fallback_used: bool = False
    budget_usage: Dict[str, float] = Field(default_factory=dict)
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
