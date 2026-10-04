"""Turn request, response, citation and SSE event contracts."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from kybernetes.contracts.enums import (
    EventType,
    RouteTag,
    TurnOutcomeStatus,
)


class Citation(BaseModel):
    """Citation metadata linked to retrieved passages."""
    citation_id: str = Field(..., description="Unique badge identifier e.g. [1]")
    passage_id: str = Field(..., description="Target passage hash/ID in Stoma/LTHCM index")
    source_id: str = Field(..., description="Source document identifier")
    source_title: str = Field(..., description="Friendly document or web title")
    text: str = Field(..., description="Verbatim cited snippet")
    score: float = Field(default=1.0, ge=0.0, description="Retrieval / verifier relevance score")


class TurnRequest(BaseModel):
    """Incoming request for a conversational turn."""
    session_id: str = Field(..., description="Session/conversation identifier")
    query: str = Field(..., description="User prompt or question")
    turn_id: Optional[str] = Field(default=None, description="Optional caller-supplied turn ID")
    deadline_ms: Optional[int] = Field(default=60000, description="Per-turn hard execution deadline in ms")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional turn metadata")


class TurnEvent(BaseModel):
    """
    Streaming event emitted by Kybernetes over SSE.
    Ordered by monotonic sequence number (seq_num) to enable safe reconnects.
    Strictly compatible with Prosopon TurnEvent schema.
    """
    turn_id: str = Field(..., description="Unique identifier for the conversational turn")
    seq_num: int = Field(..., ge=0, description="Monotonically increasing sequence number")
    event_type: str = Field(..., description="Type of turn event (started, delta, evidence, etc.)")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    status: Optional[str] = Field(
        default=None,
        description="Outcome state: 'verified', 'conflicted', 'insufficient_evidence', 'failed', 'draft', 'abstained'",
    )
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Event emission timestamp in UTC",
    )


class ShowWorkInfo(BaseModel):
    """Transparent inspection payload for the Prosopon 'show work' drawer."""
    turn_id: str
    route: str
    route_tags: List[str]
    components_used: List[str]
    citations: List[Citation] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    verifier_status: str = Field(default="draft")
    approval_status: Optional[str] = Field(default=None)
    budget_summary: Dict[str, Any] = Field(default_factory=dict)
    zero_edge_fallback_used: bool = Field(default=False)


class ApprovalResolution(BaseModel):
    """Payload to resolve a pending Phylax approval."""
    approval_id: str = Field(..., description="Approval request ID")
    decision: str = Field(..., description="'allow_once' or 'deny'")
