"""Kybernetes stream events and citation contracts."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation metadata linked to retrieved passages."""
    citation_id: str = Field(..., description="Unique badge identifier e.g. [1]")
    passage_id: str = Field(..., description="Target passage hash/ID in Stoma/LTHCM index")
    source_id: str = Field(..., description="Source document identifier")
    source_title: str = Field(..., description="Friendly document or web title")
    text: str = Field(..., description="Verbatim cited snippet")
    score: float = Field(default=1.0, ge=0.0, description="Retrieval / verifier relevance score")


class TurnEvent(BaseModel):
    """
    Streaming event emitted by Kybernetes during query deliberation and synthesis.
    Ordered by monotonic sequence number (seq_num) to enable safe reconnects.
    """
    turn_id: str = Field(..., description="Unique identifier for the conversational turn")
    seq_num: int = Field(..., ge=0, description="Monotonically increasing sequence number")
    event_type: Literal[
        "started",
        "plan_selected",
        "delta",
        "evidence",
        "ask_approval",
        "approval_resolved",
        "completed",
        "failed",
        "cancelled",
    ] = Field(..., description="Type of turn event")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    status: Optional[str] = Field(
        default=None,
        description="Outcome state: 'verified', 'conflicted', 'insufficient_evidence', 'failed', 'draft'",
    )
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Event emission timestamp",
    )
