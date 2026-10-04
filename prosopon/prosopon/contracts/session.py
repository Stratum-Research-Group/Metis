"""Session and conversation history contracts."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from prosopon.contracts.kybernetes import Citation
from prosopon.contracts.phylax import ApprovalRequest


class ChatMessage(BaseModel):
    """Single conversational turn in a session."""
    message_id: str = Field(..., description="Unique message UUID")
    role: Literal["user", "assistant", "system"] = Field(..., description="Author role")
    content: str = Field(..., description="Renderable text content")
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC creation time",
    )
    citations: List[Citation] = Field(default_factory=list, description="Citations attached to response")
    route_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Redacted execution trace: route, components, timing, verifier status",
    )
    status: str = Field(
        default="completed",
        description="Outcome state: 'verified', 'conflicted', 'insufficient_evidence', 'failed', 'draft'",
    )
    pending_approval: Optional[ApprovalRequest] = Field(
        default=None,
        description="Pending Phylax approval card if turn requires human consent",
    )


class SessionRecord(BaseModel):
    """Named conversational session containing persistent history."""
    session_id: str = Field(..., description="Unique session identifier")
    name: str = Field(default="New Conversation", description="Friendly conversation title")
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp",
    )
    updated_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Last update timestamp",
    )
    messages: List[ChatMessage] = Field(default_factory=list, description="Ordered conversation messages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session settings or tags")
