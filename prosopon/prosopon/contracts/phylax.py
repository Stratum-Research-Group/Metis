"""Phylax approval request and response contracts."""

from datetime import datetime, timezone
from typing import Any, Dict, Literal
from pydantic import BaseModel, Field


class ApprovalRequest(BaseModel):
    """
    Phylax ASK approval card payload presented to the user.
    Binds to an exact action and argument digest with strict one-use semantics.
    """
    approval_id: str = Field(..., description="Unique one-use approval handle")
    turn_id: str = Field(..., description="Associated turn ID")
    action: str = Field(..., description="Requested capability action, e.g. 'WRITE_FILE', 'EXECUTE_TOOL'")
    target: str = Field(..., description="Target file path, URL, or resource identifier")
    argument_summary: Dict[str, Any] = Field(default_factory=dict, description="Redacted arguments or diff")
    policy_rule: str = Field(..., description="Matched Phylax rule requiring human approval")
    expiry_utc: str = Field(..., description="ISO 8601 UTC expiration timestamp")
    status: Literal["pending", "approved", "denied", "expired"] = Field(
        default="pending",
        description="Approval state",
    )

    def is_expired(self) -> bool:
        """Check if approval deadline has elapsed."""
        try:
            exp = datetime.fromisoformat(self.expiry_utc)
            now = datetime.now(timezone.utc)
            return now >= exp
        except Exception:
            return False


class ApprovalResponse(BaseModel):
    """Client decision response transmitted to Kybernetes/Phylax."""
    approval_id: str = Field(..., description="Unique approval handle matching request")
    decision: Literal["allow_once", "deny"] = Field(..., description="Owner decision")
    resolved_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Resolution timestamp",
    )
