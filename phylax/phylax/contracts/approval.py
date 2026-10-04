"""Approval contract models matching the Prosopon and Phylax interface."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from phylax.contracts.enums import ApprovalStatus


class ApprovalRequest(BaseModel):
    """Pending approval record emitted when a rule evaluates to ASK.
    
    Compatible with Prosopon's ApprovalCard schema.
    """
    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(description="Unique identifier for this approval")
    turn_id: str | None = Field(default=None, description="Prosopon/Kybernetes interaction turn ID")
    trace_id: str | None = Field(default=None, description="Orchestration trace ID")
    subject: str = Field(description="Requester component")
    action: str = Field(description="Verb (e.g. send, execute, write)")
    target: str = Field(description="Canonical target (e.g. URL, file path, command)")
    argument_summary: str = Field(description="Human-readable summary of arguments")
    argument_digest: str = Field(description="SHA-256 digest of canonicalized JSON arguments")
    raw_arguments: dict[str, Any] = Field(default_factory=dict, description="Redacted raw arguments")
    policy_rule: str = Field(description="Matched policy rule ID that requested approval")
    policy_version: str = Field(description="Policy version when request was created")
    created_at_utc: str = Field(description="ISO-8601 creation timestamp")
    expiry_utc: str = Field(description="ISO-8601 expiration timestamp")
    status: ApprovalStatus = Field(default=ApprovalStatus.PENDING)
    nonce: str = Field(description="One-time secret nonce required for atomic consumption")
    consumed_at_utc: str | None = Field(default=None, description="Timestamp when consumed")
    resolution_reason: str | None = Field(default=None, description="User note or auto-deny reason")


class ApprovalResolution(BaseModel):
    """Payload to approve or reject a pending request."""
    model_config = ConfigDict(extra="forbid")

    approval_id: str
    approved: bool
    reason: str | None = None
    resolved_by: str = "owner"
