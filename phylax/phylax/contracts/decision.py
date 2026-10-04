"""Decision and execution contract models."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from phylax.contracts.enums import Effect, ResourceClass, Verb


class AuthorizationRecord(BaseModel):
    """Immutable record of an authorization decision."""
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(description="Unique UUID for this authorization query")
    timestamp_utc: str = Field(description="ISO-8601 timestamp in UTC")
    subject: str = Field(description="Verified subject/component making request")
    verb: Verb = Field(description="Operation verb")
    resource_class: ResourceClass = Field(description="Resource class")
    canonical_resource: str = Field(description="Canonicalized target path or identifier")
    effect: Effect = Field(description="ALLOW, DENY, or ASK")
    matched_rule_id: str | None = Field(default=None, description="ID of rule that matched")
    policy_version: str = Field(description="Version of policy evaluated against")
    reason: str = Field(description="Human-readable decision rationale")
    trace_id: str | None = Field(default=None, description="Optional trace ID from Kybernetes")
    approval_id: str | None = Field(default=None, description="Approval ID if effect is ASK")
    argument_digest: str | None = Field(default=None, description="SHA-256 digest of input arguments")


class ExecutionReceipt(BaseModel):
    """Receipt emitted after a capability gateway executes an authorized action."""
    model_config = ConfigDict(extra="forbid")

    receipt_id: str = Field(description="Unique UUID for this execution receipt")
    decision_id: str = Field(description="Correlated authorization decision ID")
    timestamp_utc: str = Field(description="ISO-8601 timestamp in UTC")
    success: bool = Field(description="True if execution completed without error")
    error_message: str | None = Field(default=None, description="Error message if execution failed")
    bytes_processed: int | None = Field(default=None, description="Bytes read or sent if applicable")
    execution_digest: str | None = Field(default=None, description="SHA-256 digest of returned result or artifact")
