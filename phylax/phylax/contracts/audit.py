"""Audit log models with cryptographic chaining."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class AuditRecord(BaseModel):
    """Cryptographically chained append-only audit log entry."""
    model_config = ConfigDict(extra="forbid")

    record_id: int = Field(description="Strictly monotonic record index")
    timestamp_utc: str = Field(description="ISO-8601 UTC timestamp")
    prev_hash: str = Field(description="SHA-256 hash of previous audit record")
    record_hash: str = Field(description="SHA-256 hash of current record contents")
    subject: str = Field(description="Requesting component")
    verb: str = Field(description="Capability verb (read, write, execute, send)")
    resource_class: str = Field(description="Resource category (file, tool, network, system)")
    canonical_resource: str = Field(description="Resolved canonical resource path/name")
    effect: str = Field(description="allow, deny, ask")
    matched_rule_id: str | None = Field(default=None, description="Matched rule ID")
    policy_version: str = Field(description="Evaluated policy version")
    trace_id: str | None = Field(default=None, description="Correlated trace ID")
    execution_receipt_id: str | None = Field(default=None, description="Execution receipt if executed")
    execution_status: str | None = Field(default=None, description="Status of execution")
    redacted_details: dict[str, Any] = Field(default_factory=dict, description="Redacted arguments/context")


class AuditVerificationResult(BaseModel):
    """Result of verifying audit chain integrity."""
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    total_records: int
    head_hash: str
    broken_record_id: int | None = None
    error_message: str | None = None
