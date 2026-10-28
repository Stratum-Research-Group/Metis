"""Decision output contract models emitted by Krisis selector."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from krisis.contracts.enums import DecisionStatus


class CandidateRejection(BaseModel):
    """Rejection record for an individual candidate."""
    model_config = ConfigDict(extra="allow")

    id: str
    reason: str
    detail: str | None = None
    gate_failed: str | None = None


class KrisisDecision(BaseModel):
    """The normative output emitted by Krisis."""
    model_config = ConfigDict(extra="allow")

    status: str
    selected_id: str | None = None
    reason: str | None = None
    profile: str | None = None
    scores: dict[str, str] = Field(default_factory=dict)
    rejected: list[CandidateRejection] = Field(default_factory=list)
    per_candidate_reasons: dict[str, str] = Field(default_factory=dict)
    mapped_reasons: dict[str, str] = Field(default_factory=dict)
    label: str | None = None
    note: str | None = None
    warning: str | None = None
    policy: str | None = None
    merged: list[dict[str, Any]] = Field(default_factory=list)
    kept_separate: list[str] = Field(default_factory=list)
    upstream: dict[str, int] | None = None
    trust_flags: dict[str, str] = Field(default_factory=dict)
    normalized_question_class: str | None = None
    conformal: dict[str, Any] | None = None
