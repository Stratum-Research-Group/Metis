"""Request payload contract models for Krisis selector."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from krisis.contracts.candidate import Candidate, EvidencePassage


class Capabilities(BaseModel):
    """Runtime capabilities available for evaluation."""
    model_config = ConfigDict(extra="allow")

    verifier_active: bool = True
    logprobs: bool = False


class KrisisRequest(BaseModel):
    """Input request sent to the Krisis selector."""
    model_config = ConfigDict(extra="allow")

    question_class: str | list[str] | None = None
    requires_grounding: bool = False
    capabilities: Capabilities = Field(default_factory=Capabilities)
    candidates: list[Candidate] = Field(default_factory=list)
    evidence: list[EvidencePassage] = Field(default_factory=list)
    snapshot_id: str | None = None

    # Stale-recheck snapshot fields
    selection_snapshot: str | None = None
    recheck_snapshot: str | None = None
    evidence_changed: bool = False
    rebuilds_remaining: int | None = None
    remaining_rebuild_budget: bool | None = None

    # Upstream generation tracking
    generation_attempts: int | None = None
    generation_failures: int | None = None
