"""Candidate answer, claims, and evidence representation models."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class EvidencePassage(BaseModel):
    """Retrieved evidence passage used for grounding validation."""
    model_config = ConfigDict(extra="allow")

    passage_id: str
    snapshot_id: str | None = None
    text: str | None = None
    trust_flag: str | None = None


class Claim(BaseModel):
    """Factual atomic claim extracted from candidate answer."""
    model_config = ConfigDict(extra="allow")

    claim_id: str
    text: str | None = None
    verdict: str  # supported, unsupported, contradicted, unverified
    calibrated_confidence: str | None = None
    cited_passage_ids: list[str] = Field(default_factory=list)


class GateResults(BaseModel):
    """Boolean status of pre-evaluated hard gates."""
    model_config = ConfigDict(extra="allow")

    access: bool = True
    provenance: bool = True
    contradiction: bool = True
    citation_resolution: bool = True


class Candidate(BaseModel):
    """Candidate answer model evaluated by Krisis."""
    model_config = ConfigDict(extra="allow")

    id: str
    normalized_text: str | None = None
    text: str | None = None
    citations: list[str] = Field(default_factory=list)
    cited_passage_ids: list[str] = Field(default_factory=list)
    snapshot_id: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    cal_ll: str | None = None
    raw_ll: str | None = None
    gate_results: dict[str, bool] | GateResults = Field(default_factory=GateResults)
    gate_failure: str | None = None

    def get_gate_result(self, gate_name: str) -> bool:
        if isinstance(self.gate_results, GateResults):
            return getattr(self.gate_results, gate_name, True)
        elif isinstance(self.gate_results, dict):
            return self.gate_results.get(gate_name, True)
        return True
