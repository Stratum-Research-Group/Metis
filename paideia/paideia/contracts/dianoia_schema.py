"""Dianoia walk log schema and validation contract."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class DianoiaStep(BaseModel):
    """Single step in a Dianoia deliberative walk path."""
    node: str = Field(..., description="Visited node ID")
    action: Literal["EXPAND", "BACKTRACK", "ANSWER"] = Field(..., description="Action taken at step")
    raw_output: str = Field(default="", description="Raw model generation for this step")
    reason: str = Field(default="", description="Structured reasoning or justification")
    parse_failed: bool = Field(default=False, description="Whether action parsing failed and used fallback")

    @field_validator("node")
    @classmethod
    def validate_node(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Node ID cannot be empty")
        return v


class DianoiaVerification(BaseModel):
    """Verification results emitted by Dianoia sheaf or NLI checks."""
    conflicts_found: bool = Field(default=False, description="Whether contradictions or conflicts were detected")
    rewalked: bool = Field(default=False, description="Whether a corrective re-walk was initiated")
    report: str = Field(default="", description="Diagnostic report from verification")
    conflict_detail: Optional[str] = Field(default=None, description="Detailed explanation of contradiction")


class DianoiaWalkRecord(BaseModel):
    """Complete serialized Dianoia walk log line conforming to v1 schema."""
    schema_version: str = Field(default="1.0.0", description="Schema version of walk log")
    walk_id: str = Field(..., description="UUIDv4 identifier linking initial attempt and corrective re-walks")
    question: str = Field(..., description="Input user prompt or question")
    seeded_nodes: List[str] = Field(default_factory=list, description="Initial seed nodes from retrieval")
    path: List[DianoiaStep] = Field(default_factory=list, description="Sequence of step actions executed")
    verification: DianoiaVerification = Field(default_factory=DianoiaVerification)
    final_answer: str = Field(default="", description="Final synthesized answer")
    max_steps: int = Field(default=10, ge=1, description="Configured step budget")
    steps_used: int = Field(default=0, ge=0, description="Actual step count consumed")
    model: str = Field(..., description="Model identifier used for deliberation")
    fallback_used: bool = Field(default=False, description="Whether fallback model or policy was activated")
    initial_path: Optional[List[DianoiaStep]] = Field(default=None, description="Path of failed attempt if this record is a re-walk")
    privacy_label: str = Field(default="internal", description="Privacy classification: public, internal, or restricted")

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_and_migrate_schema_version(cls, v: Any) -> str:
        if v in (1, "1", "1.0.0"):
            return "1.0.0"
        raise ValueError(f"Unsupported schema version: {v}. Expected supported v1 schema ('1.0.0', 1, '1').")

    @field_validator("walk_id", "question", "model")
    @classmethod
    def not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("String field cannot be empty")
        return v

    def is_defensible_gold_trajectory(self) -> bool:
        """
        Check if trajectory is defensible for policy supervised fine-tuning.
        Rejects walks that failed parsing, had uncorrected conflicts, or empty paths.
        Continuing conflicts are strictly rejected even if rewalked=True.
        """
        if self.verification.conflicts_found:
            return False
        if not self.path:
            return False
        if any(step.parse_failed for step in self.path):
            return False
        if not self.final_answer.strip():
            return False
        return True
