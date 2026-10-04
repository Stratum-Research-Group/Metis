"""Policy manifest contract models."""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ScoreConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    formula: str
    weights: dict[str, str] = Field(default_factory=dict)
    cal_ll_map: dict[str, Any] | None = None


class ProfileConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str
    requires_grounding: bool
    requires_capability: list[str] = Field(default_factory=list)
    hard_gates: list[str] = Field(default_factory=list)
    per_claim_min: str | None = None
    score: ScoreConfig | None = None
    tau: str | None = None
    selection: str | None = None
    fallback: str | None = None
    release_gate: str | None = None


class RoutingRule(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    question_class: str = Field(alias="class")
    requires_grounding: bool | None = None
    verifier_active: bool | None = None
    logprobs: bool | None = None
    profile: str | None = None
    abstain: str | None = None


class ClassRoutingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    rules: list[RoutingRule]
    multi_class: str
    unclassified: str
    input_class_normalization: dict[str, Any]
    matching: str


class PolicyManifest(BaseModel):
    """Pydantic model representing Krisis policy manifest (e.g. krisis-policy-0.1.json)."""
    model_config = ConfigDict(extra="allow")

    manifest_id: str
    schema_version: str
    status: str
    conventions: dict[str, str]
    governance: dict[str, Any]
    reason_enum: list[str]
    reason_precedence: list[str]
    candidate_only_reasons: list[str] = Field(default_factory=lambda: ["FEATURE_INVALID"])
    reason_mapping: dict[str, Any]
    generation_policy: dict[str, Any]
    profiles: dict[str, ProfileConfig]
    class_routing: ClassRoutingConfig
    n1_handling: str
    budgets: dict[str, Any]
    invalidation_events: list[str]
    audit_card: dict[str, Any]
    retention: dict[str, Any]
