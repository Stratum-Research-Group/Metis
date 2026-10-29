"""Contract models for ingesting Krisis active selective prediction records into Paideia (ASPEST).

Corresponds to the KrisisActiveRecord schema emitted by Krisis when selective prediction
abstains or identifies epistemic ambiguity (arXiv:2304.03870).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class KrisisActiveRecord(BaseModel):
    """An active learning query sample from a Krisis selective prediction run."""
    model_config = ConfigDict(extra="allow")

    query_id: str
    question: str
    decision_status: str
    decision_reason: Optional[str] = None
    rejection_score: float = Field(ge=0.0, le=1.0)
    prediction_set_size: Optional[int] = None
    selected_id: Optional[str] = None
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ActiveSamplingConfig(BaseModel):
    """Configuration parameters for the ASPEST active selection algorithm."""
    model_config = ConfigDict(extra="forbid")

    budget: int = Field(default=50, gt=0, description="Maximum number of queries to select for oracle annotation")
    beta: float = Field(default=0.6, ge=0.0, le=1.0, description="Weight on uncertainty U(x) vs diversity D(x, S)")
    min_rejection_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum rejection score required")
    diversity_metric: str = Field(default="token_jaccard", description="Diversity distance metric: 'token_jaccard' or 'cosine'")


class ASPESTAuditReport(BaseModel):
    """Statistical audit report summarizing the active selection outcome."""
    model_config = ConfigDict(extra="allow")

    total_pool_size: int
    eligible_count: int
    budget: int
    selected_count: int
    mean_rejection_score: float
    mean_diversity_distance: float
    reason_distribution: Dict[str, int] = Field(default_factory=dict)
