"""Boundary-aware practice selection contracts (EoG Item 2, arXiv:2610.35598).

Defines schemas for repeated-attempt case rollouts, sample statistics,
difficulty taxonomy (Boundary, Retention Check, Hard Diagnostic),
and practice batch selection reports.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CaseAttemptCategory(str, Enum):
    """Categorization of a task based on repeated-attempt score variance."""
    BOUNDARY = "BOUNDARY"                    # sigma^2 > min_variance (active learning frontier)
    RETENTION_CHECK = "RETENTION_CHECK"      # mean == 1.0, sigma^2 == 0 (all attempts passed)
    HARD_DIAGNOSTIC = "HARD_DIAGNOSTIC"      # mean == 0.0, sigma^2 == 0 (all attempts failed)
    CONSISTENT_PARTIAL = "CONSISTENT_PARTIAL"# 0 < mean < 1.0, sigma^2 == 0 (constant partial credit)


class CaseAttemptRecord(BaseModel):
    """Execution record for a single evaluation/practice rollout attempt."""
    model_config = ConfigDict(extra="allow")

    case_id: str
    attempt_index: int = Field(ge=0, description="0-indexed attempt number (e.g. 0 to N-1)")
    seed: int = Field(description="RNG seed used for this rollout")
    score: float = Field(ge=0.0, le=1.0, description="Evaluated score (e.g. answer_F1 or exact_set_match)")
    evidence_grounded: bool = Field(default=True, description="Whether supporting citations/evidence were verified")
    latency: Optional[float] = Field(default=None, description="Inference latency in seconds")
    token_use: Optional[int] = Field(default=None, description="Total tokens consumed")
    failure_label: Optional[str] = Field(default=None, description="Optional diagnostic failure mode")
    model_and_config: Optional[str] = Field(default=None, description="Model identifier and sampling parameters")
    tested_revision: Optional[str] = Field(default=None, description="Git HEAD commit or model checkpoint")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CaseAttemptStats(BaseModel):
    """Summary statistics over N repeated attempts on a specific case."""
    model_config = ConfigDict(extra="allow")

    case_id: str
    num_attempts: int
    mean_score: float = Field(ge=0.0, le=1.0)
    sample_variance: float = Field(ge=0.0, description="Unbiased sample variance s^2 over N attempts")
    sample_std: float = Field(ge=0.0)
    category: CaseAttemptCategory
    min_score: float = Field(ge=0.0, le=1.0)
    max_score: float = Field(ge=0.0, le=1.0)
    failure_counts: Dict[str, int] = Field(default_factory=dict)
    scores: List[float] = Field(default_factory=list)


class BoundarySelectionConfig(BaseModel):
    """Configuration for boundary-aware practice batch selection."""
    model_config = ConfigDict(extra="forbid")

    budget: int = Field(default=20, gt=0, description="Total number of cases to select for practice")
    boundary_ratio: float = Field(default=0.6, ge=0.0, le=1.0, description="Ratio of boundary cases (sigma^2 > 0)")
    retention_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="Ratio of easy retention checks")
    hard_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="Ratio of hard diagnostic cases")
    min_variance: float = Field(default=1e-4, ge=0.0, description="Minimum sample variance to qualify as boundary")
    prioritize_highest_variance: bool = Field(default=True, description="Prioritize cases with maximum variance")
    seed: int = Field(default=42, description="RNG seed for stratum sampling")


class BoundarySelectionReport(BaseModel):
    """Audit report detailing practice selection composition and variance metrics."""
    model_config = ConfigDict(extra="allow")

    total_cases_analyzed: int
    num_attempts_per_case: int
    boundary_pool_size: int
    retention_pool_size: int
    hard_pool_size: int
    consistent_partial_pool_size: int
    selected_count: int
    selected_boundary_count: int
    selected_retention_count: int
    selected_hard_count: int
    selected_cases: List[str]
    mean_boundary_variance: float
    selection_strategy: str
