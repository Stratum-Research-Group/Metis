"""Data contracts and schemas for the Kybernetes Budget Engine."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from kybernetes.contracts.enums import BudgetUnit, CheckpointAction


class PhasePlan(BaseModel):
    """Specification of budget allocation for an individual task phase."""
    phase_id: str
    phase_name: str
    target_criteria: str
    allocated_input_tokens: float = 0.0
    allocated_output_tokens: float = 0.0
    allocated_time_seconds: float = 0.0
    allocated_money_usd: float = 0.0
    weight: float = 1.0


class TaskPlan(BaseModel):
    """Comprehensive resource budget plan for an entire task/turn."""
    task_id: str
    turn_id: str
    task_type: str = "chat_turn"
    model_name: str = "default_local"
    cohort: str = "default_cohort"
    phases: List[PhasePlan] = Field(default_factory=list)
    limit_input_tokens: Optional[float] = None
    limit_output_tokens: Optional[float] = None
    limit_time_seconds: Optional[float] = None
    limit_money_usd: Optional[float] = None
    warning_probability_threshold: float = 0.80
    hard_stop_enabled: bool = False
    price_per_1k_input_tokens_usd: float = 0.0  # 0.0 for purely local models
    price_per_1k_output_tokens_usd: float = 0.0
    currency: str = "USD"
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class UsageMeasurement(BaseModel):
    """Specific resource usage measurement with explicit units."""
    unit: BudgetUnit
    quantity: float


class UsageEvent(BaseModel):
    """
    Immutable, idempotent resource usage event emitted during execution.
    Keyed by request_id and event_id with explicit provenance.
    """
    event_id: str
    task_id: str
    phase_id: str
    request_id: str
    is_retry: bool = False
    measurements: List[UsageMeasurement] = Field(default_factory=list)
    provenance: str = "hestia"  # hestia, wall_clock, tool, api
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProgressEvent(BaseModel):
    """Progress event indicating percentage of phase criteria achieved."""
    event_id: str
    task_id: str
    phase_id: str
    fraction_completed: float = Field(..., ge=0.0, le=1.0)
    is_rework: bool = False
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class IntervalEstimate(BaseModel):
    """Interval estimate with coverage confidence."""
    expected: float
    lower_90: float
    upper_90: float
    unit: BudgetUnit


class PhaseEstimate(BaseModel):
    """Phase estimate before dispatch or at checkpoint."""
    phase_id: str
    phase_name: str
    estimates: Dict[str, IntervalEstimate] = Field(default_factory=dict)
    sample_count: int = 0
    is_cold_start: bool = True
    assumptions: str = "local_priors"


class TaskEstimate(BaseModel):
    """Full task forecast before dispatch."""
    task_id: str
    forecast_version: str = "1.0.0"
    cohort: str
    phase_estimates: List[PhaseEstimate] = Field(default_factory=list)
    total_estimates: Dict[str, IntervalEstimate] = Field(default_factory=dict)
    sample_count: int = 0
    is_cold_start: bool = True
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class CheckpointDecision(BaseModel):
    """
    Evaluation decision at pre-dispatch or in-phase checkpoint.
    Exportable to Phylax as a bounded policy decision.
    """
    task_id: str
    phase_id: Optional[str] = None
    action: CheckpointAction
    reason: str
    overrun_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    eac_values: Dict[str, float] = Field(default_factory=dict)
    cpi_values: Dict[str, float] = Field(default_factory=dict)
    limits_exceeded: List[str] = Field(default_factory=list)
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ReconciliationReport(BaseModel):
    """Post-task reconciliation of forecast versus actual spend."""
    task_id: str
    total_actual: Dict[str, float] = Field(default_factory=dict)
    total_forecast: Dict[str, float] = Field(default_factory=dict)
    intervals_90: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    coverage_hits: Dict[str, bool] = Field(default_factory=dict)
    cpi: Dict[str, Optional[float]] = Field(default_factory=dict)
    phase_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    rework_count: int = 0
    completed: bool = True
    final_status: str = "completed"
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
