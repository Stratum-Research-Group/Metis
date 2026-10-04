"""Dokime evaluation harness adapter for prior retrieval and benchmark tracking."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvalOutcomeRecord(BaseModel):
    """Historical evaluation outcome exported by Dokime."""
    eval_id: str
    task_type: str
    model_name: str
    phase: str
    measured_input_tokens: float
    measured_output_tokens: float
    measured_time_seconds: float
    factual_correctness_score: float
    timestamp_utc: str


class DokimeAdapter:
    """
    Adapter interfacing with Dokime evaluation history.
    Supplies empirical cohorts and prior distributions to the budget engine.
    """

    def __init__(self):
        self._history: List[EvalOutcomeRecord] = []

    def load_cohort_priors(self, cohort: str) -> Optional[Dict[str, float]]:
        """Retrieve aggregated historical usage for a task cohort."""
        cohort_records = [r for r in self._history if r.task_type == cohort]
        if not cohort_records:
            return None

        n = len(cohort_records)
        return {
            "sample_count": n,
            "mean_input_tokens": sum(r.measured_input_tokens for r in cohort_records) / n,
            "mean_output_tokens": sum(r.measured_output_tokens for r in cohort_records) / n,
            "mean_time_seconds": sum(r.measured_time_seconds for r in cohort_records) / n,
        }

    def record_run(self, outcome: EvalOutcomeRecord) -> None:
        self._history.append(outcome)
