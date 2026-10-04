"""Core Kybernetes Budget Engine implementing Bayesian priors, conformal intervals, Kalman filtering, and EVM checkpoints."""

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from kybernetes.budget.kalman import KalmanBurnRateFilter
from kybernetes.budget.models import (
    CheckpointDecision,
    IntervalEstimate,
    PhaseEstimate,
    ProgressEvent,
    ReconciliationReport,
    TaskEstimate,
    TaskPlan,
    UsageEvent,
    UsageMeasurement,
)
from kybernetes.budget.storage import BudgetDatabase
from kybernetes.contracts.enums import BudgetUnit, CheckpointAction


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class BudgetEngine:
    """
    Production-grade local budget loop:
    - Pre-dispatch Bayesian forecasting and adaptive conformal intervals
    - In-flight Kalman filter burn-rate tracking and Earned Value Management (CPI, EAC)
    - Predictive distribution for overrun probability with allow/warn/block checkpoints
    - Post-task reconciliation and prior calibration
    """

    DEFAULT_LOCAL_PRIORS = {
        BudgetUnit.INPUT_TOKENS: {"mean": 350.0, "std": 120.0},
        BudgetUnit.OUTPUT_TOKENS: {"mean": 250.0, "std": 90.0},
        BudgetUnit.TIME_SECONDS: {"mean": 3.5, "std": 1.5},
    }

    def __init__(self, db: BudgetDatabase):
        self.db = db
        # Active Kalman filters per (task_id, phase_id, unit)
        self._kalman_filters: Dict[Tuple[str, str, BudgetUnit], KalmanBurnRateFilter] = {}
        # Track last emitted checkpoint action to avoid spamming identical warnings
        self._last_decisions: Dict[str, CheckpointAction] = {}

    def prepare_task_estimate(self, plan: TaskPlan) -> TaskEstimate:
        """
        Generate pre-dispatch forecast before execution starts.
        Pulls cohort history if available, else applies labelled cold-start priors.
        Saves immutable forecast snapshot in SQLite.
        """
        self.db.save_task_plan(plan)
        calib = self.db.get_cohort_calibration(plan.cohort)

        sample_count = calib["sample_count"] if calib else 0
        is_cold_start = sample_count < 5
        q_multiplier = calib["q_hat"] if (calib and not is_cold_start) else 1.645  # 90% normal z-score

        phase_estimates: List[PhaseEstimate] = []
        total_means: Dict[BudgetUnit, float] = {u: 0.0 for u in self.DEFAULT_LOCAL_PRIORS}
        total_vars: Dict[BudgetUnit, float] = {u: 0.0 for u in self.DEFAULT_LOCAL_PRIORS}

        for phase in plan.phases:
            p_est_dict: Dict[str, IntervalEstimate] = {}

            for unit, prior in self.DEFAULT_LOCAL_PRIORS.items():
                if calib and not is_cold_start:
                    if unit == BudgetUnit.INPUT_TOKENS:
                        base_mean = calib["prior_mean_input_tokens"]
                    elif unit == BudgetUnit.OUTPUT_TOKENS:
                        base_mean = calib["prior_mean_output_tokens"]
                    else:
                        base_mean = calib["prior_mean_time_seconds"]
                    base_std = prior["std"]
                else:
                    base_mean = prior["mean"]
                    base_std = prior["std"]

                # Weight by phase allocation weight
                mean_val = base_mean * phase.weight
                std_val = base_std * math.sqrt(phase.weight)

                margin = q_multiplier * std_val
                p_est_dict[unit.value] = IntervalEstimate(
                    expected=round(mean_val, 2),
                    lower_90=round(max(0.0, mean_val - margin), 2),
                    upper_90=round(mean_val + margin, 2),
                    unit=unit,
                )

                total_means[unit] += mean_val
                total_vars[unit] += std_val ** 2

                # Initialize Kalman filter for this phase & unit
                filter_key = (plan.task_id, phase.phase_id, unit)
                self._kalman_filters[filter_key] = KalmanBurnRateFilter(
                    unit=unit,
                    initial_rate=mean_val,
                    initial_variance=std_val ** 2,
                )

            phase_estimates.append(
                PhaseEstimate(
                    phase_id=phase.phase_id,
                    phase_name=phase.phase_name,
                    estimates=p_est_dict,
                    sample_count=sample_count,
                    is_cold_start=is_cold_start,
                    assumptions="calibrated_cohort" if not is_cold_start else "cold_start_priors",
                )
            )

        total_estimates: Dict[str, IntervalEstimate] = {}
        for unit, mean_val in total_means.items():
            total_std = math.sqrt(total_vars[unit])
            margin = q_multiplier * total_std
            total_estimates[unit.value] = IntervalEstimate(
                expected=round(mean_val, 2),
                lower_90=round(max(0.0, mean_val - margin), 2),
                upper_90=round(mean_val + margin, 2),
                unit=unit,
            )

        forecast = TaskEstimate(
            task_id=plan.task_id,
            forecast_version="1.0.0",
            cohort=plan.cohort,
            phase_estimates=phase_estimates,
            total_estimates=total_estimates,
            sample_count=sample_count,
            is_cold_start=is_cold_start,
        )

        self.db.save_task_estimate(forecast)
        return forecast

    def ingest_usage(self, event: UsageEvent) -> bool:
        """
        Ingest a usage event idempotently.
        Updates internal progress and persists to SQLite.
        """
        return self.db.record_usage_event(event)

    def record_progress(
        self,
        task_id: str,
        phase_id: str,
        fraction: float,
        is_rework: bool = False,
    ) -> None:
        """Record phase progress and update active Kalman burn rate filters."""
        old_progress = self.db.get_latest_phase_progress(task_id, phase_id)
        delta_p = max(0.0, fraction - old_progress)

        event = ProgressEvent(
            event_id=f"prog-{task_id}-{phase_id}-{int(fraction * 1000)}",
            task_id=task_id,
            phase_id=phase_id,
            fraction_completed=fraction,
            is_rework=is_rework,
        )
        self.db.record_progress_event(event)

        if delta_p > 0.001:
            phase_usage = self.db.get_phase_usage(task_id, phase_id)
            for unit in [BudgetUnit.INPUT_TOKENS, BudgetUnit.OUTPUT_TOKENS, BudgetUnit.TIME_SECONDS]:
                key = (task_id, phase_id, unit)
                if key in self._kalman_filters:
                    total_u = phase_usage.get(unit.value, 0.0)
                    delta_u = total_u * delta_p
                    self._kalman_filters[key].update(delta_u, delta_p)

    def evaluate_checkpoint(
        self,
        task_id: str,
        phase_id: Optional[str] = None,
    ) -> CheckpointDecision:
        """
        Evaluate budget state at a phase boundary or in-phase checkpoint.
        Computes Earned Value Metrics (CPI, EAC) and overrun probability.
        Emits ALLOW, WARN, or BLOCK decision.
        """
        plan = self.db.get_task_plan(task_id)
        if not plan:
            return CheckpointDecision(
                task_id=task_id,
                phase_id=phase_id,
                action=CheckpointAction.ALLOW,
                reason="No budget plan found for task",
            )

        actual_usage = self.db.get_task_usage(task_id)
        eac_values: Dict[str, float] = {}
        cpi_values: Dict[str, float] = {}
        overrun_probs: Dict[str, float] = {}
        limits_exceeded: List[str] = []

        limits_map = {
            BudgetUnit.INPUT_TOKENS: plan.limit_input_tokens,
            BudgetUnit.OUTPUT_TOKENS: plan.limit_output_tokens,
            BudgetUnit.TIME_SECONDS: plan.limit_time_seconds,
            BudgetUnit.MONEY_USD: plan.limit_money_usd,
        }

        # Calculate progress across task
        total_weights = sum(p.weight for p in plan.phases) or 1.0
        weighted_progress = 0.0
        for p in plan.phases:
            prog = self.db.get_latest_phase_progress(task_id, p.phase_id)
            weighted_progress += prog * (p.weight / total_weights)

        remaining_fraction = max(0.0, 1.0 - weighted_progress)

        for unit, limit in limits_map.items():
            if limit is None or limit <= 0:
                continue

            ac = actual_usage.get(unit.value, 0.0)
            ev = limit * weighted_progress

            # Cost Performance Index (CPI = EV / AC)
            cpi = (ev / ac) if (ac > 0 and ev > 0) else 1.0
            cpi_values[unit.value] = round(cpi, 3)

            # Estimate at Completion (EAC)
            remaining_mean = 0.0
            remaining_var = 0.0
            phases_to_check = [phase_id] if phase_id else [p.phase_id for p in plan.phases]
            for p_id in phases_to_check:
                f_key = (task_id, p_id, unit)
                if f_key in self._kalman_filters:
                    rem_m, rem_s = self._kalman_filters[f_key].estimate_remaining(remaining_fraction)
                    remaining_mean += rem_m
                    remaining_var += rem_s ** 2

            if remaining_mean == 0.0:
                # Fallback to EVM standard formula: EAC = AC + (Budget - EV)/CPI
                eac = limit / cpi if cpi > 0 else (ac + (limit - ev))
            else:
                eac = ac + remaining_mean

            eac_values[unit.value] = round(eac, 2)

            # Check if current actual spend already exceeded hard limit
            if ac >= limit:
                limits_exceeded.append(unit.value)
                overrun_probs[unit.value] = 1.0
                continue

            # Predictive distribution for overrun probability
            # Spend ~ Normal(eac, sigma^2)
            sigma = math.sqrt(max(1.0, remaining_var + (0.15 * eac) ** 2))
            z = (limit - eac) / sigma
            prob_overrun = 1.0 - _normal_cdf(z)
            overrun_probs[unit.value] = round(prob_overrun, 4)

            if prob_overrun >= plan.warning_probability_threshold:
                limits_exceeded.append(unit.value)

        max_overrun_prob = max(overrun_probs.values(), default=0.0)

        # Decide Action: BLOCK, WARN, or ALLOW
        action = CheckpointAction.ALLOW
        reason = "Usage within expected bounds."

        if limits_exceeded:
            units_str = ", ".join(limits_exceeded)
            if plan.hard_stop_enabled:
                action = CheckpointAction.BLOCK
                reason = f"Budget overrun imminent on: {units_str} (probability={max_overrun_prob:.2f}). Hard stop enforced."
            else:
                action = CheckpointAction.WARN
                reason = f"High probability of exceeding budget on: {units_str} (probability={max_overrun_prob:.2f})."

        # Suppress duplicate consecutive warnings for the same task
        prev_action = self._last_decisions.get(task_id)
        if action == CheckpointAction.WARN and prev_action == CheckpointAction.WARN:
            # Downgrade logged decision reason to prevent alarm fatigue
            reason += " (repeated checkpoint warning)"

        self._last_decisions[task_id] = action

        decision = CheckpointDecision(
            task_id=task_id,
            phase_id=phase_id,
            action=action,
            reason=reason,
            overrun_probability=max_overrun_prob,
            eac_values=eac_values,
            cpi_values=cpi_values,
            limits_exceeded=limits_exceeded,
        )

        self.db.record_checkpoint_decision(decision)
        return decision

    def reconcile_task(
        self,
        task_id: str,
        final_status: str = "completed",
    ) -> ReconciliationReport:
        """
        Reconcile pre-dispatch forecast with actual observations.
        Updates Bayesian priors and conformal calibration parameters in SQLite.
        """
        plan = self.db.get_task_plan(task_id)
        forecast = self.db.get_task_estimate(task_id)
        actual_usage = self.db.get_task_usage(task_id)

        if not plan or not forecast:
            return ReconciliationReport(
                task_id=task_id,
                total_actual=actual_usage,
                final_status=final_status,
            )

        total_forecast: Dict[str, float] = {}
        intervals_90: Dict[str, Dict[str, float]] = {}
        coverage_hits: Dict[str, bool] = {}
        cpi_dict: Dict[str, Optional[float]] = {}

        for unit_str, est in forecast.total_estimates.items():
            total_forecast[unit_str] = est.expected
            intervals_90[unit_str] = {"lower": est.lower_90, "upper": est.upper_90}
            actual_val = actual_usage.get(unit_str, 0.0)
            hit = est.lower_90 <= actual_val <= est.upper_90
            coverage_hits[unit_str] = hit

            # Calculate overall CPI if budget limit was defined
            limit_val = getattr(plan, f"limit_{unit_str}", None)
            if limit_val and actual_val > 0:
                cpi_dict[unit_str] = round(limit_val / actual_val, 3)
            else:
                cpi_dict[unit_str] = None

        phase_breakdown: List[Dict[str, Any]] = []
        for phase in plan.phases:
            p_actual = self.db.get_phase_usage(task_id, phase.phase_id)
            phase_breakdown.append(
                {
                    "phase_id": phase.phase_id,
                    "phase_name": phase.phase_name,
                    "actual_usage": p_actual,
                }
            )

        report = ReconciliationReport(
            task_id=task_id,
            total_actual=actual_usage,
            total_forecast=total_forecast,
            intervals_90=intervals_90,
            coverage_hits=coverage_hits,
            cpi=cpi_dict,
            phase_breakdown=phase_breakdown,
            rework_count=0,
            completed=(final_status == "completed"),
            final_status=final_status,
        )
        self.db.save_reconciliation(report)

        # Update Cohort Calibration & Bayesian Priors
        if report.completed:
            self._update_calibration(plan.cohort, actual_usage, coverage_hits)

        return report

    def _update_calibration(
        self,
        cohort: str,
        actual_usage: Dict[str, float],
        coverage_hits: Dict[str, bool],
    ) -> None:
        """Adaptive Bayesian and conformal calibration update."""
        calib = self.db.get_cohort_calibration(cohort)
        n = calib["sample_count"] if calib else 0
        new_n = n + 1

        in_tok = actual_usage.get(BudgetUnit.INPUT_TOKENS.value, 350.0)
        out_tok = actual_usage.get(BudgetUnit.OUTPUT_TOKENS.value, 250.0)
        t_sec = actual_usage.get(BudgetUnit.TIME_SECONDS.value, 3.5)

        if calib:
            # Welford/exponential Bayesian update
            alpha = 1.0 / new_n
            mean_in = (1 - alpha) * calib["prior_mean_input_tokens"] + alpha * in_tok
            mean_out = (1 - alpha) * calib["prior_mean_output_tokens"] + alpha * out_tok
            mean_time = (1 - alpha) * calib["prior_mean_time_seconds"] + alpha * t_sec

            # Adaptive conformal update for q_hat
            # If hit coverage, reduce interval multiplier slightly; if missed, widen it
            missed = not all(coverage_hits.values())
            gamma = 0.05 / math.sqrt(new_n)
            q_hat = calib["q_hat"] + (gamma * (0.90 if missed else -0.10))
            q_hat = max(1.0, min(3.5, q_hat))
            cov_acc = (1 - alpha) * calib["alpha_coverage"] + alpha * (0.0 if missed else 1.0)
        else:
            mean_in = in_tok
            mean_out = out_tok
            mean_time = t_sec
            q_hat = 1.645
            cov_acc = 1.0 if all(coverage_hits.values()) else 0.0

        self.db.update_cohort_calibration(
            cohort=cohort,
            sample_count=new_n,
            alpha_coverage=round(cov_acc, 3),
            q_hat=round(q_hat, 4),
            prior_mean_input_tokens=round(mean_in, 2),
            prior_mean_output_tokens=round(mean_out, 2),
            prior_mean_time_seconds=round(mean_time, 2),
            updated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
