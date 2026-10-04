"""Unit tests for the Budget Engine, SQLite storage, Kalman filter, and EVM checkpoints."""

import pytest
from kybernetes.budget.engine import BudgetEngine
from kybernetes.budget.kalman import KalmanBurnRateFilter
from kybernetes.budget.models import (
    PhasePlan,
    TaskPlan,
    UsageEvent,
    UsageMeasurement,
)
from kybernetes.budget.storage import BudgetDatabase
from kybernetes.contracts.enums import BudgetUnit, CheckpointAction


@pytest.fixture
def budget_db():
    return BudgetDatabase(":memory:")


@pytest.fixture
def budget_engine(budget_db):
    return BudgetEngine(budget_db)


def test_kalman_filter_startup_and_update():
    kf = KalmanBurnRateFilter(
        unit=BudgetUnit.INPUT_TOKENS,
        initial_rate=300.0,
        initial_variance=500.0,
    )
    rem_m, rem_s = kf.estimate_remaining(remaining_fraction=0.5)
    assert rem_m == 150.0
    assert rem_s > 0

    # Simulate observation
    kf.update(delta_usage=160.0, delta_progress=0.5)
    assert kf.observation_count == 1
    # Variance should decrease after measurement
    assert kf.variance < 500.0 + kf.q


def test_pre_dispatch_estimate_cold_start(budget_engine):
    plan = TaskPlan(
        task_id="t1",
        turn_id="turn-1",
        cohort="test_cohort",
        phases=[
            PhasePlan(
                phase_id="p1",
                phase_name="synth",
                target_criteria="generate answer",
                allocated_input_tokens=400.0,
                allocated_output_tokens=200.0,
            )
        ],
        limit_input_tokens=1000.0,
    )
    forecast = budget_engine.prepare_task_estimate(plan)
    assert forecast.is_cold_start is True
    assert forecast.sample_count == 0
    assert "input_tokens" in forecast.total_estimates
    est_in = forecast.total_estimates["input_tokens"]
    assert est_in.lower_90 < est_in.expected < est_in.upper_90


def test_idempotent_usage_ingestion(budget_engine):
    plan = TaskPlan(
        task_id="t2",
        turn_id="turn-2",
        cohort="test_cohort",
        phases=[PhasePlan(phase_id="p1", phase_name="s", target_criteria="c")],
    )
    budget_engine.prepare_task_estimate(plan)

    event = UsageEvent(
        event_id="ev-100",
        task_id="t2",
        phase_id="p1",
        request_id="req-1",
        measurements=[UsageMeasurement(unit=BudgetUnit.INPUT_TOKENS, quantity=150.0)],
    )
    assert budget_engine.ingest_usage(event) is True
    # Ingesting same event_id again must be ignored (idempotent)
    assert budget_engine.ingest_usage(event) is False

    totals = budget_engine.db.get_task_usage("t2")
    assert totals[BudgetUnit.INPUT_TOKENS.value] == 150.0


def test_checkpoint_allow_and_warn(budget_engine):
    plan = TaskPlan(
        task_id="t3",
        turn_id="turn-3",
        cohort="test_cohort",
        phases=[PhasePlan(phase_id="p1", phase_name="s", target_criteria="c", weight=1.0)],
        limit_input_tokens=600.0,
        warning_probability_threshold=0.50,
        hard_stop_enabled=False,
    )
    budget_engine.prepare_task_estimate(plan)

    # Initial checkpoint before any usage -> ALLOW (expected 350 < 600)
    dec1 = budget_engine.evaluate_checkpoint("t3")
    assert dec1.action == CheckpointAction.ALLOW

    # Consume 580 of 600 limit with 0 progress -> high overrun probability -> WARN
    budget_engine.ingest_usage(
        UsageEvent(
            event_id="ev-over",
            task_id="t3",
            phase_id="p1",
            request_id="req-over",
            measurements=[UsageMeasurement(unit=BudgetUnit.INPUT_TOKENS, quantity=580.0)],
        )
    )
    dec2 = budget_engine.evaluate_checkpoint("t3", phase_id="p1")
    assert dec2.action in (CheckpointAction.WARN, CheckpointAction.BLOCK)
    assert "input_tokens" in dec2.limits_exceeded


def test_checkpoint_hard_stop_blocks(budget_engine):
    plan = TaskPlan(
        task_id="t4",
        turn_id="turn-4",
        cohort="test_cohort",
        phases=[PhasePlan(phase_id="p1", phase_name="s", target_criteria="c", weight=1.0)],
        limit_input_tokens=100.0,
        hard_stop_enabled=True,
    )
    budget_engine.prepare_task_estimate(plan)

    budget_engine.ingest_usage(
        UsageEvent(
            event_id="ev-stop",
            task_id="t4",
            phase_id="p1",
            request_id="req-stop",
            measurements=[UsageMeasurement(unit=BudgetUnit.INPUT_TOKENS, quantity=120.0)],
        )
    )
    dec = budget_engine.evaluate_checkpoint("t4", phase_id="p1")
    assert dec.action == CheckpointAction.BLOCK


def test_reconciliation_updates_calibration(budget_engine):
    plan = TaskPlan(
        task_id="t5",
        turn_id="turn-5",
        cohort="calib_cohort",
        phases=[PhasePlan(phase_id="p1", phase_name="s", target_criteria="c")],
    )
    budget_engine.prepare_task_estimate(plan)

    budget_engine.ingest_usage(
        UsageEvent(
            event_id="ev-rec",
            task_id="t5",
            phase_id="p1",
            request_id="req-rec",
            measurements=[
                UsageMeasurement(unit=BudgetUnit.INPUT_TOKENS, quantity=300.0),
                UsageMeasurement(unit=BudgetUnit.OUTPUT_TOKENS, quantity=200.0),
                UsageMeasurement(unit=BudgetUnit.TIME_SECONDS, quantity=2.5),
            ],
        )
    )

    report = budget_engine.reconcile_task("t5", final_status="completed")
    assert report.completed is True
    assert report.total_actual[BudgetUnit.INPUT_TOKENS.value] == 300.0

    # Calibration should now exist in DB
    calib = budget_engine.db.get_cohort_calibration("calib_cohort")
    assert calib is not None
    assert calib["sample_count"] == 1
