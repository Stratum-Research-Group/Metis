"""SQLite persistence layer for the Kybernetes Budget Engine."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from kybernetes.budget.models import (
    CheckpointDecision,
    PhasePlan,
    ProgressEvent,
    ReconciliationReport,
    TaskEstimate,
    TaskPlan,
    UsageEvent,
    UsageMeasurement,
)
from kybernetes.contracts.enums import BudgetUnit, CheckpointAction


class BudgetDatabase:
    """
    SQLite-backed transactional persistence for budget plans, events,
    forecasts, checkpoints, and post-task reconciliations.
    """

    def __init__(self, db_path: Path | str = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
            self._mem_conn.execute("PRAGMA foreign_keys = ON;")
        else:
            self._mem_conn = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS task_plans (
                task_id TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                model_name TEXT NOT NULL,
                cohort TEXT NOT NULL,
                limit_input_tokens REAL,
                limit_output_tokens REAL,
                limit_time_seconds REAL,
                limit_money_usd REAL,
                warning_probability_threshold REAL NOT NULL,
                hard_stop_enabled INTEGER NOT NULL,
                price_per_1k_input_tokens_usd REAL NOT NULL,
                price_per_1k_output_tokens_usd REAL NOT NULL,
                currency TEXT NOT NULL,
                created_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS phase_plans (
                phase_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                phase_name TEXT NOT NULL,
                target_criteria TEXT NOT NULL,
                allocated_input_tokens REAL NOT NULL,
                allocated_output_tokens REAL NOT NULL,
                allocated_time_seconds REAL NOT NULL,
                allocated_money_usd REAL NOT NULL,
                weight REAL NOT NULL,
                FOREIGN KEY (task_id) REFERENCES task_plans(task_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS usage_events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                phase_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                is_retry INTEGER NOT NULL,
                measurements_json TEXT NOT NULL,
                provenance TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES task_plans(task_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_usage_task_phase ON usage_events(task_id, phase_id);
            CREATE INDEX IF NOT EXISTS idx_usage_request_id ON usage_events(request_id);

            CREATE TABLE IF NOT EXISTS progress_events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                phase_id TEXT NOT NULL,
                fraction_completed REAL NOT NULL,
                is_rework INTEGER NOT NULL,
                timestamp_utc TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES task_plans(task_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS task_estimates (
                task_id TEXT PRIMARY KEY,
                forecast_version TEXT NOT NULL,
                cohort TEXT NOT NULL,
                estimate_data_json TEXT NOT NULL,
                sample_count INTEGER NOT NULL,
                is_cold_start INTEGER NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES task_plans(task_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS checkpoint_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                phase_id TEXT,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                overrun_probability REAL NOT NULL,
                eac_values_json TEXT NOT NULL,
                cpi_values_json TEXT NOT NULL,
                limits_exceeded_json TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES task_plans(task_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS reconciliations (
                task_id TEXT PRIMARY KEY,
                report_json TEXT NOT NULL,
                completed INTEGER NOT NULL,
                final_status TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES task_plans(task_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS cohort_calibration (
                cohort TEXT PRIMARY KEY,
                sample_count INTEGER NOT NULL,
                alpha_coverage REAL NOT NULL,
                q_hat REAL NOT NULL,
                prior_mean_input_tokens REAL NOT NULL,
                prior_mean_output_tokens REAL NOT NULL,
                prior_mean_time_seconds REAL NOT NULL,
                updated_at_utc TEXT NOT NULL
            );
            """)

    def save_task_plan(self, plan: TaskPlan) -> None:
        """Persist a task plan and its associated phase allocations."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_plans (
                    task_id, turn_id, task_type, model_name, cohort,
                    limit_input_tokens, limit_output_tokens, limit_time_seconds, limit_money_usd,
                    warning_probability_threshold, hard_stop_enabled,
                    price_per_1k_input_tokens_usd, price_per_1k_output_tokens_usd,
                    currency, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.task_id,
                    plan.turn_id,
                    plan.task_type,
                    plan.model_name,
                    plan.cohort,
                    plan.limit_input_tokens,
                    plan.limit_output_tokens,
                    plan.limit_time_seconds,
                    plan.limit_money_usd,
                    plan.warning_probability_threshold,
                    1 if plan.hard_stop_enabled else 0,
                    plan.price_per_1k_input_tokens_usd,
                    plan.price_per_1k_output_tokens_usd,
                    plan.currency,
                    plan.created_at_utc,
                ),
            )
            for p in plan.phases:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO phase_plans (
                        phase_id, task_id, phase_name, target_criteria,
                        allocated_input_tokens, allocated_output_tokens,
                        allocated_time_seconds, allocated_money_usd, weight
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p.phase_id,
                        plan.task_id,
                        p.phase_name,
                        p.target_criteria,
                        p.allocated_input_tokens,
                        p.allocated_output_tokens,
                        p.allocated_time_seconds,
                        p.allocated_money_usd,
                        p.weight,
                    ),
                )

    def get_task_plan(self, task_id: str) -> Optional[TaskPlan]:
        """Retrieve task plan and its phase plans."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM task_plans WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return None
            phase_rows = conn.execute("SELECT * FROM phase_plans WHERE task_id = ?", (task_id,)).fetchall()
            phases = [
                PhasePlan(
                    phase_id=pr["phase_id"],
                    phase_name=pr["phase_name"],
                    target_criteria=pr["target_criteria"],
                    allocated_input_tokens=pr["allocated_input_tokens"],
                    allocated_output_tokens=pr["allocated_output_tokens"],
                    allocated_time_seconds=pr["allocated_time_seconds"],
                    allocated_money_usd=pr["allocated_money_usd"],
                    weight=pr["weight"],
                )
                for pr in phase_rows
            ]
            return TaskPlan(
                task_id=row["task_id"],
                turn_id=row["turn_id"],
                task_type=row["task_type"],
                model_name=row["model_name"],
                cohort=row["cohort"],
                phases=phases,
                limit_input_tokens=row["limit_input_tokens"],
                limit_output_tokens=row["limit_output_tokens"],
                limit_time_seconds=row["limit_time_seconds"],
                limit_money_usd=row["limit_money_usd"],
                warning_probability_threshold=row["warning_probability_threshold"],
                hard_stop_enabled=bool(row["hard_stop_enabled"]),
                price_per_1k_input_tokens_usd=row["price_per_1k_input_tokens_usd"],
                price_per_1k_output_tokens_usd=row["price_per_1k_output_tokens_usd"],
                currency=row["currency"],
                created_at_utc=row["created_at_utc"],
            )

    def record_usage_event(self, event: UsageEvent) -> bool:
        """
        Record a usage event idempotently.
        Returns True if inserted, False if already exists.
        """
        meas_json = json.dumps([m.model_dump() for m in event.measurements])
        meta_json = json.dumps(event.metadata)
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO usage_events (
                    event_id, task_id, phase_id, request_id, is_retry,
                    measurements_json, provenance, timestamp_utc, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.task_id,
                    event.phase_id,
                    event.request_id,
                    1 if event.is_retry else 0,
                    meas_json,
                    event.provenance,
                    event.timestamp_utc,
                    meta_json,
                ),
            )
            return cur.rowcount > 0

    def get_task_usage(self, task_id: str) -> Dict[str, float]:
        """Aggregate total usage across distinct measurements for a task."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT measurements_json FROM usage_events WHERE task_id = ?", (task_id,)).fetchall()
            totals: Dict[str, float] = {}
            for r in rows:
                meas_list = json.loads(r["measurements_json"])
                for m in meas_list:
                    unit = m["unit"]
                    qty = m["quantity"]
                    totals[unit] = totals.get(unit, 0.0) + qty
            return totals

    def get_phase_usage(self, task_id: str, phase_id: str) -> Dict[str, float]:
        """Aggregate phase-specific usage."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT measurements_json FROM usage_events WHERE task_id = ? AND phase_id = ?",
                (task_id, phase_id),
            ).fetchall()
            totals: Dict[str, float] = {}
            for r in rows:
                meas_list = json.loads(r["measurements_json"])
                for m in meas_list:
                    unit = m["unit"]
                    qty = m["quantity"]
                    totals[unit] = totals.get(unit, 0.0) + qty
            return totals

    def record_progress_event(self, event: ProgressEvent) -> None:
        """Record a progress update for earned value tracking."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO progress_events (
                    event_id, task_id, phase_id, fraction_completed, is_rework, timestamp_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.task_id,
                    event.phase_id,
                    event.fraction_completed,
                    1 if event.is_rework else 0,
                    event.timestamp_utc,
                ),
            )

    def get_latest_phase_progress(self, task_id: str, phase_id: str) -> float:
        """Get latest recorded completion fraction for a phase."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT fraction_completed FROM progress_events
                WHERE task_id = ? AND phase_id = ?
                ORDER BY timestamp_utc DESC LIMIT 1
                """,
                (task_id, phase_id),
            ).fetchone()
            return float(row["fraction_completed"]) if row else 0.0

    def save_task_estimate(self, estimate: TaskEstimate) -> None:
        """Record pre-dispatch forecast snapshot."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_estimates (
                    task_id, forecast_version, cohort, estimate_data_json,
                    sample_count, is_cold_start, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    estimate.task_id,
                    estimate.forecast_version,
                    estimate.cohort,
                    estimate.model_dump_json(),
                    estimate.sample_count,
                    1 if estimate.is_cold_start else 0,
                    estimate.created_at_utc,
                ),
            )

    def get_task_estimate(self, task_id: str) -> Optional[TaskEstimate]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT estimate_data_json FROM task_estimates WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return None
            return TaskEstimate.model_validate_json(row["estimate_data_json"])

    def record_checkpoint_decision(self, decision: CheckpointDecision) -> None:
        """Record warn/block checkpoint decision for audit."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO checkpoint_decisions (
                    task_id, phase_id, action, reason, overrun_probability,
                    eac_values_json, cpi_values_json, limits_exceeded_json, timestamp_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.task_id,
                    decision.phase_id,
                    decision.action.value,
                    decision.reason,
                    decision.overrun_probability,
                    json.dumps(decision.eac_values),
                    json.dumps(decision.cpi_values),
                    json.dumps(decision.limits_exceeded),
                    decision.timestamp_utc,
                ),
            )

    def get_latest_checkpoint_decision(self, task_id: str) -> Optional[CheckpointDecision]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoint_decisions WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if not row:
                return None
            return CheckpointDecision(
                task_id=row["task_id"],
                phase_id=row["phase_id"],
                action=CheckpointAction(row["action"]),
                reason=row["reason"],
                overrun_probability=row["overrun_probability"],
                eac_values=json.loads(row["eac_values_json"]),
                cpi_values=json.loads(row["cpi_values_json"]),
                limits_exceeded=json.loads(row["limits_exceeded_json"]),
                timestamp_utc=row["timestamp_utc"],
            )

    def save_reconciliation(self, report: ReconciliationReport) -> None:
        """Record post-task reconciliation report."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reconciliations (
                    task_id, report_json, completed, final_status, timestamp_utc
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    report.task_id,
                    report.model_dump_json(),
                    1 if report.completed else 0,
                    report.final_status,
                    report.timestamp_utc,
                ),
            )

    def get_reconciliation(self, task_id: str) -> Optional[ReconciliationReport]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT report_json FROM reconciliations WHERE task_id = ?", (task_id,)).fetchone()
            if not row:
                return None
            return ReconciliationReport.model_validate_json(row["report_json"])

    def get_cohort_calibration(self, cohort: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM cohort_calibration WHERE cohort = ?", (cohort,)).fetchone()
            return dict(row) if row else None

    def update_cohort_calibration(
        self,
        cohort: str,
        sample_count: int,
        alpha_coverage: float,
        q_hat: float,
        prior_mean_input_tokens: float,
        prior_mean_output_tokens: float,
        prior_mean_time_seconds: float,
        updated_at_utc: str,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cohort_calibration (
                    cohort, sample_count, alpha_coverage, q_hat,
                    prior_mean_input_tokens, prior_mean_output_tokens,
                    prior_mean_time_seconds, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cohort,
                    sample_count,
                    alpha_coverage,
                    q_hat,
                    prior_mean_input_tokens,
                    prior_mean_output_tokens,
                    prior_mean_time_seconds,
                    updated_at_utc,
                ),
            )
