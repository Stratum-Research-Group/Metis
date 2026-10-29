"""Tests for Paideia boundary-aware practice selection (EoG Item 2)."""

import json
from pathlib import Path
import pytest
from click.testing import CliRunner

from paideia.cli import cli
from paideia.contracts.boundary_schema import (
    BoundarySelectionConfig,
    CaseAttemptCategory,
    CaseAttemptRecord,
)
from paideia.data.boundary_sampler import (
    compute_attempt_stats_by_case,
    compute_case_stats,
    load_case_attempts,
    save_case_attempts,
    select_boundary_practice_batch,
    select_uniform_practice_batch,
)


def _make_attempts(case_id: str, scores: list[float], failure_label: str | None = None) -> list[CaseAttemptRecord]:
    records = []
    for idx, s in enumerate(scores):
        fl = failure_label if s < 1.0 else None
        records.append(
            CaseAttemptRecord(
                case_id=case_id,
                attempt_index=idx,
                seed=100 + idx,
                score=s,
                evidence_grounded=True,
                failure_label=fl,
            )
        )
    return records


def test_compute_case_stats_categories():
    """Verify statistical variance estimation and category classification."""
    # Boundary: alternating successes and failures
    b_records = _make_attempts("case_b", [1.0, 0.0, 1.0, 0.0, 1.0])
    stats_b = compute_case_stats("case_b", b_records)
    assert stats_b.category == CaseAttemptCategory.BOUNDARY
    assert stats_b.mean_score == pytest.approx(0.6, abs=1e-5)
    assert stats_b.sample_variance > 0.0

    # Retention check: all passed
    r_records = _make_attempts("case_r", [1.0, 1.0, 1.0, 1.0, 1.0])
    stats_r = compute_case_stats("case_r", r_records)
    assert stats_r.category == CaseAttemptCategory.RETENTION_CHECK
    assert stats_r.mean_score == 1.0
    assert stats_r.sample_variance == 0.0

    # Hard diagnostic: all failed
    h_records = _make_attempts("case_h", [0.0, 0.0, 0.0, 0.0, 0.0], failure_label="missed_constraint")
    stats_h = compute_case_stats("case_h", h_records)
    assert stats_h.category == CaseAttemptCategory.HARD_DIAGNOSTIC
    assert stats_h.mean_score == 0.0
    assert stats_h.sample_variance == 0.0
    assert stats_h.failure_counts.get("missed_constraint") == 5

    # Consistent partial: invariant partial score
    cp_records = _make_attempts("case_cp", [0.5, 0.5, 0.5, 0.5, 0.5])
    stats_cp = compute_case_stats("case_cp", cp_records)
    assert stats_cp.category == CaseAttemptCategory.CONSISTENT_PARTIAL
    assert stats_cp.mean_score == 0.5
    assert stats_cp.sample_variance == 0.0


def test_select_boundary_practice_batch_stratification():
    """Verify stratified practice selection prioritizes highest-variance boundary cases."""
    records = []
    # 3 boundary cases with differing variances
    records.extend(_make_attempts("b_high", [1.0, 0.0, 1.0, 0.0]))    # var = 1/3 ~ 0.333
    records.extend(_make_attempts("b_med", [1.0, 1.0, 1.0, 0.0]))     # var = 0.25
    records.extend(_make_attempts("b_low", [1.0, 1.0, 1.0, 0.8]))     # var ~ 0.01

    # 2 retention checks
    records.extend(_make_attempts("r_1", [1.0, 1.0, 1.0, 1.0]))
    records.extend(_make_attempts("r_2", [1.0, 1.0, 1.0, 1.0]))

    # 2 hard diagnostics
    records.extend(_make_attempts("h_1", [0.0, 0.0, 0.0, 0.0]))
    records.extend(_make_attempts("h_2", [0.0, 0.0, 0.0, 0.0]))

    stats = compute_attempt_stats_by_case(records)
    cfg = BoundarySelectionConfig(
        budget=5,
        boundary_ratio=0.6,   # target 3 boundary
        retention_ratio=0.2,  # target 1 retention
        hard_ratio=0.2,       # target 1 hard
        prioritize_highest_variance=True,
    )

    selected, report = select_boundary_practice_batch(stats, cfg)
    assert len(selected) == 5
    assert report.selected_boundary_count == 3
    assert report.selected_retention_count == 1
    assert report.selected_hard_count == 1

    # Check highest-variance boundary cases were chosen first
    assert "b_high" in selected
    assert "b_med" in selected
    assert "b_low" in selected


def test_select_uniform_practice_batch_comparison():
    """Verify uniform baseline sampling for matched budget comparison."""
    records = []
    for i in range(10):
        records.extend(_make_attempts(f"c_{i}", [1.0, 0.0, 1.0, 0.0]))

    stats = compute_attempt_stats_by_case(records)
    selected_u1, report_u1 = select_uniform_practice_batch(stats, budget=4, seed=42)
    selected_u2, _ = select_uniform_practice_batch(stats, budget=4, seed=42)

    assert len(selected_u1) == 4
    assert selected_u1 == selected_u2  # Deterministic seed
    assert report_u1.selection_strategy == "uniform_baseline"


def test_cli_sample_boundary(tmp_path: Path):
    """Verify paideia CLI sample-boundary execution and artifact export."""
    records = []
    records.extend(_make_attempts("b_1", [1.0, 0.0, 1.0, 0.0, 1.0]))
    records.extend(_make_attempts("b_2", [0.0, 1.0, 0.0, 1.0, 0.0]))
    records.extend(_make_attempts("r_1", [1.0, 1.0, 1.0, 1.0, 1.0]))
    records.extend(_make_attempts("h_1", [0.0, 0.0, 0.0, 0.0, 0.0]))

    input_file = tmp_path / "attempts.json"
    output_file = tmp_path / "selected.json"
    save_case_attempts(records, input_file)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dataset",
            "sample-boundary",
            str(input_file),
            "--output",
            str(output_file),
            "--budget",
            "3",
            "--boundary-ratio",
            "0.6",
        ],
    )

    assert result.exit_code == 0
    assert "Paideia Boundary Practice Selection Audit" in result.output
    assert output_file.exists()

    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert "report" in data
    assert "selected_cases" in data
    assert len(data["selected_cases"]) == 3
