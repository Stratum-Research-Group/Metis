"""Unit tests for AttemptRolloutVarianceSampler ingesting rollouts, Krisis logs, and Dokime runs."""

import json
from pathlib import Path
import pytest

from paideia.contracts.boundary_schema import (
    BoundarySelectionConfig,
    CaseAttemptCategory,
    CaseAttemptRecord,
)
from paideia.contracts.krisis_schema import KrisisActiveRecord
from paideia.data.boundary_sampler import AttemptRolloutVarianceSampler


def test_sampler_ingest_case_attempts():
    """Verify ingesting CaseAttemptRecord instances."""
    sampler = AttemptRolloutVarianceSampler()
    recs = [
        CaseAttemptRecord(case_id="c1", attempt_index=0, seed=1, score=1.0),
        CaseAttemptRecord(case_id="c1", attempt_index=1, seed=2, score=0.0),
        CaseAttemptRecord(case_id="c2", attempt_index=0, seed=1, score=1.0),
        CaseAttemptRecord(case_id="c2", attempt_index=1, seed=2, score=1.0),
    ]
    added = sampler.ingest_rollouts(recs)
    assert added == 4
    assert len(sampler.records) == 4

    stats = sampler.compute_stats()
    assert stats["c1"].category == CaseAttemptCategory.BOUNDARY
    assert stats["c2"].category == CaseAttemptCategory.RETENTION_CHECK


def test_sampler_ingest_krisis_records():
    """Verify ingesting KrisisActiveRecord instances with score derivation and attempt indexing."""
    sampler = AttemptRolloutVarianceSampler()
    k_recs = [
        KrisisActiveRecord(
            query_id="q_frontier",
            question="What is the topological dimension?",
            decision_status="abstained",
            decision_reason="ambiguous_prediction_set",
            rejection_score=0.4,
            metadata={"seed": 101},
        ),
        KrisisActiveRecord(
            query_id="q_frontier",
            question="What is the topological dimension?",
            decision_status="selected",
            rejection_score=0.0,
            metadata={"seed": 102},
        ),
        KrisisActiveRecord(
            query_id="q_hard",
            question="Solve intractable conjecture.",
            decision_status="abstained",
            decision_reason="conformal_empty_set",
            rejection_score=1.0,
            metadata={"seed": 101},
        ),
        KrisisActiveRecord(
            query_id="q_hard",
            question="Solve intractable conjecture.",
            decision_status="abstained",
            decision_reason="conformal_empty_set",
            rejection_score=1.0,
            metadata={"seed": 102},
        ),
    ]

    added = sampler.ingest_krisis_records(k_recs)
    assert added == 4

    # Verify attempt indices
    q_frontier_attempts = [r for r in sampler.records if r.case_id == "q_frontier"]
    assert len(q_frontier_attempts) == 2
    assert q_frontier_attempts[0].attempt_index == 0
    assert q_frontier_attempts[1].attempt_index == 1
    assert q_frontier_attempts[0].score == 0.6  # 1.0 - 0.4
    assert q_frontier_attempts[1].score == 1.0  # selected -> 1.0

    stats = sampler.compute_stats()
    assert stats["q_frontier"].category == CaseAttemptCategory.BOUNDARY
    assert stats["q_frontier"].sample_variance > 0.0
    assert stats["q_hard"].category == CaseAttemptCategory.HARD_DIAGNOSTIC
    assert stats["q_hard"].sample_variance == 0.0


def test_sampler_ingest_dokime_eval_run_file(tmp_path: Path):
    """Verify loading from JSON file containing attempt_records or active_records."""
    payload = {
        "benchmark_id": "dokime_graph_qa_rollout",
        "attempt_records": [
            {"case_id": "case_1", "attempt_index": 0, "seed": 42, "score": 1.0},
            {"case_id": "case_1", "attempt_index": 1, "seed": 43, "score": 0.0},
            {"case_id": "case_2", "attempt_index": 0, "seed": 42, "score": 1.0},
            {"case_id": "case_2", "attempt_index": 1, "seed": 43, "score": 1.0},
        ],
    }
    json_path = tmp_path / "dokime_report.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    sampler = AttemptRolloutVarianceSampler()
    added = sampler.ingest_dokime_eval_run(json_path)
    assert added == 4

    selected, report = sampler.sample(BoundarySelectionConfig(budget=1, boundary_ratio=1.0, retention_ratio=0.0, hard_ratio=0.0))
    assert selected == ["case_1"]
    assert report.selected_boundary_count == 1


def test_sampler_filter_and_export_dataset(tmp_path: Path):
    """Verify filtering practice datasets and exporting to JSON / JSONL."""
    sampler = AttemptRolloutVarianceSampler()
    recs = [
        CaseAttemptRecord(case_id="case_A", attempt_index=0, seed=1, score=1.0),
        CaseAttemptRecord(case_id="case_A", attempt_index=1, seed=2, score=0.0),
        CaseAttemptRecord(case_id="case_B", attempt_index=0, seed=1, score=1.0),
        CaseAttemptRecord(case_id="case_B", attempt_index=1, seed=2, score=1.0),
    ]
    sampler.ingest_rollouts(recs)
    sampler.sample(BoundarySelectionConfig(budget=1, boundary_ratio=1.0, retention_ratio=0.0, hard_ratio=0.0))

    # Raw training dataset items
    raw_dataset = [
        {"case_id": "case_A", "question": "Question A", "gold_answer": "ans_A"},
        {"case_id": "case_B", "question": "Question B", "gold_answer": "ans_B"},
    ]

    filtered = sampler.filter_dataset(raw_dataset)
    assert len(filtered) == 1
    assert filtered[0]["case_id"] == "case_A"

    # Export to jsonl
    out_file = tmp_path / "boundary_practice.jsonl"
    sampler.export_practice_dataset(raw_dataset, out_file)
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["case_id"] == "case_A"
