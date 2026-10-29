"""Tests for ASPEST active selective prediction sampler in Paideia."""

from pathlib import Path
import pytest
from click.testing import CliRunner

from paideia.contracts.krisis_schema import ActiveSamplingConfig, KrisisActiveRecord
from paideia.data.aspest_sampler import (
    ASPESTSampler,
    compute_distance,
    load_krisis_active_records,
    save_krisis_active_records,
)
from paideia.cli import cli


def test_compute_distance_token_jaccard():
    rec1 = KrisisActiveRecord(query_id="1", question="What is quantum mechanics?", decision_status="abstained", rejection_score=0.8)
    rec2 = KrisisActiveRecord(query_id="2", question="What is quantum physics?", decision_status="abstained", rejection_score=0.8)
    rec3 = KrisisActiveRecord(query_id="3", question="Capital city of France", decision_status="abstained", rejection_score=0.8)

    dist_near = compute_distance(rec1, rec2, metric="token_jaccard")
    dist_far = compute_distance(rec1, rec3, metric="token_jaccard")

    assert 0.0 < dist_near < 1.0
    assert dist_far == 1.0  # Disjoint vocabularies
    assert dist_near < dist_far


def test_compute_distance_cosine():
    rec1 = KrisisActiveRecord(query_id="1", question="Q1", decision_status="abstained", rejection_score=0.8, embedding=[1.0, 0.0])
    rec2 = KrisisActiveRecord(query_id="2", question="Q2", decision_status="abstained", rejection_score=0.8, embedding=[1.0, 0.0])
    rec3 = KrisisActiveRecord(query_id="3", question="Q3", decision_status="abstained", rejection_score=0.8, embedding=[0.0, 1.0])

    assert compute_distance(rec1, rec2, metric="cosine") == pytest.approx(0.0, abs=1e-5)
    assert compute_distance(rec1, rec3, metric="cosine") == pytest.approx(0.5, abs=1e-5)


def test_aspest_uncertainty_vs_diversity():
    # 3 items about "quantum": query_q1 (0.95), query_q2 (0.90), query_q3 (0.85)
    # 1 item about "france": query_f1 (0.80)
    pool = [
        KrisisActiveRecord(query_id="query_q1", question="quantum computing physics", decision_status="abstained", rejection_score=0.95),
        KrisisActiveRecord(query_id="query_q2", question="quantum information physics", decision_status="abstained", rejection_score=0.90),
        KrisisActiveRecord(query_id="query_q3", question="quantum algorithms physics", decision_status="abstained", rejection_score=0.85),
        KrisisActiveRecord(query_id="query_f1", question="capital of france history", decision_status="abstained", rejection_score=0.80),
    ]

    # Pure uncertainty (beta = 1.0, budget = 2) should pick query_q1 and query_q2 (both quantum)
    cfg_u = ActiveSamplingConfig(budget=2, beta=1.0)
    selected_u, report_u = ASPESTSampler.select_active_batch(pool, cfg_u)
    assert len(selected_u) == 2
    assert [r.query_id for r in selected_u] == ["query_q1", "query_q2"]

    # Balanced ASPEST (beta = 0.5, budget = 2) should pick query_q1 first, then query_f1 for diversity!
    cfg_balanced = ActiveSamplingConfig(budget=2, beta=0.5)
    selected_b, report_b = ASPESTSampler.select_active_batch(pool, cfg_balanced)
    assert len(selected_b) == 2
    assert selected_b[0].query_id == "query_q1"
    assert selected_b[1].query_id == "query_f1"  # Diverse topic selected over duplicate quantum queries
    assert report_b.mean_diversity_distance > report_u.mean_diversity_distance


def test_aspest_filtering_and_budget_bounds():
    pool = [
        KrisisActiveRecord(query_id="q1", question="A", decision_status="selected", rejection_score=0.0),
        KrisisActiveRecord(query_id="q2", question="B", decision_status="abstained", rejection_score=0.2),
        KrisisActiveRecord(query_id="q3", question="C", decision_status="abstained", rejection_score=0.7),
    ]
    # Filter min_rejection_score >= 0.5: only q3 eligible
    cfg = ActiveSamplingConfig(budget=5, min_rejection_score=0.5)
    selected, report = ASPESTSampler.select_active_batch(pool, cfg)
    assert len(selected) == 1
    assert selected[0].query_id == "q3"
    assert report.eligible_count == 1
    assert report.total_pool_size == 3


def test_serialization_and_cli_subcommand(tmp_path: Path):
    in_file = tmp_path / "krisis_abstains.jsonl"
    out_file = tmp_path / "active_curated.jsonl"

    records = [
        KrisisActiveRecord(query_id=f"id_{i}", question=f"Question text {i}", decision_status="abstained", rejection_score=0.5 + i * 0.1)
        for i in range(5)
    ]
    save_krisis_active_records(records, in_file)

    reloaded = load_krisis_active_records(in_file)
    assert len(reloaded) == 5

    # Run CLI command
    runner = CliRunner()
    result = runner.invoke(cli, [
        "dataset", "sample-active",
        str(in_file),
        "--output", str(out_file),
        "--budget", "3",
        "--beta", "0.7",
    ])
    assert result.exit_code == 0
    assert "ASPEST Active Selection Audit" in result.output
    assert out_file.is_file()

    curated = load_krisis_active_records(out_file)
    assert len(curated) == 3
