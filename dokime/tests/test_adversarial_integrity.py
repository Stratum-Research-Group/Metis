"""
Unit tests for Adversarial Evaluation-Gate Integrity in Dokime
per specification 02-evaluation-gate-integrity-dokime-krisis-87c5fa3f.txt.
"""

import pytest

from dokime.adversarial import (
    AdversarialGateAuditor,
    AuditFailure,
    AuditGateResult,
    ExecutionKind,
    ReconciliationRecord,
    SelectiveRiskReport,
    TaskDataKind,
)


def test_count_reconciliation_balanced():
    """Valid reconciliation when all attempts equal sum of components."""
    rec = ReconciliationRecord(
        total_attempted=100,
        passed=80,
        failed=10,
        timed_out=5,
        invalid=3,
        abstained=2,
    )
    assert rec.is_balanced
    assert rec.total_accounted == 100


def test_count_reconciliation_unbalanced():
    """Detects missing/dropped timeouts or uncounted samples."""
    rec = ReconciliationRecord(
        total_attempted=100,
        passed=80,
        failed=10,
        timed_out=0,  # Dropped!
        invalid=0,
        abstained=0,
    )
    assert not rec.is_balanced
    assert rec.total_accounted == 90


def test_auditor_rejects_dropped_timeouts():
    """Auditor fails run when total attempted does not match accounted observations."""
    manifest = {
        "execution_kind": "REAL",
        "task_data_kind": "REAL_WORLD",
        "total_attempted": 10,
    }
    # Only 8 observations supplied
    observations = [
        {"task_id": f"t{i}", "status": "PASSED", "latency_ns": 50_000_000}
        for i in range(8)
    ]
    res = AdversarialGateAuditor.audit_run(manifest, observations)
    assert not res.eligible_for_promotion
    assert any("Count reconciliation failure" in r for r in res.rejection_reasons)


def test_auditor_rejects_mock_execution_for_promotion():
    """Mock/synthetic runs are recorded accurately but rejected for production promotion."""
    manifest = {
        "execution_kind": "MOCK",
        "task_data_kind": "FIXTURE",
        "total_attempted": 5,
    }
    observations = [
        {"task_id": f"t{i}", "status": "PASSED", "latency_ns": 50_000_000}
        for i in range(5)
    ]
    res = AdversarialGateAuditor.audit_run(manifest, observations)
    assert not res.eligible_for_promotion
    assert any("is synthetic/mock. Ineligible for real promotion" in r for r in res.rejection_reasons)


def test_auditor_detects_unitless_or_implausible_latency():
    """Auditor rejects unitless 'latency' and implausible conversion errors (<5µs for real model)."""
    manifest = {
        "execution_kind": "REAL",
        "task_data_kind": "REAL_WORLD",
        "total_attempted": 2,
    }
    # First has unitless 'latency', second has 150 ns (< 5 µs)
    observations = [
        {"task_id": "t1", "status": "PASSED", "latency": 150},
        {"task_id": "t2", "status": "PASSED", "latency_ns": 150},
    ]
    res = AdversarialGateAuditor.audit_run(manifest, observations)
    assert not res.eligible_for_promotion
    assert any("uses unitless 'latency'" in r for r in res.rejection_reasons)
    assert any("reports implausible real latency: 150 ns" in r for r in res.rejection_reasons)


def test_global_p99_computation():
    """Auditor computes true global p99 from raw observations."""
    latencies = [i * 1_000_000 for i in range(1, 101)]  # 1ms to 100ms
    p99 = AdversarialGateAuditor.compute_global_p99_ns(latencies)
    assert p99 == 99_010_000 or p99 == 99_000_000 or (98_000_000 <= p99 <= 100_000_000)

    # Rejects empty list
    with pytest.raises(AuditFailure, match="Cannot compute global p99 from empty"):
        AdversarialGateAuditor.compute_global_p99_ns([])


def test_auditor_detects_training_data_leakage():
    """Detects when evaluation items share content hashes with training/dev sets."""
    manifest = {
        "execution_kind": "REAL",
        "task_data_kind": "REAL_WORLD",
        "total_attempted": 2,
    }
    observations = [
        {"task_id": "t1", "status": "PASSED", "latency_ns": 60_000_000, "source_content_hash": "leak_hash_123"},
        {"task_id": "t2", "status": "PASSED", "latency_ns": 65_000_000, "source_content_hash": "clean_hash_456"},
    ]
    training_hashes = {"leak_hash_123", "train_hash_789"}
    res = AdversarialGateAuditor.audit_run(manifest, observations, training_source_hashes=training_hashes)

    assert not res.eligible_for_promotion
    assert any("Contamination / leakage detected" in r for r in res.rejection_reasons)


def test_auditor_rejects_total_abstention_masquerading_as_accuracy():
    """Zero or low coverage due to total abstention cannot pass gate."""
    manifest = {
        "execution_kind": "REAL",
        "task_data_kind": "REAL_WORLD",
        "total_attempted": 10,
    }
    observations = [
        {"task_id": f"t{i}", "status": "ABSTAINED", "latency_ns": 10_000_000}
        for i in range(10)
    ]
    res = AdversarialGateAuditor.audit_run(manifest, observations, min_required_coverage=0.5)

    assert not res.eligible_for_promotion
    assert res.selective_risk.coverage == 0.0
    assert any("Coverage 0.0% is below required minimum" in r for r in res.rejection_reasons)


def test_clean_real_run_passes_adversarial_audit():
    """A genuine real evaluation run with balanced counts and safe latency passes."""
    manifest = {
        "execution_kind": "REAL",
        "task_data_kind": "REAL_WORLD",
        "total_attempted": 10,
    }
    observations = [
        {"task_id": f"t{i}", "status": "PASSED", "latency_ns": 45_000_000}
        for i in range(9)
    ] + [
        {"task_id": "t9", "status": "FAILED", "latency_ns": 55_000_000}
    ]
    res = AdversarialGateAuditor.audit_run(
        manifest,
        observations,
        max_allowed_p99_latency_ns=100_000_000,
        min_required_coverage=0.8,
        max_allowed_selective_risk=0.20,
    )

    assert res.eligible_for_promotion
    assert len(res.rejection_reasons) == 0
    assert res.reconciliation.is_balanced
    assert res.selective_risk.coverage == 1.0
    assert res.selective_risk.selective_risk == 0.10
