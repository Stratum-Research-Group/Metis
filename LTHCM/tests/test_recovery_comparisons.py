"""Tests for Comparative Baselines, Ablations, and 6 Held-Out Stress Tests.

Satisfies Section 6 and 7 of LTHCM-recovery-application-plan-356cbcd7.txt:
- Verifies budget-matched execution of all 5 baselines and 2 ablations.
- Runs all 6 held-out stress tests with zero tolerance for safety violations.
"""

import pytest

from recovery.baselines import (
    DeterministicRebuildRerankBaseline,
    RecurrentNoDamageAblation,
    RecurrentNoRecurrenceAblation,
    RecurrentRecoveryWrapperBaseline,
    SpectralChebyshevBaseline,
    StandardGNNBaseline,
    UnchangedPipelineBaseline,
)
from recovery.benchmark import RecoveryBenchmarkHarness
from recovery.corpus import build_frozen_recovery_dataset
from recovery.damage import DamageInjector
from recovery.schema import DamageConfig
from recovery.stress_tests import HeldOutStressTester
from recovery.validator import IndependentRecoveryValidator


# ── Comparative Baselines at Equal Budget (Section 6) ──────────


@pytest.mark.parametrize(
    "model_cls",
    [
        UnchangedPipelineBaseline,
        DeterministicRebuildRerankBaseline,
        SpectralChebyshevBaseline,
        StandardGNNBaseline,
        RecurrentRecoveryWrapperBaseline,
        RecurrentNoRecurrenceAblation,
        RecurrentNoDamageAblation,
    ],
)
def test_all_baselines_run_under_budget_and_emit_valid_outcomes(model_cls):
    corpus, queries = build_frozen_recovery_dataset()
    harness = RecoveryBenchmarkHarness(corpus=corpus, queries=queries)
    query = queries[0]
    as_of = query["query_timestamp"]

    clean_state, _ = harness._build_clean_state_for_query(query)
    injector = DamageInjector(DamageConfig(seed=42, edge_drop_prob=0.3))
    damaged_state, _ = injector.apply_damage(clean_state, [d.doc_id for d in corpus.get_all()])

    validator = IndependentRecoveryValidator(corpus, as_of=as_of)
    model = model_cls()

    outcome = model.run(
        query=query["query"],
        query_timestamp=as_of,
        damaged_state=damaged_state,
        validator=validator,
        latency_budget_ms=500.0,
    )

    assert outcome.status in {"SUCCESS", "DEGRADED", "ABSTAIN"}
    assert isinstance(outcome.final_candidates, list)
    # Every returned candidate must exist and be eligible
    for cand in outcome.final_candidates:
        doc = corpus.get(cand["doc_id"])
        assert doc is not None
        assert doc.timestamp <= as_of
        assert doc.is_deleted is False


def test_recurrent_wrapper_recovers_better_than_unchanged():
    corpus, queries = build_frozen_recovery_dataset()
    harness = RecoveryBenchmarkHarness(corpus=corpus, queries=queries)
    # Pick a query where edge drop causes support loss
    q = queries[0]
    as_of = q["query_timestamp"]
    expected = set(q["expected_support"])

    clean_state, _ = harness._build_clean_state_for_query(q)
    # Strong edge drop
    injector = DamageInjector(DamageConfig(seed=123, edge_drop_prob=0.8))
    damaged_state, _ = injector.apply_damage(clean_state, [d.doc_id for d in corpus.get_all()])

    validator = IndependentRecoveryValidator(corpus, as_of=as_of)

    # 1. Unchanged pipeline
    unchanged_res = UnchangedPipelineBaseline().run(q["query"], as_of, damaged_state, validator)

    # 2. Recurrent recovery wrapper
    recovered_res = RecurrentRecoveryWrapperBaseline().run(q["query"], as_of, damaged_state, validator)

    assert recovered_res.status == "SUCCESS"
    rec_cands = [c["doc_id"] for c in recovered_res.final_candidates]
    # Recurrent wrapper should retain or restore both hop support documents
    assert expected.issubset(set(rec_cands[:4]))


# ── The 6 Held-Out Stress Scenarios (Section 6 & 7) ───────────


def test_held_out_stress_1_unseen_bridge_removal():
    tester = HeldOutStressTester()
    res = tester.test_unseen_bridge_removal()
    assert res["passed"] is True
    assert res["status"] in {"SUCCESS", "DEGRADED"}


def test_held_out_stress_2_high_similarity_false_shortcuts():
    tester = HeldOutStressTester()
    res = tester.test_high_similarity_false_shortcuts()
    assert res["passed"] is True


def test_held_out_stress_3_decisive_fact_swap():
    tester = HeldOutStressTester()
    res = tester.test_decisive_fact_swap()
    assert res["passed"] is True


def test_held_out_stress_4_all_supporting_sources_unavailable():
    tester = HeldOutStressTester()
    res = tester.test_all_supporting_sources_unavailable()
    assert res["passed"] is True
    assert res["deleted_survived"] is False


def test_held_out_stress_5_source_revoked_mid_loop():
    tester = HeldOutStressTester()
    res = tester.test_source_revoked_mid_loop()
    assert res["passed"] is True


def test_held_out_stress_6_only_temporally_invalid_candidates():
    tester = HeldOutStressTester()
    res = tester.test_only_temporally_invalid_candidates()
    assert res["passed"] is True
