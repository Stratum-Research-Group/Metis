"""Unit tests for dataset ingestion, deduplication, attempt linkage, and splitting."""

from pathlib import Path
import pytest

from paideia.data.router_loader import load_router_dataset
from paideia.data.walk_loader import load_walk_dataset
from paideia.data.splitter import (
    split_grouped_dataset,
    derive_question_family_key,
)
from paideia.data.card import DatasetCard
from paideia.contracts.lthcm_schema import RouteUtility

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_router_dataset():
    fixture_path = FIXTURES_DIR / "lthcm_sample.json"
    util = RouteUtility(lambda_latency=0.001, lambda_cost=0.01)
    records, audit = load_router_dataset(
        fixture_path,
        utility=util,
        eligible_routes=["flat", "mined"],
    )

    assert len(records) == 5
    assert audit.valid_records == 5
    assert audit.schema_violations == 0
    assert audit.class_counts_type["comparison"] == 3
    assert audit.class_counts_type["bridge"] == 2
    # Best-route checks
    assert audit.class_counts_best_route["flat"] >= 3


def test_load_walk_dataset_and_attempt_linkage():
    fixture_path = FIXTURES_DIR / "dianoia_sample.jsonl"
    samples, audit = load_walk_dataset(fixture_path)

    # In fixture:
    # walk-uuid-1: 2 steps
    # walk-uuid-2: 2 attempts sharing the same walk_id; the first had conflict, the second re-walk resolved it
    # Expected: attempt linkage joins walk-uuid-2 and keeps only the resolved re-walk!
    assert audit.walk_attempts_joined == 1
    assert len(samples) == 4  # 2 steps from walk-1 + 2 steps from resolved walk-2
    assert audit.action_counts["EXPAND"] == 2
    assert audit.action_counts["ANSWER"] == 2


def test_split_grouped_dataset_leakage_protection():
    fixture_path = FIXTURES_DIR / "lthcm_sample.json"
    records, _ = load_router_dataset(fixture_path)

    splits = split_grouped_dataset(
        records,
        group_key_fn=lambda r: derive_question_family_key(r.question),
        train_ratio=0.60,
        dev_ratio=0.20,
        test_ratio=0.20,
        seed=42,
    )

    assert len(splits.train) + len(splits.dev) + len(splits.test) == len(records)
    assert splits.train_checksum != ""
    assert splits.dev_checksum != ""
    assert splits.test_checksum != ""

    # Verify no question stem leakage across splits
    train_stems = {derive_question_family_key(r.question) for r in splits.train}
    dev_stems = {derive_question_family_key(r.question) for r in splits.dev}
    test_stems = {derive_question_family_key(r.question) for r in splits.test}

    assert len(train_stems.intersection(dev_stems)) == 0
    assert len(train_stems.intersection(test_stems)) == 0
    assert len(dev_stems.intersection(test_stems)) == 0


def test_dataset_card_generation(tmp_path):
    card = DatasetCard(
        dataset_name="LTHCM-HotpotQA-Router",
        task="question_type_and_route_policy",
        license_name="MIT",
        source_revisions={"LTHCM": "8a71605379ee32059e714feb3648cf74f4b42dc6"},
        audit_summary={"total_records": 500, "valid_records": 500},
        split_summary={"seed": 42, "train_count": 350, "dev_count": 75, "test_count": 75},
    )

    out_file = card.save(tmp_path)
    assert out_file.exists()
    content = out_file.read_text()
    assert "LTHCM-HotpotQA-Router" in content
    assert "8a71605379ee32059e714feb3648cf74f4b42dc6" in content


def test_gold_trajectory_verification_gates():
    from paideia.contracts.dianoia_schema import DianoiaStep, DianoiaVerification, DianoiaWalkRecord

    step = DianoiaStep(node="n1", action="ANSWER", raw_output="ans", reason="ok")

    # 1. Unverified record is rejected
    rec_unverified = DianoiaWalkRecord(
        walk_id="w-1",
        question="Q?",
        model="m",
        final_answer="A",
        path=[step],
        verification=DianoiaVerification(verified=False),
    )
    assert rec_unverified.is_defensible_gold_trajectory() is False

    # 2. Simulated record is rejected
    rec_sim = DianoiaWalkRecord(
        walk_id="w-2",
        question="Q?",
        model="m",
        final_answer="A",
        path=[step],
        verification=DianoiaVerification(verified=True, simulated=True, provenance="mock"),
    )
    assert rec_sim.is_defensible_gold_trajectory() is False

    # 3. Contradiction / conflicts found is rejected
    rec_conflict = DianoiaWalkRecord(
        walk_id="w-3",
        question="Q?",
        model="m",
        final_answer="A",
        path=[step],
        verification=DianoiaVerification(verified=True, conflicts_found=True, provenance="sheaf"),
    )
    assert rec_conflict.is_defensible_gold_trajectory() is False

    # 4. Incorrect outcome is rejected
    rec_incorrect = DianoiaWalkRecord(
        walk_id="w-4",
        question="Q?",
        model="m",
        final_answer="A",
        path=[step],
        verification=DianoiaVerification(verified=True, conflicts_found=False, is_correct=False, provenance="sheaf"),
    )
    assert rec_incorrect.is_defensible_gold_trajectory() is False

    # 5. Verified non-simulated without conflicts is accepted
    rec_gold = DianoiaWalkRecord(
        walk_id="w-5",
        question="Q?",
        model="m",
        final_answer="A",
        path=[step],
        verification=DianoiaVerification(verified=True, conflicts_found=False, is_correct=True, provenance="ground_truth_eval"),
    )
    assert rec_gold.is_defensible_gold_trajectory() is True

