"""
Integration and unit tests for ContinuousPolicyRetrainingCoordinator and `paideia loop`.

Tests the complete closed-loop self-improving policy pipeline:
- Krisis active selective prediction / attempt logs ingestion.
- Boundary variance sampling.
- QLoRA training with matrix Shampoo and AdamW.
- Dokime paired promotion gate and bundle serialization.
- Hestia adapter export and crash-safe CAS model hot-swapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest

from paideia.contracts.boundary_schema import CaseAttemptRecord
from paideia.training.loop import (
    ContinuousPolicyRetrainingCoordinator,
    RetrainingCoordinatorConfig,
    RetrainingCycleResult,
)


def _generate_mock_krisis_records(count: int = 15) -> list[dict]:
    """Generate mock Krisis active selective prediction records."""
    records = []
    for i in range(count):
        # Generate some boundary/abstained records with rejection scores
        rejection = 0.6 if i % 2 == 0 else 0.8
        records.append({
            "query_id": f"query_{i:03d}",
            "question": f"How do I route query {i} for optimal execution?",
            "decision_status": "abstained",
            "decision_reason": "ambiguity_above_threshold",
            "rejection_score": rejection,
            "prediction_set_size": 2,
            "metadata": {"task": "router", "attempt": 0},
        })
    return records


def _generate_mock_attempt_records(cases: int = 6, attempts_per_case: int = 3) -> list[dict]:
    """Generate repeated rollout attempts to create true statistical variance."""
    records = []
    for c in range(cases):
        cid = f"case_{c:03d}"
        for a in range(attempts_per_case):
            # Vary scores to create boundary variance
            score = 1.0 if (c + a) % 2 == 0 else 0.0
            records.append({
                "case_id": cid,
                "attempt_index": a,
                "score": score,
                "seed": 42 + a,
                "evidence_grounded": True,
                "failure_label": "suboptimal" if score == 0.0 else None,
                "model_and_config": "qwen2.5-14b-instruct",
            })
    return records


def test_coordinator_skips_when_insufficient_samples(tmp_path: Path):
    """Verify coordinator skips training if newly appended records < min_samples."""
    log_file = tmp_path / "krisis_log.jsonl"
    few_records = _generate_mock_krisis_records(3)
    with open(log_file, "w", encoding="utf-8") as f:
        for r in few_records:
            f.write(json.dumps(r) + "\n")

    cfg = RetrainingCoordinatorConfig(
        log_file=log_file,
        output_dir=tmp_path / "runs",
        min_samples=10,
        auto_swap=False,
    )

    coordinator = ContinuousPolicyRetrainingCoordinator(cfg)
    result = coordinator.run_cycle()

    assert result.status == "SKIPPED_INSUFFICIENT_SAMPLES"
    assert result.samples_ingested == 3
    assert result.samples_selected == 0
    assert result.training_run_id is None


def test_coordinator_e2e_shampoo_training_and_promotion(tmp_path: Path):
    """
    Verify complete closed loop:
    1. Read 18 attempt records.
    2. Curate boundary batch via AttemptRolloutVarianceSampler.
    3. Execute real QLoRA training with Shampoo optimizer.
    4. Pass Dokime promotion gate.
    5. Export Hestia adapter and registry entry.
    """
    log_file = tmp_path / "dokime_attempts.jsonl"
    records = _generate_mock_attempt_records(cases=6, attempts_per_case=3)  # 18 records
    with open(log_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cfg = RetrainingCoordinatorConfig(
        log_file=log_file,
        output_dir=tmp_path / "runs",
        task="router",
        optimizer="shampoo",
        min_samples=10,
        epochs=3,
        batch_size=2,
        lr=0.005,
        rank=8,
        allow_simulated=False,
        auto_swap=False,
        seed=42,
    )

    coordinator = ContinuousPolicyRetrainingCoordinator(cfg)
    result = coordinator.run_cycle()

    assert result.status == "TRAINED_AND_PROMOTED"
    assert result.samples_ingested == 18
    assert result.samples_selected > 0
    assert result.training_run_id is not None
    assert result.adapter_digest is not None
    assert result.adapter_id is not None
    assert result.eval_verdict["decision"] == "PROMOTED"

    # Verify run manifest and exported adapter files
    checkpoint_dir = Path(result.checkpoint_dir)
    assert (checkpoint_dir / "adapter_config.json").exists()
    assert (checkpoint_dir / "adapter_model.bin").exists()
    assert (checkpoint_dir / "hestia_registry_entry.json").exists()

    with open(checkpoint_dir / "hestia_registry_entry.json") as f:
        entry = json.load(f)
    assert entry["adapter_id"] == result.adapter_id
    assert entry["task"] == "router"

    # Verify state tracking
    assert coordinator.state["last_processed_line"] == 18
    assert coordinator.state["active_adapter_id"] == result.adapter_id
    assert coordinator.state["total_cycles_completed"] == 1


def test_coordinator_rejection_path(tmp_path: Path):
    """Verify that when promotion evaluation fails, candidate is rejected and not exported."""
    log_file = tmp_path / "attempts.jsonl"
    records = _generate_mock_attempt_records(cases=5, attempts_per_case=3)
    with open(log_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cfg = RetrainingCoordinatorConfig(
        log_file=log_file,
        output_dir=tmp_path / "runs",
        min_samples=10,
        epochs=1,
        auto_swap=False,
        allow_simulated=False,
    )

    # Force evaluator to return regressed candidate scores (candidate worse than baseline)
    def failing_evaluator(ckpt_dir, eval_cases):
        n = max(len(eval_cases), 10)
        b_scores = [0.9] * n
        c_scores = [0.2] * n  # Severe regression
        latencies = [15.0] * n
        failures = 0
        return b_scores, c_scores, latencies, failures

    coordinator = ContinuousPolicyRetrainingCoordinator(cfg, evaluator=failing_evaluator)
    result = coordinator.run_cycle()

    assert result.status == "TRAINED_AND_REJECTED"
    assert result.eval_verdict["decision"] == "REJECTED"
    assert len(result.eval_verdict["rejection_reasons"]) > 0

    # Verify state advanced to prevent re-training on rejected window
    assert coordinator.state["last_processed_line"] == 15
    assert coordinator.state["active_adapter_id"] is None


def test_coordinator_log_offset_incremental_processing(tmp_path: Path):
    """Verify coordinator only processes newly appended log lines in subsequent cycles."""
    log_file = tmp_path / "stream.jsonl"
    first_batch = _generate_mock_attempt_records(cases=4, attempts_per_case=3)  # 12 records
    with open(log_file, "w", encoding="utf-8") as f:
        for r in first_batch:
            f.write(json.dumps(r) + "\n")

    cfg = RetrainingCoordinatorConfig(
        log_file=log_file,
        output_dir=tmp_path / "runs",
        min_samples=10,
        epochs=1,
        auto_swap=False,
        allow_simulated=False,
    )

    coordinator = ContinuousPolicyRetrainingCoordinator(cfg)
    res1 = coordinator.run_cycle()
    assert res1.status == "TRAINED_AND_PROMOTED"
    assert res1.samples_ingested == 12

    # Second run without appending: should detect 0 new records and skip
    res2 = coordinator.run_cycle()
    assert res2.status == "SKIPPED_INSUFFICIENT_SAMPLES"
    assert res2.samples_ingested == 0

    # Append 15 new records
    second_batch = _generate_mock_attempt_records(cases=5, attempts_per_case=3)
    with open(log_file, "a", encoding="utf-8") as f:
        for r in second_batch:
            f.write(json.dumps(r) + "\n")

    # Third run: should detect and train on the 15 newly appended records
    res3 = coordinator.run_cycle()
    assert res3.status == "TRAINED_AND_PROMOTED"
    assert res3.samples_ingested == 15
    assert coordinator.state["total_cycles_completed"] == 2
    assert coordinator.state["last_processed_line"] == 27


def test_coordinator_in_process_hestia_cas_swap(tmp_path: Path):
    """
    Integration test connecting coordinator to Hestia ServerManager duck-typed interface:
    Verifies that upon promotion, coordinator registers bundle and hot-swaps
    into Hestia, incrementing active generation.
    """
    import asyncio

    class MockSwapRecord:
        def __init__(self, generation: int, bundle_id: str):
            self.generation = generation
            self.bundle_id = bundle_id

    class MockChangeRecord:
        class State:
            value = "COMMITTED"
        state = State()

    class MockSwapManager:
        def __init__(self):
            self.active_records = {}

    class MockHestiaServerManager:
        def __init__(self):
            self.swap_manager = MockSwapManager()
            self.registered_bundles = {}

        def register_paideia_bundle(self, entry_data, adapter_path, base_model_name):
            class Bundle:
                bundle_id = entry_data["adapter_id"]
            class Entry:
                name = entry_data["adapter_id"]
            self.registered_bundles[entry_data["adapter_id"]] = (entry_data, adapter_path)
            return Bundle(), Entry()

        async def swap_model(self, slot, candidate_bundle_id, expected_generation, idempotency_key, auto_commit=True):
            self.swap_manager.active_records[slot] = MockSwapRecord(
                generation=expected_generation + 1,
                bundle_id=candidate_bundle_id,
            )
            return MockChangeRecord()

    manager = MockHestiaServerManager()

    # Prepare coordinator records
    log_file = tmp_path / "krisis_active.jsonl"
    records = _generate_mock_attempt_records(cases=5, attempts_per_case=3)  # 15 records
    with open(log_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cfg = RetrainingCoordinatorConfig(
        log_file=log_file,
        output_dir=tmp_path / "runs",
        base_model="qwen2.5-14b-instruct",
        task="router",
        min_samples=10,
        epochs=1,
        auto_swap=True,
        allow_simulated=False,
    )

    coordinator = ContinuousPolicyRetrainingCoordinator(cfg, hestia_manager=manager)
    result = asyncio.run(coordinator.run_cycle_async())

    assert result.status == "TRAINED_AND_PROMOTED"
    assert result.hestia_swap_status is not None
    assert result.hestia_swap_status["status"] == "swapped"
    assert result.hestia_swap_status["active_generation"] == 1
    assert result.hestia_swap_status["bundle_id"] == result.adapter_id

    # Check mock manager state
    active = manager.swap_manager.active_records.get("primary")
    assert active is not None
    assert active.generation == 1
    assert active.bundle_id == result.adapter_id


def test_coordinator_http_hestia_mock(tmp_path: Path):
    """Test REST HTTP swap interaction with mocked urllib request responses."""
    log_file = tmp_path / "active.jsonl"
    records = _generate_mock_attempt_records(cases=4, attempts_per_case=3)
    with open(log_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cfg = RetrainingCoordinatorConfig(
        log_file=log_file,
        output_dir=tmp_path / "runs",
        base_model="qwen2.5-14b-instruct",
        task="router",
        min_samples=10,
        epochs=1,
        hestia_url="http://127.0.0.1:8080",
        auto_swap=True,
        allow_simulated=False,
    )

    mock_responses = [
        {"active_records": {"primary": {"generation": 2, "bundle_id": "prev_bundle"}}},  # GET /manage/swap
        {"status": "registered", "bundle": {"bundle_id": "test"}},                          # POST /manage/bundles
        {"status": "swapped", "active": {"generation": 3, "bundle_id": "new_adapter"}},    # POST /manage/swap
    ]

    with patch("paideia.training.loop._execute_http_json", side_effect=mock_responses):
        coordinator = ContinuousPolicyRetrainingCoordinator(cfg)
        result = coordinator.run_cycle()

    assert result.status == "TRAINED_AND_PROMOTED"
    assert result.hestia_swap_status is not None
    assert result.hestia_swap_status["status"] == "swapped"
    assert result.hestia_swap_status["active"]["generation"] == 3


def test_cli_loop_once(tmp_path: Path):
    """Test `paideia loop --once` invocation from Click CLI runner."""
    from click.testing import CliRunner
    from paideia.cli import cli

    log_file = tmp_path / "krisis_data.jsonl"
    records = _generate_mock_attempt_records(cases=4, attempts_per_case=3)
    with open(log_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    runner = CliRunner()
    res = runner.invoke(cli, [
        "loop",
        "--log-file", str(log_file),
        "--output-dir", str(tmp_path / "cli_runs"),
        "--min-samples", "10",
        "--epochs", "1",
        "--once",
        "--no-auto-swap",
    ])

    assert res.exit_code == 0
    assert "=== Paideia Continuous Policy Retraining Loop ===" in res.output
    assert "TRAINED_AND_PROMOTED" in res.output
    assert "Adapter ID:" in res.output
