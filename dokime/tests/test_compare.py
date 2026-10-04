"""Tests for dokime compare logic, model scoping, and regression detection."""

import json
from pathlib import Path
import pytest

from dokime.compare import CompareError, run_comparison
from dokime.manifest import sha256_file


def _create_mock_run(
    run_dir: Path,
    model_id: str,
    scores: list[float],
    task_id: str = "arc_easy_25",
    dataset_checksum: str = "fixed_hash_123",
    expected_answer: str = "A",
):
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, score in enumerate(scores):
            f.write(
                json.dumps(
                    {
                        "run_id": run_dir.name,
                        "task_id": task_id,
                        "query_id": f"q_{idx:03d}",
                        "model_id": model_id,
                        "prompt_hash": "a" * 64,
                        "predicted": expected_answer if score == 1.0 else "X",
                        "expected": expected_answer,
                        "score": score,
                        "latency_ms": 10.0,
                        "timestamp": "2026-09-26T00:00:00Z",
                    }
                )
                + "\n"
            )

    # Write manifest.json with required checksum provenance
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_dir.name,
                "model_ids": [model_id],
                "task_ids": [task_id],
                "checksums": {
                    f"dataset_{task_id}": dataset_checksum,
                    "results_jsonl": sha256_file(jsonl_path),
                },
            },
            f,
            indent=2,
        )


def test_compare_improvement_detected(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    # Run A: 20% accuracy
    scores_a = [1.0] * 5 + [0.0] * 20
    # Run B: 80% accuracy
    scores_b = [1.0] * 20 + [0.0] * 5

    _create_mock_run(runs_dir / "run_base", "m8b", scores_a)
    _create_mock_run(runs_dir / "run_cand", "m8b", scores_b)

    report = run_comparison("run_base", "run_cand", model="m8b", base_dir=str(runs_dir))

    sig = report["significance"]
    assert sig["verdict"] == "IMPROVEMENT"
    assert sig["ci_excludes_zero"] is True
    assert sig["delta"] > 0.0
    assert report["paired_success_count"] == 25
    assert report["failures_baseline"] == 0
    assert report["failures_candidate"] == 0


def test_compare_regression_detected(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    # Run A: 90% accuracy
    scores_a = [1.0] * 20 + [0.0] * 2
    # Run B: 10% accuracy
    scores_b = [0.0] * 20 + [1.0] * 2

    _create_mock_run(runs_dir / "run_good", "m8b", scores_a)
    _create_mock_run(runs_dir / "run_bad", "m8b", scores_b)

    report = run_comparison("run_good", "run_bad", model="m8b", base_dir=str(runs_dir))

    sig = report["significance"]
    assert sig["verdict"] == "REGRESSION"
    assert sig["ci_excludes_zero"] is True
    assert sig["delta"] < 0.0


def test_compare_multi_model_requires_scoping(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_multi"
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        # Contains model_x and model_y
        for m in ("model_x", "model_y"):
            for idx in range(5):
                f.write(
                    json.dumps(
                        {
                            "run_id": "run_multi",
                            "task_id": "task_1",
                            "query_id": f"q_{idx}",
                            "model_id": m,
                            "expected": "A",
                            "score": 1.0,
                        }
                    )
                    + "\n"
                )

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": "run_multi",
                "model_ids": ["model_x", "model_y"],
                "task_ids": ["task_1"],
                "checksums": {
                    "dataset_task_1": "hash_task1",
                    "results_jsonl": sha256_file(jsonl_path),
                },
            },
            f,
            indent=2,
        )

    # Comparing without --model must raise CompareError
    with pytest.raises(CompareError) as exc_info:
        run_comparison("run_multi", "run_multi", model=None, base_dir=str(runs_dir))

    assert "Cannot compare without --model" in str(exc_info.value)

    # Scoped with --model succeeds
    report = run_comparison("run_multi", "run_multi", model="model_x", base_dir=str(runs_dir))
    assert report["model_id"] == "model_x"
    assert report["significance"]["verdict"] == "NO_CHANGE"
