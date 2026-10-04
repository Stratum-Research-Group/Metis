"""Focused regression tests for Dokime Round 3 fixes (Compare repair & results verification).

All tests in this module perform mock validation of execution flows and isolated filesystem artifacts;
they do not require or represent live Hestia server runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from dokime.cli import main
from dokime.compare import CompareError, run_comparison
from dokime.manifest import sha256_file


def test_item_a_truncated_candidate_jsonl_causes_checksum_mismatch_not_no_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Truncating candidate results.jsonl without updating its manifest causes a checksum mismatch error, not NO_CHANGE."""
    runs_dir = tmp_path / "runs"
    run_base = runs_dir / "run_base"
    run_cand = runs_dir / "run_cand"
    run_base.mkdir(parents=True)
    run_cand.mkdir(parents=True)

    # Two valid 2-query runs with score=1.0 for q1 and q2
    base_rows = [
        {"run_id": "run_base", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0},
        {"run_id": "run_base", "task_id": "t1", "query_id": "q2", "model_id": "m1", "expected": "B", "score": 1.0},
    ]
    cand_rows = [
        {"run_id": "run_cand", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0},
        {"run_id": "run_cand", "task_id": "t1", "query_id": "q2", "model_id": "m1", "expected": "B", "score": 1.0},
    ]

    with open(run_base / "results.jsonl", "w", encoding="utf-8") as f:
        for r in base_rows:
            f.write(json.dumps(r) + "\n")

    with open(run_cand / "results.jsonl", "w", encoding="utf-8") as f:
        for r in cand_rows:
            f.write(json.dumps(r) + "\n")

    # Write manifests with genuine checksums for the original 2-row files
    with open(run_base / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(
            {
                "run_id": "run_base",
                "model_ids": ["m1"],
                "task_ids": ["t1"],
                "checksums": {
                    "dataset_t1": "hash_dataset",
                    "results_jsonl": sha256_file(run_base / "results.jsonl"),
                },
            },
            mf,
        )

    cand_original_hash = sha256_file(run_cand / "results.jsonl")
    with open(run_cand / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(
            {
                "run_id": "run_cand",
                "model_ids": ["m1"],
                "task_ids": ["t1"],
                "checksums": {
                    "dataset_t1": "hash_dataset",
                    "results_jsonl": cand_original_hash,
                },
            },
            mf,
        )

    # Truncate candidate results.jsonl to only 1 row without updating manifest
    with open(run_cand / "results.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(cand_rows[0]) + "\n")

    # run_comparison must reject it with CompareError naming 'run_cand' and stating checksum mismatch
    with pytest.raises(CompareError) as exc_info:
        run_comparison("run_base", "run_cand", model="m1", base_dir=str(runs_dir))

    err_msg = str(exc_info.value)
    assert "run_cand" in err_msg
    assert "checksum" in err_msg.lower()

    # CLI compare must exit with status 2 and report the error
    ret = main(["compare", "run_base", "run_cand", "--model", "m1", "--output-dir", str(runs_dir)])
    assert ret == 2
    captured = capsys.readouterr()
    assert "Compare error:" in captured.err
    assert "run_cand" in captured.err


def test_item_b_missing_or_malformed_manifest_results_checksum_fails_clearly(tmp_path: Path):
    """Missing or malformed results_jsonl checksum in manifest fails clearly naming the affected run."""
    runs_dir = tmp_path / "runs"
    run_base = runs_dir / "run_base"
    run_cand = runs_dir / "run_cand"
    run_base.mkdir(parents=True)
    run_cand.mkdir(parents=True)

    with open(run_base / "results.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "run_base", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0}) + "\n")
    with open(run_cand / "results.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "run_cand", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0}) + "\n")

    with open(run_base / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(
            {
                "run_id": "run_base",
                "model_ids": ["m1"],
                "task_ids": ["t1"],
                "checksums": {
                    "dataset_t1": "hash_dataset",
                    "results_jsonl": sha256_file(run_base / "results.jsonl"),
                },
            },
            mf,
        )

    # Missing results_jsonl in candidate manifest
    with open(run_cand / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(
            {
                "run_id": "run_cand",
                "model_ids": ["m1"],
                "task_ids": ["t1"],
                "checksums": {
                    "dataset_t1": "hash_dataset",
                },
            },
            mf,
        )

    with pytest.raises(CompareError) as exc_info:
        run_comparison("run_base", "run_cand", model="m1", base_dir=str(runs_dir))

    assert "run_cand" in str(exc_info.value)
    assert "results_jsonl" in str(exc_info.value)

    # Malformed checksums (non-string value) in run_base
    with open(run_base / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(
            {
                "run_id": "run_base",
                "model_ids": ["m1"],
                "task_ids": ["t1"],
                "checksums": {
                    "dataset_t1": "hash_dataset",
                    "results_jsonl": 12345,
                },
            },
            mf,
        )

    with pytest.raises(CompareError) as exc_info:
        run_comparison("run_base", "run_cand", model="m1", base_dir=str(runs_dir))

    assert "run_base" in str(exc_info.value)


def test_item_c_different_query_sets_accurate_unpaired_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Two intact, correctly checksummed runs with different query sets show accurate baseline-only/candidate-only counts."""
    runs_dir = tmp_path / "runs"
    run_base = runs_dir / "run_base"
    run_cand = runs_dir / "run_cand"
    run_base.mkdir(parents=True)
    run_cand.mkdir(parents=True)

    # Baseline has q1 and q2 (both scored 1.0)
    # Candidate has q2 and q3 (both scored 1.0)
    base_rows = [
        {"run_id": "run_base", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0},
        {"run_id": "run_base", "task_id": "t1", "query_id": "q2", "model_id": "m1", "expected": "B", "score": 1.0},
    ]
    cand_rows = [
        {"run_id": "run_cand", "task_id": "t1", "query_id": "q2", "model_id": "m1", "expected": "B", "score": 1.0},
        {"run_id": "run_cand", "task_id": "t1", "query_id": "q3", "model_id": "m1", "expected": "C", "score": 1.0},
    ]

    with open(run_base / "results.jsonl", "w", encoding="utf-8") as f:
        for r in base_rows:
            f.write(json.dumps(r) + "\n")

    with open(run_cand / "results.jsonl", "w", encoding="utf-8") as f:
        for r in cand_rows:
            f.write(json.dumps(r) + "\n")

    with open(run_base / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(
            {
                "run_id": "run_base",
                "model_ids": ["m1"],
                "task_ids": ["t1"],
                "checksums": {
                    "dataset_t1": "hash_dataset_t1",
                    "results_jsonl": sha256_file(run_base / "results.jsonl"),
                },
            },
            mf,
        )

    with open(run_cand / "manifest.json", "w", encoding="utf-8") as mf:
        json.dump(
            {
                "run_id": "run_cand",
                "model_ids": ["m1"],
                "task_ids": ["t1"],
                "checksums": {
                    "dataset_t1": "hash_dataset_t1",
                    "results_jsonl": sha256_file(run_cand / "results.jsonl"),
                },
            },
            mf,
        )

    report = run_comparison("run_base", "run_cand", model="m1", base_dir=str(runs_dir))

    # Shared successful query is only q2
    assert report["paired_success_count"] == 1
    assert report["unpaired_baseline"] == 1
    assert report["unpaired_candidate"] == 1
    assert report["unpaired_baseline_count"] == 1
    assert report["unpaired_candidate_count"] == 1
    assert report["failures_baseline"] == 0
    assert report["failures_candidate"] == 0
    assert report["significance"]["delta"] == 0.0

    # Also test CLI output displays both counts
    ret = main(["compare", "run_base", "run_cand", "--model", "m1", "--output-dir", str(runs_dir)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Paired Successes:  1" in captured.out
    assert "Unpaired Baseline: 1" in captured.out
    assert "Unpaired Candidate: 1" in captured.out


def test_item_d_scored_zero_valid_and_errors_remain_failures_not_unpaired(tmp_path: Path):
    """Scored 0.0 remains valid in paired stats, while error rows remain failures, not unpaired successes."""
    runs_dir = tmp_path / "runs"
    run_base = runs_dir / "run_base"
    run_cand = runs_dir / "run_cand"
    run_base.mkdir(parents=True)
    run_cand.mkdir(parents=True)

    # Query 1: scored 0.0 in both runs (measured zero)
    # Query 2: baseline scored 1.0, candidate had null score (error)
    base_rows = [
        {"run_id": "run_base", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 0.0},
        {"run_id": "run_base", "task_id": "t1", "query_id": "q2", "model_id": "m1", "expected": "B", "score": 1.0},
    ]
    cand_rows = [
        {"run_id": "run_cand", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 0.0},
        {"run_id": "run_cand", "task_id": "t1", "query_id": "q2", "model_id": "m1", "expected": "B", "score": None, "error": "API timeout"},
    ]

    with open(run_base / "results.jsonl", "w", encoding="utf-8") as f:
        for r in base_rows:
            f.write(json.dumps(r) + "\n")

    with open(run_cand / "results.jsonl", "w", encoding="utf-8") as f:
        for r in cand_rows:
            f.write(json.dumps(r) + "\n")

    for rdir in (run_base, run_cand):
        with open(rdir / "manifest.json", "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "run_id": rdir.name,
                    "model_ids": ["m1"],
                    "task_ids": ["t1"],
                    "checksums": {
                        "dataset_t1": "hash_dataset",
                        "results_jsonl": sha256_file(rdir / "results.jsonl"),
                    },
                },
                mf,
            )

    report = run_comparison("run_base", "run_cand", model="m1", base_dir=str(runs_dir))

    # q1 (0.0 vs 0.0) is a valid paired success
    assert report["paired_success_count"] == 1
    # q2 in candidate is an error row, counted as failure, NOT an unpaired success or unpaired query
    assert report["failures_baseline"] == 0
    assert report["failures_candidate"] == 1
    assert report["unpaired_baseline"] == 0
    assert report["unpaired_candidate"] == 0
    assert report["significance"]["delta"] == 0.0
