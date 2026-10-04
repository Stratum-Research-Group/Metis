"""Tests for dokime v8: Within-run cross-model compare with safeguards."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from dokime.cli import main
from dokime.client import HestiaClient
from dokime.compare import (
    CompareError,
    make_cross_model_compare_report_filename,
    run_comparison,
    run_cross_model_comparison,
)
from dokime.config import RunConfig
from dokime.manifest import sha256_file
from dokime.runner import run_evaluation


def _create_single_run_matrix(
    run_dir: Path,
    model_tasks_data: dict[str, dict[str, list[dict]]],
    seed: int = 42,
    dataset_checksums: dict[str, str] | None = None,
):
    """Helper to create a single matrix run directory with results.jsonl and manifest.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "results.jsonl"
    task_ids = set()
    model_ids = sorted(model_tasks_data.keys())

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for m_id, tasks_map in model_tasks_data.items():
            for t_id, query_records in tasks_map.items():
                task_ids.add(t_id)
                for q in query_records:
                    rec = {
                        "run_id": run_dir.name,
                        "task_id": t_id,
                        "query_id": q["query_id"],
                        "model_id": m_id,
                        "prompt_hash": q.get("prompt_hash", "default_hash_abc"),
                        "predicted": q.get("predicted", "A"),
                        "expected": q.get("expected", "A"),
                        "score": q.get("score"),
                        "latency_ms": 10.0,
                        "timestamp": "2026-09-26T20:00:00Z",
                    }
                    if "error" in q:
                        rec["error"] = q["error"]
                    if "dataset_checksum" in q:
                        rec["dataset_checksum"] = q["dataset_checksum"]
                    if "seed" in q:
                        rec["seed"] = q["seed"]
                    f.write(json.dumps(rec) + "\n")

    sorted_tasks = sorted(task_ids)
    checksums = {"results_jsonl": sha256_file(jsonl_path)}
    for t_id in sorted_tasks:
        h = dataset_checksums.get(t_id, f"dataset_hash_{t_id}") if dataset_checksums else f"dataset_hash_{t_id}"
        checksums[f"dataset_{t_id}"] = h

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(
            {
                "run_id": run_dir.name,
                "model_ids": model_ids,
                "task_ids": sorted_tasks,
                "seed": seed,
                "temperature": 0.0,
                "max_tokens": 128,
                "checksums": checksums,
            },
            mf,
            indent=2,
        )


def test_make_cross_model_compare_report_filename_uniqueness_and_safety():
    """Report filename must be filesystem-safe, contain both model IDs and task ID, and never collide."""
    f1 = make_cross_model_compare_report_filename("meta/llama-3.1", "qwen/qwen-2.5", "arc_easy_25")
    f2 = make_cross_model_compare_report_filename("meta/llama-3.1", "google/gemma-2", "arc_easy_25")
    f3 = make_cross_model_compare_report_filename("meta/llama-3.1", "qwen/qwen-2.5", "mmlu_stem")

    assert f1 != f2
    assert f1 != f3
    assert f2 != f3
    assert "compare_cross_meta_llama-3_1_vs_qwen_qwen-2_5_arc_easy_25_" in f1
    assert f1.endswith(".json")


def test_cross_model_compare_success(tmp_path: Path):
    """Successful cross-model comparison within a matrix run computes bootstrap and writes report."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_mat_1"

    queries_m1 = [
        {"query_id": f"q_{i:02d}", "score": 0.0, "prompt_hash": f"hash_{i}", "expected": "A"}
        for i in range(10)
    ]
    queries_m2 = [
        {"query_id": f"q_{i:02d}", "score": 1.0, "prompt_hash": f"hash_{i}", "expected": "A"}
        for i in range(10)
    ]

    _create_single_run_matrix(
        run_dir,
        {"model_alpha": {"task_1": queries_m1}, "model_beta": {"task_1": queries_m2}},
    )

    report = run_cross_model_comparison(
        run_ref=str(run_dir),
        model_a="model_alpha",
        model_b="model_beta",
        task="task_1",
        base_dir=str(runs_dir),
        write_report=True,
    )

    assert report["comparison_type"] == "cross_model"
    assert report["run_id"] == "run_mat_1"
    assert report["model_a"] == "model_alpha"
    assert report["model_b"] == "model_beta"
    assert report["task_id"] == "task_1"
    assert report["paired_success_count"] == 10
    assert report["failures_model_a"] == 0
    assert report["failures_model_b"] == 0
    assert report["model_a_stats"]["mean"] == 0.0
    assert report["model_b_stats"]["mean"] == 1.0
    assert report["significance"]["delta"] == 1.0
    assert report["significance"]["ci_lower"] > 0.0
    assert report["significance"]["verdict"] == "IMPROVEMENT"
    assert report["query_diffs"]["improved_count"] == 10

    # Verify report written to disk
    report_file = Path(report["report_path"])
    assert report_file.is_file()
    assert report_file.parent == run_dir
    with open(report_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert disk_data["model_a"] == "model_alpha"
    assert disk_data["model_b"] == "model_beta"


def test_cross_model_reports_do_not_overwrite(tmp_path: Path):
    """Reports for different model pairs and tasks within the same run must not overwrite each other."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_mat_multi"

    def mk_q(score):
        return [{"query_id": f"q_{i}", "score": score, "prompt_hash": f"h_{i}", "expected": "A"} for i in range(5)]

    _create_single_run_matrix(
        run_dir,
        {
            "m1": {"t1": mk_q(0.0), "t2": mk_q(0.2)},
            "m2": {"t1": mk_q(0.5), "t2": mk_q(0.4)},
            "m3": {"t1": mk_q(1.0), "t2": mk_q(0.8)},
        },
    )

    r1 = run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))
    r2 = run_cross_model_comparison(str(run_dir), "m1", "m3", "t1", base_dir=str(runs_dir))
    r3 = run_cross_model_comparison(str(run_dir), "m1", "m2", "t2", base_dir=str(runs_dir))

    p1 = Path(r1["report_path"])
    p2 = Path(r2["report_path"])
    p3 = Path(r3["report_path"])

    assert p1.is_file()
    assert p2.is_file()
    assert p3.is_file()
    assert len({p1, p2, p3}) == 3


def test_cli_cross_model_flow(tmp_path: Path, capsys):
    """CLI dokime compare <run_id> --cross-model m1 m2 --task task_id executes successfully."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_cli_test"

    queries_m1 = [{"query_id": f"q_{i}", "score": 1.0, "prompt_hash": f"ph_{i}", "expected": "A"} for i in range(10)]
    queries_m2 = [{"query_id": f"q_{i}", "score": 0.0, "prompt_hash": f"ph_{i}", "expected": "A"} for i in range(10)]

    _create_single_run_matrix(
        run_dir,
        {"model_x": {"task_a": queries_m1}, "model_y": {"task_a": queries_m2}},
    )

    ret = main([
        "compare",
        str(run_dir),
        "--cross-model", "model_x", "model_y",
        "--task", "task_a",
        "--output-dir", str(runs_dir),
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "== Dokime Cross-Model Compare ==" in captured.out
    assert "Model A (Baseline):model_x" in captured.out
    assert "Model B (Candidate):model_y" in captured.out
    assert "Task:              task_a" in captured.out
    assert "Paired Successes:  10" in captured.out
    assert "[REGRESSION]" in captured.out
    assert "Report written:" in captured.out


def test_cli_cross_model_mutual_exclusions(tmp_path: Path, capsys):
    """--cross-model cannot be combined with second run ID or --model, and requires --task."""
    # 1. Combined with second run ID
    ret = main(["compare", "run1", "run2", "--cross-model", "m1", "m2", "--task", "t1"])
    assert ret == 2
    captured = capsys.readouterr()
    assert "Compare error: --cross-model cannot be combined with a second run ID." in captured.err

    # 2. Combined with --model
    ret = main(["compare", "run1", "--cross-model", "m1", "m2", "--model", "m1", "--task", "t1"])
    assert ret == 2
    captured = capsys.readouterr()
    assert "Compare error: --cross-model cannot be combined with --model." in captured.err

    # 3. Missing --task
    ret = main(["compare", "run1", "--cross-model", "m1", "m2"])
    assert ret == 2
    captured = capsys.readouterr()
    assert "Compare error: --task <task_id> is required for --cross-model compare." in captured.err

    # 4. Same model twice
    ret = main(["compare", "run1", "--cross-model", "m1", "m1", "--task", "t1"])
    assert ret == 2
    captured = capsys.readouterr()
    assert "Compare error: --cross-model requires two distinct model IDs" in captured.err

    # 5. Ordinary compare missing run_b
    ret = main(["compare", "run1"])
    assert ret == 2
    captured = capsys.readouterr()
    assert "Compare error: Two run IDs are required for ordinary compare" in captured.err


def test_cross_model_missing_model_or_task_fails(tmp_path: Path):
    """Cross-model compare refuses when a model or task is absent in the run."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_missing"

    queries = [{"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A"}]
    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries}},
    )

    # Model B missing
    with pytest.raises(CompareError, match="Model 'm2' has no results for task 't1'"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))

    # Task missing from manifest checksums
    with pytest.raises(CompareError, match="Missing required dataset checksum for task 'nonexistent'"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "nonexistent", base_dir=str(runs_dir))

    # Task in manifest but has no records in results.jsonl
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        mf = json.load(f)
    mf["checksums"]["dataset_t2"] = "hash_t2"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(mf, f, indent=2)

    with pytest.raises(CompareError, match="No records found for task 't2'"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t2", base_dir=str(runs_dir))


def test_cross_model_differing_query_id_sets_fails(tmp_path: Path):
    """Cross-model compare refuses if the two models evaluated differing query_id sets."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_diff_queries"

    queries_m1 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A"},
        {"query_id": "q2", "score": 1.0, "prompt_hash": "h2", "expected": "A"},
    ]
    queries_m2 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A"},
        {"query_id": "q3", "score": 1.0, "prompt_hash": "h3", "expected": "A"},
    ]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    with pytest.raises(CompareError, match="Query ID sets differ between model 'm1' and model 'm2'"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))


def test_cross_model_dataset_checksum_mismatch_fails(tmp_path: Path):
    """Cross-model compare refuses if recorded dataset checksum is missing or differs."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_ds_mismatch"

    queries = [{"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A"}]
    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries}, "m2": {"t1": queries}},
    )

    # Corrupt manifest to remove dataset_t1 checksum
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        mf = json.load(f)
    del mf["checksums"]["dataset_t1"]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(mf, f, indent=2)

    with pytest.raises(CompareError, match="Missing required dataset checksum for task 't1'"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))


def test_cross_model_prompt_hash_mismatch_fails(tmp_path: Path):
    """Cross-model compare refuses if models evaluated different prompt hashes on any query."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_prompt_mismatch"

    queries_m1 = [{"query_id": "q1", "score": 1.0, "prompt_hash": "hash_version_1", "expected": "A"}]
    queries_m2 = [{"query_id": "q1", "score": 1.0, "prompt_hash": "hash_version_2", "expected": "A"}]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    with pytest.raises(CompareError, match="Prompt hash mismatch on query 'q1'"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))


def test_cross_model_seed_mismatch_fails(tmp_path: Path):
    """Cross-model compare refuses if run seed is missing from manifest or differs in records."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_seed_mismatch"

    queries_m1 = [{"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A", "seed": 42}]
    queries_m2 = [{"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A", "seed": 99}]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
        seed=42,
    )

    with pytest.raises(CompareError, match="Seed mismatch between model 'm1'"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))


def test_cross_model_expected_answer_mismatch_fails(tmp_path: Path):
    """Cross-model compare refuses if expected answers differ between models on any query."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_expected_mismatch"

    queries_m1 = [{"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A"}]
    queries_m2 = [{"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "B"}]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    with pytest.raises(CompareError, match="Expected answer mismatch on query 'q1'"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))


def test_cross_model_duplicate_query_records_rejected(tmp_path: Path):
    """Cross-model compare rejects runs with duplicate query records for a model."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_duplicates"

    queries_m1 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A"},
        {"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A"},
    ]
    queries_m2 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A"},
    ]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    with pytest.raises(CompareError, match="Duplicate query record detected"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))


def test_cross_model_failures_excluded_and_reported_separately(tmp_path: Path):
    """Failure/null-result rows are excluded from paired statistics and reported separately."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_failures"

    # 10 queries total
    # m1 fails on q2 and q5
    # m2 fails on q5 and q7
    queries_m1 = []
    queries_m2 = []
    for i in range(10):
        qid = f"q_{i}"
        ph = f"hash_{i}"
        if i in (2, 5):
            queries_m1.append({"query_id": qid, "score": None, "error": "timeout", "prompt_hash": ph, "expected": "A"})
        else:
            queries_m1.append({"query_id": qid, "score": 0.0, "prompt_hash": ph, "expected": "A"})

        if i in (5, 7):
            queries_m2.append({"query_id": qid, "score": None, "error": "server error", "prompt_hash": ph, "expected": "A"})
        else:
            queries_m2.append({"query_id": qid, "score": 1.0, "prompt_hash": ph, "expected": "A"})

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    report = run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))

    # Total queries per model: 10
    # Complete pairs where BOTH succeeded: {0, 1, 3, 4, 6, 8, 9} -> 7 pairs
    assert report["paired_success_count"] == 7
    assert report["failures_model_a"] == 2
    assert report["failures_model_b"] == 2
    assert report["model_a_stats"]["count"] == 7
    assert report["model_b_stats"]["count"] == 7
    assert report["significance"]["delta"] == 1.0


def test_cross_model_no_complete_pairs_fails_clearly(tmp_path: Path):
    """If no queries succeeded on both models, compare raises clear CompareError."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_all_failed"

    queries_m1 = [{"query_id": "q1", "score": None, "error": "timeout", "prompt_hash": "h1", "expected": "A"}]
    queries_m2 = [{"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A"}]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    with pytest.raises(CompareError, match="No valid paired queries with successful scores remain"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))


def test_ordinary_compare_remains_same_model_only(tmp_path: Path):
    """Ordinary two-run compare continues to refuse cross-model pairing."""
    runs_dir = tmp_path / "runs"
    r1 = runs_dir / "run_1"
    r2 = runs_dir / "run_2"

    queries = [{"query_id": "q1", "score": 1.0, "prompt_hash": "h1", "expected": "A"}]
    _create_single_run_matrix(r1, {"model_a": {"t1": queries}})
    _create_single_run_matrix(r2, {"model_b": {"t1": queries}})

    with pytest.raises(CompareError, match="cross-model comparison is not supported"):
        run_comparison(str(r1), str(r2), base_dir=str(runs_dir))


def test_end_to_end_runner_matrix_and_cross_model_compare(tmp_path: Path, capsys):
    """End-to-end test: evaluate 2 models x 1 task with dokime run, then compare with CLI --cross-model."""
    import httpx

    # Mock server: returns A for model_1, B for model_2
    def mock_handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        model = data.get("model")
        ans = "A" if model == "model_1" else "B"
        return httpx.Response(200, json={"choices": [{"message": {"content": f"Answer: {ans}"}}]})

    transport = httpx.MockTransport(mock_handler)
    client = HestiaClient(base_url="http://mock:8000", client=httpx.Client(transport=transport))

    runs_dir = tmp_path / "runs"
    config = RunConfig(
        models=["model_1", "model_2"],
        tasks=["arc_easy_25"],
        output_dir=str(runs_dir),
        seed=42,
    )

    run_res = run_evaluation(config, client=client, run_id="run_e2e_matrix")
    run_id = run_res["run_id"]

    ret = main([
        "compare",
        run_id,
        "--cross-model", "model_1", "model_2",
        "--task", "arc_easy_25",
        "--output-dir", str(runs_dir),
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "== Dokime Cross-Model Compare ==" in captured.out
    assert "Run ID:            run_e2e_matrix" in captured.out
    assert "Model A (Baseline):model_1" in captured.out
    assert "Model B (Candidate):model_2" in captured.out
    assert "Task:              arc_easy_25" in captured.out
    assert "Paired Successes:  25" in captured.out
    assert "Model A Failures:  0" in captured.out
    assert "Model B Failures:  0" in captured.out
    assert "Delta (B - A):" in captured.out
    assert "Verdict:" in captured.out
    assert "Report written:" in captured.out

    # Check report on disk
    expected_filename = make_cross_model_compare_report_filename("model_1", "model_2", "arc_easy_25")
    report_path = runs_dir / run_id / expected_filename
    assert report_path.is_file()


def test_cross_model_null_hash_on_failed_row_fails_cli(tmp_path: Path, capsys):
    """Refuse comparison with non-zero exit if a failure row has a null/missing prompt hash, even if valid pairs exist."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_null_hash_fail"

    # q1 is a valid complete pair; q2 has null prompt_hash on model m1's failure row
    queries_m1 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "hash_q1", "expected": "A"},
        {"query_id": "q2", "score": None, "error": "timeout", "prompt_hash": None, "expected": "A"},
    ]
    queries_m2 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "hash_q1", "expected": "A"},
        {"query_id": "q2", "score": 1.0, "prompt_hash": "hash_q2", "expected": "A"},
    ]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    ret = main([
        "compare",
        str(run_dir),
        "--cross-model", "m1", "m2",
        "--task", "t1",
        "--output-dir", str(runs_dir),
    ])
    assert ret == 2
    captured = capsys.readouterr()
    assert "Compare error:" in captured.err
    assert "Prompt hash missing or null on query 'q2'" in captured.err
    assert "model 'm1' has null or empty prompt hash" in captured.err

    # Assert no report written
    reports = list(run_dir.glob("compare_*.json"))
    assert len(reports) == 0


def test_cross_model_mismatched_prompt_hash_on_failed_row_fails(tmp_path: Path):
    """Refuse comparison if non-null prompt hashes mismatch on a failure row."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_mismatch_failed"

    queries_m1 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "hash_q1", "expected": "A"},
        {"query_id": "q2", "score": None, "error": "timeout", "prompt_hash": "hash_alpha", "expected": "A"},
    ]
    queries_m2 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "hash_q1", "expected": "A"},
        {"query_id": "q2", "score": 1.0, "prompt_hash": "hash_beta", "expected": "A"},
    ]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    with pytest.raises(CompareError, match="Prompt hash mismatch on query 'q2'"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))


def test_cross_model_null_hash_on_both_failed_rows_fails(tmp_path: Path):
    """Refuse comparison if both models have null prompt hashes on a failure row."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_both_null_failed"

    queries_m1 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "hash_q1", "expected": "A"},
        {"query_id": "q2", "score": None, "error": "timeout", "prompt_hash": None, "expected": "A"},
    ]
    queries_m2 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "hash_q1", "expected": "A"},
        {"query_id": "q2", "score": None, "error": "server error", "prompt_hash": None, "expected": "A"},
    ]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    with pytest.raises(CompareError, match="Prompt hash missing or null on query 'q2'"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))


def test_small_effect_fixture_no_all_zero_displayed_interval(tmp_path: Path, capsys):
    """Small-effect fixture: ten baseline scores of 0.0 and ten candidate scores of 0.00001.

    Must report delta > 0, CI > 0, verdict IMPROVEMENT, and never display an all-zero interval.
    """
    from dokime.stats import paired_bootstrap_test

    baseline = [0.0] * 10
    candidate = [0.00001] * 10

    # 1. Stats level check
    res = paired_bootstrap_test(baseline, candidate, num_samples=10000, seed=42)
    assert res["delta"] == 0.00001
    assert res["ci_lower"] == 0.00001
    assert res["ci_upper"] == 0.00001
    assert res["ci_excludes_zero"] is True
    assert res["verdict"] == "IMPROVEMENT"
    assert res["delta"] != 0.0
    assert res["ci_lower"] != 0.0
    assert res["ci_upper"] != 0.0

    # 2. End-to-end CLI compare check
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_small_effect"

    queries_m1 = [{"query_id": f"q_{i}", "score": 0.0, "prompt_hash": f"h_{i}", "expected": "A"} for i in range(10)]
    queries_m2 = [{"query_id": f"q_{i}", "score": 0.00001, "prompt_hash": f"h_{i}", "expected": "A"} for i in range(10)]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    ret = main([
        "compare",
        str(run_dir),
        "--cross-model", "m1", "m2",
        "--task", "t1",
        "--output-dir", str(runs_dir),
    ])
    assert ret == 0
    captured = capsys.readouterr()

    # Assert that no all-zero interval is displayed for this significant effect
    assert "[+0.0000, +0.0000]" not in captured.out
    assert "Delta (B - A):     +0.0000 " not in captured.out

    # Assert correct display with exposed precision matching the significant improvement
    assert "[IMPROVEMENT] CI excludes zero (strictly positive)" in captured.out
    assert "Delta (B - A):     +0.00001" in captured.out
    assert "95% CI: [+0.00001, +0.00001]" in captured.out
    assert "Query Breakdown:   10 improved, 0 regressed, 0 unchanged" in captured.out
    assert "0 improved, 10 unchanged" not in captured.out

    # Assert serialized report agrees with CLI
    report_file = run_dir / make_cross_model_compare_report_filename("m1", "m2", "t1")
    assert report_file.is_file()
    with open(report_file, "r", encoding="utf-8") as f:
        disk_report = json.load(f)

    sig = disk_report["significance"]
    assert sig["delta"] == 0.00001
    assert sig["ci_lower"] == 0.00001
    assert sig["ci_upper"] == 0.00001
    assert sig["verdict"] == "IMPROVEMENT"
    assert sig["ci_excludes_zero"] is True

    qd = disk_report["query_diffs"]
    assert qd["improved_count"] == 10
    assert qd["regressed_count"] == 0
    assert qd["unchanged_count"] == 0
    assert len(qd["improved"]) == 10
    assert len(qd["regressed"]) == 0
    assert len(qd["unchanged"]) == 0
    assert all(e["diff"] == 0.00001 for e in qd["improved"])
    assert all(v == 0.00001 for v in qd["diffs_by_query"].values())


def test_small_effect_negative_1e5(tmp_path: Path, capsys):
    """Small-effect fixture: ten baseline scores of 0.00001 and ten candidate scores of 0.0.

    Must report delta < 0, CI < 0, verdict REGRESSION, and breakdown 10 regressed.
    """
    from dokime.stats import paired_bootstrap_test

    baseline = [0.00001] * 10
    candidate = [0.0] * 10

    # 1. Stats level check
    res = paired_bootstrap_test(baseline, candidate, num_samples=10000, seed=42)
    assert res["delta"] == -0.00001
    assert res["ci_lower"] == -0.00001
    assert res["ci_upper"] == -0.00001
    assert res["ci_excludes_zero"] is True
    assert res["verdict"] == "REGRESSION"

    # 2. End-to-end CLI compare check
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_neg_1e5"

    queries_m1 = [{"query_id": f"q_{i}", "score": 0.00001, "prompt_hash": f"h_{i}", "expected": "A"} for i in range(10)]
    queries_m2 = [{"query_id": f"q_{i}", "score": 0.0, "prompt_hash": f"h_{i}", "expected": "A"} for i in range(10)]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    ret = main([
        "compare",
        str(run_dir),
        "--cross-model", "m1", "m2",
        "--task", "t1",
        "--output-dir", str(runs_dir),
    ])
    assert ret == 0
    captured = capsys.readouterr()

    assert "[REGRESSION] CI excludes zero (strictly negative)" in captured.out
    assert "Delta (B - A):     -0.00001" in captured.out
    assert "95% CI: [-0.00001, -0.00001]" in captured.out
    assert "Query Breakdown:   0 improved, 10 regressed, 0 unchanged" in captured.out
    assert "0 regressed, 10 unchanged" not in captured.out

    report_file = run_dir / make_cross_model_compare_report_filename("m1", "m2", "t1")
    assert report_file.is_file()
    with open(report_file, "r", encoding="utf-8") as f:
        disk_report = json.load(f)

    sig = disk_report["significance"]
    assert sig["delta"] == -0.00001
    assert sig["ci_lower"] == -0.00001
    assert sig["ci_upper"] == -0.00001
    assert sig["verdict"] == "REGRESSION"
    assert sig["ci_excludes_zero"] is True

    qd = disk_report["query_diffs"]
    assert qd["improved_count"] == 0
    assert qd["regressed_count"] == 10
    assert qd["unchanged_count"] == 0
    assert len(qd["regressed"]) == 10
    assert all(e["diff"] == -0.00001 for e in qd["regressed"])
    assert all(v == -0.00001 for v in qd["diffs_by_query"].values())


def test_small_effect_below_decimal_cap_positive_1e9(tmp_path: Path, capsys):
    """Effect at +1e-9 below 8-decimal cap: must expose sign and magnitude, never reading as zero."""
    from dokime.stats import paired_bootstrap_test

    baseline = [0.0] * 10
    candidate = [1e-9] * 10

    # 1. Stats level
    res = paired_bootstrap_test(baseline, candidate, num_samples=10000, seed=42)
    assert res["delta"] == 1e-9
    assert res["ci_lower"] == 1e-9
    assert res["ci_upper"] == 1e-9
    assert res["ci_excludes_zero"] is True
    assert res["verdict"] == "IMPROVEMENT"

    # 2. Report-driven compare flow
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_pos_1e9"
    queries_m1 = [{"query_id": f"q_{i}", "score": 0.0, "prompt_hash": f"h_{i}", "expected": "A"} for i in range(10)]
    queries_m2 = [{"query_id": f"q_{i}", "score": 1e-9, "prompt_hash": f"h_{i}", "expected": "A"} for i in range(10)]
    _create_single_run_matrix(run_dir, {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}})

    ret = main([
        "compare",
        str(run_dir),
        "--cross-model", "m1", "m2",
        "--task", "t1",
        "--output-dir", str(runs_dir),
    ])
    assert ret == 0
    captured = capsys.readouterr()

    # Never display all-zero interval or zero delta for significant effect
    assert "[+0.0000, +0.0000]" not in captured.out
    assert "Delta (B - A):     +0.0000 " not in captured.out

    # Assert sign and nonzero magnitude are visible beside IMPROVEMENT verdict
    assert "[IMPROVEMENT] CI excludes zero (strictly positive)" in captured.out
    assert "Delta (B - A):     +1.0000e-09" in captured.out
    assert "95% CI: [+1.0000e-09, +1.0000e-09]" in captured.out
    assert "Query Breakdown:   10 improved, 0 regressed, 0 unchanged" in captured.out

    # Assert serialized disk report agrees
    report_file = run_dir / make_cross_model_compare_report_filename("m1", "m2", "t1")
    assert report_file.is_file()
    with open(report_file, "r", encoding="utf-8") as f:
        disk_report = json.load(f)
    sig = disk_report["significance"]
    assert sig["delta"] == 1e-9
    assert sig["ci_lower"] == 1e-9
    assert sig["ci_upper"] == 1e-9
    assert sig["verdict"] == "IMPROVEMENT"

    qd = disk_report["query_diffs"]
    assert qd["improved_count"] == 10
    assert qd["regressed_count"] == 0
    assert qd["unchanged_count"] == 0
    assert len(qd["improved"]) == 10
    assert all(e["diff"] == 1e-9 for e in qd["improved"])
    assert all(v == 1e-9 for v in qd["diffs_by_query"].values())


def test_small_effect_below_decimal_cap_negative_1e9(tmp_path: Path, capsys):
    """Effect at -1e-9 below 8-decimal cap: must expose negative sign and magnitude, never reading as zero."""
    from dokime.stats import paired_bootstrap_test

    baseline = [1e-9] * 10
    candidate = [0.0] * 10

    # 1. Stats level
    res = paired_bootstrap_test(baseline, candidate, num_samples=10000, seed=42)
    assert res["delta"] == -1e-9
    assert res["ci_lower"] == -1e-9
    assert res["ci_upper"] == -1e-9
    assert res["ci_excludes_zero"] is True
    assert res["verdict"] == "REGRESSION"

    # 2. Report-driven compare flow
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_neg_1e9"
    queries_m1 = [{"query_id": f"q_{i}", "score": 1e-9, "prompt_hash": f"h_{i}", "expected": "A"} for i in range(10)]
    queries_m2 = [{"query_id": f"q_{i}", "score": 0.0, "prompt_hash": f"h_{i}", "expected": "A"} for i in range(10)]
    _create_single_run_matrix(run_dir, {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}})

    ret = main([
        "compare",
        str(run_dir),
        "--cross-model", "m1", "m2",
        "--task", "t1",
        "--output-dir", str(runs_dir),
    ])
    assert ret == 0
    captured = capsys.readouterr()

    # Never display all-zero interval or zero delta for significant effect
    assert "[-0.0000, -0.0000]" not in captured.out
    assert "Delta (B - A):     -0.0000 " not in captured.out

    # Assert sign and nonzero magnitude are visible beside REGRESSION verdict
    assert "[REGRESSION] CI excludes zero (strictly negative)" in captured.out
    assert "Delta (B - A):     -1.0000e-09" in captured.out
    assert "95% CI: [-1.0000e-09, -1.0000e-09]" in captured.out
    assert "Query Breakdown:   0 improved, 10 regressed, 0 unchanged" in captured.out

    # Assert serialized disk report agrees
    report_file = run_dir / make_cross_model_compare_report_filename("m1", "m2", "t1")
    assert report_file.is_file()
    with open(report_file, "r", encoding="utf-8") as f:
        disk_report = json.load(f)
    sig = disk_report["significance"]
    assert sig["delta"] == -1e-9
    assert sig["ci_lower"] == -1e-9
    assert sig["ci_upper"] == -1e-9
    assert sig["verdict"] == "REGRESSION"

    qd = disk_report["query_diffs"]
    assert qd["improved_count"] == 0
    assert qd["regressed_count"] == 10
    assert qd["unchanged_count"] == 0
    assert len(qd["regressed"]) == 10
    assert all(e["diff"] == -1e-9 for e in qd["regressed"])
    assert all(v == -1e-9 for v in qd["diffs_by_query"].values())


def test_true_zero_effect_renders_zero_distinguishably(tmp_path: Path, capsys):
    """True zero effect: distinguishable as +0.0000 and [+0.0000, +0.0000]."""
    from dokime.stats import paired_bootstrap_test

    baseline = [0.0] * 10
    candidate = [0.0] * 10

    # 1. Stats level
    res = paired_bootstrap_test(baseline, candidate, num_samples=10000, seed=42)
    assert res["delta"] == 0.0
    assert res["ci_lower"] == 0.0
    assert res["ci_upper"] == 0.0
    assert res["ci_excludes_zero"] is False
    assert res["verdict"] == "NO_CHANGE"

    # 2. Report-driven compare flow
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_true_zero"
    queries_m1 = [{"query_id": f"q_{i}", "score": 0.0, "prompt_hash": f"h_{i}", "expected": "A"} for i in range(10)]
    queries_m2 = [{"query_id": f"q_{i}", "score": 0.0, "prompt_hash": f"h_{i}", "expected": "A"} for i in range(10)]
    _create_single_run_matrix(run_dir, {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}})

    ret = main([
        "compare",
        str(run_dir),
        "--cross-model", "m1", "m2",
        "--task", "t1",
        "--output-dir", str(runs_dir),
    ])
    assert ret == 0
    captured = capsys.readouterr()

    assert "[NO SIGNIFICANT CHANGE] CI includes zero" in captured.out
    assert "Delta (B - A):     +0.0000  95% CI: [+0.0000, +0.0000]" in captured.out
    assert "Query Breakdown:   0 improved, 0 regressed, 10 unchanged" in captured.out

    # Assert serialized disk report agrees
    report_file = run_dir / make_cross_model_compare_report_filename("m1", "m2", "t1")
    assert report_file.is_file()
    with open(report_file, "r", encoding="utf-8") as f:
        disk_report = json.load(f)
    sig = disk_report["significance"]
    assert sig["delta"] == 0.0
    assert sig["ci_lower"] == 0.0
    assert sig["ci_upper"] == 0.0
    assert sig["verdict"] == "NO_CHANGE"

    qd = disk_report["query_diffs"]
    assert qd["improved_count"] == 0
    assert qd["regressed_count"] == 0
    assert qd["unchanged_count"] == 10
    assert len(qd["unchanged"]) == 10
    assert all(e["diff"] == 0.0 for e in qd["unchanged"])
    assert all(v == 0.0 for v in qd["diffs_by_query"].values())


def test_cross_model_historic_jsonl_nonfinite_scores_excluded_from_stats(tmp_path: Path, capsys):
    """NaN, +Inf, -Inf in historic JSONL must be treated as failed results and never enter bootstrap stats."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_historic_nonfinite"

    # q1 is a complete valid pair (1.0 vs 1.0)
    # q_nan, q_inf, q_neginf have nonfinite scores
    queries_m1 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "h_q1", "expected": "A"},
        {"query_id": "q_nan", "score": 1.0, "prompt_hash": "h_nan", "expected": "A"},
        {"query_id": "q_inf", "score": float("inf"), "prompt_hash": "h_inf", "expected": "A"},
        {"query_id": "q_neginf", "score": float("-inf"), "prompt_hash": "h_neginf", "expected": "A"},
    ]
    queries_m2 = [
        {"query_id": "q1", "score": 1.0, "prompt_hash": "h_q1", "expected": "A"},
        {"query_id": "q_nan", "score": float("nan"), "prompt_hash": "h_nan", "expected": "A"},
        {"query_id": "q_inf", "score": 1.0, "prompt_hash": "h_inf", "expected": "A"},
        {"query_id": "q_neginf", "score": float("-inf"), "prompt_hash": "h_neginf", "expected": "A"},
    ]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    report = run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))

    # Must not yield a success count from the bad scores
    assert report["paired_success_count"] == 1
    assert report["failures_model_a"] == 2  # q_inf, q_neginf
    assert report["failures_model_b"] == 2  # q_nan, q_neginf

    # Stats computed purely from q1 (1.0 vs 1.0)
    assert report["significance"]["delta"] == 0.0
    assert report["significance"]["ci_lower"] == 0.0
    assert report["significance"]["ci_upper"] == 0.0
    assert report["significance"]["verdict"] == "NO_CHANGE"
    assert report["query_diffs"]["total_paired"] == 1

    # Check CLI rendering
    ret = main([
        "compare",
        str(run_dir),
        "--cross-model", "m1", "m2",
        "--task", "t1",
        "--output-dir", str(runs_dir),
    ])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Paired Successes:  1" in captured.out
    assert "Model A Failures:  2" in captured.out
    assert "Model B Failures:  2" in captured.out
    assert "[NO SIGNIFICANT CHANGE] CI includes zero" in captured.out


def test_cross_model_historic_jsonl_all_nonfinite_fails_clearly(tmp_path: Path):
    """When all queries in historic JSONL contain nonfinite scores, comparison fails clearly."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run_all_nonfinite"

    queries_m1 = [{"query_id": "q1", "score": float("nan"), "prompt_hash": "h_q1", "expected": "A"}]
    queries_m2 = [{"query_id": "q1", "score": float("inf"), "prompt_hash": "h_q1", "expected": "A"}]

    _create_single_run_matrix(
        run_dir,
        {"m1": {"t1": queries_m1}, "m2": {"t1": queries_m2}},
    )

    with pytest.raises(CompareError, match="No valid paired queries with successful scores remain"):
        run_cross_model_comparison(str(run_dir), "m1", "m2", "t1", base_dir=str(runs_dir))


def test_ordinary_compare_historic_jsonl_nonfinite_scores_excluded_from_stats(tmp_path: Path):
    """Ordinary compare excludes nonfinite scores from paired statistics and counts them as failures."""
    runs_dir = tmp_path / "runs"
    run_a = runs_dir / "run_a"
    run_b = runs_dir / "run_b"

    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    jsonl_a = run_a / "results.jsonl"
    jsonl_b = run_b / "results.jsonl"

    recs_a = [
        {"run_id": "run_a", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0},
        {"run_id": "run_a", "task_id": "t1", "query_id": "q_nan", "model_id": "m1", "expected": "A", "score": float("nan")},
        {"run_id": "run_a", "task_id": "t1", "query_id": "q_inf", "model_id": "m1", "expected": "A", "score": float("inf")},
    ]
    recs_b = [
        {"run_id": "run_b", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0},
        {"run_id": "run_b", "task_id": "t1", "query_id": "q_nan", "model_id": "m1", "expected": "A", "score": 1.0},
        {"run_id": "run_b", "task_id": "t1", "query_id": "q_inf", "model_id": "m1", "expected": "A", "score": 1.0},
    ]

    with open(jsonl_a, "w", encoding="utf-8") as f:
        for r in recs_a:
            f.write(json.dumps(r) + "\n")

    with open(jsonl_b, "w", encoding="utf-8") as f:
        for r in recs_b:
            f.write(json.dumps(r) + "\n")

    for rdir, jpath, rname in [(run_a, jsonl_a, "run_a"), (run_b, jsonl_b, "run_b")]:
        with open(rdir / "manifest.json", "w", encoding="utf-8") as mf:
            json.dump({
                "run_id": rname,
                "model_ids": ["m1"],
                "task_ids": ["t1"],
                "checksums": {
                    "dataset_t1": "chk_t1",
                    "results_jsonl": sha256_file(jpath),
                },
            }, mf)

    report = run_comparison("run_a", "run_b", model="m1", task="t1", base_dir=str(runs_dir))

    assert report["paired_success_count"] == 1
    assert report["failures_baseline"] == 2
    assert report["failures_candidate"] == 0
    assert report["significance"]["delta"] == 0.0
    assert report["significance"]["verdict"] == "NO_CHANGE"


