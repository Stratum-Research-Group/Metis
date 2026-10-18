"""Unit and regression tests for Dataset BOM and Experiment Lineage Tree (Phase 3)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from dokime.bom import DatasetBOM, estimate_tokens, extract_schema_signature, generate_dataset_bom
from dokime.config import RunConfig
from dokime.lineage import (
    build_experiment_tree,
    compute_parameter_diff,
    evaluate_lineage_regressions,
    get_lineage_path,
)
from dokime.manifest import create_manifest, get_git_dirty
from dokime.runner import run_evaluation
from dokime.tasks.protocol import Query


def test_token_estimation_and_schema_signature():
    text = "Hello world this is a test prompt with several tokens."
    tokens = estimate_tokens(text)
    assert tokens >= 9

    queries = [
        Query(id="q1", prompt="What is 2+2?", choices={"A": "4", "B": "3"}, expected="A", metadata={"subject": "math"}),
        Query(id="q2", prompt="What is the capital of France?", expected="Paris", metadata={"difficulty": "easy"}),
    ]
    sig = extract_schema_signature(queries)
    assert "id" in sig
    assert "prompt" in sig
    assert "expected" in sig
    assert "metadata.subject" in sig or "metadata" in sig


def test_dataset_bom_generation(tmp_path: Path):
    d_file = tmp_path / "test_dataset.json"
    queries = [
        Query(id="1", prompt="Explain gravity", expected="Attraction between masses"),
        Query(id="2", prompt="Explain entropy", expected="Measure of disorder"),
    ]
    d_file.write_text(json.dumps([{"id": q.id, "prompt": q.prompt, "expected": q.expected} for q in queries]))

    bom = generate_dataset_bom("physics_task", queries, d_file)
    assert bom.task_id == "physics_task"
    assert bom.record_count == 2
    assert bom.token_count_estimate > 0
    assert len(bom.sha256) == 64
    assert bom.lineage["format"] == "json"

    # Test roundtrip serialization
    bom_dict = bom.to_dict()
    restored = DatasetBOM.from_dict(bom_dict)
    assert restored.task_id == bom.task_id
    assert restored.sha256 == bom.sha256
    assert restored.record_count == bom.record_count


def test_get_git_dirty():
    # If run in git repo, returns a bool
    dirty = get_git_dirty()
    assert dirty is None or isinstance(dirty, bool)


def test_manifest_records_lineage_and_bom(tmp_path: Path):
    res_file = tmp_path / "results.jsonl"
    res_file.write_text('{"run_id": "test_r"}\n')
    out_manifest = tmp_path / "manifest.json"

    boms = {
        "task_1": {
            "task_id": "task_1",
            "sha256": "abc123456789",
            "record_count": 10,
        }
    }

    manifest = create_manifest(
        run_id="run_child_123",
        model_ids=["meta-llama-3"],
        task_ids=["task_1"],
        seed=100,
        temperature=0.7,
        max_tokens=512,
        results_jsonl_path=res_file,
        dataset_paths={},
        parent_run_id="run_parent_001",
        branch_point="prompt_tuning_v2",
        parameter_diff={"param_temperature": {"parent": 0.0, "child": 0.7}},
        dataset_boms=boms,
        output_path=out_manifest,
    )

    assert manifest["parent_run_id"] == "run_parent_001"
    assert manifest["branch_point"] == "prompt_tuning_v2"
    assert manifest["parameter_diff"]["param_temperature"]["child"] == 0.7
    assert manifest["dataset_boms"]["task_1"]["record_count"] == 10
    assert "git_dirty" in manifest
    assert out_manifest.is_file()


def test_compute_parameter_diff():
    parent_m = {
        "run_id": "parent",
        "model_ids": ["model_a"],
        "task_ids": ["task_1", "task_2"],
        "seed": 42,
        "temperature": 0.0,
        "max_tokens": 100,
        "prompt_template_hash": "hash_v1",
        "checksums": {"dataset_task_1": "hash_d1"},
        "git_sha": "commit_1",
    }
    child_m = {
        "run_id": "child",
        "model_ids": ["model_a", "model_b"],
        "task_ids": ["task_1"],
        "seed": 42,
        "temperature": 0.7,
        "max_tokens": 100,
        "prompt_template_hash": "hash_v2",
        "checksums": {"dataset_task_1": "hash_d1_modified"},
        "git_sha": "commit_2",
    }

    diff = compute_parameter_diff(parent_m, child_m)
    assert diff["param_temperature"] == {"parent": 0.0, "child": 0.7}
    assert diff["models"] == {"added": ["model_b"], "removed": []}
    assert diff["tasks"] == {"added": [], "removed": ["task_2"]}
    assert diff["prompt_template_hash"] == {"parent": "hash_v1", "child": "hash_v2"}
    assert diff["dataset_checksums"]["dataset_task_1"] == {"parent": "hash_d1", "child": "hash_d1_modified"}
    assert diff["git_sha"] == {"parent": "commit_1", "child": "commit_2"}


def test_build_experiment_tree_and_lineage_path(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Create root run
    root_dir = runs_dir / "run_root"
    root_dir.mkdir()
    (root_dir / "manifest.json").write_text(json.dumps({
        "run_id": "run_root",
        "model_ids": ["m1"],
        "task_ids": ["t1"],
        "timestamp_utc": "2026-10-01T00:00:00Z",
    }))

    # Create child run
    child_dir = runs_dir / "run_child"
    child_dir.mkdir()
    (child_dir / "manifest.json").write_text(json.dumps({
        "run_id": "run_child",
        "parent_run_id": "run_root",
        "branch_point": "branch_alpha",
        "model_ids": ["m1"],
        "task_ids": ["t1"],
        "timestamp_utc": "2026-10-02T00:00:00Z",
    }))

    # Create grandchild run
    grandchild_dir = runs_dir / "run_grandchild"
    grandchild_dir.mkdir()
    (grandchild_dir / "manifest.json").write_text(json.dumps({
        "run_id": "run_grandchild",
        "parent_run_id": "run_child",
        "branch_point": "branch_beta",
        "model_ids": ["m1"],
        "task_ids": ["t1"],
        "timestamp_utc": "2026-10-03T00:00:00Z",
    }))

    tree = build_experiment_tree(runs_dir)
    assert len(tree) == 3
    assert tree["run_root"].children == ["run_child"]
    assert tree["run_child"].children == ["run_grandchild"]
    assert tree["run_grandchild"].children == []

    path = get_lineage_path("run_grandchild", runs_dir)
    assert path == ["run_root", "run_child", "run_grandchild"]


def test_evaluate_lineage_regressions(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Root run with high scores (all 1.0)
    root_dir = runs_dir / "run_root"
    root_dir.mkdir()
    (root_dir / "manifest.json").write_text(json.dumps({
        "run_id": "run_root",
        "model_ids": ["m1"],
        "task_ids": ["t1"],
        "checksums": {"results_jsonl": "abc", "dataset_t1": "hash_t1"},
    }))
    (root_dir / "results.jsonl").write_text(
        '\n'.join([
            json.dumps({"model_id": "m1", "task_id": "t1", "query_id": f"q{i}", "score": 1.0})
            for i in range(25)
        ]) + '\n'
    )

    # Child run with severe regression (all 0.0)
    child_dir = runs_dir / "run_regressed"
    child_dir.mkdir()
    (child_dir / "results.jsonl").write_text(
        '\n'.join([
            json.dumps({"model_id": "m1", "task_id": "t1", "query_id": f"q{i}", "score": 0.0})
            for i in range(25)
        ]) + '\n'
    )
    (child_dir / "manifest.json").write_text(json.dumps({
        "run_id": "run_regressed",
        "parent_run_id": "run_root",
        "model_ids": ["m1"],
        "task_ids": ["t1"],
        "checksums": {"results_jsonl": "def", "dataset_t1": "hash_t1"},
    }))

    with patch("dokime.compare.sha256_file", side_effect=lambda p: "abc" if "root" in str(p) else "def"):
        report = evaluate_lineage_regressions("run_regressed", runs_dir=runs_dir)
        assert report["status"] == "REGRESSION"
        assert report["regressions_detected"] == 1
        assert report["regressions"][0]["model_id"] == "m1"
        assert report["regressions"][0]["drop_mean"] < 0


def test_runner_integration_with_bom_and_lineage(tmp_path: Path):
    output_dir = tmp_path / "runs"
    cfg = RunConfig(
        models=["mock_model"],
        tasks=["arc_easy_25"],
        output_dir=str(output_dir),
        parent_run_id="run_baseline_test",
        branch_point="experiment_tuning_1",
    )

    mock_client = MagicMock()
    mock_client.chat_completion.return_value = ("A", 15.0)
    mock_client.get_models_metadata.return_value = [{"name": "mock_model"}]

    res = run_evaluation(config=cfg, client=mock_client)
    manifest = res["manifest"]

    assert manifest["parent_run_id"] == "run_baseline_test"
    assert manifest["branch_point"] == "experiment_tuning_1"
    assert "dataset_boms" in manifest
    assert "arc_easy_25" in manifest["dataset_boms"]
    bom = manifest["dataset_boms"]["arc_easy_25"]
    assert bom["task_id"] == "arc_easy_25"
    assert bom["record_count"] == 25
    assert bom["token_count_estimate"] > 0
    assert len(bom["sha256"]) == 64
