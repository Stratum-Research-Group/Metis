"""Tests for Round 4 fixes: rejecting duplicate model/task IDs in config, CLI, runner, and compare."""

import json
from pathlib import Path
import httpx
import pytest

from dokime.cli import main
from dokime.client import HestiaClient
from dokime.compare import CompareError, run_comparison
from dokime.config import RunConfig
from dokime.manifest import sha256_file
from dokime.runner import run_evaluation
from dokime.tasks.protocol import Query
from dokime.tasks.registry import BUILTIN_TASKS


class MockPluginTask:
    """Mock task for multi-task matrix testing."""

    task_id: str = "mock_task_round4"

    def load(self) -> list[Query]:
        return [
            Query(id="m_01", prompt="What is 1+1?", expected="2"),
            Query(id="m_02", prompt="What is 2+2?", expected="4"),
        ]

    def score(self, query: Query, response: str) -> float:
        return 1.0 if query.expected in response else 0.0


# ---------------------------------------------------------------------------
# Test 1 & 2: TOML config duplicate rejection
# ---------------------------------------------------------------------------


def test_duplicate_models_in_toml_raises_value_error(tmp_path: Path):
    """TOML config specifying duplicate models must be rejected with clear error naming field and ID."""
    cfg_file = tmp_path / "dup_models.toml"
    cfg_file.write_text('models = ["model_a", "model_a"]\ntasks = ["arc_easy_25"]\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        RunConfig.from_toml_file(cfg_file)

    msg = str(exc.value)
    assert "model" in msg
    assert "model_a" in msg
    assert "Duplicate" in msg


def test_equivalent_whitespace_models_in_toml_raises_value_error(tmp_path: Path):
    """TOML config specifying repeated equivalent model IDs with whitespace must be rejected."""
    cfg_file = tmp_path / "dup_models_ws.toml"
    cfg_file.write_text('models = ["model_a", " model_a "]\ntasks = ["arc_easy_25"]\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        RunConfig.from_toml_file(cfg_file)

    msg = str(exc.value)
    assert "model" in msg
    assert "model_a" in msg


def test_model_and_models_both_specified_in_toml_raises_if_duplicate(tmp_path: Path):
    """TOML config with both single model and models containing the same ID must be rejected."""
    cfg_file = tmp_path / "both_model.toml"
    cfg_file.write_text('model = "model_a"\nmodels = ["model_a"]\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        RunConfig.from_toml_file(cfg_file)

    assert "model" in str(exc.value)
    assert "model_a" in str(exc.value)


def test_duplicate_tasks_in_toml_raises_value_error(tmp_path: Path):
    """TOML config specifying duplicate tasks must be rejected with clear error naming field and ID."""
    cfg_file = tmp_path / "dup_tasks.toml"
    cfg_file.write_text('models = ["model_a"]\ntasks = ["arc_easy_25", "arc_easy_25"]\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        RunConfig.from_toml_file(cfg_file)

    msg = str(exc.value)
    assert "task" in msg
    assert "arc_easy_25" in msg
    assert "Duplicate" in msg


def test_task_and_tasks_both_specified_in_toml_raises_if_duplicate(tmp_path: Path):
    """TOML config with both single task and tasks containing the same ID must be rejected."""
    cfg_file = tmp_path / "both_task.toml"
    cfg_file.write_text('models = ["model_a"]\ntask = "arc_easy_25"\ntasks = ["arc_easy_25"]\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        RunConfig.from_toml_file(cfg_file)

    assert "task" in str(exc.value)
    assert "arc_easy_25" in str(exc.value)


# ---------------------------------------------------------------------------
# Test 3 & 4: CLI flags duplicate rejection and exit code
# ---------------------------------------------------------------------------


def test_repeated_model_flags_in_cli_exits_nonzero_and_creates_no_files(tmp_path: Path, capsys):
    """Repeated --model flags must exit nonzero, show offending field and ID, and create no run files."""
    runs_dir = tmp_path / "runs"
    ret = main([
        "run",
        "--model", "duplicate_model",
        "--model", "duplicate_model",
        "--output-dir", str(runs_dir),
    ])

    assert ret != 0
    captured = capsys.readouterr()
    assert "model" in captured.err
    assert "duplicate_model" in captured.err
    assert not runs_dir.exists()


def test_repeated_task_flags_in_cli_exits_nonzero_and_creates_no_files(tmp_path: Path, capsys):
    """Repeated --task flags must exit nonzero, show offending field and ID, and create no run files."""
    runs_dir = tmp_path / "runs"
    ret = main([
        "run",
        "--model", "valid_model",
        "--task", "arc_easy_25",
        "--task", "arc_easy_25",
        "--output-dir", str(runs_dir),
    ])

    assert ret != 0
    captured = capsys.readouterr()
    assert "task" in captured.err
    assert "arc_easy_25" in captured.err
    assert not runs_dir.exists()


def test_cli_toml_duplicate_models_exits_nonzero_and_creates_no_files(tmp_path: Path, capsys):
    """CLI run with a TOML file containing duplicate models must exit nonzero before creating run files."""
    cfg_file = tmp_path / "bad_config.toml"
    cfg_file.write_text('models = ["m_bad", "m_bad"]\ntasks = ["arc_easy_25"]\n', encoding="utf-8")
    runs_dir = tmp_path / "runs"

    ret = main([
        "run",
        "--config", str(cfg_file),
        "--output-dir", str(runs_dir),
    ])

    assert ret != 0
    captured = capsys.readouterr()
    assert "model" in captured.err
    assert "m_bad" in captured.err
    assert not runs_dir.exists()


# ---------------------------------------------------------------------------
# Test 5: Direct RunConfig instantiation and merge_cli_args
# ---------------------------------------------------------------------------


def test_direct_run_config_instantiation_rejects_duplicate_models():
    """Direct RunConfig instantiation with duplicate models must raise ValueError immediately."""
    with pytest.raises(ValueError) as exc:
        RunConfig(models=["m1", "m1"])

    assert "model" in str(exc.value)
    assert "m1" in str(exc.value)


def test_direct_run_config_instantiation_rejects_duplicate_tasks():
    """Direct RunConfig instantiation with duplicate tasks must raise ValueError immediately."""
    with pytest.raises(ValueError) as exc:
        RunConfig(tasks=["t1", "t1"])

    assert "task" in str(exc.value)
    assert "t1" in str(exc.value)


def test_merge_cli_args_rejects_duplicate_models_and_tasks():
    """merge_cli_args must reject duplicate models or tasks passed via overrides."""
    base_config = RunConfig(models=["m1"])

    with pytest.raises(ValueError) as exc_m:
        base_config.merge_cli_args(models=["m2", "m2"])
    assert "model" in str(exc_m.value)
    assert "m2" in str(exc_m.value)

    with pytest.raises(ValueError) as exc_t:
        base_config.merge_cli_args(tasks=["arc_easy_25", "arc_easy_25"])
    assert "task" in str(exc_t.value)
    assert "arc_easy_25" in str(exc_t.value)


# ---------------------------------------------------------------------------
# Test 6: Direct runner guard before artifact creation
# ---------------------------------------------------------------------------


def test_runner_direct_call_guards_against_duplicate_models_without_creating_files(tmp_path: Path):
    """Direct run_evaluation call with post-mutation duplicate models must fail before creating run artifacts."""
    config = RunConfig(models=["model_ok"])
    # Bypass __post_init__ via attribute mutation
    config.models = ["m_dup", "m_dup"]
    config.output_dir = str(tmp_path / "runs")

    run_target = tmp_path / "runs" / "guarded_run_models"

    with pytest.raises(ValueError) as exc:
        run_evaluation(config, run_id="guarded_run_models")

    assert "model" in str(exc.value)
    assert "m_dup" in str(exc.value)
    assert not run_target.exists()
    assert not (tmp_path / "runs").exists()


def test_runner_direct_call_guards_against_duplicate_tasks_without_creating_files(tmp_path: Path):
    """Direct run_evaluation call with post-mutation duplicate tasks must fail before creating run artifacts."""
    config = RunConfig(models=["model_ok"])
    # Bypass __post_init__ via attribute mutation
    config.tasks = ["arc_easy_25", "arc_easy_25"]
    config.output_dir = str(tmp_path / "runs")

    run_target = tmp_path / "runs" / "guarded_run_tasks"

    with pytest.raises(ValueError) as exc:
        run_evaluation(config, run_id="guarded_run_tasks")

    assert "task" in str(exc.value)
    assert "arc_easy_25" in str(exc.value)
    assert not run_target.exists()
    assert not (tmp_path / "runs").exists()


# ---------------------------------------------------------------------------
# Test 7: Valid multi-model x multi-task matrix execution preserved
# ---------------------------------------------------------------------------


def test_distinct_ids_execute_valid_matrix(tmp_path: Path):
    """Valid multi-model x multi-task matrix with distinct IDs executes properly and writes correct accounting."""
    BUILTIN_TASKS["mock_task_round4"] = MockPluginTask

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer: A or 2 or 4"}}]})

    transport = httpx.MockTransport(mock_handler)
    http_client = httpx.Client(transport=transport)
    client = HestiaClient(base_url="http://mock:8000", client=http_client)

    output_dir = tmp_path / "runs"
    config = RunConfig(
        models=["model_alpha", "model_beta"],
        tasks=["arc_easy_25", "mock_task_round4"],
        output_dir=str(output_dir),
        seed=42,
    )

    result = run_evaluation(config, client=client, run_id="valid_matrix_run")
    assert result["run_id"] == "valid_matrix_run"

    # 2 models x (25 arc items + 2 mock items) = 54 queries
    assert result["total_queries"] == 54

    run_dir = output_dir / "valid_matrix_run"
    results_jsonl = run_dir / "results.jsonl"
    summary_json = run_dir / "summary.json"
    manifest_json = run_dir / "manifest.json"

    assert results_jsonl.exists()
    assert summary_json.exists()
    assert manifest_json.exists()

    with open(results_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 54

    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 4 distinct matrix cells
    assert set(summary.keys()) == {
        "model_alpha/arc_easy_25",
        "model_alpha/mock_task_round4",
        "model_beta/arc_easy_25",
        "model_beta/mock_task_round4",
    }
    assert summary["model_alpha/arc_easy_25"]["total_queries"] == 25
    assert summary["model_alpha/mock_task_round4"]["total_queries"] == 2
    assert summary["model_beta/arc_easy_25"]["total_queries"] == 25
    assert summary["model_beta/mock_task_round4"]["total_queries"] == 2


# ---------------------------------------------------------------------------
# Test 8: Compare rejects duplicate query records
# ---------------------------------------------------------------------------


def test_compare_rejects_duplicate_query_records_instead_of_collapsing(tmp_path: Path):
    """Compare must raise CompareError when a run contains duplicate query records for the same task."""
    runs_dir = tmp_path / "runs"
    run_a = runs_dir / "run_dup_a"
    run_b = runs_dir / "run_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    # run_a has duplicate query "q1" for the same task and model
    with open(run_a / "results.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "run_dup_a", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0}) + "\n")
        f.write(json.dumps({"run_id": "run_dup_a", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 0.0}) + "\n")

    with open(run_b / "results.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "run_b", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0}) + "\n")

    for rdir in (run_a, run_b):
        with open(rdir / "manifest.json", "w", encoding="utf-8") as mf:
            json.dump({
                "run_id": rdir.name,
                "model_ids": ["m1"],
                "task_ids": ["t1"],
                "checksums": {
                    "dataset_t1": "hash_t1",
                    "results_jsonl": sha256_file(rdir / "results.jsonl"),
                },
            }, mf)

    with pytest.raises(CompareError) as exc:
        run_comparison("run_dup_a", "run_b", model="m1", base_dir=str(runs_dir))

    assert "Duplicate query record detected" in str(exc.value)
