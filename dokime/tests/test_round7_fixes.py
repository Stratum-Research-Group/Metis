"""Tests for Round 7 fixes:
1. Stable, collision-free compare report filenames scoped by model and task.
2. Deterministic mock answer selection across fresh processes and PYTHONHASHSEED settings.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from dokime.compare import CompareError, make_compare_report_filename, run_comparison
from dokime.manifest import sha256_file

ROOT_DIR = Path(__file__).resolve().parent.parent
MOCK_SERVER_PATH = ROOT_DIR / "examples" / "mock_server.py"

spec = importlib.util.spec_from_file_location("mock_server", MOCK_SERVER_PATH)
mock_server_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mock_server_mod)
generate_mock_answer = mock_server_mod.generate_mock_answer


def _create_multi_model_task_run(
    run_dir: Path,
    models_scores: dict[str, dict[str, list[float]]],
    dataset_checksums: dict[str, str] | None = None,
):
    """Helper to create a run with multiple models and tasks."""
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "results.jsonl"
    task_ids = set()
    model_ids = sorted(models_scores.keys())

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for m_id, tasks_map in models_scores.items():
            for t_id, scores in tasks_map.items():
                task_ids.add(t_id)
                for idx, score in enumerate(scores):
                    f.write(
                        json.dumps(
                            {
                                "run_id": run_dir.name,
                                "task_id": t_id,
                                "query_id": f"q_{idx:03d}",
                                "model_id": m_id,
                                "expected": "A",
                                "score": score,
                                "prompt_hash": "a" * 64,
                            }
                        )
                        + "\n"
                    )

    sorted_tasks = sorted(task_ids)
    checksums = {"results_jsonl": sha256_file(jsonl_path)}
    for t_id in sorted_tasks:
        h = dataset_checksums.get(t_id, f"hash_{t_id}") if dataset_checksums else f"hash_{t_id}"
        checksums[f"dataset_{t_id}"] = h

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_dir.name,
                "model_ids": model_ids,
                "task_ids": sorted_tasks,
                "checksums": checksums,
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Item 1 Tests: Scoped Compare Report Identity & Overwrite Prevention
# ---------------------------------------------------------------------------


def test_compare_reports_for_m1_and_m2_do_not_overwrite(tmp_path: Path):
    """Comparing m1 and m2 for the same run pair must write distinct reports and preserve both."""
    runs_dir = tmp_path / "runs"
    run_base = runs_dir / "run_base"
    run_cand = runs_dir / "run_cand"

    # In baseline: m1 has 0.0, m2 has 1.0
    # In candidate: m1 has 1.0 (improvement), m2 has 0.0 (regression)
    _create_multi_model_task_run(
        run_base,
        {
            "m1": {"task_1": [0.0] * 10},
            "m2": {"task_1": [1.0] * 10},
        },
    )
    _create_multi_model_task_run(
        run_cand,
        {
            "m1": {"task_1": [1.0] * 10},
            "m2": {"task_1": [0.0] * 10},
        },
    )

    # 1. Compare m1
    rep_m1 = run_comparison("run_base", "run_cand", model="m1", base_dir=str(runs_dir))
    path_m1 = Path(rep_m1["report_path"])
    assert path_m1.is_file()
    assert rep_m1["significance"]["verdict"] == "IMPROVEMENT"

    # 2. Compare m2
    rep_m2 = run_comparison("run_base", "run_cand", model="m2", base_dir=str(runs_dir))
    path_m2 = Path(rep_m2["report_path"])
    assert path_m2.is_file()
    assert rep_m2["significance"]["verdict"] == "REGRESSION"

    # Distinct file paths
    assert path_m1 != path_m2

    # Both report files remain present on disk and retain their own values
    data_m1 = json.loads(path_m1.read_text(encoding="utf-8"))
    data_m2 = json.loads(path_m2.read_text(encoding="utf-8"))

    assert data_m1["model_id"] == "m1"
    assert data_m1["significance"]["verdict"] == "IMPROVEMENT"
    assert data_m1["run_baseline"] == "run_base"
    assert data_m1["run_candidate"] == "run_cand"

    assert data_m2["model_id"] == "m2"
    assert data_m2["significance"]["verdict"] == "REGRESSION"
    assert data_m2["run_baseline"] == "run_base"
    assert data_m2["run_candidate"] == "run_cand"


def test_compare_reports_for_two_task_scopes_do_not_overwrite(tmp_path: Path):
    """Comparing two task scopes for the same model must write distinct reports and preserve both."""
    runs_dir = tmp_path / "runs"
    run_base = runs_dir / "run_base"
    run_cand = runs_dir / "run_cand"

    # m1 on task_alpha (improvement) and task_beta (regression)
    _create_multi_model_task_run(
        run_base,
        {
            "m1": {
                "task_alpha": [0.0] * 10,
                "task_beta": [1.0] * 10,
            }
        },
    )
    _create_multi_model_task_run(
        run_cand,
        {
            "m1": {
                "task_alpha": [1.0] * 10,
                "task_beta": [0.0] * 10,
            }
        },
    )

    rep_t1 = run_comparison("run_base", "run_cand", model="m1", task="task_alpha", base_dir=str(runs_dir))
    path_t1 = Path(rep_t1["report_path"])

    rep_t2 = run_comparison("run_base", "run_cand", model="m1", task="task_beta", base_dir=str(runs_dir))
    path_t2 = Path(rep_t2["report_path"])

    assert path_t1 != path_t2
    assert path_t1.is_file()
    assert path_t2.is_file()

    data_t1 = json.loads(path_t1.read_text(encoding="utf-8"))
    data_t2 = json.loads(path_t2.read_text(encoding="utf-8"))

    assert data_t1["task_filter"] == "task_alpha"
    assert data_t1["significance"]["verdict"] == "IMPROVEMENT"

    assert data_t2["task_filter"] == "task_beta"
    assert data_t2["significance"]["verdict"] == "REGRESSION"


def test_compare_report_same_scope_reuses_path(tmp_path: Path):
    """Repeating the same comparison scope reuses its own report path."""
    runs_dir = tmp_path / "runs"
    run_base = runs_dir / "run_base"
    run_cand = runs_dir / "run_cand"

    _create_multi_model_task_run(run_base, {"m1": {"t1": [1.0] * 5}})
    _create_multi_model_task_run(run_cand, {"m1": {"t1": [1.0] * 5}})

    rep_1 = run_comparison("run_base", "run_cand", model="m1", task="t1", base_dir=str(runs_dir))
    rep_2 = run_comparison("run_base", "run_cand", model="m1", task="t1", base_dir=str(runs_dir))

    assert rep_1["report_path"] == rep_2["report_path"]


def test_make_compare_report_filename_safety_and_collision_freedom():
    """make_compare_report_filename must handle arbitrary identifiers without collisions."""
    # 1. Characters requiring sanitization
    fn1 = make_compare_report_filename("base/1", "org/model-1:v2 (alpha)", "task/sub:01")
    # Must be filesystem safe (no /, :, parens, spaces)
    for forbidden in ("/", "\\", ":", " ", "(", ")"):
        assert forbidden not in fn1

    # 2. Collision freedom: two distinct models that sanitize to the same naive slug
    model_a = "org/model-1"
    model_b = "org_model_1"
    fn_a = make_compare_report_filename("base", model_a)
    fn_b = make_compare_report_filename("base", model_b)
    assert fn_a != fn_b

    # 3. Task distinction
    fn_no_task = make_compare_report_filename("base", "m1", None)
    fn_task = make_compare_report_filename("base", "m1", "t1")
    assert fn_no_task != fn_task


# ---------------------------------------------------------------------------
# Item 2 Tests: Deterministic Mock Answer Selection Across PYTHONHASHSEED
# ---------------------------------------------------------------------------


def test_mock_answer_deterministic_across_pythonhashseed():
    """generate_mock_answer must return identical results across different PYTHONHASHSEED subprocesses."""
    test_prompts = [
        "What is the boiling point of water?\nA) 50C\nB) 100C\nC) 150C\nD) 200C",
        "Please eliminate incorrect options and choose: What causes ocean tides?",
        "Explain how guitar strings produce sound.",
        "What is the mass of a rock?",
        "Sample prompt without special keywords 12345.",
    ]

    code = """
import sys
import json
from examples.mock_server import generate_mock_answer

prompts = json.loads(sys.argv[1])
results = [generate_mock_answer(p) for p in prompts]
print(json.dumps(results))
"""

    seed_values = ["0", "42", "12345", "999999", "random"]
    outputs = []

    for seed in seed_values:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = str(ROOT_DIR)
        res = subprocess.run(
            [sys.executable, "-c", code, json.dumps(test_prompts)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        answers = json.loads(res.stdout.strip())
        outputs.append(answers)

    # All subprocess outputs must be identical
    first = outputs[0]
    for other in outputs[1:]:
        assert other == first

    # Directly verify in-process generate_mock_answer matches
    in_process = [generate_mock_answer(p) for p in test_prompts]
    assert in_process == first

    # Verify special-case answers remain intact
    assert generate_mock_answer("guitar strings") == "B"
    assert generate_mock_answer("water vapor") == "C"
    assert generate_mock_answer("pumping blood") == "D"
    assert generate_mock_answer("conductor") == "A"
    assert generate_mock_answer("mass of a rock") == "B"
    assert generate_mock_answer("eliminate incorrect options and mass of a rock") == "A"
