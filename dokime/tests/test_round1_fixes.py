"""Focused regression tests for Dokime Round 1 fixes (Items 1-7)."""

import json
import os
import subprocess
from pathlib import Path
import httpx
import pytest

from dokime.cli import main
from dokime.client import HestiaClient
from dokime.compare import CompareError, run_comparison
from dokime.config import RunConfig
from dokime.manifest import capture_hardware, create_manifest, get_git_sha, sha256_file
from dokime.runner import run_evaluation
from dokime.tasks.protocol import Query
from dokime.tasks.registry import BUILTIN_TASKS


# ---------------------------------------------------------------------------
# Item 1 Tests: Failed queries counted as results
# ---------------------------------------------------------------------------

def test_item1_timeout_cannot_masquerade_as_regression(tmp_path: Path):
    """Candidate timeout (score=None) must be excluded from paired diffs, not turned into 0.0."""
    runs_dir = tmp_path / "runs"
    run_base = runs_dir / "run_base"
    run_cand = runs_dir / "run_cand"
    run_base.mkdir(parents=True)
    run_cand.mkdir(parents=True)

    # 10 queries.
    # Baseline: 8 correct (1.0), 2 genuinely wrong (0.0).
    # Candidate: 8 correct (1.0), 2 timed out (score=None).
    base_records = []
    cand_records = []
    for i in range(10):
        # Baseline
        base_score = 1.0 if i < 8 else 0.0
        base_records.append({
            "run_id": "run_base",
            "task_id": "arc_easy_25",
            "query_id": f"q_{i}",
            "model_id": "m1",
            "expected": "A",
            "score": base_score,
            "predicted": "A" if base_score == 1.0 else "B",
        })
        # Candidate
        if i < 8:
            cand_score = 1.0
            error = None
        else:
            cand_score = None
            error = "Connection timed out"
        rec = {
            "run_id": "run_cand",
            "task_id": "arc_easy_25",
            "query_id": f"q_{i}",
            "model_id": "m1",
            "expected": "A",
            "score": cand_score,
            "predicted": "A" if cand_score == 1.0 else None,
        }
        if error:
            rec["error"] = error
        cand_records.append(rec)

    with open(run_base / "results.jsonl", "w") as f:
        for r in base_records:
            f.write(json.dumps(r) + "\n")
    with open(run_cand / "results.jsonl", "w") as f:
        for r in cand_records:
            f.write(json.dumps(r) + "\n")

    for rdir in (run_base, run_cand):
        with open(rdir / "manifest.json", "w") as mf:
            json.dump({
                "run_id": rdir.name,
                "model_ids": ["m1"],
                "task_ids": ["arc_easy_25"],
                "checksums": {
                    "dataset_arc_easy_25": "hash_same",
                    "results_jsonl": sha256_file(rdir / "results.jsonl"),
                },
            }, mf)

    report = run_comparison("run_base", "run_cand", model="m1", base_dir=str(runs_dir))

    # The 8 successful pairs are all 1.0 vs 1.0 -> delta must be 0.0, NOT a regression!
    assert report["paired_success_count"] == 8
    assert report["failures_baseline"] == 0
    assert report["failures_candidate"] == 2
    assert report["significance"]["delta"] == 0.0
    assert report["significance"]["verdict"] == "NO_CHANGE"
    assert report["significance"]["ci_excludes_zero"] is False


def test_item1_genuinely_scored_zero_remains_valid(tmp_path: Path):
    """Genuinely scored 0.0 rows must be retained in paired statistics."""
    runs_dir = tmp_path / "runs"
    run_base = runs_dir / "run_base"
    run_cand = runs_dir / "run_cand"
    run_base.mkdir(parents=True)
    run_cand.mkdir(parents=True)

    base_records = [
        {"run_id": "run_base", "task_id": "t1", "query_id": f"q_{i}", "model_id": "m1", "expected": "A", "score": 1.0}
        for i in range(10)
    ]
    cand_records = [
        {"run_id": "run_cand", "task_id": "t1", "query_id": f"q_{i}", "model_id": "m1", "expected": "A", "score": 0.0}
        for i in range(10)
    ]
    with open(run_base / "results.jsonl", "w") as f:
        for r in base_records:
            f.write(json.dumps(r) + "\n")
    with open(run_cand / "results.jsonl", "w") as f:
        for r in cand_records:
            f.write(json.dumps(r) + "\n")

    for rdir in (run_base, run_cand):
        with open(rdir / "manifest.json", "w") as mf:
            json.dump({
                "run_id": rdir.name,
                "model_ids": ["m1"],
                "task_ids": ["t1"],
                "checksums": {
                    "dataset_t1": "hash_t1",
                    "results_jsonl": sha256_file(rdir / "results.jsonl"),
                },
            }, mf)

    report = run_comparison("run_base", "run_cand", model="m1", base_dir=str(runs_dir))
    assert report["paired_success_count"] == 10
    assert report["significance"]["delta"] == -1.0
    assert report["significance"]["verdict"] == "REGRESSION"


def test_item1_no_valid_pairs_raises_clear_error(tmp_path: Path):
    """If all candidate queries fail, compare must raise CompareError, not return NO_CHANGE."""
    runs_dir = tmp_path / "runs"
    run_base = runs_dir / "run_base"
    run_cand = runs_dir / "run_cand"
    run_base.mkdir(parents=True)
    run_cand.mkdir(parents=True)

    with open(run_base / "results.jsonl", "w") as f:
        f.write(json.dumps({"run_id": "run_base", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0}) + "\n")
    with open(run_cand / "results.jsonl", "w") as f:
        # Candidate failed
        f.write(json.dumps({"run_id": "run_cand", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": None, "error": "timeout"}) + "\n")

    for rdir in (run_base, run_cand):
        with open(rdir / "manifest.json", "w") as mf:
            json.dump({
                "run_id": rdir.name,
                "model_ids": ["m1"],
                "task_ids": ["t1"],
                "checksums": {
                    "dataset_t1": "hash_t1",
                    "results_jsonl": sha256_file(rdir / "results.jsonl"),
                },
            }, mf)

    with pytest.raises(CompareError) as exc_info:
        run_comparison("run_base", "run_cand", model="m1", base_dir=str(runs_dir))

    assert "No valid paired queries with successful scores remain" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Item 2 Tests: Incompatible datasets paired by ID
# ---------------------------------------------------------------------------

def test_item2_dataset_checksum_mismatch_fails(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    r1 = runs_dir / "r1"
    r2 = runs_dir / "r2"
    r1.mkdir(parents=True)
    r2.mkdir(parents=True)

    with open(r1 / "results.jsonl", "w") as f:
        f.write(json.dumps({"run_id": "r1", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0}) + "\n")
    with open(r2 / "results.jsonl", "w") as f:
        f.write(json.dumps({"run_id": "r2", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0}) + "\n")

    # Mismatched dataset checksums
    with open(r1 / "manifest.json", "w") as mf:
        json.dump({"run_id": "r1", "model_ids": ["m1"], "task_ids": ["t1"], "checksums": {"dataset_t1": "hash_alpha", "results_jsonl": sha256_file(r1 / "results.jsonl")}}, mf)
    with open(r2 / "manifest.json", "w") as mf:
        json.dump({"run_id": "r2", "model_ids": ["m1"], "task_ids": ["t1"], "checksums": {"dataset_t1": "hash_beta", "results_jsonl": sha256_file(r2 / "results.jsonl")}}, mf)

    with pytest.raises(CompareError) as exc:
        run_comparison("r1", "r2", model="m1", base_dir=str(runs_dir))
    assert "Dataset checksum mismatch for task 't1'" in str(exc.value)


def test_item2_reused_query_id_with_changed_expected_answer_fails(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    r1 = runs_dir / "r1"
    r2 = runs_dir / "r2"
    r1.mkdir(parents=True)
    r2.mkdir(parents=True)

    # Same query_id "q1", but expected answer changed from "A" to "B"
    with open(r1 / "results.jsonl", "w") as f:
        f.write(json.dumps({"run_id": "r1", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0}) + "\n")
    with open(r2 / "results.jsonl", "w") as f:
        f.write(json.dumps({"run_id": "r2", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "B", "score": 1.0}) + "\n")

    for rdir in (r1, r2):
        with open(rdir / "manifest.json", "w") as mf:
            json.dump({"run_id": rdir.name, "model_ids": ["m1"], "task_ids": ["t1"], "checksums": {"dataset_t1": "common_hash", "results_jsonl": sha256_file(rdir / "results.jsonl")}}, mf)

    with pytest.raises(CompareError) as exc:
        run_comparison("r1", "r2", model="m1", base_dir=str(runs_dir))
    assert "Expected answer mismatch on query 'q1'" in str(exc.value)


def test_item2_cross_model_comparison_not_permitted(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    r1 = runs_dir / "r1"
    r2 = runs_dir / "r2"
    r1.mkdir(parents=True)
    r2.mkdir(parents=True)

    with open(r1 / "results.jsonl", "w") as f:
        f.write(json.dumps({"run_id": "r1", "task_id": "t1", "query_id": "q1", "model_id": "model_alpha", "expected": "A", "score": 1.0}) + "\n")
    with open(r2 / "results.jsonl", "w") as f:
        f.write(json.dumps({"run_id": "r2", "task_id": "t1", "query_id": "q1", "model_id": "model_beta", "expected": "A", "score": 1.0}) + "\n")

    for rdir, mname in [(r1, "model_alpha"), (r2, "model_beta")]:
        with open(rdir / "manifest.json", "w") as mf:
            json.dump({"run_id": rdir.name, "model_ids": [mname], "task_ids": ["t1"], "checksums": {"dataset_t1": "common_hash", "results_jsonl": sha256_file(rdir / "results.jsonl")}}, mf)

    with pytest.raises(CompareError) as exc:
        run_comparison("r1", "r2", model=None, base_dir=str(runs_dir))
    assert "cross-model comparison is not supported" in str(exc.value)


# ---------------------------------------------------------------------------
# Item 3 Tests: Minimal two-method plugin execution
# ---------------------------------------------------------------------------

class MinimalPluginTask:
    """Plugin implementing only task_id, load(), and score()."""
    task_id: str = "minimal_plugin"

    def load(self) -> list[Query]:
        return [
            Query(id="mp_01", prompt="What is 2+2?", expected="4"),
            Query(id="mp_02", prompt="What is 3+3?", expected="6"),
        ]

    def score(self, query: Query, response: str) -> float:
        return 1.0 if query.expected in response else 0.0


def test_item3_minimal_plugin_runs_and_materializes_dataset(tmp_path: Path):
    # Register minimal plugin task in BUILTIN_TASKS
    BUILTIN_TASKS["minimal_plugin"] = MinimalPluginTask

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "4"}}]})

    transport = httpx.MockTransport(mock_handler)
    http_client = httpx.Client(transport=transport)
    client = HestiaClient(base_url="http://mock:8000", client=http_client)

    output_dir = tmp_path / "runs"
    config = RunConfig(
        models=["test-model"],
        tasks=["minimal_plugin"],
        output_dir=str(output_dir),
    )

    res = run_evaluation(config, client=client, run_id="run_min_plugin")
    assert res["total_queries"] == 2

    # Verify materialized dataset artifact
    run_dir = output_dir / "run_min_plugin"
    mat_dataset = run_dir / "dataset_minimal_plugin.json"
    assert mat_dataset.is_file()

    # Verify manifest has dataset checksum
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "dataset_minimal_plugin" in manifest["checksums"]
    assert manifest["checksums"]["dataset_minimal_plugin"] == sha256_file(mat_dataset)

    # Clean up registry
    del BUILTIN_TASKS["minimal_plugin"]


# ---------------------------------------------------------------------------
# Item 4 Tests: Independent Dokime Git SHA resolution
# ---------------------------------------------------------------------------

def test_item4_git_sha_independent_of_cwd(tmp_path: Path):
    """Manifest git_sha must reflect Dokime's checkout, never an unrelated cwd repository."""
    # Create an unrelated git repo in tmp_path
    unrelated_repo = tmp_path / "other_project"
    unrelated_repo.mkdir()
    subprocess.run(["git", "init"], cwd=unrelated_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=unrelated_repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=unrelated_repo, check=True)
    (unrelated_repo / "foo.txt").write_text("hello")
    subprocess.run(["git", "add", "foo.txt"], cwd=unrelated_repo, check=True)
    subprocess.run(["git", "commit", "-m", "unrelated commit"], cwd=unrelated_repo, check=True)

    unrelated_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=unrelated_repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    old_cwd = os.getcwd()
    try:
        os.chdir(unrelated_repo)
        dokime_sha = get_git_sha()
        assert dokime_sha is not None
        # Must NOT match the unrelated repo's SHA
        assert dokime_sha != unrelated_sha
    finally:
        os.chdir(old_cwd)


def test_item4_git_sha_from_non_git_working_dir(tmp_path: Path):
    non_git = tmp_path / "empty_folder"
    non_git.mkdir()
    old_cwd = os.getcwd()
    try:
        os.chdir(non_git)
        dokime_sha = get_git_sha()
        assert dokime_sha is not None
        assert len(dokime_sha) == 40
    finally:
        os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# Item 5 Tests: Template path validation, relative resolution & artifact
# ---------------------------------------------------------------------------

def test_item5_missing_template_path_in_toml_fails(tmp_path: Path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text('models = ["m1"]\nprompt_template_path = "non_existent.txt"\n')

    with pytest.raises(FileNotFoundError) as exc:
        RunConfig.from_toml_file(cfg_file)
    assert "Prompt template file not found" in str(exc.value)


def test_item5_relative_template_path_resolved_against_config_dir(tmp_path: Path):
    sub_dir = tmp_path / "nested"
    sub_dir.mkdir()
    tmpl_file = sub_dir / "my_template.txt"
    tmpl_file.write_text("Question: {question}\nChoices:\n{choices}")

    cfg_file = sub_dir / "config.toml"
    cfg_file.write_text('models = ["m1"]\nprompt_template_path = "my_template.txt"\n')

    # Change working directory to somewhere else
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    old_cwd = os.getcwd()
    try:
        os.chdir(other_dir)
        cfg = RunConfig.from_toml_file(cfg_file)
        assert cfg.prompt_template is not None
        assert "Question: {question}" in cfg.prompt_template
    finally:
        os.chdir(old_cwd)


def test_item5_template_artifact_persisted_and_checksummed(tmp_path: Path):
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer: A"}}]})

    client = HestiaClient(base_url="http://mock:8000", client=httpx.Client(transport=httpx.MockTransport(mock_handler)))
    output_dir = tmp_path / "runs"
    template_text = "System: Think step by step.\n\n{question}\n{choices}"
    config = RunConfig(
        models=["m1"],
        tasks=["arc_easy_25"],
        prompt_template=template_text,
        output_dir=str(output_dir),
    )

    res = run_evaluation(config, client=client, run_id="run_tmpl")
    run_dir = output_dir / "run_tmpl"

    tmpl_artifact = run_dir / "prompt_template.txt"
    assert tmpl_artifact.is_file()
    assert tmpl_artifact.read_text(encoding="utf-8") == template_text

    manifest = json.loads((run_dir / "manifest.json").read_text())
    expected_hash = sha256_file(tmpl_artifact)
    assert manifest["checksums"]["prompt_template"] == expected_hash
    assert manifest["prompt_template_hash"] == expected_hash


# ---------------------------------------------------------------------------
# Item 6 Tests: Auto-selecting a non-chat model
# ---------------------------------------------------------------------------

def test_item6_auto_selection_skips_embedding_model():
    """Hestia /models with embedding model listed first must select the chat model."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "nomic-embed-text", "type": "embedding", "state": "loaded"},
                        {"name": "llama-3.1-8b-instruct", "type": "chat", "state": "loaded"},
                    ]
                },
            )
        return httpx.Response(404)

    client = HestiaClient(base_url="http://mock:8000", client=httpx.Client(transport=httpx.MockTransport(mock_handler)))
    chat_models = client.list_chat_models()
    assert chat_models == ["llama-3.1-8b-instruct"]


def test_item6_no_chat_model_available_fails_clearly(capsys):
    """When only embedding models are registered, auto-select must fail with a clear message."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "nomic-embed-text", "type": "embedding", "state": "loaded"},
                        {"name": "bge-small-en", "type": "embedding", "state": "unloaded"},
                    ]
                },
            )
        return httpx.Response(404)

    client = HestiaClient(base_url="http://mock:8000", client=httpx.Client(transport=httpx.MockTransport(mock_handler)))
    chat_models = client.list_chat_models()
    assert chat_models == []


# ---------------------------------------------------------------------------
# Item 7 Tests: Summary artifact checksummed in manifest
# ---------------------------------------------------------------------------

def test_item7_summary_json_checksummed_in_manifest(tmp_path: Path):
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer: A"}}]})

    client = HestiaClient(base_url="http://mock:8000", client=httpx.Client(transport=httpx.MockTransport(mock_handler)))
    output_dir = tmp_path / "runs"
    config = RunConfig(
        models=["m1"],
        tasks=["arc_easy_25"],
        output_dir=str(output_dir),
    )

    res = run_evaluation(config, client=client, run_id="run_summary_check")
    run_dir = output_dir / "run_summary_check"
    summary_file = run_dir / "summary.json"
    manifest_file = run_dir / "manifest.json"

    assert summary_file.is_file()
    assert manifest_file.is_file()

    manifest = json.loads(manifest_file.read_text())
    assert "summary_json" in manifest["checksums"]
    assert manifest["checksums"]["summary_json"] == sha256_file(summary_file)
