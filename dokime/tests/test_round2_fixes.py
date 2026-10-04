"""Focused regression tests for Dokime Round 2 fixes (Items 1-3).

All tests in this module perform mock validation of HTTP endpoints and execution flows;
they do not require or represent live Hestia server runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
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


# ---------------------------------------------------------------------------
# Item 1 Tests: Auto-selected model excluding failed models, preferring loaded
# ---------------------------------------------------------------------------


def test_item1_failed_chat_listed_before_loaded_chat_selects_loaded():
    """Hestia /models returns a failed chat before a loaded chat.

    Auto-selection must exclude the failed model and select the loaded chat model.
    """
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "llama-broken", "type": "chat", "state": "failed"},
                        {"name": "llama-ready", "type": "chat", "state": "loaded"},
                    ]
                },
            )
        return httpx.Response(404)

    client = HestiaClient(
        base_url="http://mock:8000",
        client=httpx.Client(transport=httpx.MockTransport(mock_handler)),
    )
    # get_auto_selectable_chat_models must only return usable models with loaded first
    auto_models = client.get_auto_selectable_chat_models()
    assert auto_models == ["llama-ready"]

    # list_chat_models also excludes failed and prefers loaded
    chat_models = client.list_chat_models()
    assert chat_models == ["llama-ready"]


def test_item1_only_failed_chats_exits_with_clear_error(capsys: pytest.CaptureFixture[str]):
    """When all registered chat models are in 'failed' state, auto-selection must fail cleanly.

    It should not attempt running queries into inevitable failures.
    """
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "crashed-chat-1", "type": "chat", "state": "failed"},
                        {"name": "crashed-chat-2", "type": "chat", "state": "failed"},
                    ]
                },
            )
        return httpx.Response(404)

    client = HestiaClient(
        base_url="http://mock:8000",
        client=httpx.Client(transport=httpx.MockTransport(mock_handler)),
    )
    assert client.get_auto_selectable_chat_models() == []

    # Run CLI without --model pointing to this mock server
    import unittest.mock
    with unittest.mock.patch("dokime.cli.HestiaClient", return_value=client):
        ret = main(["run", "--task", "arc_easy_25"])
        assert ret == 1

    captured = capsys.readouterr()
    assert "No usable chat models available on Hestia" in captured.err
    assert "crashed-chat-1, crashed-chat-2" in captured.err
    assert "failed" in captured.err


def test_item1_unloaded_chat_can_be_selected_when_no_loaded_chat_exists():
    """When no chat model is loaded, an unloaded (loadable on demand) chat model must be selected."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "failed-model", "type": "chat", "state": "failed"},
                        {"name": "dormant-model", "type": "chat", "state": "unloaded"},
                    ]
                },
            )
        return httpx.Response(404)

    client = HestiaClient(
        base_url="http://mock:8000",
        client=httpx.Client(transport=httpx.MockTransport(mock_handler)),
    )
    auto_models = client.get_auto_selectable_chat_models()
    assert auto_models == ["dormant-model"]


def test_item1_explicit_model_not_blocked_by_auto_selection():
    """Explicit --model is preserved and does not run through the auto-selection filter."""
    config = RunConfig(models=["failed-model"], tasks=["arc_easy_25"])
    assert config.models == ["failed-model"]


# ---------------------------------------------------------------------------
# Item 2 Tests: Prompt formatter error handling per query
# ---------------------------------------------------------------------------


class MockFaultyFormatterTask:
    """Plugin task whose format_prompt raises an exception for one of two queries."""

    task_id = "mock_faulty_formatter_task"

    def __init__(self) -> None:
        self.queries = [
            Query(id="q1_fail", prompt="Query 1 prompt", expected="A"),
            Query(id="q2_ok", prompt="Query 2 prompt", expected="B"),
        ]

    def load(self) -> list[Query]:
        return self.queries

    def format_prompt(self, query: Query, template: str | None = None) -> str:
        if query.id == "q1_fail":
            raise ValueError("Template rendering syntax error on query q1_fail")
        return f"Prompt for {query.id}: {query.prompt}"

    def extract_prediction(self, query: Query, raw_response: str) -> str:
        return raw_response.strip()

    def score(self, query: Query, raw_response: str) -> float:
        return 1.0 if raw_response.strip() == query.expected else 0.0


def test_item2_formatter_error_produces_null_score_and_continues(tmp_path: Path):
    """If a task prompt formatter raises on one query, it logs a null score with error and prompt_hash: null.

    The rest of the matrix must continue to execute, summary counts must reflect 1 success and 1 failure,
    and manifest checksums must be valid.
    """
    task_id = MockFaultyFormatterTask.task_id
    BUILTIN_TASKS[task_id] = MockFaultyFormatterTask

    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "B"}}]},
            )
        return httpx.Response(404)

    client = HestiaClient(
        base_url="http://mock:8000",
        client=httpx.Client(transport=httpx.MockTransport(mock_handler)),
    )

    output_dir = tmp_path / "runs"
    config = RunConfig(
        models=["test-model"],
        tasks=[task_id],
        output_dir=str(output_dir),
    )

    results = run_evaluation(config=config, client=client, run_id="run_formatter_test")

    run_dir = output_dir / "run_formatter_test"
    results_jsonl = run_dir / "results.jsonl"
    summary_json = run_dir / "summary.json"
    manifest_json = run_dir / "manifest.json"

    # Verify both records written to JSONL
    records = []
    with open(results_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 2, f"Expected 2 records, got {len(records)}"

    # Record 1: Failed during formatting
    rec1 = records[0]
    assert rec1["query_id"] == "q1_fail"
    assert rec1["prompt_hash"] is None
    assert rec1["predicted"] is None
    assert rec1["score"] is None
    assert "error" in rec1
    assert "Template rendering syntax error on query q1_fail" in rec1["error"]

    # Record 2: Succeeded
    rec2 = records[1]
    assert rec2["query_id"] == "q2_ok"
    assert rec2["prompt_hash"] is not None
    assert len(rec2["prompt_hash"]) == 64  # SHA-256
    assert rec2["predicted"] == "B"
    assert rec2["score"] == 1.0
    assert "error" not in rec2

    # Verify summary counts
    cell_info = results["stats"]["test-model/mock_faulty_formatter_task"]
    assert cell_info["total_queries"] == 2
    assert cell_info["successful_queries"] == 1
    assert cell_info["failed_queries"] == 1
    assert cell_info["stats"]["count"] == 1
    assert cell_info["stats"]["mean"] == 1.0

    # Verify summary.json content matches
    with open(summary_json, "r", encoding="utf-8") as sf:
        summary_on_disk = json.load(sf)
    assert summary_on_disk["test-model/mock_faulty_formatter_task"]["failed_queries"] == 1

    # Verify manifest.json checksums
    with open(manifest_json, "r", encoding="utf-8") as mf:
        manifest_on_disk = json.load(mf)

    checksums = manifest_on_disk["checksums"]
    assert checksums["results_jsonl"] == sha256_file(results_jsonl)
    assert checksums["summary_json"] == sha256_file(summary_json)
    assert f"dataset_{task_id}" in checksums

    # Clean up test task registration
    BUILTIN_TASKS.pop(task_id, None)


# ---------------------------------------------------------------------------
# Item 3 Tests: All-failed run reports not-measured; 0.0 scores report measured zero
# ---------------------------------------------------------------------------


def test_item3_all_requests_failing_reports_not_measured(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """When all queries fail (n=0 valid scores), mean and CIs must be null / not-measured, never 0.0."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={"models": [{"name": "flakey-chat", "type": "chat", "state": "loaded"}]},
            )
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(500, text="Internal Server Error: Out of Memory")
        return httpx.Response(404)

    client = HestiaClient(
        base_url="http://mock:8000",
        max_retries=0,
        backoff_base=0.0,
        client=httpx.Client(transport=httpx.MockTransport(mock_handler)),
    )

    output_dir = tmp_path / "runs"
    config = RunConfig(
        models=["flakey-chat"],
        tasks=["arc_easy_25"],
        output_dir=str(output_dir),
    )

    results = run_evaluation(config=config, client=client, run_id="run_all_failed")
    cell_info = results["stats"]["flakey-chat/arc_easy_25"]

    assert cell_info["total_queries"] == 25
    assert cell_info["successful_queries"] == 0
    assert cell_info["failed_queries"] == 25

    stats = cell_info["stats"]
    assert stats["count"] == 0
    assert stats["mean"] is None
    assert stats["ci_lower"] is None
    assert stats["ci_upper"] is None
    assert stats["std"] is None
    assert stats["status"] == "not-measured"

    # Check summary.json on disk
    summary_path = Path(results["summary_path"])
    with open(summary_path, "r", encoding="utf-8") as f:
        disk_summary = json.load(f)

    disk_stats = disk_summary["flakey-chat/arc_easy_25"]["stats"]
    assert disk_stats["mean"] is None
    assert disk_stats["ci_lower"] is None
    assert disk_stats["ci_upper"] is None
    assert disk_stats["status"] == "not-measured"

    # Test CLI rendering via main
    import unittest.mock
    with unittest.mock.patch("dokime.cli.HestiaClient", return_value=client):
        ret = main(["run", "--model", "flakey-chat", "--task", "arc_easy_25", "--output-dir", str(output_dir)])
        assert ret == 0

    captured = capsys.readouterr()
    assert "Mean: not-measured" in captured.out
    assert "95% CI: N/A" in captured.out
    assert "Mean: 0.0000" not in captured.out


def test_item3_successful_requests_scored_zero_reports_measured_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """When queries succeed but score 0.0 (measured zero, n>0), mean and CI must be 0.0, not null/not-measured."""
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={"models": [{"name": "wrong-chat", "type": "chat", "state": "loaded"}]},
            )
        if request.url.path == "/v1/chat/completions":
            # Return incorrect answer for ARC questions (so score is 0.0)
            return httpx.Response(200, json={"choices": [{"message": {"content": "Answer: Z"}}]})
        return httpx.Response(404)

    client = HestiaClient(
        base_url="http://mock:8000",
        client=httpx.Client(transport=httpx.MockTransport(mock_handler)),
    )

    output_dir = tmp_path / "runs"
    config = RunConfig(
        models=["wrong-chat"],
        tasks=["arc_easy_25"],
        output_dir=str(output_dir),
    )

    results = run_evaluation(config=config, client=client, run_id="run_measured_zero")
    cell_info = results["stats"]["wrong-chat/arc_easy_25"]

    assert cell_info["total_queries"] == 25
    assert cell_info["successful_queries"] == 25
    assert cell_info["failed_queries"] == 0

    stats = cell_info["stats"]
    assert stats["count"] == 25
    assert stats["mean"] == 0.0
    assert stats["ci_lower"] == 0.0
    assert stats["ci_upper"] == 0.0
    assert stats["std"] == 0.0

    # Test CLI rendering via main
    import unittest.mock
    with unittest.mock.patch("dokime.cli.HestiaClient", return_value=client):
        ret = main(["run", "--model", "wrong-chat", "--task", "arc_easy_25", "--output-dir", str(output_dir)])
        assert ret == 0

    captured = capsys.readouterr()
    assert "Mean: 0.0000" in captured.out
    assert "95% CI: [0.0000, 0.0000]" in captured.out
    assert "not-measured" not in captured.out


def test_item3_compare_fails_clearly_when_all_requests_failed(tmp_path: Path):
    """When all requests fail in one or both runs, compare must raise CompareError cleanly."""
    runs_dir = tmp_path / "runs"
    run_base = runs_dir / "run_base"
    run_cand = runs_dir / "run_cand"
    run_base.mkdir(parents=True)
    run_cand.mkdir(parents=True)

    with open(run_base / "results.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "run_base", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": 1.0}) + "\n")

    # In candidate, all requests failed (score=None)
    with open(run_cand / "results.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"run_id": "run_cand", "task_id": "t1", "query_id": "q1", "model_id": "m1", "expected": "A", "score": None, "error": "timeout"}) + "\n")

    for rdir in (run_base, run_cand):
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

    with pytest.raises(CompareError) as exc_info:
        run_comparison("run_base", "run_cand", model="m1", base_dir=str(runs_dir))

    assert "No valid paired queries with successful scores remain" in str(exc_info.value)
