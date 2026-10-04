"""Integration tests for runner matrix execution and JSONL logging."""

import json
from pathlib import Path
import httpx

from dokime.client import HestiaClient
from dokime.config import RunConfig
from dokime.runner import run_evaluation


def test_runner_matrix_execution(tmp_path: Path):
    # Mock server that returns "A" for every question
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Answer: A"}}],
            },
        )

    transport = httpx.MockTransport(mock_handler)
    http_client = httpx.Client(transport=transport)
    client = HestiaClient(base_url="http://mock:8000", client=http_client)

    config = RunConfig(
        models=["model_1", "model_2"],
        tasks=["arc_easy_25"],
        output_dir=str(tmp_path / "runs"),
        seed=42,
    )

    result = run_evaluation(config, client=client, run_id="test_run_matrix")
    assert result["run_id"] == "test_run_matrix"
    assert result["total_queries"] == 50  # 2 models x 25 queries

    # Check results.jsonl
    results_path = Path(result["results_path"])
    assert results_path.exists()

    with open(results_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 50
    first = lines[0]
    required_fields = {
        "run_id",
        "task_id",
        "query_id",
        "model_id",
        "prompt_hash",
        "predicted",
        "expected",
        "score",
        "latency_ms",
        "timestamp",
    }
    assert required_fields.issubset(first.keys())
    assert first["predicted"] == "A"
    assert first["score"] in (0.0, 1.0)
    assert len(first["prompt_hash"]) == 64

    # Check manifest.json
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["run_id"] == "test_run_matrix"
    assert "results_jsonl" in manifest["checksums"]


def test_runner_error_logging_never_skips(tmp_path: Path):
    # Mock server that always fails with 500
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Crash")

    transport = httpx.MockTransport(mock_handler)
    http_client = httpx.Client(transport=transport)
    client = HestiaClient(
        base_url="http://mock:8000",
        max_retries=1,
        backoff_base=0.001,
        client=http_client,
    )

    config = RunConfig(
        models=["broken_model"],
        tasks=["arc_easy_25"],
        output_dir=str(tmp_path / "runs"),
    )

    result = run_evaluation(config, client=client, run_id="error_run")
    assert result["total_queries"] == 25

    with open(result["results_path"], "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 25
    for record in lines:
        assert record["score"] is None
        assert record["predicted"] is None
        assert "error" in record
        assert "500" in record["error"]


def test_runner_nonfinite_scorer_output_treated_as_failure(tmp_path: Path, monkeypatch):
    from dokime.tasks.protocol import Query
    import dokime.runner as runner_mod

    class NonfiniteScorerTask:
        task_id = "dummy_task"

        def load(self) -> list[Query]:
            return [
                Query(id="q_nan", prompt="prompt_nan", expected="A"),
                Query(id="q_inf", prompt="prompt_inf", expected="A"),
                Query(id="q_neginf", prompt="prompt_neginf", expected="A"),
                Query(id="q_zero", prompt="prompt_zero", expected="A"),
                Query(id="q_one", prompt="prompt_one", expected="A"),
            ]

        def score(self, query: Query, raw_response: str) -> float:
            if query.id == "q_nan":
                return float("nan")
            elif query.id == "q_inf":
                return float("inf")
            elif query.id == "q_neginf":
                return float("-inf")
            elif query.id == "q_zero":
                return 0.0
            return 1.0

    monkeypatch.setattr(runner_mod, "get_task", lambda task_id: NonfiniteScorerTask())

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "Answer: A"}}]})

    transport = httpx.MockTransport(mock_handler)
    http_client = httpx.Client(transport=transport)
    client = HestiaClient(base_url="http://mock:8000", client=http_client)

    config = RunConfig(
        models=["model_1"],
        tasks=["dummy_task"],
        output_dir=str(tmp_path / "runs"),
        seed=42,
    )

    result = run_evaluation(config, client=client, run_id="nonfinite_scorer_run")
    assert result["total_queries"] == 5

    # Check cell statistics in result: only 2 valid queries (0.0 and 1.0)
    cell_summary = result["stats"]["model_1/dummy_task"]
    assert cell_summary["total_queries"] == 5
    assert cell_summary["successful_queries"] == 2
    assert cell_summary["failed_queries"] == 3
    assert cell_summary["stats"]["count"] == 2
    assert cell_summary["stats"]["mean"] == 0.5

    # Check results.jsonl records
    with open(result["results_path"], "r", encoding="utf-8") as f:
        records = {r["query_id"]: r for r in (json.loads(line) for line in f)}

    assert len(records) == 5

    # Nonfinite queries must have score=None and explicit error
    for qid in ["q_nan", "q_inf", "q_neginf"]:
        rec = records[qid]
        assert rec["score"] is None
        assert "Non-finite score returned by task scorer" in rec["error"]
        assert rec["prompt_hash"] is not None and len(rec["prompt_hash"]) == 64

    # Legitimate 0.0 score must remain valid
    rec_zero = records["q_zero"]
    assert rec_zero["score"] == 0.0
    assert "error" not in rec_zero

    rec_one = records["q_one"]
    assert rec_one["score"] == 1.0
    assert "error" not in rec_one

