"""Tests for Round 5 fixes: decoupling model-request seed from statistical bootstrap seed."""

import json
from pathlib import Path
import httpx
import pytest

from dokime.client import HestiaClient
from dokime.config import RunConfig
from dokime.runner import run_evaluation
from dokime.stats import single_run_bootstrap
from dokime.tasks.protocol import Query
from dokime.tasks.registry import BUILTIN_TASKS


class SevenScoresTask:
    """Mock task producing seven queries to yield scores [1, 1, 0, 0, 0, 0, 0]."""

    task_id: str = "seven_scores_task"

    def load(self) -> list[Query]:
        return [
            Query(id=f"q_{i}", prompt=f"Query {i}", expected="PASS")
            for i in range(7)
        ]

    def score(self, query: Query, response: str) -> float:
        # Score 1.0 if response contains PASS, 0.0 otherwise
        return 1.0 if "PASS" in response else 0.0


def test_request_seed_decoupled_from_bootstrap_seed(tmp_path: Path):
    """Verify that config.seed controls the request and manifest, while summary CI uses bootstrap seed 42.

    Seven scores [1, 1, 0, 0, 0, 0, 0] yield:
    - seed-42 CI: [0.0, 0.575]
    - seed-1234 CI: [0.0, 0.5714]
    Summary mean and CI must be identical across runs with request seeds 42 and 1234, matching seed 42.
    """
    BUILTIN_TASKS["seven_scores_task"] = SevenScoresTask

    # Handlers that capture outgoing requests and return PASS for q_0 and q_1, FAIL for q_2..q_6
    requests_run1: list[dict] = []
    requests_run2: list[dict] = []

    def mock_handler_run1(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests_run1.append(body)
        prompt = body["messages"][-1]["content"]
        ans = "PASS" if any(prompt.endswith(f"Query {i}") for i in (0, 1)) else "FAIL"
        return httpx.Response(200, json={"choices": [{"message": {"content": ans}}]})

    def mock_handler_run2(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests_run2.append(body)
        prompt = body["messages"][-1]["content"]
        ans = "PASS" if any(prompt.endswith(f"Query {i}") for i in (0, 1)) else "FAIL"
        return httpx.Response(200, json={"choices": [{"message": {"content": ans}}]})

    runs_dir = tmp_path / "runs"

    # --- Run 1: Request seed 42 ---
    client_1 = HestiaClient(
        base_url="http://mock:8000",
        client=httpx.Client(transport=httpx.MockTransport(mock_handler_run1)),
    )
    config_1 = RunConfig(
        models=["test-model"],
        tasks=["seven_scores_task"],
        seed=42,
        output_dir=str(runs_dir),
    )
    res_1 = run_evaluation(config_1, client=client_1, run_id="run_seed_42")

    # --- Run 2: Request seed 1234 ---
    client_2 = HestiaClient(
        base_url="http://mock:8000",
        client=httpx.Client(transport=httpx.MockTransport(mock_handler_run2)),
    )
    config_2 = RunConfig(
        models=["test-model"],
        tasks=["seven_scores_task"],
        seed=1234,
        output_dir=str(runs_dir),
    )
    res_2 = run_evaluation(config_2, client=client_2, run_id="run_seed_1234")

    # 1. Verify outgoing HTTP JSON bodies contain the user's chosen request seed
    assert len(requests_run1) == 7
    for req in requests_run1:
        assert req["seed"] == 42

    assert len(requests_run2) == 7
    for req in requests_run2:
        assert req["seed"] == 1234

    # 2. Verify manifests record the user's chosen request seed
    with open(runs_dir / "run_seed_42" / "manifest.json", "r", encoding="utf-8") as f:
        manifest_1 = json.load(f)
    assert manifest_1["seed"] == 42

    with open(runs_dir / "run_seed_1234" / "manifest.json", "r", encoding="utf-8") as f:
        manifest_2 = json.load(f)
    assert manifest_2["seed"] == 1234

    # 3. Verify summary mean and CI are identical across the runs
    with open(runs_dir / "run_seed_42" / "summary.json", "r", encoding="utf-8") as f:
        summary_1 = json.load(f)
    with open(runs_dir / "run_seed_1234" / "summary.json", "r", encoding="utf-8") as f:
        summary_2 = json.load(f)

    stats_1 = summary_1["test-model/seven_scores_task"]["stats"]
    stats_2 = summary_2["test-model/seven_scores_task"]["stats"]

    assert stats_1 == stats_2

    # 4. Verify summary stats match single_run_bootstrap with statistical seed 42, NOT seed 1234
    expected_seed_42_stats = single_run_bootstrap([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], seed=42)
    expected_seed_1234_stats = single_run_bootstrap([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], seed=1234)

    assert stats_1 == expected_seed_42_stats
    assert stats_2 == expected_seed_42_stats

    # Specifically check the differing endpoint: seed 42 is 0.575, seed 1234 is 0.5714
    assert stats_1["ci_upper"] == 0.575
    assert stats_2["ci_upper"] == 0.575
    assert expected_seed_1234_stats["ci_upper"] == 0.5714
    assert stats_2["ci_upper"] != expected_seed_1234_stats["ci_upper"]


def test_all_failing_requests_preserves_not_measured_with_custom_seed(tmp_path: Path):
    """When all queries fail, summary stats must report not-measured regardless of request seed."""
    BUILTIN_TASKS["seven_scores_task"] = SevenScoresTask

    def mock_handler_fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "Server error"})

    runs_dir = tmp_path / "runs"
    client = HestiaClient(
        base_url="http://mock:8000",
        client=httpx.Client(transport=httpx.MockTransport(mock_handler_fail)),
        max_retries=0,
    )
    config = RunConfig(
        models=["test-model"],
        tasks=["seven_scores_task"],
        seed=1234,
        output_dir=str(runs_dir),
    )
    run_evaluation(config, client=client, run_id="run_fail_1234")

    with open(runs_dir / "run_fail_1234" / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    stats = summary["test-model/seven_scores_task"]["stats"]
    assert stats["status"] == "not-measured"
    assert stats["mean"] is None
    assert stats["ci_lower"] is None
    assert stats["ci_upper"] is None
    assert stats["count"] == 0
    assert stats["num_samples"] == 10000
