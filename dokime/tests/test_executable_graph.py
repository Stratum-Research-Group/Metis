"""Tests for Dokime Executable Graph QA task and deterministic query executor."""

import json
import pytest
from dokime.tasks.executable_graph import (
    FIXTURE_HASH,
    SYNTHETIC_KNOWLEDGE_FIXTURE,
    ExecutableGraphTask,
    Fact,
    GraphQueryExecutor,
    compute_f1_score,
    compute_fixture_hash,
    parse_entities_and_citations,
)
from dokime.tasks.protocol import Query
from dokime.tasks.registry import discover_tasks, get_task


def test_fixture_hash_integrity():
    """Verify fixture hash is deterministic and tamper-evident."""
    h1 = compute_fixture_hash()
    assert h1 == FIXTURE_HASH
    assert len(h1) == 64

    # Tampering test
    tampered = list(SYNTHETIC_KNOWLEDGE_FIXTURE) + [Fact("f_bad", "x", "y", "z")]
    h2 = compute_fixture_hash(tampered)
    assert h1 != h2


def test_executor_chains_and_paths():
    """Verify forward and reverse multi-hop query execution."""
    executor = GraphQueryExecutor()

    # 1-hop
    ans, facts = executor.execute_chain("apollo_11", ["launch_vehicle"])
    assert ans == {"saturn_v"}
    assert facts == ["f_aero_01"]

    # 2-hop
    ans, facts = executor.execute_chain("python", ["invented_by", "employed_at"])
    assert ans == {"cwi_amsterdam"}
    assert facts == ["f_cs_01", "f_cs_15"]

    # 3-hop
    ans, facts = executor.execute_chain("python", ["invented_by", "employed_at", "located_in"])
    assert ans == {"amsterdam"}
    assert facts == ["f_cs_01", "f_cs_15", "f_cs_16"]

    # Reverse chain
    ans, facts = executor.execute_reverse_chain("pbp", ["inhibits"])
    assert ans == {"penicillin"}
    assert facts == ["f_bio_06"]

    # Non-existent chain
    ans, facts = executor.execute_chain("non_existent_entity", ["invented_by"])
    assert ans == set()
    assert facts == []


def test_executor_intersection_and_empty():
    """Verify intersection queries across multiple constraints."""
    executor = GraphQueryExecutor()

    # 2-constraint intersection
    ans, facts = executor.execute_intersection([
        ("leader_based", "consensus_family"),
        ("leslie_lamport", "designed_by"),
    ])
    assert ans == {"paxos"}
    assert facts == ["f_dist_02", "f_dist_08"]

    # Empty intersection (mutually exclusive constraints)
    ans, facts = executor.execute_intersection([
        ("pbp", "inhibits"),
        ("antifolate", "drug_class"),
    ])
    assert ans == set()
    assert facts == []


def test_suite_data_conformance():
    """Verify all benchmark cases in the suite conform to the schema."""
    task = ExecutableGraphTask()
    queries = task.load()
    assert len(queries) == 20

    splits = set()
    patterns = set()

    for q in queries:
        meta = q.metadata
        assert meta["fixture_hash"] == FIXTURE_HASH
        assert meta["wording_review_status"] == "APPROVED"
        assert meta["schema_version"] == "1.0.0"
        assert meta["query_executor_version"] == "1.0.0"
        assert isinstance(meta["gold_answer_ids"], list)
        assert isinstance(meta["evidence_fact_ids"], list)
        splits.add(meta["split"])
        patterns.add(meta["query_pattern"])

    assert "test" in splits
    assert "practice" in splits
    assert "chain_2hop" in patterns
    assert "chain_3hop" in patterns
    assert "intersection" in patterns
    assert "negative_empty" in patterns


def test_parse_entities_and_citations():
    """Verify robust parsing across JSON and unstructured text responses."""
    # JSON dict format
    resp_json = '{"answers": ["python", "c_lang"], "evidence": ["f_cs_01", "f_cs_02"]}'
    ans, ev = parse_entities_and_citations(resp_json)
    assert ans == ["python", "c_lang"]
    assert ev == ["f_cs_01", "f_cs_02"]

    # JSON array format
    resp_arr = '["saturn_v"]'
    ans, ev = parse_entities_and_citations(resp_arr)
    assert ans == ["saturn_v"]
    assert ev == []

    # Text format with Answer prefix and citation tokens
    resp_text = "Answer: paxos. Based on evidence f_dist_02 and f_dist_08."
    ans, ev = parse_entities_and_citations(resp_text)
    assert "paxos" in ans
    assert ev == ["f_dist_02", "f_dist_08"]


def test_compute_f1_score():
    """Verify set-based F1 computation."""
    # Perfect match
    p, r, f1 = compute_f1_score(["a", "b"], ["b", "a"])
    assert p == 1.0 and r == 1.0 and f1 == 1.0

    # Partial match
    p, r, f1 = compute_f1_score(["a"], ["a", "b"])
    assert p == 1.0
    assert r == 0.5
    assert round(f1, 4) == 0.6667

    # Empty gold match
    p, r, f1 = compute_f1_score([], [])
    assert f1 == 1.0

    # Empty gold mismatch
    p, r, f1 = compute_f1_score(["unexpected"], [])
    assert f1 == 0.0


def test_evidence_support_separation():
    """Verify that unsupported guesses do not get evidence-grounded success credit."""
    task = ExecutableGraphTask()
    q = Query(
        id="test_case",
        prompt="Where was the employer of Python creator located?",
        expected="amsterdam",
        metadata={
            "gold_answer_ids": ["amsterdam"],
            "evidence_fact_ids": ["f_cs_01", "f_cs_15", "f_cs_16"],
            "query_pattern": "chain_3hop",
        },
    )

    # 1. Correct answer with full supporting evidence
    full_resp = '{"answers": ["amsterdam"], "evidence": ["f_cs_01", "f_cs_15", "f_cs_16"]}'
    res_full = task.score_detailed(q, full_resp)
    assert res_full["answer_F1"] == 1.0
    assert res_full["evidence_F1"] == 1.0
    assert res_full["evidence_grounded_success"] is True

    # 2. Correct answer without evidence (unsupported hallucination/guess)
    guess_resp = '{"answers": ["amsterdam"], "evidence": []}'
    res_guess = task.score_detailed(q, guess_resp)
    assert res_guess["answer_F1"] == 1.0
    assert res_guess["evidence_recall"] == 0.0
    assert res_guess["evidence_grounded_success"] is False

    # 3. Wrong answer with full evidence
    wrong_resp = '{"answers": ["paris"], "evidence": ["f_cs_01", "f_cs_15", "f_cs_16"]}'
    res_wrong = task.score_detailed(q, wrong_resp)
    assert res_wrong["answer_F1"] == 0.0
    assert res_wrong["evidence_grounded_success"] is False


def test_registry_integration():
    """Verify task discovery via dokime registry."""
    tasks = discover_tasks()
    assert "executable_graph_qa" in tasks
    task = get_task("executable_graph_qa")
    assert isinstance(task, ExecutableGraphTask)
    queries = task.load()
    assert len(queries) == 20
