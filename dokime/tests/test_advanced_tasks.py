"""Unit tests for GPQA Diamond, AIME Math, Zebra Logic, and Epistemic Adversarial benchmark tasks."""

import pytest
from dokime.tasks.base_mc import MultipleChoiceTask
from dokime.tasks.epistemic import EpistemicAdversarialTask
from dokime.tasks.gpqa import GPQADiamondTask
from dokime.tasks.math_competition import AIMECompetitionMathTask
from dokime.tasks.protocol import Query
from dokime.tasks.registry import discover_tasks, get_task
from dokime.tasks.zebra import ZebraHardLogicTask


def test_registry_discovers_all_tasks():
    tasks = discover_tasks()
    expected = {"arc_easy_25", "gpqa_diamond", "aime_math", "zebra_hard_logic", "epistemic_adversarial"}
    assert expected.issubset(set(tasks.keys()))


def test_gpqa_diamond_task():
    task = get_task("gpqa_diamond")
    queries = task.load()
    assert len(queries) == 15
    q1 = queries[0]
    assert q1.id == "gpqa_001"
    assert "spin-1/2 particle" in q1.prompt
    assert q1.expected == "A"

    # Test scoring with thought tags
    resp_with_thoughts = (
        "<|channel>thought\nLet me solve the Schrodinger equation for S_x...\n<channel|>\n"
        "Final Answer: A"
    )
    assert task.extract_prediction(q1, resp_with_thoughts) == "A"
    assert task.score(q1, resp_with_thoughts) == 1.0

    # Wrong answer
    assert task.score(q1, "Final Answer: B") == 0.0


def test_aime_math_task():
    task = get_task("aime_math")
    queries = task.load()
    assert len(queries) == 10
    q1 = queries[0]
    assert q1.id == "aime_001"
    assert q1.expected == "8"

    # Test \boxed extraction
    boxed_resp = (
        "<|channel>thought\n(a-12)(b-12) = 144, 15 divisors, ceil(15/2) = 8\n<channel|>\n"
        "Therefore, the number of ordered pairs is \\boxed{8}."
    )
    assert task.extract_prediction(q1, boxed_resp) == "8"
    assert task.score(q1, boxed_resp) == 1.0

    # Test plain final answer
    plain_resp = "The total number of ordered pairs is Final Answer: 8"
    assert task.extract_prediction(q1, plain_resp) == "8"
    assert task.score(q1, plain_resp) == 1.0


def test_zebra_hard_logic_task():
    task = get_task("zebra_hard_logic")
    queries = task.load()
    assert len(queries) == 10
    q1 = queries[0]
    assert q1.id == "zebra_001"
    assert q1.expected == "B"

    resp = "Based on spatial deduction, House 2 is blue. Final Answer: B"
    assert task.extract_prediction(q1, resp) == "B"
    assert task.score(q1, resp) == 1.0


def test_epistemic_adversarial_task():
    task = get_task("epistemic_adversarial")
    queries = task.load()
    assert len(queries) == 10
    q1 = queries[0]
    assert q1.id == "epist_001"
    assert q1.expected == "A"

    resp = (
        "<thought>Modus tollens applies validly here.</thought>\n"
        "The correct answer is (A)."
    )
    assert task.extract_prediction(q1, resp) == "A"
    assert task.score(q1, resp) == 1.0
