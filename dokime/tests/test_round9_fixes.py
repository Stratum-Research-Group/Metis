"""Regression tests for Round 9 fixes: ARC explicit answer phrase extractor tightening.

Verifies:
- The three B=blue reproductions after explicit phrases:
    "Answer: a blue object."
    "The answer is a blue object."
    "My final answer is a blue object."
- Genuine explicit A and B letter responses ("Answer: A", "Answer: A.", "The final answer is A", "Answer: (A)").
- Explanatory named-choice text.
- Article-followed-by-unknown text that must not become A ("Answer: a completely unknown phenomenon.").
- Conflicting explicit answers with named choices.
- Verification of both extract_prediction() and score().
"""

from __future__ import annotations

import pytest

from dokime.tasks.arc import ArcEasy25Task
from dokime.tasks.protocol import Query


@pytest.fixture
def arc_task() -> ArcEasy25Task:
    return ArcEasy25Task()


@pytest.fixture
def color_query_b() -> Query:
    return Query(
        id="q_color_b",
        prompt="What color is the object?",
        choices={"A": "red", "B": "blue", "C": "green", "D": "yellow"},
        expected="B",
    )


@pytest.fixture
def color_query_a() -> Query:
    return Query(
        id="q_color_a",
        prompt="What color is the object?",
        choices={"A": "red", "B": "blue", "C": "green", "D": "yellow"},
        expected="A",
    )


@pytest.fixture
def rock_query() -> Query:
    return Query(
        id="q_rock",
        prompt="Which tool is best used to measure the mass of a rock?",
        choices={
            "A": "Balance scale",
            "B": "Thermometer",
            "C": "Graduated cylinder",
            "D": "Ruler",
        },
        expected="A",
    )


def test_three_blue_reproductions(arc_task: ArcEasy25Task, color_query_b: Query):
    """Verify each of the three Round 9 reproduction cases extracts B and scores 1.0."""
    repros = [
        "Answer: a blue object.",
        "The answer is a blue object.",
        "My final answer is a blue object.",
    ]
    for text in repros:
        assert arc_task.extract_prediction(color_query_b, text) == "B", f"Failed on {text!r}"
        assert arc_task.score(color_query_b, text) == 1.0, f"Score failed on {text!r}"


def test_genuine_explicit_a_letter_responses(
    arc_task: ArcEasy25Task,
    color_query_a: Query,
    color_query_b: Query,
):
    """Verify genuine standalone A letter responses parse as A and score appropriately."""
    genuine_a = [
        "Answer: A",
        "Answer: A.",
        "The final answer is A",
        "The final answer is A.",
        "Answer: (A)",
        "Answer: [A]",
        "Answer: **A**",
        "Choice A",
        "Choice A.",
        "Option A",
        "Option A.",
        "The answer is A",
        "The answer is A.",
        "The correct answer is A.",
    ]
    for text in genuine_a:
        pred = arc_task.extract_prediction(color_query_a, text)
        assert pred == "A", f"Failed to extract 'A' from {text!r}, got {pred!r}"
        # With expected=A -> score 1.0
        assert arc_task.score(color_query_a, text) == 1.0, f"Score with expected A failed on {text!r}"
        # With expected=B -> score 0.0
        assert arc_task.score(color_query_b, text) == 0.0, f"Score with expected B failed on {text!r}"


def test_genuine_explicit_b_letter_responses(arc_task: ArcEasy25Task, color_query_b: Query):
    """Verify genuine standalone B letter responses parse as B and score 1.0."""
    genuine_b = [
        "Answer: B",
        "Answer: B.",
        "The final answer is B",
        "The final answer is B.",
        "Answer: (B)",
        "Answer: [B]",
        "Answer: **B**",
        "Choice B",
        "Choice B.",
        "Option B",
        "Option B.",
        "The answer is B",
        "The answer is B.",
        "The correct answer is B.",
    ]
    for text in genuine_b:
        pred = arc_task.extract_prediction(color_query_b, text)
        assert pred == "B", f"Failed to extract 'B' from {text!r}, got {pred!r}"
        assert arc_task.score(color_query_b, text) == 1.0, f"Score failed on {text!r}"


def test_explanatory_named_choice_text(
    arc_task: ArcEasy25Task,
    color_query_b: Query,
    rock_query: Query,
):
    """Verify named choices in explanatory text parse unambiguously."""
    assert arc_task.extract_prediction(color_query_b, "The answer is blue.") == "B"
    assert arc_task.score(color_query_b, "The answer is blue.") == 1.0

    assert arc_task.extract_prediction(color_query_b, "The correct choice is blue.") == "B"
    assert arc_task.score(color_query_b, "The correct choice is blue.") == 1.0

    assert arc_task.extract_prediction(color_query_b, "A blue object is the answer.") == "B"
    assert arc_task.score(color_query_b, "A blue object is the answer.") == 1.0

    resp_rock = "The balance scale is the proper instrument."
    assert arc_task.extract_prediction(rock_query, resp_rock) == "A"
    assert arc_task.score(rock_query, resp_rock) == 1.0


def test_article_followed_by_unknown_text_must_not_become_a(
    arc_task: ArcEasy25Task,
    color_query_b: Query,
):
    """Verify that an indefinite article followed by non-matching text returns None, never A."""
    unknown_cases = [
        "Answer: a completely unknown phenomenon.",
        "The answer is a random guess.",
        "My final answer is a mysterious force.",
        "Choice: a strange idea.",
        "Option: a perplexing result.",
        "Answer: a non-existent color spectrum.",
        "The answer is a very complicated question.",
        "Option: a curious observation that defies classification.",
        "Choice: a novel hypothesis.",
    ]
    for text in unknown_cases:
        pred = arc_task.extract_prediction(color_query_b, text)
        assert pred is None, f"Expected None for {text!r}, but got {pred!r}"
        assert arc_task.score(color_query_b, text) == 0.0, f"Expected score 0.0 for {text!r}"


def test_conflicting_explicit_answers_with_named_choices(
    arc_task: ArcEasy25Task,
    color_query_b: Query,
):
    """Verify that multiple conflicting named choices after explicit phrases return None."""
    conflicts = [
        "Answer: a blue object or green object.",
        "The answer is a red object or blue object.",
        "Choice: a blue object or red object.",
        "Option: a green object or yellow object.",
        "My final answer is a red or blue object.",
    ]
    for text in conflicts:
        pred = arc_task.extract_prediction(color_query_b, text)
        assert pred is None, f"Expected None for conflicting {text!r}, but got {pred!r}"
        assert arc_task.score(color_query_b, text) == 0.0
