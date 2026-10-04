"""Regression tests for Round 8 fixes: ARC answer extractor tightening.

Verifies:
- Prose-article reproduction: "A blue object is the answer." extracts B (not choice A).
- Explanatory text with a named choice.
- Bare A/B/C/D responses.
- Explicit answer forms: "Answer: B", "(B)", "Choice B", "The answer is (B)".
- Uncertain/conflicting text does not receive false positives (returns None, score 0.0).
- Explicit final-answer rule resolves deliberation.
- score() uses the corrected extraction.
"""

from __future__ import annotations

import pytest

from dokime.tasks.arc import ArcEasy25Task
from dokime.tasks.protocol import Query


@pytest.fixture
def arc_task() -> ArcEasy25Task:
    return ArcEasy25Task()


@pytest.fixture
def color_query() -> Query:
    return Query(
        id="q_color",
        prompt="What color is the object?",
        choices={"A": "red", "B": "blue", "C": "green", "D": "yellow"},
        expected="B",
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


def test_prose_article_reproduction(arc_task: ArcEasy25Task, color_query: Query):
    """The article 'A' in prose must not be misinterpreted as choice A before actual choice B."""
    response = "A blue object is the answer."
    assert arc_task.extract_prediction(color_query, response) == "B"
    assert arc_task.score(color_query, response) == 1.0

    # Additional prose-article variations
    resp2 = "A blue light was observed in the laboratory."
    assert arc_task.extract_prediction(color_query, resp2) == "B"
    assert arc_task.score(color_query, resp2) == 1.0

    # Prose starting with "A" when choice A was actually intended
    resp_a = "A red object is the answer."
    assert arc_task.extract_prediction(color_query, resp_a) == "A"
    assert arc_task.score(color_query, resp_a) == 0.0

    # Prose starting with "A" where no choice matches should return None (not A)
    resp_none = "A study was conducted to analyze spectral properties."
    assert arc_task.extract_prediction(color_query, resp_none) is None
    assert arc_task.score(color_query, resp_none) == 0.0


def test_explanatory_text_with_named_choice(arc_task: ArcEasy25Task, rock_query: Query, color_query: Query):
    """Explanatory text naming exactly one choice unambiguously extracts that choice."""
    resp1 = "To measure the mass of a rock, a balance scale is the most suitable instrument."
    assert arc_task.extract_prediction(rock_query, resp1) == "A"
    assert arc_task.score(rock_query, resp1) == 1.0

    resp2 = "The instrument required for this task is the balance scale."
    assert arc_task.extract_prediction(rock_query, resp2) == "A"
    assert arc_task.score(rock_query, resp2) == 1.0

    resp3 = "The thermometer is used for temperature, not mass."
    assert arc_task.extract_prediction(rock_query, resp3) == "B"
    assert arc_task.score(rock_query, resp3) == 0.0

    resp4 = "The sky is clearly blue."
    assert arc_task.extract_prediction(color_query, resp4) == "B"
    assert arc_task.score(color_query, resp4) == 1.0


def test_bare_single_letter_responses(arc_task: ArcEasy25Task, color_query: Query):
    """Bare single letters with various punctuation or markdown must parse cleanly."""
    letters = ["A", "B", "C", "D"]
    for letter in letters:
        pred = arc_task.extract_prediction(color_query, letter)
        assert pred == letter
        expected_score = 1.0 if letter == color_query.expected else 0.0
        assert arc_task.score(color_query, letter) == expected_score

    # Lowercase bare letter
    assert arc_task.extract_prediction(color_query, "b") == "B"
    assert arc_task.score(color_query, "b") == 1.0

    # Whitespace padding
    assert arc_task.extract_prediction(color_query, "   B   ") == "B"
    assert arc_task.score(color_query, "   B   ") == 1.0

    # Punctuation and bracketed variants
    variants = [
        "(B)",
        "[B]",
        "**B**",
        "*B*",
        "__B__",
        "`B`",
        "B.",
        "B:",
        "B)",
        "(B).",
        "'B'",
        '"B"',
    ]
    for var in variants:
        assert arc_task.extract_prediction(color_query, var) == "B", f"Failed on {var!r}"
        assert arc_task.score(color_query, var) == 1.0, f"Score failed on {var!r}"


def test_explicit_answer_syntax(arc_task: ArcEasy25Task, color_query: Query):
    """Explicit answer syntax like 'Answer: B', '(B)', 'Choice B', 'The answer is (B)'."""
    valid_explicit = [
        "Answer: B",
        "Answer: (B)",
        "Answer: [B]",
        "Answer: **B**",
        "ANSWER: B",
        "Answer is B",
        "Answer is: B",
        "Answer is (B)",
        "The answer is B",
        "The answer is (B)",
        "The answer is: B",
        "The answer is choice B",
        "The answer is option (B)",
        "The correct answer is B.",
        "The correct answer is (B).",
        "The best answer is B.",
        "Choice B",
        "Choice: B",
        "Choice B.",
        "Choice (B)",
        "Choice is B",
        "CHOICE B",
        "Option B",
        "Option: B",
        "Option (B)",
        "Option is B",
        "Therefore, (B) is the right selection.",
        "I select (B) for this question.",
        "Final answer: B",
        "The final answer is: (B)",
        "My final answer is B",
    ]
    for text in valid_explicit:
        pred = arc_task.extract_prediction(color_query, text)
        assert pred == "B", f"Failed to extract 'B' from {text!r}, got {pred!r}"
        assert arc_task.score(color_query, text) == 1.0


def test_uncertain_and_conflicting_text_no_false_positive(arc_task: ArcEasy25Task, color_query: Query):
    """Uncertain and conflicting texts must not guess an incidental letter or give false positives."""
    conflicting_or_uncertain = [
        # Explicit conflicting answers
        "Answer: A or Answer: B",
        "The answer is A. Or wait, the answer is B.",
        "Choice A or Choice B",
        "Option A or Option B",
        "The answer is A or B.",
        "The answer is either A or B.",
        "The correct answer is (A) or (B).",
        "It could be (A) or (B).",
        "Between (C) and (D), I am undecided.",
        # Uncertain explanatory prose
        "It could be A or it could be B.",
        "I am not sure if it is A or B.",
        "Either A or B might be correct.",
        "A is possible, but D is also possible.",
        "I think it might be A, or perhaps B.",
        # General prose containing incidental 'a'
        "A study was conducted to observe this reaction.",
        "In a standard physics laboratory, this holds true.",
        "This is a difficult question with multiple nuances.",
        # Conflicting named choices
        "It could be red or blue.",
        "Both red and blue are primary colors.",
        "Neither red nor yellow seems right.",
        # Multiple conflicting final answers
        "Final answer: A... Wait, final answer: B",
        # Generic unanswerable text
        "I don't know the answer.",
        "No conclusive evidence exists.",
        "",
        "   ",
    ]
    for text in conflicting_or_uncertain:
        pred = arc_task.extract_prediction(color_query, text)
        assert pred is None, f"Expected None for uncertain text {text!r}, but got {pred!r}"
        assert arc_task.score(color_query, text) == 0.0, f"Expected score 0.0 for {text!r}"


def test_explicit_final_answer_rule_resolves_deliberation(arc_task: ArcEasy25Task, color_query: Query):
    """When a model discusses earlier options but states an explicit final answer, that answer is used."""
    deliberations = [
        (
            "At first choice A (red) seemed plausible, but on closer analysis the final answer is B.",
            "B",
        ),
        (
            "We might consider choice C or D, but my final answer is: (B).",
            "B",
        ),
        (
            "Option B is tempting, but upon reflection the final answer is A.",
            "A",
        ),
    ]
    for text, expected in deliberations:
        pred = arc_task.extract_prediction(color_query, text)
        assert pred == expected, f"Failed deliberation resolution on {text!r}: got {pred!r}"
        expected_score = 1.0 if expected == color_query.expected else 0.0
        assert arc_task.score(color_query, text) == expected_score


def test_score_uses_corrected_extraction(arc_task: ArcEasy25Task, color_query: Query):
    """Verify score() strictly reflects extract_prediction() results against expected."""
    # Correct extraction -> 1.0
    assert arc_task.score(color_query, "The answer is (B)") == 1.0
    assert arc_task.score(color_query, "A blue object is the answer.") == 1.0

    # Wrong extraction -> 0.0
    assert arc_task.score(color_query, "Answer: C") == 0.0
    assert arc_task.score(color_query, "A red object is the answer.") == 0.0

    # Ambiguous / None -> 0.0
    assert arc_task.score(color_query, "Choice A or Choice B") == 0.0
    assert arc_task.score(color_query, "A study was conducted.") == 0.0


def test_real_arc_dataset_queries_extraction_and_scoring(arc_task: ArcEasy25Task):
    """Test extraction and scoring on real queries loaded from arc_easy_25.json."""
    queries = arc_task.load()
    q_by_id = {q.id: q for q in queries}

    # arc_001: Which tool is best used to measure the mass of a rock? A: Balance scale, expected A
    q1 = q_by_id["arc_001"]
    assert arc_task.extract_prediction(q1, "A") == "A"
    assert arc_task.extract_prediction(q1, "Answer: A") == "A"
    assert arc_task.extract_prediction(q1, "A balance scale should be used to measure mass.") == "A"
    assert arc_task.score(q1, "A balance scale should be used to measure mass.") == 1.0

    # arc_002: guitar strings vibrate -> B: Sound energy, expected B
    q2 = q_by_id["arc_002"]
    assert arc_task.extract_prediction(q2, "B) Sound energy") == "B"
    assert arc_task.extract_prediction(q2, "Vibrating guitar strings produce sound energy.") == "B"
    assert arc_task.score(q2, "Vibrating guitar strings produce sound energy.") == 1.0

    # arc_022: Sound cannot travel through -> B: A vacuum, expected B
    q22 = q_by_id["arc_022"]
    assert arc_task.extract_prediction(q22, "A vacuum cannot transmit sound vibrations.") == "B"
    assert arc_task.score(q22, "A vacuum cannot transmit sound vibrations.") == 1.0

