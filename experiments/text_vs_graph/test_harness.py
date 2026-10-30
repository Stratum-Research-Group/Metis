"""Tests for prompt rendering, answer parsing, and exact-match scoring."""

from __future__ import annotations

import pytest
from experiments.text_vs_graph.generator import generate_story
from experiments.text_vs_graph.harness import (
    format_user_prompt,
    parse_model_answer,
    score_answer,
)


def test_format_user_prompt_all_arms():
    story = generate_story("story_prompt_test", num_events=8, seed=42)
    q = story.questions[0]
    
    # Arm A
    sys_a, user_a = format_user_prompt(story, q, "A", story.graph_block)
    assert "EVIDENCE (prose):" in user_a
    assert "EVIDENCE (structured facts):" not in user_a
    assert "treat the text as correct" not in sys_a

    # Arm B
    sys_b, user_b = format_user_prompt(story, q, "B", story.graph_block)
    assert "EVIDENCE (structured facts):" in user_b
    assert "treat the text as correct" not in sys_b

    # Arm C
    sys_c, user_c = format_user_prompt(story, q, "C", story.graph_block)
    assert "TEXT:" in user_c
    assert "STRUCTURED FACTS:" in user_c
    assert "treat the text as correct" in sys_c

    # Arm D pass 1
    sys_d1, user_d1 = format_user_prompt(story, q, "D", story.graph_block, pass_number=1)
    assert "list each structured fact that the text does not support" in user_d1
    assert "Do not answer the question" in sys_d1

    # Arm D pass 2
    sys_d2, user_d2 = format_user_prompt(story, q, "D", story.graph_block, pass_number=2, pass1_notes="[DEP_E3] contradicted")
    assert "Notes from the earlier check:" in user_d2
    assert "[DEP_E3] contradicted" in user_d2
    assert "treat the text as correct" in sys_d2

    # Arm F pass 1
    sys_f1, user_f1 = format_user_prompt(story, q, "F", story.graph_block, pass_number=1)
    assert "restate the facts relevant to this question and your plan" in user_f1
    assert "Do not answer the question" in sys_f1

    # Arm F pass 2
    sys_f2, user_f2 = format_user_prompt(story, q, "F", story.graph_block, pass_number=2, pass1_notes="Key facts noted.")
    assert "Notes:\nKey facts noted." in user_f2
    assert "treat the text as correct" in sys_f2


def test_parse_model_answer():
    # 1. Event ID
    raw = 'Thinking...\nTherefore the earlier event is:\n{"answer": "E1"}'
    ok, val = parse_model_answer(raw, "event_id")
    assert ok is True
    assert val == "E1"

    # 2. Integer
    raw_int = '```json\n{"answer": 14}\n```'
    ok, val = parse_model_answer(raw_int, "integer")
    assert ok is True
    assert val == 14

    # 3. ID List
    raw_list = 'Conclusion:\n{"answer": ["E3", "E2"]}'
    ok, val = parse_model_answer(raw_list, "id_list")
    assert ok is True
    assert val == ["E2", "E3"]  # sorted

    # 4. Value
    raw_val = 'Operational parameter:\n{"answer": "cryogenic_alpha"}'
    ok, val = parse_model_answer(raw_val, "value")
    assert ok is True
    assert val == "cryogenic_alpha"

    # 5. Fallback unparseable
    bad_raw = "I do not know the answer."
    ok, val = parse_model_answer(bad_raw, "integer")
    assert ok is False


def test_score_answer():
    # Event ID
    assert score_answer("E1", "E1", "event_id") is True
    assert score_answer("e1", "E1", "event_id") is True
    assert score_answer("E2", "E1", "event_id") is False

    # Integer
    assert score_answer(15, 15, "integer") is True
    assert score_answer("15", 15, "integer") is True
    assert score_answer(14, 15, "integer") is False

    # ID list
    assert score_answer(["E2", "E3"], ["E3", "E2"], "id_list") is True
    assert score_answer(["E2"], ["E2", "E3"], "id_list") is False

    # Value
    assert score_answer("cryogenic_alpha", "cryogenic_alpha", "value") is True
    assert score_answer("CRYOGENIC_ALPHA", "cryogenic_alpha", "value") is True
    assert score_answer("grid_shunted", "cryogenic_alpha", "value") is False

