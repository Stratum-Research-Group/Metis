"""Tests for VRDUTask and Visually-Rich Document Understanding extraction benchmark."""

import json
from pathlib import Path
import pytest

from dokime.tasks.protocol import Query, Task
from dokime.tasks.registry import discover_tasks, get_task, list_task_info
from dokime.tasks.vrdu import (
    VRDUTask,
    _extract_json_from_text,
    _normalize_date,
    _normalize_string,
    _values_match,
)


def test_vrdu_discovery_and_registry():
    """Verify VRDUTask is properly registered in BUILTIN_TASKS and discoverable."""
    tasks = discover_tasks()
    assert "vrdu_document_extraction" in tasks
    task = get_task("vrdu_document_extraction")
    assert isinstance(task, VRDUTask)
    assert task.task_id == "vrdu_document_extraction"

    info_list = list_task_info()
    vrdu_info = next((item for item in info_list if item["task_id"] == "vrdu_document_extraction"), None)
    assert vrdu_info is not None
    assert vrdu_info["items"] == 20


def test_vrdu_dataset_load():
    """Verify all 20 benchmark document queries are loaded from JSON."""
    task = VRDUTask()
    queries = task.load()
    assert len(queries) == 20

    for q in queries:
        assert isinstance(q, Query)
        assert q.id.startswith("vrdu_")
        assert len(q.prompt) > 20
        assert "target_fields" in q.metadata
        assert len(q.metadata["target_fields"]) > 0


def test_json_extraction_helpers():
    """Test robust extraction of JSON from markdown blocks, raw text, and embedded strings."""
    # 1. Raw JSON
    raw = '{"order_number": "PO-100", "total": 500.0}'
    assert _extract_json_from_text(raw) == {"order_number": "PO-100", "total": 500.0}

    # 2. Markdown fenced JSON
    markdown_wrapped = 'Here is the extracted data:\n```json\n{"invoice": "INV-01", "amount": 1200}\n```\nHope this helps!'
    assert _extract_json_from_text(markdown_wrapped) == {"invoice": "INV-01", "amount": 1200}

    # 3. Outer braces in conversational text
    conversational = 'Based on the document, the result is: {"status": "APPROVED", "code": 200}. End of transmission.'
    assert _extract_json_from_text(conversational) == {"status": "APPROVED", "code": 200}

    # 4. Invalid or empty
    assert _extract_json_from_text("") is None
    assert _extract_json_from_text("No JSON here whatsoever") is None


def test_value_matching_normalization():
    """Test flexible normalization across numbers, dates, strings, and entities."""
    # Numeric tolerance and string currency stripping
    assert _values_match("$14,200,000.00", 14200000.0) is True
    assert _values_match("54,250,000.00", 54250000.0) is True
    assert _values_match(54250000, 54250000.0) is True
    assert _values_match(100.0, 200.0) is False

    # Date normalization
    assert _normalize_date("October 14, 2026") == "2026-10-14"
    assert _normalize_date("14 October 2026") == "2026-10-14"
    assert _values_match("October 14, 2026", "2026-10-14") is True

    # String normalization & entity containment
    assert _values_match("ASML Netherlands B.V.", "asml netherlands b.v.") is True
    assert _values_match("Global Foundries Fab-21, Phoenix AZ, USA", "Global Foundries Fab-21") is True
    assert _values_match("Carl Zeiss SMT GmbH (Oberkochen, Germany)", "Carl Zeiss SMT GmbH") is True

    # Lists of dicts
    target_items = [
        {"description": "EUV Collector Optics", "quantity": 2, "total": 45200000.0},
        {"description": "High-NA Stage", "quantity": 1, "total": 8400000.0},
    ]
    pred_items = [
        {"description": "EUV Collector Optics", "quantity": 2, "total": 45200000.0},
        {"description": "High-NA Stage", "quantity": 1, "total": 8400000.0},
    ]
    assert _values_match(pred_items, target_items) is True


def test_vrdu_score_exact_match():
    """Verify perfect extraction achieves 1.0 F1 score."""
    task = VRDUTask()
    queries = task.load()
    first_query = queries[0]
    target_fields = first_query.metadata["target_fields"]

    response = json.dumps(target_fields)
    score = task.score(first_query, response)
    assert score == 1.0

    detailed = task.evaluate_extraction(first_query, response)
    assert detailed["f1"] == 1.0
    assert detailed["precision"] == 1.0
    assert detailed["recall"] == 1.0
    assert all(detailed["field_matches"].values())


def test_vrdu_score_partial_and_zero_match():
    """Verify partial extraction gets proportional F1 score and garbage gets 0.0."""
    task = VRDUTask()
    queries = task.load()
    first_query = queries[0]
    target_fields = first_query.metadata["target_fields"]

    # Partial: only 2 fields correct out of 9
    partial_pred = {
        "order_number": target_fields["order_number"],
        "date": target_fields["date"],
    }
    score = task.score(first_query, json.dumps(partial_pred))
    assert 0.0 < score < 1.0

    detailed = task.evaluate_extraction(first_query, json.dumps(partial_pred))
    assert detailed["tp"] == 2
    assert detailed["fn"] == 7
    assert detailed["precision"] == 1.0  # no wrong fields predicted, but incomplete
    assert detailed["recall"] == round(2 / 9, 4)

    # Garbage / invalid response
    zero_score = task.score(first_query, "I cannot extract this document.")
    assert zero_score == 0.0
