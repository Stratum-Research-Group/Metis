"""Unit tests for Phase 2 extractor and fidelity analyzer."""

import pytest
from experiments.text_vs_graph.extractor import (
    analyze_extraction_fidelity,
    parse_extracted_events,
)


def test_parse_extracted_events():
    graph_text = """
=== STRUCTURED FACT GRAPH ===
# Events and Timestamps:
EVENT E1: name=Thermal Sensor Calibration, start_hour=10, duration_hours=4
EVENT E2: name=Coolant Loop Flush, start_hour=14, duration_hours=2
"""
    events = parse_extracted_events(graph_text)
    assert len(events) == 2
    assert "E1" in events
    assert events["E1"]["start_hour"] == 10
    assert events["E1"]["duration_hours"] == 4
    assert events["E2"]["start_hour"] == 14


def test_analyze_extraction_fidelity_perfect():
    graph_text = """
EVENT E1: name=Thermal Sensor Calibration, start_hour=10, duration_hours=4
EVENT E2: name=Coolant Loop Flush, start_hour=14, duration_hours=2
"""
    metrics = analyze_extraction_fidelity(graph_text, graph_text)
    assert metrics.gold_event_count == 2
    assert metrics.extracted_event_count == 2
    assert len(metrics.missing_events) == 0
    assert len(metrics.hallucinated_events) == 0
    assert len(metrics.timestamp_mismatches) == 0
    assert metrics.overall_fidelity == 1.0


def test_analyze_extraction_fidelity_with_errors():
    gold = """
EVENT E1: name=Thermal Sensor Calibration, start_hour=10, duration_hours=4
EVENT E2: name=Coolant Loop Flush, start_hour=14, duration_hours=2
EVENT E3: name=Reactor Vent, start_hour=16, duration_hours=3
"""
    # Extracted has shifted E1 timestamp (12 instead of 10), missing E3, and hallucinated E4
    extracted = """
EVENT E1: name=Thermal Sensor Calibration, start_hour=12, duration_hours=4
EVENT E2: name=Coolant Loop Flush, start_hour=14, duration_hours=2
EVENT E4: name=Auxiliary Purge, start_hour=20, duration_hours=1
"""
    metrics = analyze_extraction_fidelity(gold, extracted)
    assert metrics.gold_event_count == 3
    assert metrics.extracted_event_count == 3
    assert "E3" in metrics.missing_events
    assert "E4" in metrics.hallucinated_events
    assert len(metrics.timestamp_mismatches) == 1
    assert metrics.timestamp_mismatches[0] == ("E1", 10, 12)
    assert metrics.overall_fidelity < 1.0
