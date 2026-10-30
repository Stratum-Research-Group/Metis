"""
Unit tests for DeterministicSpanFilter (Tier 1 of Zero-Hallucination Pipeline).
"""

import pytest
from stoma.filters import DeterministicSpanFilter


def test_verify_event_node_grounded():
    source = "Event E2 (Thermal Radiator Deployment) started at hour 12 with a duration of 3 hours."
    node = {
        "id": "E2",
        "name": "Thermal Radiator Deployment",
        "timestamp": 12,
        "duration": 3,
    }
    is_valid, msg = DeterministicSpanFilter.verify_event_node(node, source)
    assert is_valid is True
    assert msg == "Grounded"


def test_verify_event_node_shifted_timestamp_rejected():
    source = "Event E2 (Thermal Radiator Deployment) started at hour 12 with a duration of 3 hours."
    corrupted_node = {
        "id": "E2",
        "name": "Thermal Radiator Deployment",
        "timestamp": 18,  # Shifted timestamp!
        "duration": 3,
    }
    is_valid, msg = DeterministicSpanFilter.verify_event_node(corrupted_node, source)
    assert is_valid is False
    assert "lacks textual span support" in msg


def test_verify_event_node_fake_entity_rejected():
    source = "Event E2 (Thermal Radiator Deployment) started at hour 12."
    fake_node = {
        "id": "FAKE_E1",
        "name": "Anomalous Secondary Sequence",
        "timestamp": 12,
    }
    is_valid, msg = DeterministicSpanFilter.verify_event_node(fake_node, source)
    assert is_valid is False
    assert "no span grounding" in msg


def test_verify_edge_grounded_vs_unsupported():
    source = "By system design, Event E3 requires parent events (E1, E2) under rule AND."
    # E1 and E3 co-occur
    ok, msg = DeterministicSpanFilter.verify_edge("E1", "E3", source)
    assert ok is True

    # Missing node
    ok_bad, msg_bad = DeterministicSpanFilter.verify_edge("E1", "E99", source)
    assert ok_bad is False
    assert "Target node 'E99' not found" in msg_bad


def test_filter_graph_block_consistency_ratio():
    source = (
        "Event E1 (Reactor Spinup) started at hour 4 with duration 2 hours. "
        "Event E2 (Turbine Sync) started at hour 10 with duration 4 hours. "
        "Event E3 requires E1 and E2."
    )
    nodes = [
        {"id": "E1", "name": "Reactor Spinup", "timestamp": 4, "duration": 2},
        {"id": "E2", "name": "Turbine Sync", "timestamp": 10, "duration": 4},
        {"id": "E2_CORRUPTED", "name": "Turbine Sync", "timestamp": 99, "duration": 4},
        {"id": "FAKE_E1", "name": "Fake Node", "timestamp": 5, "duration": 1},
    ]
    edges = [
        {"src": "E1", "dst": "E2"},
        {"src": "E1", "dst": "FAKE_E1"},
    ]

    filter_engine = DeterministicSpanFilter()
    v_nodes, v_edges, ratio = filter_engine.filter_graph_block(nodes, edges, source)

    assert len(v_nodes) == 2
    assert {n["id"] for n in v_nodes} == {"E1", "E2"}
    assert len(v_edges) == 1
    assert v_edges[0] == {"src": "E1", "dst": "E2"}
    assert ratio == 3 / 6  # 3 verified out of 6 extracted = 0.50

