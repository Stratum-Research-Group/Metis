"""Stage 1 Unit and Hand-Verification Test Suite.

Per test plan Section 12 (Run Order):
1. Build generator, key, scorer, parity and validity audits.
2. Unit-test the key by hand on 10 stories, including AND/OR,
   alternative-cause, and denied-root Q3 cases.
"""

from __future__ import annotations

import pytest
from typing import Dict, List

from experiments.text_vs_graph.audit import (
    audit_information_parity,
    audit_world_validity,
    extract_fact_ids,
)
from experiments.text_vs_graph.corruptor import corrupt_story_graph
from experiments.text_vs_graph.generator import generate_story
from experiments.text_vs_graph.models import Event
from experiments.text_vs_graph.solver import (
    solve_q1_ordering,
    solve_q2_arithmetic,
    solve_q3_counterfactual,
    solve_q4_conditional_rule,
)


class TestParityAndValidityAudits:
    def test_parity_positive(self):
        prose = "Event E1 [EV_E1] happened before E2 [EV_E2]. Depend: [DEP_E2]."
        graph = "EVENT E1 [EV_E1]\nEVENT E2 [EV_E2]\nDEP: [DEP_E2]"
        valid, missing_g, missing_p = audit_information_parity(prose, graph)
        assert valid is True
        assert len(missing_g) == 0
        assert len(missing_p) == 0

    def test_parity_negative_missing_in_graph(self):
        prose = "Event E1 [EV_E1] happened before E2 [EV_E2] [RULE_CR1]."
        graph = "EVENT E1 [EV_E1]\nEVENT E2 [EV_E2]"
        valid, missing_g, missing_p = audit_information_parity(prose, graph)
        assert valid is False
        assert missing_g == {"RULE_CR1"}

    def test_parity_negative_missing_in_prose(self):
        prose = "Event E1 [EV_E1]."
        graph = "EVENT E1 [EV_E1]\nEVENT E2 [EV_E2]"
        valid, missing_g, missing_p = audit_information_parity(prose, graph)
        assert valid is False
        assert missing_p == {"EV_E2"}

    def test_world_validity_cycle_detection(self):
        events = {
            "E1": Event(id="E1", name="A", timestamp=0, duration=1, parents=["E2"], rule="AND"),
            "E2": Event(id="E2", name="B", timestamp=2, duration=1, parents=["E1"], rule="AND"),
        }
        valid, msg = audit_world_validity(events)
        assert valid is False
        assert "cycles" in msg.lower()


class TestHandVerifiedTenStories:
    """10 Explicit hand-constructed stories covering AND/OR, alternative causes, denied roots."""

    def test_story_1_simple_linear_chain(self):
        """E1 (root) -> E2 (AND: [E1]) -> E3 (AND: [E2]). Deny E1 -> [E2, E3] fail."""
        events = {
            "E1": Event(id="E1", name="Ignition", timestamp=0, duration=2, parents=[], rule="ROOT"),
            "E2": Event(id="E2", name="Liftoff", timestamp=3, duration=2, parents=["E1"], rule="AND"),
            "E3": Event(id="E3", name="Stage Sep", timestamp=6, duration=2, parents=["E2"], rule="AND"),
        }
        failed, crit = solve_q3_counterfactual(events, "E1")
        assert failed == ["E2", "E3"]
        assert "E1" in crit and "E2" in crit and "E3" in crit

    def test_story_2_fork(self):
        """E1 (root) -> E2 (AND: [E1]), E1 -> E3 (AND: [E1]). Deny E1 -> [E2, E3] fail."""
        events = {
            "E1": Event(id="E1", name="Power Bus A", timestamp=0, duration=1, parents=[], rule="ROOT"),
            "E2": Event(id="E2", name="Telemetry", timestamp=2, duration=1, parents=["E1"], rule="AND"),
            "E3": Event(id="E3", name="Life Support", timestamp=2, duration=2, parents=["E1"], rule="AND"),
        }
        failed, crit = solve_q3_counterfactual(events, "E1")
        assert failed == ["E2", "E3"]

    def test_story_3_join_with_and(self):
        """E1 (root), E2 (root) -> E3 (AND: [E1, E2]). Deny E1 -> E3 fails (needs all parents)."""
        events = {
            "E1": Event(id="E1", name="Key Turn 1", timestamp=1, duration=1, parents=[], rule="ROOT"),
            "E2": Event(id="E2", name="Key Turn 2", timestamp=1, duration=1, parents=[], rule="ROOT"),
            "E3": Event(id="E3", name="Silo Launch", timestamp=3, duration=1, parents=["E1", "E2"], rule="AND"),
        }
        failed, crit = solve_q3_counterfactual(events, "E1")
        assert failed == ["E3"]

    def test_story_4_join_with_or_alternative_cause(self):
        """E1 (root), E2 (root) -> E3 (OR: [E1, E2]). Deny E1 -> E3 survives via E2! Failed: []."""
        events = {
            "E1": Event(id="E1", name="Solar Array", timestamp=0, duration=5, parents=[], rule="ROOT"),
            "E2": Event(id="E2", name="Battery Aux", timestamp=0, duration=5, parents=[], rule="ROOT"),
            "E3": Event(id="E3", name="Avionics Power", timestamp=6, duration=2, parents=["E1", "E2"], rule="OR"),
        }
        failed, crit = solve_q3_counterfactual(events, "E1")
        assert failed == []  # E3 still occurs because E2 is intact!

    def test_story_5_diamond_all_parents_lost(self):
        """E1 -> E2, E1 -> E3; E2 & E3 -> E4 (OR). Deny E1 -> E2 & E3 fail -> E4 has no survivors and fails."""
        events = {
            "E1": Event(id="E1", name="Main Steam Line", timestamp=0, duration=2, parents=[], rule="ROOT"),
            "E2": Event(id="E2", name="Turbine Alpha", timestamp=3, duration=2, parents=["E1"], rule="AND"),
            "E3": Event(id="E3", name="Turbine Beta", timestamp=3, duration=2, parents=["E1"], rule="AND"),
            "E4": Event(id="E4", name="Generator Output", timestamp=6, duration=2, parents=["E2", "E3"], rule="OR"),
        }
        failed, crit = solve_q3_counterfactual(events, "E1")
        assert failed == ["E2", "E3", "E4"]

    def test_story_6_diamond_with_independent_root_survivor(self):
        """E1 -> E2; E2 & E_indep -> E3 (OR). Deny E1 -> E2 fails, but E3 survives via E_indep."""
        events = {
            "E1": Event(id="E1", name="Coolant Line 1", timestamp=0, duration=2, parents=[], rule="ROOT"),
            "E2": Event(id="E2", name="Chiller Unit", timestamp=3, duration=2, parents=["E1"], rule="AND"),
            "E_indep": Event(id="E_indep", name="Emergency Ice Pack", timestamp=0, duration=10, parents=[], rule="ROOT"),
            "E3": Event(id="E3", name="Core Thermal Safe", timestamp=6, duration=2, parents=["E2", "E_indep"], rule="OR"),
        }
        failed, crit = solve_q3_counterfactual(events, "E1")
        assert failed == ["E2"]  # E3 survived!

    def test_story_7_denied_leaf_node(self):
        """Denying a leaf node that has zero children causes zero other events to fail."""
        events = {
            "E1": Event(id="E1", name="Sensor Boot", timestamp=0, duration=1, parents=[], rule="ROOT"),
            "E2": Event(id="E2", name="Data Log Entry", timestamp=2, duration=1, parents=["E1"], rule="AND"),
        }
        failed, crit = solve_q3_counterfactual(events, "E2")
        assert failed == []

    def test_story_8_three_layer_cascade(self):
        """E1 -> E2 (AND) -> E3 (AND) -> E4 (AND). Deny E2 -> [E3, E4] fail. E1 intact."""
        events = {
            "E1": Event(id="E1", name="Step 1", timestamp=0, duration=1, parents=[], rule="ROOT"),
            "E2": Event(id="E2", name="Step 2", timestamp=2, duration=1, parents=["E1"], rule="AND"),
            "E3": Event(id="E3", name="Step 3", timestamp=4, duration=1, parents=["E2"], rule="AND"),
            "E4": Event(id="E4", name="Step 4", timestamp=6, duration=1, parents=["E3"], rule="AND"),
        }
        failed, crit = solve_q3_counterfactual(events, "E2")
        assert failed == ["E3", "E4"]

    def test_story_9_multi_parent_and_single_denial(self):
        """E3 requires [E1, E2, E0] under AND. Only E2 denied -> E3 fails."""
        events = {
            "E0": Event(id="E0", name="Auth Signal", timestamp=0, duration=1, parents=[], rule="ROOT"),
            "E1": Event(id="E1", name="Pressure OK", timestamp=1, duration=1, parents=[], rule="ROOT"),
            "E2": Event(id="E2", name="Interlock Released", timestamp=2, duration=1, parents=[], rule="ROOT"),
            "E3": Event(id="E3", name="Ignition Gate", timestamp=4, duration=1, parents=["E0", "E1", "E2"], rule="AND"),
        }
        failed, crit = solve_q3_counterfactual(events, "E2")
        assert failed == ["E3"]

    def test_story_10_multi_parent_or_both_denied(self):
        """E3 requires [E1, E2] under OR. E1 and E2 both depend on E0 (AND). Deny E0 -> E1, E2, E3 all fail."""
        events = {
            "E0": Event(id="E0", name="Main Feed", timestamp=0, duration=1, parents=[], rule="ROOT"),
            "E1": Event(id="E1", name="Branch Left", timestamp=2, duration=1, parents=["E0"], rule="AND"),
            "E2": Event(id="E2", name="Branch Right", timestamp=2, duration=1, parents=["E0"], rule="AND"),
            "E3": Event(id="E3", name="Collector Basin", timestamp=4, duration=1, parents=["E1", "E2"], rule="OR"),
        }
        failed, crit = solve_q3_counterfactual(events, "E0")
        assert failed == ["E1", "E2", "E3"]


class TestSolversQ1Q2Q4:
    def test_q1_ordering(self):
        e_early = Event(id="E1", name="Early", timestamp=10, duration=2)
        e_late = Event(id="E2", name="Late", timestamp=20, duration=2)
        earlier, crit = solve_q1_ordering(e_early, e_late)
        assert earlier == "E1"
        assert crit == ["E1", "E2"]
        earlier_rev, _ = solve_q1_ordering(e_late, e_early)
        assert earlier_rev == "E1"

    def test_q2_arithmetic(self):
        ea = Event(id="E1", name="A", timestamp=10, duration=5)
        eb = Event(id="E2", name="B", timestamp=20, duration=8)
        tot, crit = solve_q2_arithmetic(ea, eb)
        assert tot == 13
        assert crit == ["E1", "E2"]

    def test_q4_conditional_rule(self):
        trig = Event(id="E1", name="Core Reactor Spinup", timestamp=5, duration=3)
        val, crit = solve_q4_conditional_rule(trig, "auxiliary_cooling", "cryogenic_alpha")
        assert val == "cryogenic_alpha"
        assert crit == ["E1"]


class TestCorruptorMechanics:
    def test_corruptor_level_zero(self):
        story = generate_story("story_test_c0", num_events=8, seed=42)
        record = corrupt_story_graph(story, level=0.0, seed=123)
        assert record.level == 0.0
        assert len(record.ops) == 0
        assert len(record.touched_original_nodes) == 0
        assert record.fake_node_count == 0
        assert record.corrupted_graph_block == story.graph_block

    def test_corruptor_level_thirty(self):
        story = generate_story("story_test_c30", num_events=10, seed=42)
        record = corrupt_story_graph(story, level=0.30, seed=123)
        assert record.level == 0.30
        # For n=10, round(0.3 * 10) = 3 nodes touched
        assert len(record.touched_original_nodes) == 3
        # Each touched node is distinct
        assert len(set(record.touched_original_nodes)) == 3
        # Fake node count = ceil(0.3 * 10) = 3
        assert record.fake_node_count == 3
        # Fake nodes must appear in corrupted block
        assert "FAKE_E1" in record.corrupted_graph_block
        # Corrupted graph block differs from original
        assert record.corrupted_graph_block != story.graph_block


class TestEndToEndGenerator:
    def test_generate_ten_disjoint_stories(self):
        for seed in range(10):
            story = generate_story(f"story_gen_{seed}", num_events=8, seed=seed)
            assert len(story.events) == 8
            assert len(story.questions) == 4
            
            # Check question types
            qtypes = [q.qtype for q in story.questions]
            assert qtypes == ["Q1", "Q2", "Q3", "Q4"]
            
            # Verify parity
            valid_p, missing_g, missing_p = audit_information_parity(story.prose, story.graph_block)
            assert valid_p is True, f"Seed {seed} failed parity: {missing_g}, {missing_p}"
            
            # Verify world validity
            valid_w, msg = audit_world_validity(story.events)
            assert valid_w is True, f"Seed {seed} failed world validity: {msg}"

