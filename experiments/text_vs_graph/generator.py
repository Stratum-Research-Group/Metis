"""Synthetic story, graph block, and question generator.

Per test plan Section 4:
Fact set per story (6-12 events):
- events with timestamps and durations
- TEMPORAL edges: 'before' precedence, from timestamps
- DEPENDENCY edges: parent list and rule (AND/OR). Roots always occur.
- conditional rules for Q4
- 2-3 distractor facts
- 100% bidirectional fact-tag parity between prose and structured block.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from experiments.text_vs_graph.audit import (
    audit_information_parity,
    audit_world_validity,
)
from experiments.text_vs_graph.models import (
    ConditionalRule,
    DistractorFact,
    Event,
    Fact,
    Question,
    Story,
)
from experiments.text_vs_graph.solver import (
    solve_q1_ordering,
    solve_q2_arithmetic,
    solve_q3_counterfactual,
    solve_q4_conditional_rule,
)

EVENT_NAMES_POOL = [
    "Core Reactor Spinup",
    "Auxiliary Turbine Sync",
    "Coolant Valve Depressurization",
    "Telemetry Relay Handshake",
    "Inertial Damper Calibration",
    "Cryogenic Manifold Flush",
    "Subspace Beacon Ping",
    "Power Distribution Routing",
    "Plasma Exhaust Venting",
    "Thermal Radiator Deployment",
    "Quantum Registry Sync",
    "Primary Thruster Ignition",
]

TARGET_PROPERTIES_POOL = [
    ("auxiliary_cooling", "cryogenic_alpha"),
    ("secondary_power_bus", "grid_shunted"),
    ("safety_containment_field", "level_4_engaged"),
    ("radiation_shielding", "polarized_maximum"),
    ("telemetry_downlink", "high_gain_active"),
]

DISTRACTOR_METADATA_POOL = [
    ("Sector 7 Habitation Ring", "atmospheric pressure", "101.3 kPa"),
    ("Cargo Hold Beta", "inventory reconciliation", "verified complete"),
    ("Sub-deck Maintenance Log", "shift inspection status", "certified normal"),
    ("Orbital Beacon Transponder", "firmware checksum", "valid 0x7F2A"),
    ("Environmental Scrubbers", "particulate filter efficiency", "99.8 percent"),
]


def generate_story(
    story_id: str,
    num_events: int = 8,
    seed: int = 42,
) -> Story:
    """Generates a valid synthetic story, questions, and parity-audited renderings."""
    rng = random.Random(seed)
    
    assert 6 <= num_events <= 12, "num_events must be between 6 and 12"
    
    # 1. Select event names and assign timestamps in topological order
    selected_names = rng.sample(EVENT_NAMES_POOL, num_events)
    
    events: Dict[str, Event] = {}
    current_time = rng.randint(2, 6)
    
    # Structure layers: Layer 0 (roots), Layer 1, Layer 2
    # Ensure at least 2 roots and multiple layers
    num_roots = max(2, min(3, num_events // 3))
    
    layer_roots: List[str] = []
    layer_intermediates: List[str] = []
    layer_leaves: List[str] = []
    
    for i in range(num_events):
        eid = f"E{i+1}"
        name = selected_names[i]
        duration = rng.randint(2, 6)
        start_hour = current_time
        current_time += duration + rng.randint(1, 4)
        
        events[eid] = Event(
            id=eid,
            name=name,
            timestamp=start_hour,
            duration=duration,
            parents=[],
            rule="ROOT",
        )
        
        if i < num_roots:
            layer_roots.append(eid)
        elif i < num_events - 2:
            layer_intermediates.append(eid)
        else:
            layer_leaves.append(eid)
            
    # Assign dependencies to intermediates (from roots or earlier intermediates)
    for eid in layer_intermediates:
        # Choose 1 or 2 parents from earlier nodes
        earlier_nodes = [k for k in events.keys() if int(k[1:]) < int(eid[1:])]
        parent_count = rng.choice([1, 2]) if len(earlier_nodes) >= 2 else 1
        chosen_parents = sorted(rng.sample(earlier_nodes, parent_count))
        rule = rng.choice(["AND", "OR"]) if parent_count > 1 else "AND"
        
        events[eid].parents = chosen_parents
        events[eid].rule = rule
        
    # Assign dependencies to leaves (ensure at least one AND and one OR in the story)
    has_and = any(e.rule == "AND" for e in events.values() if e.parents)
    has_or = any(e.rule == "OR" for e in events.values() if e.parents)
    
    for idx, eid in enumerate(layer_leaves):
        earlier_nodes = layer_roots + layer_intermediates
        parent_count = min(2, len(earlier_nodes))
        chosen_parents = sorted(rng.sample(earlier_nodes, parent_count))
        
        if not has_or:
            rule = "OR"
            has_or = True
        elif not has_and:
            rule = "AND"
            has_and = True
        else:
            rule = rng.choice(["AND", "OR"])
            
        events[eid].parents = chosen_parents
        events[eid].rule = rule

    # 2. Temporal edges (sample key precedence relations)
    temporal_pairs: List[Tuple[str, str]] = []
    sorted_eids = sorted(events.keys(), key=lambda k: events[k].timestamp)
    for i in range(len(sorted_eids) - 1):
        temporal_pairs.append((sorted_eids[i], sorted_eids[i+1]))
        
    # 3. Conditional rule for Q4
    # Pick one root or intermediate event
    q4_trigger = events[layer_roots[0]]
    q4_prop, q4_val = rng.choice(TARGET_PROPERTIES_POOL)
    cond_rule = ConditionalRule(
        id="CR1",
        trigger_event_id=q4_trigger.id,
        target_property=q4_prop,
        target_value=q4_val,
    )
    
    # 4. Distractors (2 facts)
    chosen_distractors = rng.sample(DISTRACTOR_METADATA_POOL, 2)
    distractors = [
        DistractorFact(id=f"D{i+1}", subject=d[0], property_name=d[1], value=d[2])
        for i, d in enumerate(chosen_distractors)
    ]
    
    # 5. Build facts list with strict 1:1 parity tags
    facts: List[Fact] = []
    
    # Events facts
    for eid, ev in sorted(events.items()):
        tag = f"EV_{eid}"
        prose_line = (
            f"Event {ev.id} ({ev.name}) started at hour {ev.timestamp} "
            f"with a duration of {ev.duration} hours [{tag}]."
        )
        graph_line = (
            f"EVENT {ev.id}: {ev.name} [start_hour={ev.timestamp}, duration_hours={ev.duration}] [{tag}]"
        )
        facts.append(Fact(id=tag, kind="event", details={"id": ev.id, "timestamp": ev.timestamp, "duration": ev.duration}, prose_sentence=prose_line, graph_line=graph_line))
        
    # Temporal facts
    for a_id, b_id in temporal_pairs:
        tag = f"TEMP_{a_id}_{b_id}"
        prose_line = f"Operationally, Event {a_id} finished before Event {b_id} began [{tag}]."
        graph_line = f"TEMPORAL: {a_id} precedes {b_id} [{tag}]"
        facts.append(Fact(id=tag, kind="temporal", details={"before": a_id, "after": b_id}, prose_sentence=prose_line, graph_line=graph_line))
        
    # Dependency facts
    for eid, ev in sorted(events.items()):
        tag = f"DEP_{eid}"
        if ev.parents:
            p_str = ", ".join(ev.parents)
            prose_line = f"By system design, Event {ev.id} requires parent events ({p_str}) under rule {ev.rule} [{tag}]."
            graph_line = f"DEPENDENCY: {ev.id} requires ({p_str}) with rule {ev.rule} [{tag}]"
        else:
            prose_line = f"Event {ev.id} is an autonomous root event and always occurs [{tag}]."
            graph_line = f"DEPENDENCY: {ev.id} is a ROOT event (always occurs) [{tag}]"
        facts.append(Fact(id=tag, kind="dependency", details={"id": ev.id, "parents": ev.parents, "rule": ev.rule}, prose_sentence=prose_line, graph_line=graph_line))

    # Conditional rule fact
    tag = f"RULE_{cond_rule.id}"
    prose_line = f"Safety protocol dictates: IF Event {cond_rule.trigger_event_id} occurs, THEN {cond_rule.target_property} is set to {cond_rule.target_value} [{tag}]."
    graph_line = f"RULE {cond_rule.id}: IF {cond_rule.trigger_event_id} occurs THEN {cond_rule.target_property} = {cond_rule.target_value} [{tag}]"
    facts.append(Fact(id=tag, kind="conditional", details={"trigger": cond_rule.trigger_event_id, "prop": cond_rule.target_property, "val": cond_rule.target_value}, prose_sentence=prose_line, graph_line=graph_line))
    
    # Distractor facts
    for d in distractors:
        tag = f"DIST_{d.id}"
        prose_line = f"General station telemetry notes that {d.subject} recorded {d.property_name} as {d.value} [{tag}]."
        graph_line = f"METADATA {d.id}: {d.subject} - {d.property_name} is {d.value} [{tag}]"
        facts.append(Fact(id=tag, kind="distractor", details={"subject": d.subject, "prop": d.property_name, "val": d.value}, prose_sentence=prose_line, graph_line=graph_line))
        
    # 6. Assemble Prose and Structured Graph Block
    # Group prose into coherent narrative paragraphs
    prose_paragraphs = [
        "MISSION TIMELINE AND LOG:\n",
        " ".join([f.prose_sentence for f in facts if f.kind == "event"]),
        "\nSEQUENCE RELATIONS:\n",
        " ".join([f.prose_sentence for f in facts if f.kind == "temporal"]),
        "\nDEPENDENCY AND GOVERNANCE RULES:\n",
        " ".join([f.prose_sentence for f in facts if f.kind == "dependency"]),
        "\nOPERATIONAL STANDING ORDERS & TELEMETRY:\n",
        " ".join([f.prose_sentence for f in facts if f.kind in ("conditional", "distractor")]),
    ]
    prose_text = "\n".join(prose_paragraphs)
    
    graph_lines = [
        "=== STRUCTURED FACT GRAPH ===",
        "# Events and Timestamps:",
        *[f.graph_line for f in facts if f.kind == "event"],
        "\n# Temporal Precedence:",
        *[f.graph_line for f in facts if f.kind == "temporal"],
        "\n# Dependency Rules:",
        *[f.graph_line for f in facts if f.kind == "dependency"],
        "\n# Conditional Operational Rules:",
        *[f.graph_line for f in facts if f.kind == "conditional"],
        "\n# System Metadata:",
        *[f.graph_line for f in facts if f.kind == "distractor"],
    ]
    graph_block_text = "\n".join(graph_lines)
    
    # 7. Audits: validity and parity
    valid_world, msg = audit_world_validity(events)
    assert valid_world, f"Generated world invalid: {msg}"
    
    valid_parity, missing_g, missing_p = audit_information_parity(prose_text, graph_block_text)
    assert valid_parity, f"Parity audit failure: missing_in_graph={missing_g}, missing_in_prose={missing_p}"
    
    # 8. Create questions Q1, Q2, Q3, Q4
    # Q1: Pick two events with different timestamps
    cand_pairs = [(a, b) for a in events.values() for b in events.values() if a.id < b.id and a.timestamp != b.timestamp]
    q1_a, q1_b = rng.choice(cand_pairs)
    q1_ans, q1_crit = solve_q1_ordering(q1_a, q1_b)
    q1 = Question(
        qid=f"{story_id}_q1",
        qtype="Q1",
        prompt=f"Which event started earlier: {q1_a.id} or {q1_b.id}? Answer with the event ID (e.g. \"{q1_a.id}\").",
        answer_format="event_id",
        gold_answer=q1_ans,
        critical_event_ids=q1_crit,
        critical_fact_ids=[f"EV_{q1_a.id}", f"EV_{q1_b.id}"],
    )
    
    # Q2: Pick two events for combined duration arithmetic
    cand_dur_pairs = [(a, b) for a in events.values() for b in events.values() if a.id < b.id]
    q2_a, q2_b = rng.choice(cand_dur_pairs)
    q2_ans, q2_crit = solve_q2_arithmetic(q2_a, q2_b)
    q2 = Question(
        qid=f"{story_id}_q2",
        qtype="Q2",
        prompt=f"What is the total combined duration in hours of {q2_a.id} and {q2_b.id}? Answer with a single integer.",
        answer_format="integer",
        gold_answer=q2_ans,
        critical_event_ids=q2_crit,
        critical_fact_ids=[f"EV_{q2_a.id}", f"EV_{q2_b.id}"],
    )
    
    # Q3: Target X with children for counterfactual propagation
    # Find a root or intermediate node that has at least one child
    nodes_with_children = [
        eid for eid in events
        if any(eid in other.parents for other in events.values())
    ]
    target_x_id = rng.choice(nodes_with_children) if nodes_with_children else layer_roots[0]
    q3_ans, q3_crit = solve_q3_counterfactual(events, target_x_id)
    q3 = Question(
        qid=f"{story_id}_q3",
        qtype="Q3",
        prompt=(
            f"Suppose {target_x_id} had not happened. Applying the dependency rules stated "
            f"(an event with rule AND occurs only if all its parents occur; with rule OR it occurs "
            f"if at least one parent occurs; events with no parents always occur unless denied), "
            f"which events (excluding {target_x_id}) would no longer occur? "
            f"Answer with a sorted JSON list of event IDs, e.g. [\"E2\", \"E4\"], or [] if none."
        ),
        answer_format="id_list",
        gold_answer=q3_ans,
        critical_event_ids=q3_crit,
        critical_fact_ids=[f"DEP_{eid}" for eid in q3_crit],
    )
    
    # Q4: Conditional rule application
    q4_ans, q4_crit = solve_q4_conditional_rule(q4_trigger, cond_rule.target_property, cond_rule.target_value)
    q4 = Question(
        qid=f"{story_id}_q4",
        qtype="Q4",
        prompt=(
            f"Applying the stated operational conditional rules, if Event {cond_rule.trigger_event_id} occurs, "
            f"what value is assigned to {cond_rule.target_property}? Answer with the exact value string."
        ),
        answer_format="value",
        gold_answer=q4_ans,
        critical_event_ids=q4_crit,
        critical_fact_ids=[f"RULE_{cond_rule.id}"],
    )
    
    return Story(
        story_id=story_id,
        title=f"Station Incident Log {story_id}",
        events=events,
        conditional_rules=[cond_rule],
        distractors=distractors,
        facts=facts,
        prose=prose_text,
        graph_block=graph_block_text,
        questions=[q1, q2, q3, q4],
    )

