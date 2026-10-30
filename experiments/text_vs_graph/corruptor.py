"""Deterministic graph corruption engine.

Per test plan Section 4:
Operators: drop-node, rewire-dependency, flip-AND/OR, shift-time, add-fake-event.
Corruption level = (number of ORIGINAL nodes touched) / (original node count n).
Each original node is touched at most once.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from experiments.text_vs_graph.models import (
    CorruptionRecord,
    Event,
    Fact,
    Story,
)


def corrupt_story_graph(
    story: Story, level: float, seed: int
) -> CorruptionRecord:
    """Applies graph corruption to the structured block of a story.
    
    Level: fraction of original nodes to touch (0.0 .. 1.0).
    Level 0.0 applies zero operators and adds zero fake nodes.
    """
    if level <= 0.0:
        return CorruptionRecord(
            level=0.0,
            ops=[],
            touched_original_nodes=[],
            fake_node_count=0,
            corrupted_graph_block=story.graph_block,
        )

    rng = random.Random(seed)
    orig_events = {eid: copy.deepcopy(e) for eid, e in story.events.items()}
    all_eids = sorted(orig_events.keys())
    n = len(all_eids)
    
    k = max(1, round(level * n))
    k = min(k, n)
    
    # Sample k distinct nodes to touch
    touched_nodes = sorted(rng.sample(all_eids, k))
    
    ops_log: List[Dict[str, Any]] = []
    dropped_nodes: Set[str] = set()
    
    for eid in touched_nodes:
        # Choose operator:
        # If node has parents, can rewire or flip.
        # If node is root, can shift time, drop, or flip.
        ev = orig_events[eid]
        possible_ops = ["shift-time", "drop-node"]
        if ev.parents:
            possible_ops.extend(["rewire-dependency", "flip-AND/OR"])
        else:
            possible_ops.append("flip-AND/OR")  # Can add a parent with AND/OR
            
        op_choice = rng.choice(possible_ops)
        
        if op_choice == "drop-node":
            dropped_nodes.add(eid)
            ops_log.append({
                "op": "drop-node",
                "node": eid,
                "description": f"Dropped node {eid} from structured graph block",
            })
            
        elif op_choice == "shift-time":
            delta = rng.choice([-12, -8, -6, 6, 8, 12, 16])
            old_time = ev.timestamp
            new_time = max(0, old_time + delta)
            if new_time == old_time:
                new_time = old_time + 10
            ev.timestamp = new_time
            ops_log.append({
                "op": "shift-time",
                "node": eid,
                "old_timestamp": old_time,
                "new_timestamp": new_time,
                "description": f"Shifted timestamp of {eid} from {old_time} to {new_time}",
            })
            
        elif op_choice == "flip-AND/OR":
            old_rule = ev.rule
            if old_rule == "AND":
                ev.rule = "OR"
            elif old_rule == "OR":
                ev.rule = "AND"
            else:
                # ROOT node flipped to requiring an arbitrary other node as parent
                other_candidates = [other for other in all_eids if other != eid]
                if other_candidates:
                    p = rng.choice(other_candidates)
                    ev.parents = [p]
                    ev.rule = "AND"
            ops_log.append({
                "op": "flip-AND/OR",
                "node": eid,
                "old_rule": old_rule,
                "new_rule": ev.rule,
                "description": f"Flipped rule of {eid} from {old_rule} to {ev.rule}",
            })
            
        elif op_choice == "rewire-dependency":
            other_candidates = [other for other in all_eids if other != eid and other not in dropped_nodes]
            if other_candidates:
                new_parent = rng.choice(other_candidates)
                old_parents = list(ev.parents)
                ev.parents = [new_parent]
                ops_log.append({
                    "op": "rewire-dependency",
                    "node": eid,
                    "old_parents": old_parents,
                    "new_parents": ev.parents,
                    "description": f"Rewired parents of {eid} from {old_parents} to {ev.parents}",
                })
            else:
                # Fallback to shift-time
                ev.timestamp += 10
                ops_log.append({
                    "op": "shift-time",
                    "node": eid,
                    "description": f"Shifted timestamp of {eid} by +10h (fallback for rewire)",
                })

    # Fake events (added as separate count, kept out of denominator)
    fake_node_count = math.ceil(level * n)
    fake_events: List[Dict[str, Any]] = []
    for i in range(fake_node_count):
        fake_id = f"FAKE_E{i+1}"
        fake_time = rng.randint(5, 80)
        fake_dur = rng.randint(2, 8)
        fake_events.append({
            "id": fake_id,
            "name": f"Anomalous Secondary Sequence {i+1}",
            "timestamp": fake_time,
            "duration": fake_dur,
            "parents": [],
            "rule": "ROOT",
        })
        ops_log.append({
            "op": "add-fake-event",
            "node": fake_id,
            "description": f"Inserted synthetic fake event {fake_id}",
        })

    # Render corrupted graph block
    lines: List[str] = ["=== STRUCTURED FACT GRAPH ==="]
    
    # 1. Events and timestamps
    lines.append("# Events:")
    for eid, ev in sorted(orig_events.items()):
        if eid in dropped_nodes:
            continue
        lines.append(f"EVENT {ev.id}: {ev.name} [start_hour={ev.timestamp}, duration_hours={ev.duration}]")
    for fe in fake_events:
        lines.append(f"EVENT {fe['id']}: {fe['name']} [start_hour={fe['timestamp']}, duration_hours={fe['duration']}]")

    # 2. Dependencies
    lines.append("\n# Dependencies:")
    for eid, ev in sorted(orig_events.items()):
        if eid in dropped_nodes:
            continue
        # Clean parents: if a parent was dropped, omit it
        active_parents = [p for p in ev.parents if p not in dropped_nodes]
        if active_parents:
            parents_str = ", ".join(active_parents)
            lines.append(f"DEPENDENCY: {ev.id} requires ({parents_str}) with rule {ev.rule}")
        else:
            lines.append(f"DEPENDENCY: {ev.id} is a ROOT event (always occurs)")

    # 3. Conditional rules
    lines.append("\n# Conditional Rules:")
    for cr in story.conditional_rules:
        if cr.trigger_event_id not in dropped_nodes:
            lines.append(f"RULE {cr.id}: IF {cr.trigger_event_id} occurs THEN {cr.target_property} = {cr.target_value}")

    # 4. Distractors
    lines.append("\n# System Metadata:")
    for d in story.distractors:
        lines.append(f"METADATA {d.id}: {d.subject} - {d.property_name} is {d.value}")

    corrupted_graph_str = "\n".join(lines)
    
    return CorruptionRecord(
        level=level,
        ops=ops_log,
        touched_original_nodes=touched_nodes,
        fake_node_count=fake_node_count,
        corrupted_graph_block=corrupted_graph_str,
    )

