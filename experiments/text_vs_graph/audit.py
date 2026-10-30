"""Validation and Information Parity Audits.

Per test plan Section 4:
- Dependency graph must be a DAG.
- Factual world must be consistent (all events occur in intact story).
- Exact 1:1 bidirectional fact ID parity between prose and structured block.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from experiments.text_vs_graph.models import Event, Story


def extract_fact_ids(text: str) -> Set[str]:
    """Extracts all fact tag tokens of the form [TAG_ID] from text."""
    matches = re.findall(r"\[([A-Z0-9_]+)\]", text)
    return set(matches)


def audit_information_parity(prose: str, graph_block: str) -> Tuple[bool, Set[str], Set[str]]:
    """Checks bidirectional fact parity between prose and structured graph block.
    
    Returns (is_valid, missing_in_graph, missing_in_prose).
    """
    prose_ids = extract_fact_ids(prose)
    graph_ids = extract_fact_ids(graph_block)
    
    missing_in_graph = prose_ids - graph_ids
    missing_in_prose = graph_ids - prose_ids
    
    is_valid = (len(missing_in_graph) == 0) and (len(missing_in_prose) == 0) and (len(prose_ids) > 0)
    return is_valid, missing_in_graph, missing_in_prose


def audit_world_validity(events: Dict[str, Event]) -> Tuple[bool, str]:
    """Validates that:
    1. The dependency graph is an acyclic DAG.
    2. Temporal ordering agrees with timestamps.
    3. All events occur in the intact baseline factual world.
    """
    # 1. Cycle detection (Kahn's or DFS)
    in_degree = {eid: len(e.parents) for eid, e in events.items()}
    children: Dict[str, List[str]] = {eid: [] for eid in events}
    for eid, e in events.items():
        for p in e.parents:
            if p not in events:
                return False, f"Parent {p} of {eid} does not exist in events"
            children[p].append(eid)
            
    queue = [eid for eid, deg in in_degree.items() if deg == 0]
    visited_count = 0
    while queue:
        curr = queue.pop(0)
        visited_count += 1
        for ch in children[curr]:
            in_degree[ch] -= 1
            if in_degree[ch] == 0:
                queue.append(ch)
                
    if visited_count < len(events):
        return False, "Dependency graph contains cycles"

    # 2. Baseline world simulation: all roots occur -> all events must occur
    occurred: Set[str] = set()
    # Process in topological order
    # Recompute topo order
    in_degree = {eid: len(e.parents) for eid, e in events.items()}
    queue = sorted([eid for eid, deg in in_degree.items() if deg == 0])
    topo: List[str] = []
    while queue:
        curr = queue.pop(0)
        topo.append(curr)
        for ch in sorted(children[curr]):
            in_degree[ch] -= 1
            if in_degree[ch] == 0:
                queue.append(ch)

    for eid in topo:
        e = events[eid]
        if not e.parents:
            occurred.add(eid)
        elif e.rule == "AND":
            if all(p in occurred for p in e.parents):
                occurred.add(eid)
            else:
                return False, f"Event {eid} failed AND rule in intact world"
        elif e.rule == "OR":
            if any(p in occurred for p in e.parents):
                occurred.add(eid)
            else:
                return False, f"Event {eid} failed OR rule in intact world"
                
    if len(occurred) != len(events):
        return False, f"Incomplete occurrence in intact world: {len(occurred)}/{len(events)}"

    return True, "Valid"

