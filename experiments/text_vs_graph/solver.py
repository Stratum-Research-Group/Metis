"""Programmatic solvers and ground-truth calculators.

Per test plan Section 4:
No LLM grades or computes any answer.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
from experiments.text_vs_graph.models import Event, Question, Story


def solve_q1_ordering(event_a: Event, event_b: Event) -> Tuple[str, List[str]]:
    """Q1 ordering: which of X, Y is earlier?
    
    Returns (earlier_event_id, critical_event_ids).
    """
    if event_a.timestamp < event_b.timestamp:
        earlier = event_a.id
    elif event_b.timestamp < event_a.timestamp:
        earlier = event_b.id
    else:
        # Tie breaker by duration or id
        earlier = event_a.id if event_a.id < event_b.id else event_b.id
    return earlier, [event_a.id, event_b.id]


def solve_q2_arithmetic(event_a: Event, event_b: Event) -> Tuple[int, List[str]]:
    """Q2 duration arithmetic: What is the total combined duration in hours of X and Y?
    
    Returns (total_duration, critical_event_ids).
    """
    total = event_a.duration + event_b.duration
    return total, [event_a.id, event_b.id]


def solve_q3_counterfactual(
    events: Dict[str, Event], denied_event_id: str
) -> Tuple[List[str], List[str]]:
    """Q3 counterfactual: Suppose X had not happened.
    
    Applying the dependency rules stated, which events (excluding X)
    would no longer occur?
    
    Key: remove X, then propagate in topological order:
    - Root events (no parents) always occur, UNLESS denied.
    - AND rule: occurs iff ALL listed parents occur.
    - OR rule: occurs iff AT LEAST ONE listed parent occurs.
    
    Returns (sorted_failed_event_ids, critical_event_ids).
    """
    # 1. Topological sort
    # In-degree based on parent relations
    children_map: Dict[str, List[str]] = {eid: [] for eid in events}
    parent_count: Dict[str, int] = {eid: len(e.parents) for eid, e in events.items()}
    
    for eid, e in events.items():
        for p in e.parents:
            if p in children_map:
                children_map[p].append(eid)
                
    # Kahn's algorithm
    queue = [eid for eid, cnt in parent_count.items() if cnt == 0]
    topo_order: List[str] = []
    
    # Sort initial roots deterministically
    queue.sort()
    
    temp_in_degree = dict(parent_count)
    while queue:
        curr = queue.pop(0)
        topo_order.append(curr)
        for child in sorted(children_map[curr]):
            temp_in_degree[child] -= 1
            if temp_in_degree[child] == 0:
                queue.append(child)
                
    # If not all nodes processed, fallback to sorted keys (though graph must be a DAG)
    if len(topo_order) < len(events):
        remaining = [eid for eid in events if eid not in topo_order]
        topo_order.extend(sorted(remaining))
        
    # 2. Simulate baseline factual world (verify all occur)
    baseline_occurred: Set[str] = set()
    for eid in topo_order:
        e = events[eid]
        if not e.parents:
            baseline_occurred.add(eid)
        elif e.rule == "AND":
            if all(p in baseline_occurred for p in e.parents):
                baseline_occurred.add(eid)
        elif e.rule == "OR":
            if any(p in baseline_occurred for p in e.parents):
                baseline_occurred.add(eid)

    # 3. Counterfactual world with denied_event_id removed
    cf_occurred: Set[str] = set()
    for eid in topo_order:
        if eid == denied_event_id:
            continue
        e = events[eid]
        if not e.parents:
            cf_occurred.add(eid)
        elif e.rule == "AND":
            if e.parents and all(p in cf_occurred for p in e.parents):
                cf_occurred.add(eid)
        elif e.rule == "OR":
            if e.parents and any(p in cf_occurred for p in e.parents):
                cf_occurred.add(eid)
                
    # Failed events are those in baseline_occurred that are not in cf_occurred, excluding denied_event_id
    failed_events = sorted([
        eid for eid in baseline_occurred
        if eid != denied_event_id and eid not in cf_occurred
    ])
    
    # Critical events are X plus any event whose status or parents were evaluated
    # To be precise: any node reachable from X in the dependency DAG, plus X itself and their parents
    evaluated_events: Set[str] = {denied_event_id}
    frontier = [denied_event_id]
    while frontier:
        nxt = frontier.pop(0)
        for ch in children_map.get(nxt, []):
            if ch not in evaluated_events:
                evaluated_events.add(ch)
                # Also include all parents of ch since they influence its survival
                for p in events[ch].parents:
                    evaluated_events.add(p)
                frontier.append(ch)
                
    return failed_events, sorted(list(evaluated_events))


def solve_q4_conditional_rule(
    trigger_event: Event, target_property: str, target_value: str
) -> Tuple[str, List[str]]:
    """Q4 rule application: apply a stated conditional to reach a value or event.
    
    Returns (target_value, critical_event_ids).
    """
    return target_value, [trigger_event.id]

