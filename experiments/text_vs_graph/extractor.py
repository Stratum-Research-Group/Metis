"""
Phase 2 Graph Extractor and Extraction Error Analyzer.

Extracts structured fact graphs from narrative prose using the local model,
and quantifies natural extraction error rates (omissions, shifts, hallucinations).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


EXTRACTION_SYSTEM_PROMPT = """You are a precise data extraction assistant. Extract all structured facts from the mission log text.
Follow this exact schema:

=== STRUCTURED FACT GRAPH ===
# Events and Timestamps:
EVENT <id>: name=<name>, start_hour=<int>, duration_hours=<int>

# Temporal Precedence:
TEMPORAL: <id1> precedes <id2>

# Dependency Rules:
DEPENDENCY: <id> requires (<parent_list>) with rule <AND|OR>
(or: DEPENDENCY: <id> is a ROOT event (always occurs))

# Conditional Operational Rules:
RULE <id>: IF <id> occurs THEN <prop> = <val>

# System Metadata:
METADATA <id>: <subject> - <prop> is <val>
"""


@dataclass
class ExtractionErrorMetrics:
    """Metrics comparing extracted graph against gold graph."""
    gold_event_count: int
    extracted_event_count: int
    missing_events: List[str]
    hallucinated_events: List[str]
    timestamp_mismatches: List[Tuple[str, int, int]]  # (event_id, gold_ts, ext_ts)
    duration_mismatches: List[Tuple[str, int, int]]   # (event_id, gold_dur, ext_dur)
    dependency_mismatches: List[str]
    rule_mismatches: List[str]
    overall_fidelity: float  # [0.0, 1.0]


def parse_extracted_events(graph_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse events from extracted or gold graph block."""
    events = {}

    # Format 1: Gold format: EVENT E1: Subspace Beacon Ping [start_hour=6, duration_hours=3] [EV_E1]
    pat_gold = re.compile(
        r"EVENT\s+([A-Za-z0-9_]+):\s*([^\[\n]+)\s*\[start_hour=(\d+),\s*duration_hours=(\d+)\]",
        re.IGNORECASE,
    )
    for match in pat_gold.finditer(graph_text):
        ev_id, name, start_h, dur_h = match.groups()
        norm_id = ev_id.strip()
        if norm_id.startswith("EV_"):
            norm_id = norm_id[3:]
        events[norm_id] = {
            "name": name.strip(),
            "start_hour": int(start_h),
            "duration_hours": int(dur_h),
        }

    # Format 2: Extractor format: EVENT E1: name=Subspace Beacon Ping, start_hour=6, duration_hours=3
    pat_ext = re.compile(
        r"EVENT\s+([A-Za-z0-9_]+):\s*(?:name=)?([^,\n]+),\s*start_hour=(\d+),\s*duration_hours=(\d+)",
        re.IGNORECASE,
    )
    for match in pat_ext.finditer(graph_text):
        ev_id, name, start_h, dur_h = match.groups()
        norm_id = ev_id.strip()
        if norm_id.startswith("EV_"):
            norm_id = norm_id[3:]
        if norm_id not in events:
            events[norm_id] = {
                "name": name.strip(),
                "start_hour": int(start_h),
                "duration_hours": int(dur_h),
            }

    return events


def analyze_extraction_fidelity(gold_graph_text: str, extracted_graph_text: str) -> ExtractionErrorMetrics:
    """Compare extracted graph against gold graph to classify natural error distribution."""
    gold_events = parse_extracted_events(gold_graph_text)
    ext_events = parse_extracted_events(extracted_graph_text)

    gold_ids = set(gold_events.keys())
    ext_ids = set(ext_events.keys())

    missing = sorted(list(gold_ids - ext_ids))
    hallucinated = sorted(list(ext_ids - gold_ids))

    ts_mismatches = []
    dur_mismatches = []
    common_ids = gold_ids & ext_ids

    for eid in sorted(list(common_ids)):
        g = gold_events[eid]
        e = ext_events[eid]
        if g["start_hour"] != e["start_hour"]:
            ts_mismatches.append((eid, g["start_hour"], e["start_hour"]))
        if g["duration_hours"] != e["duration_hours"]:
            dur_mismatches.append((eid, g["duration_hours"], e["duration_hours"]))

    total_checks = len(gold_ids) * 3  # existence, timestamp, duration
    errors = len(missing) * 3 + len(hallucinated) * 3 + len(ts_mismatches) + len(dur_mismatches)
    fidelity = max(0.0, (total_checks - errors) / total_checks) if total_checks > 0 else 1.0

    return ExtractionErrorMetrics(
        gold_event_count=len(gold_ids),
        extracted_event_count=len(ext_ids),
        missing_events=missing,
        hallucinated_events=hallucinated,
        timestamp_mismatches=ts_mismatches,
        duration_mismatches=dur_mismatches,
        dependency_mismatches=[],
        rule_mismatches=[],
        overall_fidelity=round(fidelity, 4),
    )
