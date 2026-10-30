"""Data models for text vs graph vs both experiment.

Per test plan: arXiv:2610.34094 probe (Section 4, 13).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class Event:
    id: str  # e.g. "E1"
    name: str  # e.g. "Thermal Sensor Calibration"
    timestamp: int  # start hour (0..100)
    duration: int  # duration in hours (1..24)
    parents: List[str] = field(default_factory=list)  # parent event IDs
    rule: Literal["AND", "OR", "ROOT"] = "ROOT"

    @property
    def end_time(self) -> int:
        return self.timestamp + self.duration


@dataclass
class ConditionalRule:
    id: str  # e.g. "CR1"
    trigger_event_id: str  # event that must occur
    target_property: str  # e.g. "auxiliary_chiller"
    target_value: str  # e.g. "mode_cryo_beta"


@dataclass
class DistractorFact:
    id: str  # e.g. "D1"
    subject: str
    property_name: str
    value: str


@dataclass
class Fact:
    id: str
    kind: Literal["event", "temporal", "dependency", "conditional", "distractor"]
    details: Dict[str, Any]
    prose_sentence: str
    graph_line: str


@dataclass
class Question:
    qid: str
    qtype: Literal["Q1", "Q2", "Q3", "Q4"]
    prompt: str
    answer_format: Literal["event_id", "integer", "id_list", "value"]
    gold_answer: Any
    critical_event_ids: List[str] = field(default_factory=list)
    critical_fact_ids: List[str] = field(default_factory=list)


@dataclass
class Story:
    story_id: str
    title: str
    events: Dict[str, Event]
    conditional_rules: List[ConditionalRule]
    distractors: List[DistractorFact]
    facts: List[Fact]
    prose: str
    graph_block: str
    questions: List[Question]


@dataclass
class CorruptionRecord:
    level: float
    ops: List[Dict[str, Any]]
    touched_original_nodes: List[str]
    fake_node_count: int
    corrupted_graph_block: str

