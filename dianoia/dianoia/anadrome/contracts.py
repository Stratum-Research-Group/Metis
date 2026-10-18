"""Data contracts for Anadrome walk replay and decision-tree logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from dianoia.anadrome.canonical import canonical_json_dumps, compute_observation_hash


@dataclass
class EnvironmentIdentity:
    """
    Immutable identity of the execution environment.
    Shared by all candidate policies during replay.
    """
    corpus_snapshot_hash: str
    index_build_id: str
    embedder_id: str
    controller_model_id: str
    sampling_params: Dict[str, Any]
    prompt_template_version: str
    answer_generator_config: Dict[str, Any]
    seeding_config: Dict[str, Any]
    budget_config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_env_hash(self) -> str:
        return compute_observation_hash(self.to_dict())


@dataclass
class EligibleAction:
    """An action available to the controller at a specific decision point."""
    action_type: str  # "EXPAND" | "BACKTRACK" | "ANSWER"
    target_node_id: Optional[str] = None
    score: Optional[float] = None
    edge_type: Optional[str] = None

    def format_string(self) -> str:
        if self.action_type == "EXPAND" and self.target_node_id:
            return f"EXPAND {self.target_node_id}"
        return self.action_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target_node_id": self.target_node_id,
            "score": round(self.score, 4) if self.score is not None else None,
            "edge_type": self.edge_type,
        }


@dataclass
class DecisionObservation:
    """Controller-visible state at a specific step in the walk."""
    env_hash: str
    current_node_id: str
    current_content: str
    path_prefix: List[str]
    visited_nodes: List[str]
    remaining_budget: int
    eligible_actions: List[EligibleAction]
    realization_ids: Dict[str, Any] = field(default_factory=dict)
    observation_hash: str = ""

    def __post_init__(self):
        if not self.observation_hash:
            obs_payload = {
                "env_hash": self.env_hash,
                "current_node_id": self.current_node_id,
                "current_content": self.current_content,
                "path_prefix": self.path_prefix,
                "visited_nodes": sorted(list(set(self.visited_nodes))),
                "remaining_budget": self.remaining_budget,
                "eligible_actions": [a.to_dict() for a in self.eligible_actions],
                "realization_ids": self.realization_ids,
            }
            self.observation_hash = compute_observation_hash(obs_payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env_hash": self.env_hash,
            "current_node_id": self.current_node_id,
            "current_content": self.current_content,
            "path_prefix": self.path_prefix,
            "visited_nodes": sorted(list(set(self.visited_nodes))),
            "remaining_budget": self.remaining_budget,
            "eligible_actions": [a.to_dict() for a in self.eligible_actions],
            "realization_ids": self.realization_ids,
            "observation_hash": self.observation_hash,
        }


@dataclass
class RecordedTransition:
    """A single recorded decision, execution, and resulting observation/probe."""
    step_index: int
    observation_hash: str
    observation: DecisionObservation
    taken_action: str
    raw_model_output: str
    parsed_reason: str
    parse_failed: bool
    next_node_id: Optional[str]
    answer_probe: Optional[Dict[str, Any]] = None
    stochastic_draws: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "observation_hash": self.observation_hash,
            "observation": self.observation.to_dict(),
            "taken_action": self.taken_action,
            "raw_model_output": self.raw_model_output,
            "parsed_reason": self.parsed_reason,
            "parse_failed": self.parse_failed,
            "next_node_id": self.next_node_id,
            "answer_probe": self.answer_probe,
            "stochastic_draws": self.stochastic_draws,
        }


@dataclass
class ReplayTreeRecord:
    """Full decision-tree record representing an episode or shadow branch."""
    schema_version: str
    tree_id: str
    walk_id: str
    question: str
    seeded_nodes: List[str]
    environment: EnvironmentIdentity
    is_shadow: bool
    parent_walk_id: Optional[str]
    branch_point_step: Optional[int]
    branch_point_node: Optional[str]
    transitions: List[RecordedTransition]
    final_answer: str
    steps_used: int
    terminal_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tree_id": self.tree_id,
            "walk_id": self.walk_id,
            "question": self.question,
            "seeded_nodes": self.seeded_nodes,
            "environment": self.environment.to_dict(),
            "is_shadow": self.is_shadow,
            "parent_walk_id": self.parent_walk_id,
            "branch_point_step": self.branch_point_step,
            "branch_point_node": self.branch_point_node,
            "transitions": [t.to_dict() for t in self.transitions],
            "final_answer": self.final_answer,
            "steps_used": self.steps_used,
            "terminal_status": self.terminal_status,
        }
