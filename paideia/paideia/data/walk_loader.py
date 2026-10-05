"""Dianoia walk log ingestion, attempt linkage, and policy step generator."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

from paideia.contracts.dianoia_schema import DianoiaWalkRecord


class PolicyStepSample(BaseModel):
    """Supervised training instance for a single deliberative walk policy step."""
    sample_id: str = Field(..., description="Unique step identifier: walk_id:step_idx")
    walk_id: str = Field(..., description="Parent walk identifier")
    question: str = Field(..., description="Original user prompt")
    visited_history: List[str] = Field(..., description="Ordered list of previous node IDs visited")
    current_node: str = Field(..., description="Current node ID being deliberated")
    action: str = Field(..., description="Target action: EXPAND, BACKTRACK, ANSWER")
    target_node: Optional[str] = Field(default=None, description="Target node ID for EXPAND action")
    reason: str = Field(default="", description="Structured reasoning justification")
    provenance: str = Field(default="verified_walk", description="Why this action is deemed a target")


class WalkDatasetAudit:
    """Audit report for Dianoia walk logs."""
    def __init__(self):
        self.total_lines: int = 0
        self.valid_records: int = 0
        self.schema_violations: int = 0
        self.unverified_excluded: int = 0
        self.simulated_excluded: int = 0
        self.unresolved_conflicts_excluded: int = 0
        self.incorrect_excluded: int = 0
        self.parse_failures_excluded: int = 0
        self.empty_answers_excluded: int = 0
        self.action_counts: Dict[str, int] = {}
        self.generated_step_samples: int = 0
        self.walk_attempts_joined: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "valid_records": self.valid_records,
            "schema_violations": self.schema_violations,
            "unverified_excluded": self.unverified_excluded,
            "simulated_excluded": self.simulated_excluded,
            "unresolved_conflicts_excluded": self.unresolved_conflicts_excluded,
            "incorrect_excluded": self.incorrect_excluded,
            "parse_failures_excluded": self.parse_failures_excluded,
            "empty_answers_excluded": self.empty_answers_excluded,
            "action_counts": self.action_counts,
            "generated_step_samples": self.generated_step_samples,
            "walk_attempts_joined": self.walk_attempts_joined,
        }


def load_walk_dataset(
    path: Path,
    max_history_nodes: int = 8,
) -> Tuple[List[PolicyStepSample], WalkDatasetAudit]:
    """
    Ingest Dianoia JSONL walks, link re-walk attempts by walk_id, and extract policy steps.
    
    Args:
        path: Path to dianoia walk JSONL file.
        max_history_nodes: Bounded context length for previous node history.
        
    Returns:
        Tuple of (clean policy step samples, audit report).
    """
    audit = WalkDatasetAudit()
    records_by_walk_id: Dict[str, List[DianoiaWalkRecord]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            audit.total_lines += 1
            try:
                data = json.loads(line)
                rec = DianoiaWalkRecord.model_validate(data)
                records_by_walk_id.setdefault(rec.walk_id, []).append(rec)
                audit.valid_records += 1
            except (ValidationError, json.JSONDecodeError):
                audit.schema_violations += 1

    samples: List[PolicyStepSample] = []

    for walk_id, attempts in records_by_walk_id.items():
        if len(attempts) > 1:
            audit.walk_attempts_joined += (len(attempts) - 1)
            # Choose the successful corrective re-walk
            # A re-walk has initial_path present or verification.rewalked=True
            successful = [a for a in attempts if a.is_defensible_gold_trajectory()]
            if not successful:
                audit.unresolved_conflicts_excluded += 1
                continue
            target_record = successful[-1]  # Take the final verified re-walk
        else:
            target_record = attempts[0]
            if not target_record.is_defensible_gold_trajectory():
                if target_record.verification.conflicts_found is not False:
                    audit.unresolved_conflicts_excluded += 1
                elif not target_record.verification.verified:
                    audit.unverified_excluded += 1
                elif target_record.is_simulated or target_record.verification.simulated or target_record.fallback_used:
                    audit.simulated_excluded += 1
                elif target_record.verification.is_correct is False:
                    audit.incorrect_excluded += 1
                elif any(s.parse_failed for s in target_record.path):
                    audit.parse_failures_excluded += 1
                elif not target_record.final_answer.strip():
                    audit.empty_answers_excluded += 1
                else:
                    audit.unverified_excluded += 1
                continue

        # Extract bounded state -> action step pairs
        history: List[str] = []
        for idx, step in enumerate(target_record.path):
            action_name = step.action.upper()
            audit.action_counts[action_name] = audit.action_counts.get(action_name, 0) + 1

            target_node_val = None
            if action_name == "EXPAND" and idx + 1 < len(target_record.path):
                target_node_val = target_record.path[idx + 1].node

            sample = PolicyStepSample(
                sample_id=f"{walk_id}:{idx}",
                walk_id=walk_id,
                question=target_record.question,
                visited_history=list(history[-max_history_nodes:]),
                current_node=step.node,
                action=action_name,
                target_node=target_node_val,
                reason=step.reason,
                provenance=target_record.verification.provenance or "verified_trajectory_gold",
            )
            samples.append(sample)
            history.append(step.node)

    audit.generated_step_samples = len(samples)
    return samples, audit
