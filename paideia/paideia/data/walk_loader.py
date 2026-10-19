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


def load_anadrome_replay_dataset(
    path: Path,
    max_history_nodes: int = 8,
) -> Tuple[List[PolicyStepSample], WalkDatasetAudit]:
    """
    Ingest Anadrome continuation trees and decision records (JSONL format)
    and extract clean policy step training samples with provenance and audit statistics.
    """
    audit = WalkDatasetAudit()
    samples: List[PolicyStepSample] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            audit.total_lines += 1
            try:
                data = json.loads(line)
            except Exception:
                audit.schema_violations += 1
                continue

            if not isinstance(data, dict):
                audit.schema_violations += 1
                continue

            # Check required fields for ReplayTreeRecord
            if not ("walk_id" in data and "question" in data and "transitions" in data):
                audit.schema_violations += 1
                continue

            audit.valid_records += 1

            # Verification gates:
            # 1. Terminal status must indicate completion
            status = str(data.get("terminal_status", "")).upper()
            if status not in ("ANSWER", "COMPLETED", "ANSWERED", "VERIFIED"):
                audit.unverified_excluded += 1
                continue

            # 2. Final answer must be non-empty
            final_ans = str(data.get("final_answer", "")).strip()
            if not final_ans:
                audit.empty_answers_excluded += 1
                continue

            transitions = data.get("transitions", [])
            if not transitions:
                audit.empty_answers_excluded += 1
                continue

            # 3. Check for parsing failures in transitions
            if any(t.get("parse_failed", False) for t in transitions):
                audit.parse_failures_excluded += 1
                continue

            walk_id = str(data["walk_id"])
            question = str(data["question"])
            env = data.get("environment") or {}
            model_id = env.get("controller_model_id", "anadrome")

            history: List[str] = []
            for idx, t in enumerate(transitions):
                obs = t.get("observation") or {}
                current_node = str(obs.get("current_node_id", ""))
                taken_action = str(t.get("taken_action", "")).strip()

                parts = taken_action.split(maxsplit=1)
                if parts:
                    first = parts[0].upper()
                    if first == "EXPAND":
                        action_name = "EXPAND"
                        target_node_val = parts[1] if len(parts) > 1 else t.get("next_node_id")
                    elif first == "BACKTRACK":
                        action_name = "BACKTRACK"
                        target_node_val = None
                    elif first == "ANSWER":
                        action_name = "ANSWER"
                        target_node_val = None
                    else:
                        action_name = first
                        target_node_val = t.get("next_node_id")
                else:
                    action_name = "ANSWER" if idx == len(transitions) - 1 else "EXPAND"
                    target_node_val = t.get("next_node_id")

                audit.action_counts[action_name] = audit.action_counts.get(action_name, 0) + 1

                step_sample = PolicyStepSample(
                    sample_id=f"{walk_id}:{t.get('step_index', idx)}",
                    walk_id=walk_id,
                    question=question,
                    visited_history=list(history[-max_history_nodes:]),
                    current_node=current_node,
                    action=action_name,
                    target_node=target_node_val,
                    reason=str(t.get("parsed_reason", "")),
                    provenance=f"anadrome_replay_{model_id}",
                )
                samples.append(step_sample)
                if current_node:
                    history.append(current_node)

    audit.generated_step_samples = len(samples)
    return samples, audit


def load_walk_dataset(
    path: Path,
    max_history_nodes: int = 8,
) -> Tuple[List[PolicyStepSample], WalkDatasetAudit]:
    """
    Ingest Dianoia JSONL walks or Anadrome replay trees, link re-walk attempts by walk_id,
    and extract policy steps.
    
    Args:
        path: Path to dianoia walk JSONL or Anadrome replay tree JSONL file.
        max_history_nodes: Bounded context length for previous node history.
        
    Returns:
        Tuple of (clean policy step samples, audit report).
    """
    # Peek first valid line to auto-detect format
    first_line = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                first_line = line.strip()
                break
    if first_line:
        try:
            peek_data = json.loads(first_line)
            if isinstance(peek_data, dict) and ("transitions" in peek_data or "tree_id" in peek_data):
                return load_anadrome_replay_dataset(path, max_history_nodes=max_history_nodes)
        except Exception:
            pass

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
