"""
Prefix Executability Auditor for Agent and Deliberation Traces (Item 114).

Inspired by 'Prefix Executability: Evaluating Tool-Using Agents Beyond Final Success'
(arXiv:2610.40444). Evaluates multi-step agent traces step-by-step to verify that
intermediate tool calls, file accesses, and state transitions are genuinely valid,
satisfy preconditions, and execute without error, rather than crediting an agent
solely for a final answer arrived at via corrupted or invalid intermediate steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class ActionStep:
    """An individual action or tool step in an agent execution trace."""
    step_index: int
    action_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    preconditions: Dict[str, Any] = field(default_factory=dict)
    return_code: int = 0
    error: Optional[str] = None
    output: Any = None


@dataclass
class StepExecutabilityAudit:
    """Audit outcome for a single action step in a trace."""
    step_index: int
    action_name: str
    is_executable: bool
    failure_category: Optional[str] = None
    diagnostic: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action_name": self.action_name,
            "is_executable": self.is_executable,
            "failure_category": self.failure_category,
            "diagnostic": self.diagnostic,
        }


@dataclass
class PrefixExecutabilityReport:
    """Consolidated prefix executability metrics across an agent trajectory."""
    total_steps: int
    executable_steps: int
    first_failure_index: Optional[int]
    prefix_executability_ratio: float  # Fraction of executable steps across trace
    strict_prefix_ratio: float         # Continuous unbroken executable prefix length / total steps
    is_trace_fully_executable: bool
    step_audits: List[StepExecutabilityAudit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "executable_steps": self.executable_steps,
            "first_failure_index": self.first_failure_index,
            "prefix_executability_ratio": round(self.prefix_executability_ratio, 4),
            "strict_prefix_ratio": round(self.strict_prefix_ratio, 4),
            "is_trace_fully_executable": self.is_trace_fully_executable,
            "step_audits": [a.to_dict() for a in self.step_audits],
        }


class PrefixExecutabilityAuditor:
    """Validates step-by-step prefix executability of agent traces."""

    @classmethod
    def default_step_validator(cls, step: ActionStep) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Default validator checking return code, errors, and schema completeness.
        Returns: (is_executable, failure_category, diagnostic)
        """
        # 1. Explicit execution error or non-zero exit code
        if step.error or step.return_code != 0:
            return (
                False,
                "execution_error",
                f"Step {step.step_index} ('{step.action_name}') failed with code {step.return_code}: {step.error or 'Non-zero exit'}"
            )

        # 2. Check preconditions if present
        for pre_key, expected_val in step.preconditions.items():
            if expected_val is False or expected_val is None:
                return (
                    False,
                    "precondition_violation",
                    f"Precondition '{pre_key}' not satisfied in step {step.step_index}."
                )

        # 3. Missing action name
        if not step.action_name or not step.action_name.strip():
            return (False, "invalid_schema", f"Empty action name in step {step.step_index}.")

        return (True, None, None)

    @classmethod
    def audit_trace(
        cls,
        trace: Sequence[ActionStep],
        custom_validator: Optional[Callable[[ActionStep], tuple[bool, Optional[str], Optional[str]]]] = None,
    ) -> PrefixExecutabilityReport:
        """Audit an agent trajectory for step-by-step executability."""
        num_steps = len(trace)
        if num_steps == 0:
            return PrefixExecutabilityReport(
                total_steps=0,
                executable_steps=0,
                first_failure_index=None,
                prefix_executability_ratio=1.0,
                strict_prefix_ratio=1.0,
                is_trace_fully_executable=True,
            )

        validator = custom_validator or cls.default_step_validator
        step_audits: List[StepExecutabilityAudit] = []
        executable_count = 0
        first_failure_idx: Optional[int] = None

        for idx, step in enumerate(trace):
            is_valid, fail_cat, diag = validator(step)
            if is_valid:
                executable_count += 1
            else:
                if first_failure_idx is None:
                    first_failure_idx = idx

            step_audits.append(
                StepExecutabilityAudit(
                    step_index=step.step_index,
                    action_name=step.action_name,
                    is_executable=is_valid,
                    failure_category=fail_cat,
                    diagnostic=diag,
                )
            )

        prefix_ratio = executable_count / num_steps
        unbroken_len = first_failure_idx if first_failure_idx is not None else num_steps
        strict_ratio = unbroken_len / num_steps

        return PrefixExecutabilityReport(
            total_steps=num_steps,
            executable_steps=executable_count,
            first_failure_index=first_failure_idx,
            prefix_executability_ratio=round(prefix_ratio, 4),
            strict_prefix_ratio=round(strict_ratio, 4),
            is_trace_fully_executable=(executable_count == num_steps),
            step_audits=step_audits,
        )
