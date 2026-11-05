"""
Causal Deliberation Step Audit for Multi-Step Reasoning and MCTS Walks (Item 112).

Inspired by 'Does Latent Chain-of-Thought Actually Reason? A Unifying Taxonomy'
(arXiv:2610.21182). Separates genuine reasoning where intermediate steps causally
mediate decisions from 'ritual placeholders' where intermediate compute is merely
ceremonial buffer that doesn't affect downstream outcomes when corrupted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


class StepCausalRole(str, Enum):
    """Causal taxonomy classification for an intermediate deliberation step."""
    CAUSALLY_GROUNDED = "causally_grounded"      # Content actively mediates outcome; corruption or deletion degrades decision.
    RITUAL_PLACEHOLDER = "ritual_placeholder"    # Content has zero impact when corrupted; acts as ceremonial buffer.
    FRAGILE_BYSTANDER = "fragile_bystander"      # Deletion fails due to length/format, but any filler content succeeds equally.
    REDUNDANT_INERT = "redundant_inert"          # Neither deletion nor corruption changes the outcome; entirely inactive.


@dataclass(frozen=True)
class DeliberationStep:
    """An individual step in a deliberation sequence or MCTS walk trace."""
    index: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepInterventionAudit:
    """Audit record for a single deliberation step under counterfactual interventions."""
    step_index: int
    original_content: str
    role: StepCausalRole
    baseline_outcome: Any
    deletion_outcome: Any
    corruption_outcome: Any
    deletion_delta: float
    corruption_delta: float
    diagnostic: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "original_content": self.original_content,
            "role": self.role.value,
            "baseline_outcome": str(self.baseline_outcome),
            "deletion_outcome": str(self.deletion_outcome),
            "corruption_outcome": str(self.corruption_outcome),
            "deletion_delta": round(self.deletion_delta, 4),
            "corruption_delta": round(self.corruption_delta, 4),
            "diagnostic": self.diagnostic,
        }


@dataclass
class DeliberationAuditReport:
    """Consolidated audit report across an entire multi-step reasoning trace."""
    total_steps: int
    causally_grounded_steps: int
    ritual_placeholder_steps: int
    fragile_bystander_steps: int
    redundant_inert_steps: int
    causal_grounding_ratio: float
    ritual_placeholder_ratio: float
    deliberation_integrity_score: float
    is_deliberation_valid: bool
    step_audits: List[StepInterventionAudit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "causally_grounded_steps": self.causally_grounded_steps,
            "ritual_placeholder_steps": self.ritual_placeholder_steps,
            "fragile_bystander_steps": self.fragile_bystander_steps,
            "redundant_inert_steps": self.redundant_inert_steps,
            "causal_grounding_ratio": round(self.causal_grounding_ratio, 4),
            "ritual_placeholder_ratio": round(self.ritual_placeholder_ratio, 4),
            "deliberation_integrity_score": round(self.deliberation_integrity_score, 4),
            "is_deliberation_valid": self.is_deliberation_valid,
            "step_audits": [a.to_dict() for a in self.step_audits],
        }


class CausalDeliberationAuditor:
    """
    Audits deliberation traces using counterfactual deletion and content-corruption interventions.
    Determines whether intermediate steps perform causal computation or serve as ritual placeholders.
    """

    @classmethod
    def default_corruptor(cls, text: str, seed: int = 42) -> str:
        """Deterministically scramble or replace tokens with neutral uninformative filler."""
        words = text.split()
        if len(words) <= 3:
            return "neutral placeholder computation"
        # Deterministic shuffle to preserve word-count/token-length while destroying semantic assertions
        rng = random.Random(seed + len(text))
        shuffled = list(words)
        rng.shuffle(shuffled)
        return " ".join(shuffled)

    @classmethod
    def audit_trace(
        cls,
        steps: Sequence[Union[str, DeliberationStep]],
        evaluate_fn: Callable[[List[str]], Any],
        scorer_fn: Optional[Callable[[Any], float]] = None,
        corruptor_fn: Optional[Callable[[str], str]] = None,
        score_delta_threshold: float = 0.05,
        min_causal_ratio: float = 0.5,
    ) -> DeliberationAuditReport:
        """
        Audit a multi-step deliberation trace against an evaluation endpoint.

        Args:
            steps: List of step strings or DeliberationStep instances.
            evaluate_fn: Callable that accepts a list of string steps and returns a final answer or outcome.
            scorer_fn: Optional scoring function converting evaluate_fn's outcome to float in [0, 1].
                       Defaults to identity float conversion or exact equality with baseline.
            corruptor_fn: Counterfactual mutator for step content.
            score_delta_threshold: Minimum drop in score to consider an intervention causally impactful.
            min_causal_ratio: Threshold ratio of grounded steps required for is_deliberation_valid.
        """
        raw_steps: List[str] = [
            s.content if isinstance(s, DeliberationStep) else str(s)
            for s in steps
        ]
        num_steps = len(raw_steps)
        if num_steps == 0:
            return DeliberationAuditReport(
                total_steps=0,
                causally_grounded_steps=0,
                ritual_placeholder_steps=0,
                fragile_bystander_steps=0,
                redundant_inert_steps=0,
                causal_grounding_ratio=0.0,
                ritual_placeholder_ratio=0.0,
                deliberation_integrity_score=0.0,
                is_deliberation_valid=False,
            )

        corruptor = corruptor_fn or cls.default_corruptor

        # 1. Baseline evaluation with all steps intact
        baseline_outcome = evaluate_fn(raw_steps)
        default_scorer = (lambda res: float(res)) if isinstance(baseline_outcome, (int, float)) else (lambda res: 1.0 if res == baseline_outcome else 0.0)
        scorer = scorer_fn or default_scorer

        baseline_score = scorer(baseline_outcome)

        step_audits: List[StepInterventionAudit] = []
        counts = {role: 0 for role in StepCausalRole}

        for idx, orig_text in enumerate(raw_steps):
            # Intervention 1: Deletion (remove step idx entirely)
            deleted_trace = [s for j, s in enumerate(raw_steps) if j != idx]
            deletion_outcome = evaluate_fn(deleted_trace)
            deletion_score = scorer(deletion_outcome)
            deletion_delta = max(0.0, baseline_score - deletion_score)
            is_deletion_impactful = (deletion_delta >= score_delta_threshold) or (deletion_outcome != baseline_outcome)

            # Intervention 2: Content Corruption (preserve slot/token-budget, corrupt semantics)
            corrupted_content = corruptor(orig_text)
            corrupted_trace = [
                corrupted_content if j == idx else s
                for j, s in enumerate(raw_steps)
            ]
            corruption_outcome = evaluate_fn(corrupted_trace)
            corruption_score = scorer(corruption_outcome)
            corruption_delta = max(0.0, baseline_score - corruption_score)
            is_corruption_impactful = (corruption_delta >= score_delta_threshold) or (corruption_outcome != baseline_outcome)

            # Classify step causal role
            if is_corruption_impactful:
                # Content semantics actively matter!
                role = StepCausalRole.CAUSALLY_GROUNDED
                diag = f"Step {idx} is causally grounded: corrupting content dropped score by {corruption_delta:.3f}."
            elif is_deletion_impactful and not is_corruption_impactful:
                # Deleting hurts, but corrupting content makes no difference!
                # This is the exact signature of a ritual placeholder / length bystander (arXiv:2610.21182)
                role = StepCausalRole.RITUAL_PLACEHOLDER
                diag = (
                    f"Step {idx} is a ritual placeholder: deleting changed outcome, "
                    f"but arbitrary corrupted content produced unchanged outcome."
                )
            elif not is_deletion_impactful and not is_corruption_impactful:
                # Neither deletion nor corruption made any difference
                role = StepCausalRole.REDUNDANT_INERT
                diag = f"Step {idx} is redundant/inert: neither deletion nor corruption affected outcome."
            else:
                role = StepCausalRole.FRAGILE_BYSTANDER
                diag = f"Step {idx} displays bystander sensitivity."

            counts[role] += 1
            step_audits.append(
                StepInterventionAudit(
                    step_index=idx,
                    original_content=orig_text,
                    role=role,
                    baseline_outcome=baseline_outcome,
                    deletion_outcome=deletion_outcome,
                    corruption_outcome=corruption_outcome,
                    deletion_delta=deletion_delta,
                    corruption_delta=corruption_delta,
                    diagnostic=diag,
                )
            )

        causal_ratio = counts[StepCausalRole.CAUSALLY_GROUNDED] / num_steps
        ritual_ratio = counts[StepCausalRole.RITUAL_PLACEHOLDER] / num_steps

        # Integrity score rewards causal steps, penalizes ritual placeholders
        integrity_score = max(
            0.0,
            causal_ratio - (0.5 * ritual_ratio) - (0.2 * (counts[StepCausalRole.REDUNDANT_INERT] / num_steps))
        )
        is_valid = (causal_ratio >= min_causal_ratio) and (ritual_ratio < 0.4)

        return DeliberationAuditReport(
            total_steps=num_steps,
            causally_grounded_steps=counts[StepCausalRole.CAUSALLY_GROUNDED],
            ritual_placeholder_steps=counts[StepCausalRole.RITUAL_PLACEHOLDER],
            fragile_bystander_steps=counts[StepCausalRole.FRAGILE_BYSTANDER],
            redundant_inert_steps=counts[StepCausalRole.REDUNDANT_INERT],
            causal_grounding_ratio=round(causal_ratio, 4),
            ritual_placeholder_ratio=round(ritual_ratio, 4),
            deliberation_integrity_score=round(integrity_score, 4),
            is_deliberation_valid=is_valid,
            step_audits=step_audits,
        )
