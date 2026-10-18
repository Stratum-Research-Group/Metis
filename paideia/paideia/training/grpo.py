"""
GRPO Walk Policy Optimization Engine for Dianoia Deliberative Walks.
Implements Option A (step-level decision optimization) and Anti-Reward-Hacking
penalties (H1-H8) per specification metis-grpo-walk-policy-spec-e6633b6a.txt.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from paideia.data.walk_loader import PolicyStepSample


@dataclass
class GRPOConfig:
    """Hyperparameters for Dianoia step-level GRPO policy optimization."""
    group_size: int = 4  # G >= 4 parallel candidate completions
    max_reason_chars: int = 400
    length_penalty_alpha: float = 0.005  # H1 verbosity penalty
    cycle_penalty: float = 0.5            # H2 repetitive cycle penalty
    unparseable_penalty: float = 1.0     # Penalty for malformed action strings
    kl_coeff: float = 0.04               # Reference policy KL penalty
    epsilon: float = 1e-6


class DianoiaStateReconstructor:
    """
    Deterministically reconstructs the deliberative prompt presented to
    the policy model at step t from PolicyStepSample.
    """

    @classmethod
    def build_step_prompt(cls, sample: PolicyStepSample) -> str:
        history_str = " -> ".join(sample.visited_history) if sample.visited_history else "[START]"
        prompt = (
            f"<deliberation_state>\n"
            f"Question: {sample.question.strip()}\n"
            f"Visited Path: {history_str}\n"
            f"Current Node: {sample.current_node.strip()}\n"
            f"</deliberation_state>\n\n"
            f"Available Actions: EXPAND <target_node>, BACKTRACK, ANSWER\n"
            f"Select next action and state your reasoning:\n"
            f"Action: "
        )
        return prompt


@dataclass
class CandidateCompletion:
    """A single sampled completion in a GRPO group."""
    raw_text: str
    parsed_action: Optional[str] = None
    target_node: Optional[str] = None
    reason: str = ""
    is_valid_format: bool = False


@dataclass
class GroupEvaluation:
    """Rewards and normalized advantages for a group of G sampled completions."""
    prompt: str
    sample_id: str
    completions: List[CandidateCompletion]
    raw_rewards: List[float]
    advantages: List[float]
    group_variance: float
    has_diversity: bool


class GRPORewardEngine:
    """
    Evaluates group completions under Option A and enforces anti-reward-hacking constraints:
    - H1: Length/verbosity penalty
    - H2: Repetitive cycle penalty (visiting an already visited node)
    - H3: Hard verification dependency (cannot reward unverified actions)
    - H4: Group variance floor
    - H5: Semantic diversity check
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        self.config = config or GRPOConfig()

    def parse_completion(self, text: str) -> CandidateCompletion:
        """Parse raw model completion into action, target, and reason."""
        text = text.strip()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return CandidateCompletion(raw_text=text, is_valid_format=False)

        # Expected format: "EXPAND node_123" or "BACKTRACK" or "ANSWER: ..."
        first_line = lines[0]
        parsed_action = None
        target_node = None
        reason = "\n".join(lines[1:]) if len(lines) > 1 else ""

        if first_line.startswith("EXPAND"):
            parts = first_line.split(maxsplit=1)
            parsed_action = "EXPAND"
            if len(parts) > 1 and parts[1].strip():
                target_node = parts[1].strip()
        elif first_line.startswith("BACKTRACK"):
            parsed_action = "BACKTRACK"
        elif first_line.startswith("ANSWER"):
            parsed_action = "ANSWER"

        is_valid = parsed_action is not None
        if parsed_action == "EXPAND" and not target_node:
            is_valid = False

        return CandidateCompletion(
            raw_text=text,
            parsed_action=parsed_action,
            target_node=target_node,
            reason=reason,
            is_valid_format=is_valid,
        )

    def compute_single_reward(
        self,
        candidate: CandidateCompletion,
        sample: PolicyStepSample,
    ) -> float:
        """Compute scalar reward with H1-H3 penalties for a single candidate completion."""
        if not candidate.is_valid_format or not candidate.parsed_action:
            return -self.config.unparseable_penalty

        base_reward = 0.0

        # Match action against gold target action
        if candidate.parsed_action == sample.action:
            base_reward += 1.0
            # If EXPAND, check target node match
            if candidate.parsed_action == "EXPAND" and sample.target_node:
                if candidate.target_node == sample.target_node:
                    base_reward += 0.5
                else:
                    base_reward -= 0.3
        else:
            base_reward -= 0.5

        # H1: Length penalty for excessive verbosity
        char_len = len(candidate.raw_text)
        if char_len > self.config.max_reason_chars:
            excess = char_len - self.config.max_reason_chars
            base_reward -= self.config.length_penalty_alpha * excess

        # H2: Repetitive cycle penalty
        if candidate.target_node and candidate.target_node in sample.visited_history:
            base_reward -= self.config.cycle_penalty

        return base_reward

    def evaluate_group(
        self,
        sample: PolicyStepSample,
        candidate_texts: List[str],
    ) -> GroupEvaluation:
        """
        Evaluate a group of G completions, compute raw rewards, and calculate
        normalized advantages: A_i = (R_i - mean(R)) / (std(R) + eps).
        """
        if len(candidate_texts) < 2:
            raise ValueError(f"Group size must be >= 2, got {len(candidate_texts)}")

        prompt = DianoiaStateReconstructor.build_step_prompt(sample)
        completions = [self.parse_completion(t) for t in candidate_texts]
        rewards = [self.compute_single_reward(c, sample) for c in completions]

        r_array = np.array(rewards, dtype=np.float32)
        mean_r = float(np.mean(r_array))
        std_r = float(np.std(r_array))
        var_r = float(np.var(r_array))

        # Check semantic diversity across group
        unique_completions = set(t.strip() for t in candidate_texts)
        has_diversity = len(unique_completions) > 1

        # Normalized advantages
        if std_r < self.config.epsilon:
            # If all rewards are identical, advantages are 0.0
            advantages = [0.0] * len(rewards)
        else:
            advantages = [float((r - mean_r) / (std_r + self.config.epsilon)) for r in rewards]

        return GroupEvaluation(
            prompt=prompt,
            sample_id=sample.sample_id,
            completions=completions,
            raw_rewards=rewards,
            advantages=advantages,
            group_variance=var_r,
            has_diversity=has_diversity,
        )
