"""
Unit tests for GRPO Walk Policy Optimization and Anti-Reward Hacking Penalties
per specification metis-grpo-walk-policy-spec-e6633b6a.txt.
"""

import pytest

from paideia.data.walk_loader import PolicyStepSample
from paideia.training.grpo import (
    CandidateCompletion,
    DianoiaStateReconstructor,
    GRPOConfig,
    GRPORewardEngine,
    GroupEvaluation,
)


@pytest.fixture
def sample_step():
    return PolicyStepSample(
        sample_id="walk_123:0",
        walk_id="walk_123",
        question="What is the capital of France?",
        visited_history=["node_root", "node_geography"],
        current_node="node_europe",
        action="EXPAND",
        target_node="node_paris",
        reason="France is in Europe, so expand into Paris.",
        provenance="verified_sheaf",
    )


def test_dianoia_state_reconstructor(sample_step):
    """Verify deterministic prompt reconstruction from policy step sample."""
    prompt = DianoiaStateReconstructor.build_step_prompt(sample_step)

    assert "Question: What is the capital of France?" in prompt
    assert "Visited Path: node_root -> node_geography" in prompt
    assert "Current Node: node_europe" in prompt
    assert "Available Actions: EXPAND <target_node>, BACKTRACK, ANSWER" in prompt
    assert prompt.endswith("Action: ")


def test_grpo_completion_parsing():
    """Verify parsing of EXPAND, BACKTRACK, ANSWER and unparseable outputs."""
    engine = GRPORewardEngine()

    c_expand = engine.parse_completion("EXPAND node_target\nBecause it connects.")
    assert c_expand.is_valid_format
    assert c_expand.parsed_action == "EXPAND"
    assert c_expand.target_node == "node_target"
    assert c_expand.reason == "Because it connects."

    c_backtrack = engine.parse_completion("BACKTRACK\nDead end reached.")
    assert c_backtrack.is_valid_format
    assert c_backtrack.parsed_action == "BACKTRACK"
    assert c_backtrack.target_node is None

    c_invalid = engine.parse_completion("RANDOM_TEXT_NOT_AN_ACTION")
    assert not c_invalid.is_valid_format
    assert c_invalid.parsed_action is None


def test_grpo_reward_penalties(sample_step):
    """Verify H1 length penalty and H2 cycle penalty behavior."""
    cfg = GRPOConfig(
        max_reason_chars=50,
        length_penalty_alpha=0.01,
        cycle_penalty=0.5,
    )
    engine = GRPORewardEngine(config=cfg)

    # 1. Correct concise target completion
    c_good = engine.parse_completion("EXPAND node_paris\nDirect link.")
    r_good = engine.compute_single_reward(c_good, sample_step)
    assert r_good == 1.5  # 1.0 (action) + 0.5 (target node)

    # 2. H1: Excessively verbose completion
    verbose_reason = "A" * 150  # 100 chars excess
    c_verbose = engine.parse_completion(f"EXPAND node_paris\n{verbose_reason}")
    r_verbose = engine.compute_single_reward(c_verbose, sample_step)
    # Penalized by 0.01 * 100 = 1.0
    assert r_verbose < r_good

    # 3. H2: Repetitive cycle penalty (proposing node_geography which is in visited_history)
    c_cycle = engine.parse_completion("EXPAND node_geography\nGoing back.")
    r_cycle = engine.compute_single_reward(c_cycle, sample_step)
    # Action matches EXPAND, but target mismatch (-0.3) and cycle penalty (-0.5)
    assert r_cycle < 1.0


def test_grpo_group_evaluation_and_advantage_normalization(sample_step):
    """Verify group evaluation computes normalized advantages across G candidate completions."""
    engine = GRPORewardEngine(config=GRPOConfig(group_size=4))

    completions = [
        "EXPAND node_paris\nCorrect step.",         # High reward
        "EXPAND node_wrong\nDifferent step.",       # Medium-low
        "BACKTRACK\nUnnecessary backtrack.",        # Low
        "UNPARSEABLE GIBBERISH",                   # Heavy penalty
    ]

    eval_result = engine.evaluate_group(sample_step, completions)

    assert len(eval_result.raw_rewards) == 4
    assert len(eval_result.advantages) == 4
    assert eval_result.has_diversity
    assert eval_result.group_variance > 0.0

    # Best completion should have the highest positive advantage
    assert eval_result.advantages[0] > 0
    # Worst completion should have negative advantage
    assert eval_result.advantages[3] < 0

    # Mean of advantages should be approximately 0.0
    adv_mean = sum(eval_result.advantages) / len(eval_result.advantages)
    assert abs(adv_mean) < 1e-4


def test_grpo_group_identical_completions(sample_step):
    """When all completions in group are identical, advantages collapse safely to 0."""
    engine = GRPORewardEngine()
    identical = ["EXPAND node_paris\nExact same."] * 4

    eval_result = engine.evaluate_group(sample_step, identical)
    assert eval_result.group_variance == 0.0
    assert not eval_result.has_diversity
    assert eval_result.advantages == [0.0, 0.0, 0.0, 0.0]
