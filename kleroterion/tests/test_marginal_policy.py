"""
Tests for marginal distribution aggregation, entropy diagnostics, and action selection (Section 3.3, 4.1, 10.4).
"""

import numpy as np
import pytest

from kleroterion.errors import InvalidInputError, NoEligibleActionError
from kleroterion.interfaces import SelectionMode
from kleroterion.marginal_policy import (
    compute_entropy_diagnostics,
    compute_marginal_distribution,
    select_action,
    shannon_entropy,
)
from kleroterion.masked_softmax import stable_masked_softmax


def test_expected_softmax_is_not_softmax_of_expected_logits():
    """
    Demonstrate Section 3.3 invariant:
    Average probabilities, not logits. In general, expected softmax is not softmax of expected logits.
    """
    tau = 1.0
    mask = [True, True]
    # Two proposals with disparate logits
    logits_m1 = np.array([10.0, 0.0])
    logits_m2 = np.array([0.0, 10.0])

    p1 = stable_masked_softmax(logits_m1, mask, tau)
    p2 = stable_masked_softmax(logits_m2, mask, tau)

    # Marginal distribution = average of probabilities
    marginal_q = compute_marginal_distribution([p1, p2])
    # p1 is approx [1.0, 0.0], p2 is approx [0.0, 1.0], so average is [0.5, 0.5]
    np.testing.assert_allclose(marginal_q, [0.5, 0.5], atol=1e-4)

    # Softmax of expected logits:
    # expected logits = [5.0, 5.0]
    expected_logits = 0.5 * (logits_m1 + logits_m2)
    softmax_of_expected = stable_masked_softmax(expected_logits, mask, tau)
    # While for this symmetric case both give [0.5, 0.5], consider an asymmetric case:
    logits_a = np.array([0.0, 0.0])
    logits_b = np.array([10.0, 0.0])
    pa = stable_masked_softmax(logits_a, mask, tau)  # [0.5, 0.5]
    pb = stable_masked_softmax(logits_b, mask, tau)  # [1.0, 0.0]
    q_avg_probs = compute_marginal_distribution([pa, pb])  # [0.75, 0.25]

    mean_logits_ab = 0.5 * (logits_a + logits_b)  # [5.0, 0.0]
    q_softmax_mean_logits = stable_masked_softmax(mean_logits_ab, mask, tau)  # [0.9933, 0.0067]

    assert not np.allclose(q_avg_probs, q_softmax_mean_logits, atol=0.1)
    assert np.isclose(q_avg_probs[0], 0.75, atol=1e-4)
    assert np.isclose(q_softmax_mean_logits[0], 1.0 / (1.0 + np.exp(-5.0)), atol=1e-4)


def test_shannon_entropy():
    # Uniform over 2 elements: H = log(2) ~= 0.693147
    p_uniform = np.array([0.5, 0.5])
    assert np.isclose(shannon_entropy(p_uniform), np.log(2.0))

    # Deterministic: H = 0
    p_det = np.array([1.0, 0.0, 0.0])
    assert np.isclose(shannon_entropy(p_det), 0.0)


def test_entropy_diagnostics_and_nonnegative_disagreement():
    p1 = np.array([0.9, 0.1, 0.0])
    p2 = np.array([0.1, 0.9, 0.0])
    marginal = compute_marginal_distribution([p1, p2])

    h_cond, h_marg, j = compute_entropy_diagnostics([p1, p2], marginal)

    assert h_marg >= h_cond
    assert j >= 0.0
    assert np.isclose(j, h_marg - h_cond)


def test_select_action_argmax_stable_tie_breaking():
    actions = ["A", "B", "C"]
    mask = [True, True, True]
    rng = np.random.default_rng(42)

    # Clear winner
    probs = np.array([0.2, 0.7, 0.1])
    assert select_action(probs, mask, actions, SelectionMode.ARGMAX, rng) == "B"

    # Exact tie between A and B: must pick lowest index (A) for stability
    tie_probs = np.array([0.45, 0.45, 0.1])
    assert select_action(tie_probs, mask, actions, SelectionMode.ARGMAX, rng) == "A"


def test_select_action_sample_respects_mask():
    actions = ["A", "B", "C"]
    mask = [True, False, True]
    marginal = np.array([0.6, 0.0, 0.4])
    rng = np.random.default_rng(100)

    # 1000 draws should only yield A or C, never B
    drawn = set()
    for _ in range(1000):
        act = select_action(marginal, mask, actions, SelectionMode.SAMPLE, rng)
        assert act != "B"
        drawn.add(act)

    assert drawn == {"A", "C"}
