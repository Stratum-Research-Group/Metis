"""
Tests for stable masked softmax (Section 3.3 & Section 10.4).
"""

import numpy as np
import pytest

from kleroterion.errors import InvalidInputError, NoEligibleActionError
from kleroterion.masked_softmax import stable_masked_softmax


def test_basic_probabilities_and_unit_sum():
    logits = np.array([2.0, 1.0, 0.1, -1.0])
    mask = [True, True, True, False]
    temperature = 1.0

    probs = stable_masked_softmax(logits, mask, temperature)

    assert probs.shape == (4,)
    assert np.all(probs >= 0.0)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-12)
    assert probs[3] == 0.0  # Masked out action must have exactly 0 probability


def test_shift_invariance():
    logits = np.array([1.5, 3.2, -0.5, 0.0])
    mask = [True, True, False, True]
    tau = 0.8

    probs1 = stable_masked_softmax(logits, mask, tau)
    # Add large constant C
    probs2 = stable_masked_softmax(logits + 100.0, mask, tau)

    np.testing.assert_allclose(probs1, probs2, atol=1e-12)


def test_equal_logit_uniformity():
    logits = np.array([5.0, 5.0, 5.0, 1.0])
    mask = [True, True, True, False]

    for tau in [0.1, 0.5, 1.0, 5.0, 10.0]:
        probs = stable_masked_softmax(logits, mask, tau)
        assert np.isclose(probs[0], 1.0 / 3.0)
        assert np.isclose(probs[1], 1.0 / 3.0)
        assert np.isclose(probs[2], 1.0 / 3.0)
        assert probs[3] == 0.0


def test_extreme_logits_numerical_stability():
    # Extremely large positive and negative logits
    logits = np.array([1000.0, 999.0, -1000.0, 0.0])
    mask = [True, True, True, False]

    probs = stable_masked_softmax(logits, mask, temperature=1.0)

    assert np.all(np.isfinite(probs))
    assert np.isclose(np.sum(probs), 1.0, atol=1e-10)
    # exp(0) / (exp(0) + exp(-1)) = 1 / (1 + 1/e) ~= 0.731
    assert np.isclose(probs[0], 1.0 / (1.0 + np.exp(-1.0)), atol=1e-6)
    assert np.isclose(probs[2], 0.0, atol=1e-12)
    assert probs[3] == 0.0


def test_singleton_eligible_action():
    logits = np.array([-10.0, 42.0, 5.0])
    mask = [False, True, False]

    probs = stable_masked_softmax(logits, mask, temperature=1.0)
    assert probs[1] == 1.0
    assert probs[0] == 0.0
    assert probs[2] == 0.0


def test_no_eligible_action_raises():
    logits = np.array([1.0, 2.0])
    mask = [False, False]

    with pytest.raises(NoEligibleActionError, match="NO_ELIGIBLE_ACTION"):
        stable_masked_softmax(logits, mask, temperature=1.0)


def test_non_finite_eligible_logit_raises():
    logits = np.array([np.nan, 1.0])
    mask = [True, True]

    with pytest.raises(InvalidInputError, match="Non-finite"):
        stable_masked_softmax(logits, mask, temperature=1.0)


def test_invalid_temperature_raises():
    logits = np.array([1.0, 2.0])
    mask = [True, True]

    with pytest.raises(InvalidInputError, match="temperature"):
        stable_masked_softmax(logits, mask, temperature=-0.5)

    with pytest.raises(InvalidInputError, match="temperature"):
        stable_masked_softmax(logits, mask, temperature=0.0)
