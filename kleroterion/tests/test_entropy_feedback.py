"""
Tests for entropy feedback and temperature adaptation (Section 4, 10.4).
"""

import numpy as np
import pytest

from kleroterion.entropy_feedback import (
    assess_entropy_estimate_quality,
    compute_support_fingerprint,
    reset_smoothing_if_needed,
    update_controller_state,
)
from kleroterion.interfaces import (
    ControllerState,
    EntropyEstimateStatus,
    KleroterionConfig,
    PolicyProfile,
    PolicyProfileType,
    RoutingRequest,
)
from kleroterion.marginal_policy import (
    compute_entropy_diagnostics,
    compute_marginal_distribution,
)


def test_two_action_random_flip_counterexample():
    """
    Section 4.1 & 10.4:
    Reproduce the two-action random-flip counterexample:
    Noise flips a huge two-action logit gap with equal probability.
    Each proposal has nearly zero conditional entropy, while marginal choice
    entropy approaches log(2). A conditional-entropy controller would incorrectly
    raise temperature if its stated goal were aggregate exploration.
    Verify the controller consumes estimated marginal entropy.
    """
    # M proposals where half strongly favor action 0, half strongly favor action 1
    M = 100
    p_action0 = np.array([0.9999, 0.0001])
    p_action1 = np.array([0.0001, 0.9999])

    proposals = [p_action0] * (M // 2) + [p_action1] * (M // 2)
    marginal_q = compute_marginal_distribution(proposals)

    h_cond, h_marg, j = compute_entropy_diagnostics(proposals, marginal_q)

    # Conditional entropy is near zero:
    assert h_cond < 0.01
    # Marginal entropy is near log(2) ~= 0.6931:
    assert np.isclose(h_marg, np.log(2.0), atol=1e-3)
    # Disagreement J is large:
    assert j > 0.68


def test_controller_consumes_marginal_entropy_not_conditional():
    config = KleroterionConfig(
        proposal_count=4,
        temperature_initial=1.0,
        temperature_bounds=(0.1, 5.0),
        adapt_temperature=True,
        entropy_target=0.5,
        entropy_ema_alpha=1.0,  # Immediate update
        controller_gain=0.2,
        entropy_deadband=0.0,
    )
    ctrl_state = ControllerState(
        log_temperature=0.0,  # tau = 1.0
        smoothed_marginal_entropy=None,
    )
    status = EntropyEstimateStatus(accepted=True, reason="", variance=0.01)

    # Case A: Marginal entropy is high (e.g. 0.693 / log(2) = 1.0). Error = 0.5 - 1.0 = -0.5.
    # Gain * error = 0.2 * (-0.5) = -0.1 => log_tau decreases => tau decreases!
    next_state, tau_next = update_controller_state(
        controller_state=ctrl_state,
        h_marginal=np.log(2.0),
        num_eligible=2,
        config=config,
        estimate_status=status,
    )
    assert next_state.smoothed_marginal_entropy == 1.0
    assert next_state.log_temperature < 0.0
    assert tau_next < 1.0


def test_quality_gate_rejects_single_proposal():
    config = KleroterionConfig(proposal_count=1)
    status = assess_entropy_estimate_quality([np.array([0.5, 0.5])], config)
    assert not status.accepted
    assert "M=1 < 2" in status.reason


def test_singleton_support_freezes_feedback():
    config = KleroterionConfig(
        proposal_count=4,
        temperature_bounds=(0.1, 5.0),
        adapt_temperature=True,
        entropy_target=0.5,
        entropy_ema_alpha=0.5,
        controller_gain=0.1,
    )
    ctrl_state = ControllerState(log_temperature=0.0)
    status = EntropyEstimateStatus(accepted=True, reason="")

    next_state, tau_next = update_controller_state(
        controller_state=ctrl_state,
        h_marginal=0.0,
        num_eligible=1,  # Singleton support!
        config=config,
        estimate_status=status,
    )
    # Log temperature and state remain unchanged
    assert next_state.log_temperature == 0.0
    assert tau_next == 1.0


def test_deadband_suppresses_small_errors():
    config = KleroterionConfig(
        proposal_count=4,
        temperature_bounds=(0.1, 5.0),
        adapt_temperature=True,
        entropy_target=0.5,
        entropy_ema_alpha=1.0,
        controller_gain=0.2,
        entropy_deadband=0.1,  # Deadband delta = 0.1
    )
    ctrl_state = ControllerState(log_temperature=0.0)
    status = EntropyEstimateStatus(accepted=True, reason="")

    # Error = 0.5 - 0.55 = -0.05. |error| = 0.05 <= 0.1 deadband => no change
    h_marg = 0.55 * np.log(2.0)
    next_state, tau_next = update_controller_state(
        controller_state=ctrl_state,
        h_marginal=h_marg,
        num_eligible=2,
        config=config,
        estimate_status=status,
    )
    assert next_state.log_temperature == 0.0
    assert tau_next == 1.0


def test_reset_smoothing_on_episode_or_support_change(simple_routing_request):
    ctrl_state = ControllerState(
        log_temperature=0.0,
        smoothed_marginal_entropy=0.75,
        support_fingerprint="old_fp",
        profile_version="1.0",
        episode_id="ep_001",
        eligible_update_count=5,
    )

    # Same episode, different support
    new_fp = "new_fp"
    reset_state = reset_smoothing_if_needed(ctrl_state, simple_routing_request, new_fp)
    assert reset_state.smoothed_marginal_entropy is None
    assert reset_state.eligible_update_count == 0
    assert reset_state.support_fingerprint == new_fp
