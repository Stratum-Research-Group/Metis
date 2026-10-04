"""
Tests for reproducibility, replay, and independent RNG streams (Section 10.4).
"""

import numpy as np
import pytest

from kleroterion.core import propose_action
from kleroterion.interfaces import (
    ControllerState,
    HostCallbacks,
    KleroterionConfig,
    PolicyProfile,
    PolicyProfileType,
    RngState,
    RoutingRequest,
    SelectionMode,
)


def test_reproducibility_with_pinned_seed():
    """
    Verify that identical seeds and configurations produce identical decisions.
    """
    profile = PolicyProfile("explore", PolicyProfileType.EXPLORATORY, allow_noise=True, allow_sampling=True)
    cfg = KleroterionConfig(
        dt=1.0,
        noise_rms_budget=0.3,
        noise_hard_radius=0.5,
        proposal_count=3,
        selection_mode=SelectionMode.SAMPLE,
        temperature_initial=1.2,
    )

    req = RoutingRequest(
        request_id="req_rep",
        episode_id="ep_rep",
        query_context=None,
        walk_state=None,
        snapshot_id="snap1",
        lthcm_revision="rev1",
        feature_schema_version="v1",
        policy_features=np.array([0.1, -0.2, 0.5, 1.0]),
        actions=["A", "B", "C"],
        authoritative_mask=[True, True, True],
        policy_profile=profile,
    )

    def scorer(r, x, acts):
        # Deterministic linear function of perturbed features
        return np.array([float(np.dot(x, [1, 0, 0, 0])), float(np.dot(x, [0, 1, 0, 0])), 0.5])

    callbacks = HostCallbacks(
        proposal_features=lambda x, inc: x + inc,
        proposal_mask=lambda r, x: [True, True, True],
        score_actions=scorer,
        revalidate_before_execute=lambda a, s: (True, None),
    )

    ctrl1 = ControllerState(log_temperature=np.log(1.2))
    ctrl2 = ControllerState(log_temperature=np.log(1.2))
    rng1 = RngState.from_seed(99999)
    rng2 = RngState.from_seed(99999)

    dec1 = propose_action(req, cfg, ctrl1, rng1, callbacks)
    dec2 = propose_action(req, cfg, ctrl2, rng2, callbacks)

    assert dec1.selected_action_id == dec2.selected_action_id
    np.testing.assert_allclose(dec1.marginal_probabilities, dec2.marginal_probabilities)
    assert np.isclose(dec1.conditional_entropy, dec2.conditional_entropy)
    assert np.isclose(dec1.marginal_entropy, dec2.marginal_entropy)


def test_independent_rng_streams():
    """
    Verify proposal RNG and selection RNG operate independently.
    """
    rng = RngState.from_seed(12345)
    prop_sample1 = rng.proposal_rng.standard_normal(5)
    sel_sample1 = rng.selection_rng.standard_normal(5)

    # They should not be identical
    assert not np.allclose(prop_sample1, sel_sample1)
