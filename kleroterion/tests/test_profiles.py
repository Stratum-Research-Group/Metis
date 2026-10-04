"""
Tests for PolicyProfile overrides and behavior (Section 7, 8, 10.4).
"""

import numpy as np
import pytest

from kleroterion.core import propose_action, resolve_effective_config
from kleroterion.interfaces import (
    HostCallbacks,
    KleroterionConfig,
    PolicyProfile,
    PolicyProfileType,
    RoutingRequest,
    SelectionMode,
)


def test_exact_fact_profile_overrides_exploratory_config(rng_state, initial_controller_state):
    """
    Section 7 & 8:
    Exact-fact profiles enforce noise=0, argmax, and adaptation=false
    regardless of exploratory defaults.
    """
    exploratory_config = KleroterionConfig(
        dt=1.0,
        noise_rms_budget=0.5,
        noise_hard_radius=1.0,
        proposal_count=5,
        selection_mode=SelectionMode.SAMPLE,
        temperature_initial=1.0,
        temperature_bounds=(0.1, 5.0),
        adapt_temperature=True,
        entropy_target=0.8,
        entropy_ema_alpha=0.2,
        controller_gain=0.1,
    )

    exact_fact_profile = PolicyProfile(
        profile_name="fact_check",
        profile_type=PolicyProfileType.EXACT_FACT,
        version="1.0",
    )

    req = RoutingRequest(
        request_id="req_fact",
        episode_id="ep_1",
        query_context=None,
        walk_state=None,
        snapshot_id="snap1",
        lthcm_revision="rev1",
        feature_schema_version="v1",
        policy_features=np.array([1.0, 2.0]),
        actions=["A", "B"],
        authoritative_mask=[True, True],
        policy_profile=exact_fact_profile,
    )

    effective = resolve_effective_config(req, exploratory_config)
    assert effective.noise_rms_budget == 0.0
    assert effective.noise_hard_radius is None
    assert effective.proposal_count == 1
    assert effective.selection_mode == SelectionMode.ARGMAX
    assert effective.adapt_temperature is False

    callbacks = HostCallbacks(
        proposal_features=lambda x, inc: x + inc,
        proposal_mask=lambda r, x: [True, True],
        score_actions=lambda r, x, acts: np.array([1.0, 2.0]),
        revalidate_before_execute=lambda act, snap: (True, None),
    )

    decision = propose_action(req, exploratory_config, initial_controller_state, rng_state, callbacks)
    assert decision.proposal_count == 1
    assert decision.selected_action_id == "B"  # Argmax of [1.0, 2.0]
    assert decision.temperature_next == decision.temperature_used
