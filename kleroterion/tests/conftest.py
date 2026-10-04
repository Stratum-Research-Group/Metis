"""
Pytest configuration and common fixtures.
"""

import numpy as np
import pytest

from kleroterion.interfaces import (
    ControllerState,
    KleroterionConfig,
    PolicyProfile,
    PolicyProfileType,
    RngState,
    RoutingRequest,
    SelectionMode,
)


@pytest.fixture
def rng_state():
    return RngState.from_seed(42)


@pytest.fixture
def default_config():
    return KleroterionConfig(
        proposal_mode="independent",
        dt=1.0,
        noise_rms_budget=0.0,
        proposal_count=1,
        selection_mode=SelectionMode.ARGMAX,
        temperature_initial=1.0,
        adapt_temperature=False,
    )


@pytest.fixture
def exact_fact_profile():
    return PolicyProfile(
        profile_name="fact_check",
        profile_type=PolicyProfileType.EXACT_FACT,
        version="1.0",
    )


@pytest.fixture
def exploratory_profile():
    return PolicyProfile(
        profile_name="graph_explore",
        profile_type=PolicyProfileType.EXPLORATORY,
        version="1.0",
        entropy_target=0.693,
        allow_noise=True,
        allow_sampling=True,
        allow_adaptation=True,
    )


@pytest.fixture
def simple_routing_request(exact_fact_profile):
    return RoutingRequest(
        request_id="req_001",
        episode_id="ep_001",
        query_context={"text": "What is the capital of France?"},
        walk_state={"current_node": "node_paris"},
        snapshot_id="snap_20260929",
        lthcm_revision="lthcm_rev_abc123",
        feature_schema_version="schema_v1",
        policy_features=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        actions=["EXPAND:node_1", "EXPAND:node_2", "BACKTRACK", "ANSWER"],
        authoritative_mask=[True, True, True, True],
        policy_profile=exact_fact_profile,
    )


@pytest.fixture
def initial_controller_state():
    return ControllerState(
        log_temperature=0.0,  # tau = 1.0
        smoothed_marginal_entropy=None,
        support_fingerprint="",
        profile_version="1.0",
        episode_id="ep_001",
        eligible_update_count=0,
    )
