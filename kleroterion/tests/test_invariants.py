"""
Tests for Safety and Identity Invariants (Section 5, 10.4).
"""

import numpy as np
import pytest

from kleroterion.core import host_accept, propose_action
from kleroterion.errors import (
    NoEligibleActionError,
    RevalidationError,
    TransactionAbortedError,
)
from kleroterion.interfaces import (
    ActionID,
    ControllerState,
    HostCallbacks,
    KleroterionConfig,
    PolicyProfile,
    PolicyProfileType,
    RngState,
    RoutingRequest,
    SelectionMode,
)
from kleroterion.weighted import WeightedFeatureVector, compute_weighted_features


def test_invariant_1_authoritative_filters_cannot_be_bypassed(
    default_config, initial_controller_state, rng_state
):
    """
    Invariant 1: A host-rejected action in authoritative_mask cannot become eligible.
    """
    req = RoutingRequest(
        request_id="req_inv1",
        episode_id="ep_1",
        query_context=None,
        walk_state=None,
        snapshot_id="snap1",
        lthcm_revision="rev1",
        feature_schema_version="v1",
        policy_features=np.array([1.0, 2.0]),
        actions=["A", "B", "C"],
        authoritative_mask=[True, False, True],  # B is authoritatively rejected!
        policy_profile=PolicyProfile(
            profile_name="test",
            profile_type=PolicyProfileType.EXPLORATORY,
        ),
    )

    # Even if proposal_mask tries to enable B:
    callbacks = HostCallbacks(
        proposal_features=lambda x, inc: x + inc,
        proposal_mask=lambda r, x: [True, True, True],
        score_actions=lambda r, x, acts: np.array([10.0, 100.0, 5.0]),  # B has huge score!
        revalidate_before_execute=lambda act, snap: (True, None),
    )

    decision = propose_action(req, default_config, initial_controller_state, rng_state, callbacks)
    # B must have 0 probability and must not be selected
    assert decision.marginal_probabilities[1] == 0.0
    assert decision.selected_action_id != "B"
    assert decision.diagnostics["common_mask"][1] is False


def test_invariant_2_eligibility_can_only_narrow_and_fails_closed(
    default_config, initial_controller_state, rng_state
):
    """
    Invariant 2: Eligibility can only narrow; empty intersection fails closed.
    """
    req = RoutingRequest(
        request_id="req_inv2",
        episode_id="ep_1",
        query_context=None,
        walk_state=None,
        snapshot_id="snap1",
        lthcm_revision="rev1",
        feature_schema_version="v1",
        policy_features=np.array([1.0, 2.0]),
        actions=["A", "B"],
        authoritative_mask=[True, True],
        policy_profile=PolicyProfile(profile_name="test", profile_type=PolicyProfileType.EXPLORATORY),
    )

    # Proposal mask rejects all actions
    callbacks = HostCallbacks(
        proposal_features=lambda x, inc: x + inc,
        proposal_mask=lambda r, x: [False, False],
        score_actions=lambda r, x, acts: np.array([1.0, 1.0]),
        revalidate_before_execute=lambda act, snap: (True, None),
    )

    with pytest.raises(NoEligibleActionError, match="NO_ELIGIBLE_ACTION"):
        propose_action(req, default_config, initial_controller_state, rng_state, callbacks)


def test_invariant_3_revalidate_before_execution_aborts_on_stale():
    """
    Invariant 3: A stale or newly invalid action aborts the transaction.
    """
    from kleroterion.interfaces import ActionDecision, EntropyEstimateStatus

    decision = ActionDecision(
        request_id="req_stale",
        snapshot_id="snap_v1",
        lthcm_revision="rev_v1",
        selected_action_id="EXPAND:node_x",
        action_ids=["EXPAND:node_x"],
        marginal_probabilities=np.array([1.0]),
        conditional_entropy=0.0,
        marginal_entropy=0.0,
        proposal_disagreement=0.0,
        proposal_count=1,
        entropy_estimate_status=EntropyEstimateStatus(True, ""),
        temperature_used=1.0,
        temperature_next=1.0,
        controller_state_next=ControllerState(0.0),
        rng_state_next=RngState.from_seed(1),
    )

    executed_ids = set()

    # Revalidation fails because snapshot is stale
    callbacks = HostCallbacks(
        proposal_features=lambda x, inc: x + inc,
        proposal_mask=lambda r, x: [True],
        score_actions=lambda r, x, acts: np.array([1.0]),
        revalidate_before_execute=lambda act, snap: (False, "Stale snapshot detected"),
    )

    with pytest.raises(RevalidationError, match="failed pre-execution revalidation"):
        host_accept(decision, executed_ids, callbacks)

    assert "req_stale" not in executed_ids


def test_invariant_4_and_5_weighted_feature_vector_type_distinction():
    """
    Invariant 4 & 5: WeightedFeatureVector is a distinct type, cannot be an ActionDecision or ActionID.
    """
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([0.6, 0.4])

    wfv = compute_weighted_features(features, weights)
    assert isinstance(wfv, WeightedFeatureVector)
    assert not isinstance(wfv, str)  # Not an ActionID
    np.testing.assert_allclose(wfv.features, [0.6 * 1 + 0.4 * 3, 0.6 * 2 + 0.4 * 4])


def test_invariant_6_no_implicit_writes_and_idempotency(
    default_config, initial_controller_state, rng_state
):
    """
    Invariant 6: State commits only upon successful host acceptance; retries cannot duplicate execute.
    """
    req = RoutingRequest(
        request_id="req_tx1",
        episode_id="ep_1",
        query_context=None,
        walk_state=None,
        snapshot_id="snap1",
        lthcm_revision="rev1",
        feature_schema_version="v1",
        policy_features=np.array([1.0, 2.0]),
        actions=["A", "B"],
        authoritative_mask=[True, True],
        policy_profile=PolicyProfile(profile_name="test", profile_type=PolicyProfileType.EXACT_FACT),
    )

    callbacks = HostCallbacks(
        proposal_features=lambda x, inc: x + inc,
        proposal_mask=lambda r, x: [True, True],
        score_actions=lambda r, x, acts: np.array([2.0, 1.0]),
        revalidate_before_execute=lambda act, snap: (True, None),
    )

    executed_ids = set()
    decision = propose_action(req, default_config, initial_controller_state, rng_state, callbacks)

    # First accept commits
    ok, msg = host_accept(decision, executed_ids, callbacks)
    assert ok and msg == "ACCEPTED"
    assert "req_tx1" in executed_ids

    # Second accept must raise TransactionAbortedError (duplicate execution)
    with pytest.raises(TransactionAbortedError, match="Duplicate execution"):
        host_accept(decision, executed_ids, callbacks)
