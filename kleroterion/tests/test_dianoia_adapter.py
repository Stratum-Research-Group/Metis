"""
Tests for Dianoia adapter (Section 2, 6, 9).
"""

import numpy as np
import pytest

from adapters.dianoia.adapter import DianoiaActionAdapter, DianoiaWalkAction
from kleroterion.errors import InvalidInputError, RevalidationError
from kleroterion.interfaces import (
    ControllerState,
    KleroterionConfig,
    PolicyProfile,
    PolicyProfileType,
    RngState,
    SelectionMode,
)


def test_dianoia_walk_action_mapping():
    # Expand
    act1 = DianoiaWalkAction(action_type="EXPAND", target="node_42")
    act_id1 = act1.to_action_id()
    assert act_id1 == "EXPAND:node_42"
    assert DianoiaWalkAction.from_action_id(act_id1) == act1

    # Backtrack
    act2 = DianoiaWalkAction(action_type="BACKTRACK")
    assert act2.to_action_id() == "BACKTRACK"
    assert DianoiaWalkAction.from_action_id("BACKTRACK") == act2

    # Answer
    act3 = DianoiaWalkAction(action_type="ANSWER")
    assert act3.to_action_id() == "ANSWER"
    assert DianoiaWalkAction.from_action_id("ANSWER") == act3

    # Invalid action
    with pytest.raises(InvalidInputError):
        DianoiaWalkAction(action_type="EXPAND", target=None).to_action_id()

    with pytest.raises(InvalidInputError):
        DianoiaWalkAction.from_action_id("UNKNOWN:123")


def test_dianoia_adapter_end_to_end_route_and_commit():
    def dummy_scorer(req, x, actions):
        # Prefer EXPAND:neighbour_2
        scores = []
        for a in actions:
            if a == "EXPAND:neighbour_2":
                scores.append(10.0)
            elif a == "EXPAND:neighbour_1":
                scores.append(5.0)
            else:
                scores.append(1.0)
        return np.array(scores, dtype=np.float64)

    def dummy_revalidation(action: DianoiaWalkAction, snapshot_id: str):
        # Valid if snapshot_id matches
        if snapshot_id == "snap_valid":
            return True, None
        return False, "Snapshot mismatch"

    adapter = DianoiaActionAdapter(
        scorer_fn=dummy_scorer,
        revalidation_fn=dummy_revalidation,
    )

    actions = [
        DianoiaWalkAction("EXPAND", "neighbour_1"),
        DianoiaWalkAction("EXPAND", "neighbour_2"),
        DianoiaWalkAction("BACKTRACK"),
    ]
    mask = [True, True, True]

    req = adapter.build_routing_request(
        request_id="req_dianoia_1",
        episode_id="ep_d1",
        snapshot_id="snap_valid",
        lthcm_revision="rev_1",
        policy_features=np.array([0.5, 0.5]),
        valid_actions=actions,
        authoritative_mask=mask,
        policy_profile=PolicyProfile("explore", PolicyProfileType.EXACT_FACT),
    )

    cfg = KleroterionConfig()
    ctrl = ControllerState(0.0)
    rng = RngState.from_seed(42)

    # Route step
    dianoia_action, decision = adapter.route_step(req, cfg, ctrl, rng)
    assert dianoia_action.action_type == "EXPAND"
    assert dianoia_action.target == "neighbour_2"

    # Commit step
    executed = set()
    committed_action, next_ctrl, next_rng = adapter.commit_step(decision, executed)
    assert committed_action == dianoia_action
    assert "req_dianoia_1" in executed


def test_dianoia_adapter_commit_revalidation_failure():
    adapter = DianoiaActionAdapter(
        scorer_fn=lambda r, x, a: np.array([1.0]),
        revalidation_fn=lambda act, snap: (False, "Target node pruned from memory graph"),
    )

    actions = [DianoiaWalkAction("EXPAND", "pruned_node")]
    req = adapter.build_routing_request(
        request_id="req_fail",
        episode_id="ep_1",
        snapshot_id="snap_1",
        lthcm_revision="rev_1",
        policy_features=np.array([0.1]),
        valid_actions=actions,
        authoritative_mask=[True],
        policy_profile=PolicyProfile("fact", PolicyProfileType.EXACT_FACT),
    )

    _, decision = adapter.route_step(req, KleroterionConfig(), ControllerState(0.0), RngState.from_seed(1))
    executed = set()
    with pytest.raises(RevalidationError, match="Target node pruned"):
        adapter.commit_step(decision, executed)
