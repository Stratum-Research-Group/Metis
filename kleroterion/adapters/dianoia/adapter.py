"""
Adapter mapping Dianoia walk actions and host constraints to Kleroterion contracts.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from kleroterion.core import host_accept, propose_action
from kleroterion.errors import InvalidInputError, RevalidationError
from kleroterion.interfaces import (
    ActionDecision,
    ActionID,
    ControllerState,
    HostCallbacks,
    KleroterionConfig,
    PolicyProfile,
    RngState,
    RoutingRequest,
)


@dataclass(frozen=True)
class DianoiaWalkAction:
    """
    Representation of an action in Dianoia's deliberative walk.
    """
    action_type: str  # "EXPAND", "BACKTRACK", or "ANSWER"
    target: Optional[str] = None  # neighbour_id for EXPAND

    def to_action_id(self) -> ActionID:
        if self.action_type == "EXPAND":
            if not self.target:
                raise InvalidInputError("EXPAND action requires a non-empty target neighbour ID.")
            return f"EXPAND:{self.target}"
        elif self.action_type in ("BACKTRACK", "ANSWER"):
            return self.action_type
        else:
            raise InvalidInputError(f"Unsupported action type for Dianoia: {self.action_type}")

    @classmethod
    def from_action_id(cls, action_id: ActionID) -> "DianoiaWalkAction":
        if action_id.startswith("EXPAND:"):
            target = action_id[len("EXPAND:"):]
            return cls(action_type="EXPAND", target=target)
        elif action_id in ("BACKTRACK", "ANSWER"):
            return cls(action_type=action_id, target=None)
        else:
            raise InvalidInputError(f"Cannot map unknown ActionID '{action_id}' to DianoiaWalkAction.")


class DianoiaActionAdapter:
    """
    Adapter between Dianoia WalkController and Kleroterion.
    """

    def __init__(
        self,
        scorer_fn: Callable[[RoutingRequest, np.ndarray, List[ActionID]], np.ndarray],
        revalidation_fn: Callable[[DianoiaWalkAction, str], Tuple[bool, Optional[str]]],
        proposal_mask_fn: Optional[Callable[[RoutingRequest, np.ndarray], List[bool]]] = None,
        feature_transform_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    ):
        self.scorer_fn = scorer_fn
        self.revalidation_fn = revalidation_fn
        self.proposal_mask_fn = proposal_mask_fn
        self.feature_transform_fn = feature_transform_fn

    def build_routing_request(
        self,
        request_id: str,
        episode_id: str,
        snapshot_id: str,
        lthcm_revision: str,
        policy_features: np.ndarray,
        valid_actions: List[DianoiaWalkAction],
        authoritative_mask: List[bool],
        policy_profile: PolicyProfile,
        walk_state: Optional[Any] = None,
        query_context: Optional[Any] = None,
        feature_schema_version: str = "dianoia.v1",
    ) -> RoutingRequest:
        """
        Construct a validated RoutingRequest from Dianoia candidates and state.
        """
        action_ids = [act.to_action_id() for act in valid_actions]
        return RoutingRequest(
            request_id=request_id,
            episode_id=episode_id,
            query_context=query_context,
            walk_state=walk_state,
            snapshot_id=snapshot_id,
            lthcm_revision=lthcm_revision,
            feature_schema_version=feature_schema_version,
            policy_features=policy_features,
            actions=action_ids,
            authoritative_mask=authoritative_mask,
            policy_profile=policy_profile,
        )

    def build_host_callbacks(self) -> HostCallbacks:
        """
        Assemble side-effect-free HostCallbacks for Kleroterion core.
        """
        def default_proposal_features(x: np.ndarray, increment: np.ndarray) -> np.ndarray:
            if self.feature_transform_fn is not None:
                return self.feature_transform_fn(x, increment)
            return x + increment

        def default_proposal_mask(req: RoutingRequest, x_proposed: np.ndarray) -> List[bool]:
            if self.proposal_mask_fn is not None:
                return self.proposal_mask_fn(req, x_proposed)
            # Default: preserve authoritative mask
            return list(req.authoritative_mask)

        def wrapped_revalidate(action_id: ActionID, snapshot_id: str) -> Tuple[bool, Optional[str]]:
            action = DianoiaWalkAction.from_action_id(action_id)
            return self.revalidation_fn(action, snapshot_id)

        return HostCallbacks(
            proposal_features=default_proposal_features,
            proposal_mask=default_proposal_mask,
            score_actions=self.scorer_fn,
            revalidate_before_execute=wrapped_revalidate,
        )

    def route_step(
        self,
        request: RoutingRequest,
        config: KleroterionConfig,
        controller_state: ControllerState,
        rng_state: RngState,
    ) -> Tuple[DianoiaWalkAction, ActionDecision]:
        """
        Propose stochastic walk decision and map to DianoiaWalkAction.
        """
        callbacks = self.build_host_callbacks()
        decision = propose_action(
            request=request,
            config=config,
            controller_state=controller_state,
            rng_state=rng_state,
            callbacks=callbacks,
        )
        dianoia_action = DianoiaWalkAction.from_action_id(decision.selected_action_id)
        return dianoia_action, decision

    def commit_step(
        self,
        decision: ActionDecision,
        executed_request_ids: Set[str],
    ) -> Tuple[DianoiaWalkAction, ControllerState, RngState]:
        """
        Commit host acceptance transaction. Aborts on revalidation failure.
        """
        callbacks = self.build_host_callbacks()
        host_accept(decision, executed_request_ids, callbacks)
        dianoia_action = DianoiaWalkAction.from_action_id(decision.selected_action_id)
        return dianoia_action, decision.controller_state_next, decision.rng_state_next
