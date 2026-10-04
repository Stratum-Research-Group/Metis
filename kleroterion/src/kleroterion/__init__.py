"""
Kleroterion: Stochastic Decision Submodule.
"""

from kleroterion.core import host_accept, propose_action, resolve_effective_config
from kleroterion.entropy_feedback import (
    assess_entropy_estimate_quality,
    compute_support_fingerprint,
    reset_smoothing_if_needed,
    update_controller_state,
)
from kleroterion.errors import (
    InvalidConfigError,
    InvalidInputError,
    InvalidProposalError,
    KleroterionError,
    NoEligibleActionError,
    RevalidationError,
    ScorerError,
    TransactionAbortedError,
)
from kleroterion.interfaces import (
    ActionDecision,
    ActionID,
    ControllerState,
    EntropyEstimateStatus,
    HostCallbacks,
    KleroterionConfig,
    PolicyProfile,
    PolicyProfileType,
    RngState,
    RoutingRequest,
    SelectionMode,
)
from kleroterion.marginal_policy import (
    compute_entropy_diagnostics,
    compute_marginal_distribution,
    select_action,
    shannon_entropy,
)
from kleroterion.masked_softmax import stable_masked_softmax
from kleroterion.proposals import (
    clip_increment_to_radius,
    generate_budgeted_gaussian_increment,
    verify_proposed_features,
)
from kleroterion.weighted import WeightedFeatureVector, compute_weighted_features

__version__ = "0.1.0"

__all__ = [
    "propose_action",
    "host_accept",
    "resolve_effective_config",
    "stable_masked_softmax",
    "generate_budgeted_gaussian_increment",
    "clip_increment_to_radius",
    "verify_proposed_features",
    "compute_marginal_distribution",
    "compute_entropy_diagnostics",
    "select_action",
    "shannon_entropy",
    "assess_entropy_estimate_quality",
    "compute_support_fingerprint",
    "reset_smoothing_if_needed",
    "update_controller_state",
    "compute_weighted_features",
    "WeightedFeatureVector",
    "ActionID",
    "SelectionMode",
    "PolicyProfileType",
    "PolicyProfile",
    "KleroterionConfig",
    "RoutingRequest",
    "HostCallbacks",
    "ControllerState",
    "RngState",
    "EntropyEstimateStatus",
    "ActionDecision",
    "KleroterionError",
    "InvalidConfigError",
    "InvalidInputError",
    "InvalidProposalError",
    "NoEligibleActionError",
    "ScorerError",
    "RevalidationError",
    "TransactionAbortedError",
]
