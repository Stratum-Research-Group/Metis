"""
Core interfaces, data contracts, and configuration for Kleroterion.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from kleroterion.errors import InvalidConfigError, InvalidInputError

ActionID = str


class SelectionMode(str, Enum):
    ARGMAX = "argmax"
    SAMPLE = "sample"


class PolicyProfileType(str, Enum):
    EXACT_FACT = "exact_fact"
    EXPLORATORY = "exploratory"


@dataclass(frozen=True)
class PolicyProfile:
    """
    Policy profile governing execution mode.
    
    Exact-fact profiles enforce zero noise, argmax selection, and no adaptation.
    Exploratory profiles require justified entropy targets and validated parameters.
    """
    profile_name: str
    profile_type: PolicyProfileType = PolicyProfileType.EXACT_FACT
    version: str = "1.0"
    entropy_target: Optional[float] = None
    allow_noise: bool = True
    allow_sampling: bool = True
    allow_adaptation: bool = True

    def is_exact_fact(self) -> bool:
        return self.profile_type == PolicyProfileType.EXACT_FACT


@dataclass(frozen=True)
class KleroterionConfig:
    """
    Configuration parameters for Kleroterion stochastic decisions.
    Defaults are conservative baselines.
    """
    proposal_mode: str = "independent"
    dt: float = 1.0
    noise_rms_budget: float = 0.0
    noise_hard_radius: Optional[float] = None
    diffusion_matrix: Optional[np.ndarray] = None
    proposal_count: int = 1
    selection_mode: SelectionMode = SelectionMode.ARGMAX
    temperature_initial: float = 1.0
    temperature_bounds: Optional[Tuple[float, float]] = None  # (tau_min, tau_max)
    adapt_temperature: bool = False
    entropy_target: Optional[float] = None
    entropy_ema_alpha: Optional[float] = None
    controller_gain: Optional[float] = None
    entropy_deadband: float = 0.0
    entropy_quality_gate: Optional[Dict[str, Any]] = None
    stability_limits: Optional[Dict[str, float]] = None

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise InvalidConfigError(f"dt must be positive (got {self.dt}).")
        if self.proposal_count < 1:
            raise InvalidConfigError(f"proposal_count must be >= 1 (got {self.proposal_count}).")
        if self.noise_rms_budget < 0.0:
            raise InvalidConfigError(f"noise_rms_budget must be >= 0.0 (got {self.noise_rms_budget}).")
        
        # When noise is enabled (noise_rms_budget > 0), a hard radius R >= b > 0 is required
        if self.noise_rms_budget > 0.0:
            if self.noise_hard_radius is None:
                raise InvalidConfigError(
                    "noise_hard_radius is required when noise_rms_budget > 0."
                )
            if self.noise_hard_radius < self.noise_rms_budget:
                raise InvalidConfigError(
                    f"noise_hard_radius ({self.noise_hard_radius}) must be >= noise_rms_budget ({self.noise_rms_budget})."
                )

        if self.temperature_initial <= 0.0:
            raise InvalidConfigError(
                f"temperature_initial must be positive (got {self.temperature_initial})."
            )

        if self.temperature_bounds is not None:
            tau_min, tau_max = self.temperature_bounds
            if tau_min <= 0.0 or tau_max <= 0.0 or tau_min > tau_max:
                raise InvalidConfigError(
                    f"Invalid temperature_bounds ({tau_min}, {tau_max}): require 0 < min <= max."
                )
            if not (tau_min <= self.temperature_initial <= tau_max):
                raise InvalidConfigError(
                    f"temperature_initial ({self.temperature_initial}) must be within bounds [{tau_min}, {tau_max}]."
                )

        if self.adapt_temperature:
            if self.temperature_bounds is None:
                raise InvalidConfigError("temperature_bounds are required when adapt_temperature is True.")
            if self.entropy_target is None or self.entropy_target < 0.0:
                raise InvalidConfigError("A non-negative entropy_target is required when adapt_temperature is True.")
            if self.entropy_ema_alpha is None or not (0.0 < self.entropy_ema_alpha <= 1.0):
                raise InvalidConfigError("entropy_ema_alpha must be in (0, 1] when adapt_temperature is True.")
            if self.controller_gain is None or self.controller_gain < 0.0:
                raise InvalidConfigError("controller_gain must be non-negative when adapt_temperature is True.")
            if not (0.0 <= self.entropy_deadband <= 1.0):
                raise InvalidConfigError(f"entropy_deadband must be in [0, 1] (got {self.entropy_deadband}).")
            if self.proposal_count < 2:
                raise InvalidConfigError(
                    f"adapt_temperature requires proposal_count >= 2 (got {self.proposal_count})."
                )


@dataclass
class RoutingRequest:
    """
    Authoritative routing request supplied by the host (Dianoia).
    """
    request_id: str
    episode_id: str
    query_context: Any
    walk_state: Any
    snapshot_id: str
    lthcm_revision: str
    feature_schema_version: str
    policy_features: np.ndarray
    actions: List[ActionID]
    authoritative_mask: List[bool]
    policy_profile: PolicyProfile

    def validate(self) -> None:
        if not self.request_id:
            raise InvalidInputError("request_id cannot be empty.")
        if not self.snapshot_id:
            raise InvalidInputError("snapshot_id cannot be empty.")
        if not self.lthcm_revision:
            raise InvalidInputError("lthcm_revision cannot be empty.")
        if not isinstance(self.policy_features, np.ndarray) or not np.all(np.isfinite(self.policy_features)):
            raise InvalidInputError("policy_features must be a finite numpy ndarray.")
        if self.policy_features.ndim != 1:
            raise InvalidInputError("policy_features must be a 1D vector.")
        
        k = len(self.actions)
        if k == 0:
            raise InvalidInputError("actions list cannot be empty.")
        if len(set(self.actions)) != k:
            raise InvalidInputError("actions list must contain unique ActionIDs.")
        if len(self.authoritative_mask) != k:
            raise InvalidInputError(
                f"authoritative_mask length ({len(self.authoritative_mask)}) must match actions length ({k})."
            )


@dataclass
class HostCallbacks:
    """
    Side-effect-free callbacks into host logic.
    """
    proposal_features: Callable[[np.ndarray, np.ndarray], np.ndarray]
    proposal_mask: Callable[[RoutingRequest, np.ndarray], List[bool]]
    score_actions: Callable[[RoutingRequest, np.ndarray, List[ActionID]], np.ndarray]
    revalidate_before_execute: Callable[[ActionID, str], Tuple[bool, Optional[str]]]


@dataclass
class ControllerState:
    """
    Persistent state for the marginal entropy feedback controller.
    """
    log_temperature: float
    smoothed_marginal_entropy: Optional[float] = None
    support_fingerprint: str = ""
    profile_version: str = ""
    episode_id: str = ""
    eligible_update_count: int = 0

    def copy(self) -> "ControllerState":
        return ControllerState(
            log_temperature=self.log_temperature,
            smoothed_marginal_entropy=self.smoothed_marginal_entropy,
            support_fingerprint=self.support_fingerprint,
            profile_version=self.profile_version,
            episode_id=self.episode_id,
            eligible_update_count=self.eligible_update_count,
        )


@dataclass
class RngState:
    """
    Separate random streams for proposals and action selection.
    """
    proposal_rng: np.random.Generator
    selection_rng: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int) -> "RngState":
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(2)
        return cls(
            proposal_rng=np.random.default_rng(child_seeds[0]),
            selection_rng=np.random.default_rng(child_seeds[1]),
        )

    def copy(self) -> "RngState":
        # Deep copy bit generators to preserve deterministic trajectory
        import copy
        return copy.deepcopy(self)


@dataclass(frozen=True)
class EntropyEstimateStatus:
    accepted: bool
    reason: str
    variance: float = 0.0


@dataclass(frozen=True)
class ActionDecision:
    """
    Result of a Kleroterion decision proposal.
    
    Safety Invariant 4:
    Kleroterion returns action IDs and probabilities, never a document or citation.
    """
    request_id: str
    snapshot_id: str
    lthcm_revision: str
    selected_action_id: ActionID
    action_ids: List[ActionID]
    marginal_probabilities: np.ndarray
    conditional_entropy: float
    marginal_entropy: float
    proposal_disagreement: float
    proposal_count: int
    entropy_estimate_status: EntropyEstimateStatus
    temperature_used: float
    temperature_next: float
    controller_state_next: ControllerState
    rng_state_next: RngState
    diagnostics: Dict[str, Any] = field(default_factory=dict)
