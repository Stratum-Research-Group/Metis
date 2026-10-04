"""
Entropy-controlled exploration and temperature adaptation.
"""

import hashlib
from typing import List, Optional, Sequence, Tuple
import numpy as np

from kleroterion.errors import InvalidConfigError
from kleroterion.interfaces import (
    ActionID,
    ControllerState,
    EntropyEstimateStatus,
    KleroterionConfig,
    RoutingRequest,
)


def compute_support_fingerprint(actions: Sequence[ActionID], mask: Sequence[bool]) -> str:
    """
    Compute deterministic fingerprint of eligible action support.
    """
    eligible_actions = [actions[i] for i, m in enumerate(mask) if m]
    raw = "|".join(eligible_actions)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def assess_entropy_estimate_quality(
    distributions: Sequence[np.ndarray],
    config: KleroterionConfig,
) -> EntropyEstimateStatus:
    """
    Quality gate assessment for marginal entropy estimate.
    
    Section 4.2:
    Requires M >= 2 plus an evaluation-validated proposal count;
    log M and proposal variability, and freeze adaptation when a configured
    uncertainty check rejects the estimate.
    """
    m = len(distributions)
    if m < 2:
        return EntropyEstimateStatus(
            accepted=False,
            reason=f"Proposal count M={m} < 2; between-proposal exploration requires M >= 2.",
            variance=0.0,
        )

    # Compute proposal distribution variance across M proposals
    stacked = np.stack(distributions, axis=0)  # (M, K)
    # Variance of probabilities across proposals per action, averaged over actions
    proposal_var = float(np.mean(np.var(stacked, axis=0)))

    # Check custom quality gate thresholds if configured
    if config.entropy_quality_gate is not None:
        max_var = config.entropy_quality_gate.get("max_variance")
        if max_var is not None and proposal_var > max_var:
            return EntropyEstimateStatus(
                accepted=False,
                reason=f"Proposal variance {proposal_var:.6f} exceeded quality gate maximum {max_var:.6f}.",
                variance=proposal_var,
            )
        min_proposals = config.entropy_quality_gate.get("min_proposals", 2)
        if m < min_proposals:
            return EntropyEstimateStatus(
                accepted=False,
                reason=f"Proposal count M={m} < min_proposals={min_proposals}.",
                variance=proposal_var,
            )

    return EntropyEstimateStatus(
        accepted=True,
        reason="Proposal count and variance within quality gate limits.",
        variance=proposal_var,
    )


def reset_smoothing_if_needed(
    controller_state: ControllerState,
    request: RoutingRequest,
    current_support_fingerprint: str,
) -> ControllerState:
    """
    Reset EMA smoothing when episode, policy profile, or action support changes.
    """
    state = controller_state.copy()
    should_reset = False

    if state.episode_id and state.episode_id != request.episode_id:
        should_reset = True
    if state.profile_version and state.profile_version != request.policy_profile.version:
        should_reset = True
    if state.support_fingerprint and state.support_fingerprint != current_support_fingerprint:
        should_reset = True

    if should_reset:
        state.smoothed_marginal_entropy = None
        state.eligible_update_count = 0

    state.episode_id = request.episode_id
    state.profile_version = request.policy_profile.version
    state.support_fingerprint = current_support_fingerprint
    return state


def update_controller_state(
    controller_state: ControllerState,
    h_marginal: float,
    num_eligible: int,
    config: KleroterionConfig,
    estimate_status: EntropyEstimateStatus,
) -> Tuple[ControllerState, float]:
    """
    Compute bounded marginal entropy update for next decision.
    
    Formula (Section 4.2):
        h_norm = H_marginal / log(|V|)
        h_bar_t = (1 - alpha) * h_bar_{t-1} + alpha * h_norm
        e_t = h_target - h_bar_t
        if |e_t| <= delta: e_t = 0 (deadband)
        log(tau_{t+1}) = clip(log(tau_t) + eta * e_t, log(tau_min), log(tau_max))
        
    Returns:
        Tuple of (updated_controller_state, tau_next).
    """
    next_state = controller_state.copy()
    current_tau = float(np.exp(next_state.log_temperature))

    # If adaptation is disabled, or singleton support, or estimate rejected: freeze
    if not config.adapt_temperature or num_eligible <= 1 or not estimate_status.accepted:
        return next_state, current_tau

    if config.entropy_target is None or config.entropy_ema_alpha is None or config.controller_gain is None:
        raise InvalidConfigError("Missing required adaptation parameters in config.")

    if config.temperature_bounds is None:
        raise InvalidConfigError("temperature_bounds are required when adaptation is enabled.")

    tau_min, tau_max = config.temperature_bounds
    log_tau_min = float(np.log(tau_min))
    log_tau_max = float(np.log(tau_max))

    # Normalize marginal entropy by log(|V|)
    h_norm = h_marginal / np.log(num_eligible)

    # EMA update
    alpha = config.entropy_ema_alpha
    if next_state.smoothed_marginal_entropy is None:
        h_bar = h_norm
    else:
        h_bar = (1.0 - alpha) * next_state.smoothed_marginal_entropy + alpha * h_norm

    next_state.smoothed_marginal_entropy = float(h_bar)
    next_state.eligible_update_count += 1

    # Error and deadband
    error = config.entropy_target - h_bar
    if abs(error) <= config.entropy_deadband:
        error = 0.0

    # Bounded log-temperature update
    gain = config.controller_gain
    new_log_tau = float(np.clip(next_state.log_temperature + gain * error, log_tau_min, log_tau_max))
    next_state.log_temperature = new_log_tau
    tau_next = float(np.exp(new_log_tau))

    return next_state, tau_next
