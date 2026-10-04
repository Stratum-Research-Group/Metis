"""
Core stochastic decision logic implementing Section 8 pseudocode and safety invariants.
"""

import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from kleroterion.entropy_feedback import (
    assess_entropy_estimate_quality,
    compute_support_fingerprint,
    reset_smoothing_if_needed,
    update_controller_state,
)
from kleroterion.errors import (
    InvalidConfigError,
    InvalidInputError,
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
    PolicyProfileType,
    RngState,
    RoutingRequest,
    SelectionMode,
)
from kleroterion.masked_softmax import stable_masked_softmax
from kleroterion.marginal_policy import (
    compute_entropy_diagnostics,
    compute_marginal_distribution,
    select_action,
)
from kleroterion.proposals import (
    clip_increment_to_radius,
    generate_budgeted_gaussian_increment,
    verify_proposed_features,
)


def resolve_effective_config(
    request: RoutingRequest,
    config: KleroterionConfig,
) -> KleroterionConfig:
    """
    Apply profile rules to resolve effective execution configuration.
    
    Section 7:
    Exact-fact profiles enforce noise=0, argmax, and adaptation=false
    regardless of exploratory defaults.
    """
    profile = request.policy_profile
    if profile.is_exact_fact() or not profile.allow_noise:
        # Override to strict deterministic exact-fact baseline
        return KleroterionConfig(
            proposal_mode=config.proposal_mode,
            dt=config.dt,
            noise_rms_budget=0.0,
            noise_hard_radius=None,
            diffusion_matrix=None,
            proposal_count=1,
            selection_mode=SelectionMode.ARGMAX,
            temperature_initial=config.temperature_initial,
            temperature_bounds=config.temperature_bounds,
            adapt_temperature=False,
            entropy_target=None,
            entropy_ema_alpha=None,
            controller_gain=None,
            entropy_deadband=0.0,
            entropy_quality_gate=None,
            stability_limits=config.stability_limits,
        )

    # For exploratory profile, allow profile's entropy target to override if specified
    entropy_target = profile.entropy_target if profile.entropy_target is not None else config.entropy_target
    selection_mode = config.selection_mode if profile.allow_sampling else SelectionMode.ARGMAX
    adapt_temperature = config.adapt_temperature if profile.allow_adaptation else False

    return KleroterionConfig(
        proposal_mode=config.proposal_mode,
        dt=config.dt,
        noise_rms_budget=config.noise_rms_budget,
        noise_hard_radius=config.noise_hard_radius,
        diffusion_matrix=config.diffusion_matrix,
        proposal_count=config.proposal_count,
        selection_mode=selection_mode,
        temperature_initial=config.temperature_initial,
        temperature_bounds=config.temperature_bounds,
        adapt_temperature=adapt_temperature,
        entropy_target=entropy_target,
        entropy_ema_alpha=config.entropy_ema_alpha,
        controller_gain=config.controller_gain,
        entropy_deadband=config.entropy_deadband,
        entropy_quality_gate=config.entropy_quality_gate,
        stability_limits=config.stability_limits,
    )


def propose_action(
    request: RoutingRequest,
    config: KleroterionConfig,
    controller_state: ControllerState,
    rng_state: RngState,
    callbacks: HostCallbacks,
) -> ActionDecision:
    """
    Execute Kleroterion action proposal procedure.
    
    Implements Section 8 pseudocode with strict invariant enforcement.
    """
    start_time = time.perf_counter()

    # 1. Validate inputs and configuration
    request.validate()
    effective_cfg = resolve_effective_config(request, config)
    effective_cfg.validate()

    # Work on isolated local copies to prevent unintended side effects on failure
    local_rng = rng_state.copy()
    next_ctrl = controller_state.copy()

    # Determine temperature tau to use for this decision
    tau = float(np.exp(next_ctrl.log_temperature))

    # Initialize common eligibility mask from authoritative request mask
    # Safety Invariant 2: Eligibility can only narrow.
    common_mask = list(request.authoritative_mask)
    if not any(common_mask):
        raise NoEligibleActionError("NO_ELIGIBLE_ACTION: Authoritative mask has zero eligible actions.")

    d = request.policy_features.shape[0]
    proposals: List[np.ndarray] = []
    clipping_count = 0
    total_displacement = 0.0

    # 2. Generate M bounded Gaussian proposals
    for _ in range(effective_cfg.proposal_count):
        raw_inc, _ = generate_budgeted_gaussian_increment(
            d=d,
            dt=effective_cfg.dt,
            noise_rms_budget=effective_cfg.noise_rms_budget,
            rng=local_rng.proposal_rng,
            diffusion_matrix=effective_cfg.diffusion_matrix,
        )

        clipped_inc, was_clipped = clip_increment_to_radius(
            increment=raw_inc,
            hard_radius=effective_cfg.noise_hard_radius,
        )
        if was_clipped:
            clipping_count += 1

        # Host applies bounded increment to feature schema
        x_prop = callbacks.proposal_features(request.policy_features, clipped_inc)
        verify_proposed_features(
            original=request.policy_features,
            proposed=x_prop,
            hard_radius=effective_cfg.noise_hard_radius,
        )

        # Host proposal mask callback: intersection with proposal-dependent checks
        prop_mask = callbacks.proposal_mask(request, x_prop)
        if len(prop_mask) != len(common_mask):
            raise InvalidInputError(
                f"proposal_mask length ({len(prop_mask)}) does not match actions length ({len(common_mask)})."
            )

        # Narrow eligibility mask: common_mask = common_mask & prop_mask
        for i in range(len(common_mask)):
            common_mask[i] = common_mask[i] and prop_mask[i]

        proposals.append(x_prop)
        total_displacement += float(np.linalg.norm(x_prop - request.policy_features))

    # Check for empty intersection across all proposals
    eligible_count = sum(common_mask)
    if eligible_count == 0:
        raise NoEligibleActionError(
            "NO_ELIGIBLE_ACTION: Intersection of proposal eligibility masks is empty."
        )

    # 3. Score actions and compute masked softmax per proposal
    distributions: List[np.ndarray] = []
    scorer_calls = 0

    for x_prop in proposals:
        logits = callbacks.score_actions(request, x_prop, request.actions)
        scorer_calls += 1
        if not isinstance(logits, np.ndarray) or logits.shape != (len(request.actions),):
            raise ScorerError(
                f"score_actions must return 1D numpy array of length {len(request.actions)}."
            )
        p = stable_masked_softmax(logits=logits, mask=common_mask, temperature=tau)
        distributions.append(p)

    # 4. Average proposal probabilities -> marginal distribution
    marginal_q = compute_marginal_distribution(distributions)

    # 5. Diagnostics: conditional entropy, marginal entropy, disagreement J
    h_cond, h_marg, disagreement = compute_entropy_diagnostics(distributions, marginal_q)

    # 6. Action selection (Section 3.3 & 8)
    selected_action = select_action(
        marginal=marginal_q,
        common_mask=common_mask,
        actions=request.actions,
        mode=effective_cfg.selection_mode,
        rng=local_rng.selection_rng,
    )

    # 7. Entropy feedback controller update (Section 4.2 & 8)
    support_fingerprint = compute_support_fingerprint(request.actions, common_mask)
    next_ctrl = reset_smoothing_if_needed(next_ctrl, request, support_fingerprint)
    estimate_quality = assess_entropy_estimate_quality(distributions, effective_cfg)

    next_ctrl, tau_next = update_controller_state(
        controller_state=next_ctrl,
        h_marginal=h_marg,
        num_eligible=eligible_count,
        config=effective_cfg,
        estimate_status=estimate_quality,
    )

    elapsed_time = time.perf_counter() - start_time
    diagnostics = {
        "displacement_mean": total_displacement / effective_cfg.proposal_count,
        "clipping_count": clipping_count,
        "eligible_count": eligible_count,
        "saturation": (
            effective_cfg.temperature_bounds is not None and (
                np.isclose(tau_next, effective_cfg.temperature_bounds[0]) or
                np.isclose(tau_next, effective_cfg.temperature_bounds[1])
            )
        ),
        "scorer_calls": scorer_calls,
        "elapsed_time_sec": elapsed_time,
        "common_mask": common_mask,
    }

    return ActionDecision(
        request_id=request.request_id,
        snapshot_id=request.snapshot_id,
        lthcm_revision=request.lthcm_revision,
        selected_action_id=selected_action,
        action_ids=list(request.actions),
        marginal_probabilities=marginal_q,
        conditional_entropy=h_cond,
        marginal_entropy=h_marg,
        proposal_disagreement=disagreement,
        proposal_count=effective_cfg.proposal_count,
        entropy_estimate_status=estimate_quality,
        temperature_used=tau,
        temperature_next=tau_next,
        controller_state_next=next_ctrl,
        rng_state_next=local_rng,
        diagnostics=diagnostics,
    )


def host_accept(
    decision: ActionDecision,
    executed_request_ids: Set[str],
    callbacks: HostCallbacks,
) -> Tuple[bool, str]:
    """
    Host acceptance transaction coordinating action execution and state commitment.
    
    Section 8 pseudocode:
        check_request_not_already_executed(decision.request_id)
        revalidate_before_execute(decision.action, decision.snapshot_id)
        execute_and_commit_with_host_transaction(decision)
    """
    if decision.request_id in executed_request_ids:
        raise TransactionAbortedError(
            f"Duplicate execution attempted for request_id {decision.request_id}."
        )

    is_valid, reason = callbacks.revalidate_before_execute(
        decision.selected_action_id,
        decision.snapshot_id,
    )
    if not is_valid:
        raise RevalidationError(
            f"Action {decision.selected_action_id} failed pre-execution revalidation: {reason}"
        )

    # Commit execution
    executed_request_ids.add(decision.request_id)
    return True, "ACCEPTED"
