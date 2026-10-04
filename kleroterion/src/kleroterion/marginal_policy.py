"""
Marginal policy aggregation, entropy calculations, and action selection.
"""

from typing import List, Sequence, Tuple
import numpy as np

from kleroterion.errors import InvalidInputError, NoEligibleActionError
from kleroterion.interfaces import ActionID, SelectionMode


def compute_marginal_distribution(
    distributions: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Estimate marginal action distribution by averaging proposal probability vectors.
    
    q_hat = (1 / M) * sum_{m=1}^M p^{(m)}.
    
    Averages probabilities, NOT logits.
    """
    if not distributions:
        raise InvalidInputError("Cannot compute marginal distribution from empty list of distributions.")

    k = distributions[0].shape[0]
    for m, p in enumerate(distributions):
        if p.shape != (k,):
            raise InvalidInputError(
                f"Distribution at index {m} has shape {p.shape}, expected ({k},)."
            )
        if not np.all(np.isfinite(p)) or np.any(p < 0.0):
            raise InvalidInputError(f"Distribution at index {m} contains invalid probabilities.")

    stacked = np.stack(distributions, axis=0)  # Shape: (M, K)
    marginal = np.mean(stacked, axis=0)
    
    # Ensure numerical normalization
    total_q = float(np.sum(marginal))
    if total_q > 0.0:
        marginal = marginal / total_q

    return marginal


def shannon_entropy(p: np.ndarray) -> float:
    """
    Shannon entropy H(p) = - sum_{p_i > 0} p_i * log(p_i) in nats.
    Zero terms are omitted.
    """
    nonzero = p[p > 0.0]
    if len(nonzero) == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log(nonzero)))


def compute_entropy_diagnostics(
    distributions: Sequence[np.ndarray],
    marginal: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute conditional entropy, marginal entropy, and proposal disagreement J.
    
    H_cond = (1 / M) * sum_m H(p^{(m)})
    H_marg = H(q_hat)
    J = H_marg - H_cond >= 0
    
    Returns:
        Tuple of (H_conditional, H_marginal, proposal_disagreement).
    """
    cond_entropies = [shannon_entropy(p) for p in distributions]
    h_cond = float(np.mean(cond_entropies))
    h_marg = shannon_entropy(marginal)
    disagreement = max(0.0, h_marg - h_cond)
    return h_cond, h_marg, disagreement


def select_action(
    marginal: np.ndarray,
    common_mask: Sequence[bool],
    actions: Sequence[ActionID],
    mode: SelectionMode,
    rng: np.random.Generator,
) -> ActionID:
    """
    Select an action from marginal distribution according to selection mode.
    
    In ARGMAX mode: picks stable-order argmax among eligible actions.
    In SAMPLE mode: draws a categorical sample from marginal distribution.
    """
    k = len(actions)
    if len(common_mask) != k or marginal.shape != (k,):
        raise InvalidInputError("Dimension mismatch between actions, mask, and marginal distribution.")

    eligible_indices = [i for i, is_eligible in enumerate(common_mask) if is_eligible]
    if not eligible_indices:
        raise NoEligibleActionError("NO_ELIGIBLE_ACTION: No actions eligible for selection.")

    if len(eligible_indices) == 1:
        return actions[eligible_indices[0]]

    if mode == SelectionMode.ARGMAX:
        # Stable-order argmax: in case of exact tie, lowest index in actions list is chosen
        best_idx = eligible_indices[0]
        best_prob = marginal[best_idx]
        for idx in eligible_indices[1:]:
            if marginal[idx] > best_prob:
                best_prob = marginal[idx]
                best_idx = idx
        return actions[best_idx]

    elif mode == SelectionMode.SAMPLE:
        # Filter probabilities to eligible indices
        eligible_probs = marginal[eligible_indices]
        sum_p = np.sum(eligible_probs)
        if sum_p <= 0.0:
            # Fallback to uniform over eligible if all eligible have 0 probability
            chosen_sub_idx = rng.integers(0, len(eligible_indices))
            return actions[eligible_indices[chosen_sub_idx]]

        norm_probs = eligible_probs / sum_p
        chosen_sub_idx = rng.choice(len(eligible_indices), p=norm_probs)
        return actions[eligible_indices[chosen_sub_idx]]

    else:
        raise InvalidInputError(f"Unsupported selection mode: {mode}")
