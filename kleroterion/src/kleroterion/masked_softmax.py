"""
Stable masked softmax over host action logits.
"""

from typing import List, Sequence
import numpy as np

from kleroterion.errors import InvalidInputError, NoEligibleActionError


def stable_masked_softmax(
    logits: np.ndarray,
    mask: Sequence[bool],
    temperature: float,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Compute numerically stable masked softmax over eligible action logits.
    
    For eligible set V and temperature tau > 0:
        c = max_{j in V} z_j
        p_i = exp((z_i - c) / tau) / sum_{j in V} exp((z_j - c) / tau) for i in V
        p_i = 0 for i not in V
        
    Args:
        logits: 1D array of real-valued action scores of length K.
        mask: Sequence of booleans of length K indicating eligibility.
        temperature: Positive temperature scalar tau > 0.
        atol: Absolute tolerance for unit sum validation.
        
    Returns:
        Probability distribution vector of shape (K,).
        
    Raises:
        NoEligibleActionError: If no actions are eligible.
        InvalidInputError: If inputs are malformed or non-finite.
    """
    if not isinstance(logits, np.ndarray):
        logits = np.asarray(logits, dtype=np.float64)

    if logits.ndim != 1:
        raise InvalidInputError("logits must be a 1D array.")

    k = logits.shape[0]
    if len(mask) != k:
        raise InvalidInputError(
            f"mask length ({len(mask)}) does not match logits length ({k})."
        )

    if temperature <= 0.0 or not np.isfinite(temperature):
        raise InvalidInputError(f"temperature must be finite and positive (got {temperature}).")

    eligible_indices = [i for i, is_eligible in enumerate(mask) if is_eligible]
    if not eligible_indices:
        raise NoEligibleActionError("NO_ELIGIBLE_ACTION: No actions are eligible in the provided mask.")

    # Validate that eligible logits are finite
    eligible_logits = logits[eligible_indices]
    if not np.all(np.isfinite(eligible_logits)):
        raise InvalidInputError("Non-finite (NaN or Inf) logits found among eligible actions.")

    probs = np.zeros(k, dtype=np.float64)

    # Special case: single eligible action receives exact probability 1.0
    if len(eligible_indices) == 1:
        probs[eligible_indices[0]] = 1.0
        return probs

    # Numerical stability: subtract maximum of eligible logits before scaling by tau
    c = np.max(eligible_logits)
    scaled_shifted = (eligible_logits - c) / temperature
    
    # Avoid underflow/overflow issues by clipping to safe float exp range [-700, 0]
    # Since scaled_shifted <= 0 by definition (c = max), we just bound below
    clipped_shifted = np.clip(scaled_shifted, -700.0, 0.0)
    exp_vals = np.exp(clipped_shifted)
    sum_exp = np.sum(exp_vals)

    if sum_exp <= 0.0 or not np.isfinite(sum_exp):
        # Fallback to uniform over eligible if extreme numerical underflow
        probs[eligible_indices] = 1.0 / len(eligible_indices)
        return probs

    probs[eligible_indices] = exp_vals / sum_exp

    # Enforce unit sum
    total_p = np.sum(probs)
    if not np.isclose(total_p, 1.0, atol=1e-7):
        raise InvalidInputError(f"Softmax probabilities failed to sum to 1.0 (got {total_p}).")

    return probs
