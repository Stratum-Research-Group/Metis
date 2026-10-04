"""
Gaussian proposal generation with dimension-aware noise budget and clipping.
"""

from typing import Optional, Tuple
import numpy as np

from kleroterion.errors import InvalidConfigError, InvalidProposalError


def generate_budgeted_gaussian_increment(
    d: int,
    dt: float,
    noise_rms_budget: float,
    rng: np.random.Generator,
    diffusion_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an unclipped Gaussian increment satisfying the RMS displacement budget b.
    
    E[||L sqrt(dt) epsilon||_2^2] = dt * tr(L L^T) <= b^2.
    
    For isotropic proposals:
        L = sigma * I_d, where sigma = b / sqrt(d * dt).
        L * sqrt(dt) * epsilon = (b / sqrt(d)) * epsilon, epsilon ~ N(0, I_d).
        
    Args:
        d: Dimension of active feature coordinates.
        dt: Positive time step.
        noise_rms_budget: RMS displacement budget b >= 0.
        rng: Proposal random number generator.
        diffusion_matrix: Optional anisotropic matrix L of shape (d, r).
        
    Returns:
        Tuple of (increment_vector, L_matrix).
    """
    if d <= 0:
        raise InvalidProposalError(f"Feature dimension must be positive (got {d}).")
    if dt <= 0.0:
        raise InvalidConfigError(f"dt must be positive (got {dt}).")
    if noise_rms_budget < 0.0:
        raise InvalidConfigError(f"noise_rms_budget must be >= 0.0 (got {noise_rms_budget}).")

    if noise_rms_budget == 0.0:
        zero_inc = np.zeros(d, dtype=np.float64)
        zero_L = np.zeros((d, d), dtype=np.float64)
        return zero_inc, zero_L

    if diffusion_matrix is None:
        # Isotropic case: sigma = b / sqrt(d * dt)
        sigma = noise_rms_budget / np.sqrt(d * dt)
        L = sigma * np.eye(d, dtype=np.float64)
        # Increment: L * sqrt(dt) * eps = sigma * sqrt(dt) * eps = (b / sqrt(d)) * eps
        eps = rng.standard_normal(d)
        increment = (noise_rms_budget / np.sqrt(d)) * eps
        return increment, L

    # Anisotropic matrix L of shape (d, r)
    if not isinstance(diffusion_matrix, np.ndarray) or not np.all(np.isfinite(diffusion_matrix)):
        raise InvalidConfigError("diffusion_matrix must be a finite numpy ndarray.")
    if diffusion_matrix.ndim != 2 or diffusion_matrix.shape[0] != d:
        raise InvalidConfigError(
            f"diffusion_matrix must have shape ({d}, r), got {diffusion_matrix.shape}."
        )

    r = diffusion_matrix.shape[1]
    # Check trace budget: dt * tr(L L^T) <= b^2
    trace_LLT = np.trace(diffusion_matrix @ diffusion_matrix.T)
    expected_sq_disp = dt * trace_LLT
    if expected_sq_disp > (noise_rms_budget ** 2) + 1e-7:
        raise InvalidConfigError(
            f"diffusion_matrix violates RMS budget: dt * tr(LL^T) = {expected_sq_disp:.6f} > {noise_rms_budget**2:.6f}."
        )

    eps = rng.standard_normal(r)
    increment = np.sqrt(dt) * (diffusion_matrix @ eps)
    return increment, diffusion_matrix


def clip_increment_to_radius(
    increment: np.ndarray,
    hard_radius: Optional[float],
) -> Tuple[np.ndarray, bool]:
    """
    Clip raw increment vector to hard radius R if norm exceeds R.
    
    Args:
        increment: Raw Gaussian increment vector.
        hard_radius: Maximum permitted displacement radius R.
        
    Returns:
        Tuple of (clipped_increment, was_clipped).
    """
    if hard_radius is None:
        return increment, False
    
    if hard_radius <= 0.0:
        raise InvalidConfigError(f"hard_radius must be positive (got {hard_radius}).")

    norm = float(np.linalg.norm(increment))
    if norm > hard_radius:
        # Scale down to hard radius
        clipped = increment * (hard_radius / norm)
        return clipped, True
    
    return increment, False


def verify_proposed_features(
    original: np.ndarray,
    proposed: np.ndarray,
    hard_radius: Optional[float] = None,
    atol: float = 1e-6,
) -> None:
    """
    Enforce invariant checks on proposed feature vector.
    Fails closed with InvalidProposalError without hidden fallbacks.
    """
    if proposed.shape != original.shape:
        raise InvalidProposalError(
            f"Proposed features shape {proposed.shape} does not match original {original.shape}."
        )
    if not np.all(np.isfinite(proposed)):
        raise InvalidProposalError("Proposed features contain non-finite (NaN or Inf) values.")

    if hard_radius is not None and hard_radius > 0.0:
        displacement = float(np.linalg.norm(proposed - original))
        if displacement > hard_radius + atol:
            raise InvalidProposalError(
                f"Proposed features displacement {displacement:.6f} exceeds hard radius {hard_radius:.6f}."
            )
