"""
Type-safe weighted feature utility respecting Safety Invariant 5.

A WeightedFeatureVector is a purely diagnostic or intermediate mathematical
representation. It must never be accepted as an ActionID, ActionDecision,
document, node identity, citation, or executable walk action.
"""

from dataclasses import dataclass
import numpy as np
from kleroterion.errors import InvalidInputError


@dataclass(frozen=True)
class WeightedFeatureVector:
    """
    Distinct type for convex combinations of feature vectors sum(q_i * v_i).
    
    Safety Invariant 5:
    A WeightedFeatureVector must never be treated as or converted to
    a document ID, node identity, citation, or executable ActionID.
    """
    features: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        if not isinstance(self.features, np.ndarray) or not np.all(np.isfinite(self.features)):
            raise InvalidInputError("WeightedFeatureVector features must be a finite numpy ndarray.")
        if not isinstance(self.weights, np.ndarray) or not np.all(np.isfinite(self.weights)):
            raise InvalidInputError("WeightedFeatureVector weights must be a finite numpy ndarray.")


def compute_weighted_features(
    feature_matrix: np.ndarray,
    weights: np.ndarray,
) -> WeightedFeatureVector:
    """
    Compute convex combination of feature vectors: sum_i weights[i] * feature_matrix[i].
    
    Args:
        feature_matrix: Array of shape (K, d).
        weights: 1D probability array of shape (K,).
        
    Returns:
        WeightedFeatureVector object.
    """
    if feature_matrix.ndim != 2:
        raise InvalidInputError("feature_matrix must have 2 dimensions (K, d).")
    if weights.ndim != 1:
        raise InvalidInputError("weights must be a 1D array of length K.")
    if feature_matrix.shape[0] != weights.shape[0]:
        raise InvalidInputError(
            f"Shape mismatch: feature_matrix has {feature_matrix.shape[0]} rows, "
            f"but weights has length {weights.shape[0]}."
        )
    if not np.all(np.isfinite(feature_matrix)):
        raise InvalidInputError("feature_matrix contains non-finite values.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise InvalidInputError("weights must be non-negative and finite.")
    sum_w = float(np.sum(weights))
    if not np.isclose(sum_w, 1.0, atol=1e-5):
        raise InvalidInputError(f"weights must sum to 1.0 (got {sum_w}).")

    combined = np.sum(feature_matrix * weights[:, np.newaxis], axis=0)
    return WeightedFeatureVector(features=combined, weights=weights.copy())
