"""
Fisher-Rao Information Geometry Retrieval Metric
=================================================
Replaces flat, isotropic cosine similarity with a Fisher-Rao metric
on the statistical manifold of embedding distributions.

Standard cosine: sim(q, d) = <q, d> / (||q|| ||d||)   -- treats all dimensions equally
Fisher-Rao:      sim(q, d) = <q, Λ d> / sqrt(<q, Λ q> <d, Λ d>)

where Λ = diag(λ_1, ..., λ_D) is a learned diagonal Fisher Information matrix
that assigns per-dimension confidence weights.

Key advantages:
  - Dimensions with high semantic precision (large λ_i) get amplified.
  - Noisy or redundant dimensions (small λ_i) get suppressed.
  - Provably reduces to standard cosine when Λ = I (identity).
  - Differentiable and GPU-friendly: O(D) per comparison.

Mathematical foundation:
  The Fisher Information Matrix for a parametric distribution p(x|θ) is:
    I(θ)_{ij} = E[ ∂ log p / ∂θ_i · ∂ log p / ∂θ_j ]

  For diagonal approximation (independent dimensions):
    I(θ)_i = Var[ ∂ log p / ∂θ_i ]

  The Fisher-Rao distance on the statistical manifold is:
    d_FR(p, q)^2 = (θ_p - θ_q)^T I(θ) (θ_p - θ_q)

  We implement this as a Mahalanobis-type weighted inner product for retrieval scoring.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FisherRaoMetric(nn.Module):
    """
    Learnable diagonal Fisher Information metric for uncertainty-aware retrieval.

    Replaces cosine similarity with a weighted inner product:
        score(q, d) = <q, Λ d> / sqrt(<q, Λ q> · <d, Λ d>)

    where Λ = softplus(raw_weights) ensures strictly positive scaling.

    The metric can be:
      1. Initialized uniformly (equivalent to cosine similarity).
      2. Learned from retrieval feedback via backpropagation.
      3. Estimated from embedding variance statistics.
    """

    def __init__(self, d_model: int, init_mode: str = "uniform"):
        """
        Args:
            d_model: Embedding dimensionality.
            init_mode: Initialization strategy:
                - "uniform": All weights = 1.0 (equivalent to cosine, safe default).
                - "inverse_variance": Requires calling estimate_from_data() after init.
        """
        super().__init__()
        self.d_model = d_model
        self.init_mode = init_mode

        # Raw (unconstrained) Fisher weights — softplus ensures λ_i > 0
        if init_mode == "uniform":
            # softplus(0.5413) ≈ 1.0, so this initializes Λ ≈ I
            init_val = torch.full((d_model,), 0.5413)
        else:
            init_val = torch.zeros(d_model)

        self.raw_weights = nn.Parameter(init_val)

    @property
    def fisher_weights(self) -> torch.Tensor:
        """Strictly positive Fisher Information weights Λ = softplus(raw)."""
        return F.softplus(self.raw_weights)

    def weighted_inner_product(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute <a, Λ b> for batched vectors.
        Args:
            a: [..., D]
            b: [..., D]
        Returns:
            inner: [...]
        """
        lam = self.fisher_weights  # [D]
        return torch.sum(a * lam * b, dim=-1)

    def weighted_norm(self, a: torch.Tensor) -> torch.Tensor:
        """Compute sqrt(<a, Λ a>) for batched vectors."""
        return torch.sqrt(self.weighted_inner_product(a, a).clamp(min=1e-12))

    def similarity(
        self,
        query: torch.Tensor,
        documents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fisher-Rao weighted cosine similarity.

        score(q, d_i) = <q, Λ d_i> / sqrt(<q, Λ q> · <d_i, Λ d_i>)

        Args:
            query: [D] or [1, D] query vector.
            documents: [N, D] document vectors.
        Returns:
            scores: [N] similarity scores in [-1, 1].
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, D]

        lam = self.fisher_weights  # [D]

        # Weighted inner products: <q, Λ d_i> for each document
        # query: [1, D], documents: [N, D], lam: [D]
        q_weighted = query * lam  # [1, D]
        numerator = torch.mm(q_weighted, documents.T).squeeze(0)  # [N]

        # Weighted norms
        q_norm = torch.sqrt(torch.sum(query * lam * query, dim=-1).clamp(min=1e-12))  # [1]
        d_norms = torch.sqrt(torch.sum(documents * lam * documents, dim=-1).clamp(min=1e-12))  # [N]

        denominator = q_norm * d_norms  # [N]
        return numerator / denominator

    def distance(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fisher-Rao Mahalanobis distance: d_FR(a, b) = sqrt((a-b)^T Λ (a-b)).
        Args:
            a: [..., D]
            b: [..., D]
        Returns:
            dist: [...]
        """
        diff = a - b
        return torch.sqrt(self.weighted_inner_product(diff, diff).clamp(min=1e-12))

    def estimate_from_data(
        self,
        embeddings: torch.Tensor,
        method: str = "inverse_variance",
    ):
        """
        Estimate Fisher weights from a corpus of embeddings.
        High-variance dimensions are less informative → lower weight.
        Low-variance dimensions are more precise → higher weight.

        Args:
            embeddings: [N, D] corpus embeddings.
            method: "inverse_variance" (default) or "normalized_precision".
        """
        with torch.no_grad():
            var = torch.var(embeddings, dim=0).clamp(min=1e-8)  # [D]

            if method == "inverse_variance":
                # λ_i = 1 / var_i — high precision = high weight
                target_weights = 1.0 / var
            elif method == "normalized_precision":
                # λ_i = mean(var) / var_i — normalized to average 1.0
                target_weights = var.mean() / var
            else:
                raise ValueError(f"Unknown estimation method: {method}")

            # Invert softplus: raw = log(exp(target) - 1). For target > 20, softplus(x) ≈ x.
            clamped = target_weights.clamp(min=0.01, max=50.0)
            raw = torch.where(clamped > 20.0, clamped, torch.log(torch.expm1(clamped).clamp(min=1e-8)))
            self.raw_weights.data.copy_(raw)

    def get_dimension_importance(self) -> torch.Tensor:
        """Return normalized importance scores per dimension (sum to 1)."""
        lam = self.fisher_weights
        return lam / lam.sum()

    def forward(
        self,
        query: torch.Tensor,
        documents: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computes Fisher-Rao similarity scores."""
        return self.similarity(query, documents)

    def __repr__(self) -> str:
        lam = self.fisher_weights
        return (
            f"FisherRaoMetric(d_model={self.d_model}, "
            f"weight_range=[{lam.min().item():.3f}, {lam.max().item():.3f}], "
            f"weight_mean={lam.mean().item():.3f})"
        )
