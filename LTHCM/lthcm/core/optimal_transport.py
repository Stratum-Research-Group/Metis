"""
Optimal Transport & Wasserstein Document Alignment
===================================================
Treats multi-faceted documents and token sequences as discrete probability measures
in embedding space:
    mu = sum_{i=1}^N a_i * delta_{x_i},    nu = sum_{j=1}^M b_j * delta_{y_j}

Computes the Entropic Optimal Transport (Wasserstein Distance) via the GPU-accelerated
Sinkhorn-Knopp algorithm:
    W_eps(mu, nu) = min_{gamma in Pi(a, b)} <gamma, C> - eps * H(gamma)
where C_{ij} is the ground cost matrix between facets/tokens (cosine distance or Euclidean),
and H(gamma) is the Shannon entropy regularizer.

Connects directly with LTHCM's Quantum Superposition Memory Store, where documents
are naturally multi-faceted probability distributions: rho_i = sum_k w_k |v_k><v_k|.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cost_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    """
    Compute pairwise ground cost matrix C between two sets of vectors.
    Args:
        x: Point cloud 1 [..., N, D].
        y: Point cloud 2 [..., M, D].
        metric: 'cosine' (1 - cos_sim), 'euclidean' (||x - y||_2), or 'sqeuclidean' (||x - y||_2^2).
    Returns:
        Cost matrix C [..., N, M].
    """
    if metric == "cosine":
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        sim = torch.matmul(x_norm, y_norm.transpose(-1, -2))
        return torch.clamp(1.0 - sim, min=0.0)
    elif metric == "sqeuclidean":
        x2 = torch.sum(x ** 2, dim=-1, keepdim=True)  # [..., N, 1]
        y2 = torch.sum(y ** 2, dim=-1, keepdim=True).transpose(-1, -2)  # [..., 1, M]
        xy = torch.matmul(x, y.transpose(-1, -2))  # [..., N, M]
        dist2 = x2 + y2 - 2 * xy
        return torch.clamp(dist2, min=0.0)
    elif metric == "euclidean":
        sq = compute_cost_matrix(x, y, metric="sqeuclidean")
        return torch.sqrt(sq + 1e-8)
    else:
        raise ValueError(f"Unknown metric '{metric}'. Supported: 'cosine', 'euclidean', 'sqeuclidean'.")


def sinkhorn_knopp(
    cost: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    eps: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Entropic Regularized Optimal Transport using log-stabilized Sinkhorn iterations.
    Args:
        cost: Pairwise cost matrix [..., N, M].
        a: Source probability vector [..., N] (uniform if None).
        b: Target probability vector [..., M] (uniform if None).
        eps: Entropic regularization parameter (smaller = closer to exact OT, larger = smoother/faster).
        max_iter: Maximum number of dual coordinate ascent iterations.
        tol: Stopping threshold for marginal error.
    Returns:
        distance: Optimal transport cost scalar/tensor [...].
        plan: Optimal transport plan matrix [..., N, M].
    """
    N, M = cost.shape[-2], cost.shape[-1]
    batch_shape = cost.shape[:-2]
    device = cost.device
    dtype = cost.dtype

    if a is None:
        a = torch.full((*batch_shape, N), 1.0 / N, device=device, dtype=dtype)
    else:
        a = a / torch.sum(a, dim=-1, keepdim=True)

    if b is None:
        b = torch.full((*batch_shape, M), 1.0 / M, device=device, dtype=dtype)
    else:
        b = b / torch.sum(b, dim=-1, keepdim=True)

    # Log-stabilized iterations: f in R^N, g in R^M dual potentials
    # Gibbs kernel K = exp(-cost / eps)
    K = torch.exp(-cost / eps)  # [..., N, M]
    u = torch.ones_like(a)  # [..., N]
    v = torch.ones_like(b)  # [..., M]

    for _ in range(max_iter):
        u_prev = u
        # u = a / (K @ v)
        Kv = torch.matmul(K, v.unsqueeze(-1)).squeeze(-1)  # [..., N]
        u = a / (Kv + 1e-12)

        # v = b / (K.T @ u)
        Ktu = torch.matmul(K.transpose(-1, -2), u.unsqueeze(-1)).squeeze(-1)  # [..., M]
        v = b / (Ktu + 1e-12)

        # Check convergence
        err = torch.max(torch.abs(u - u_prev))
        if err < tol:
            break

    # Transport plan gamma = diag(u) @ K @ diag(v)
    plan = u.unsqueeze(-1) * K * v.unsqueeze(-2)  # [..., N, M]
    distance = torch.sum(plan * cost, dim=(-2, -1))

    return distance, plan


class WassersteinDocumentAligner(nn.Module):
    """
    Wasserstein Optimal Transport Document Alignment Layer.

    Compares documents as distributions of token or facet vectors rather than single centroids.
    Particularly effective at capturing fine-grained structural alignments, paraphrases,
    and cross-document semantic flows.
    """

    def __init__(
        self,
        d_model: int = 64,
        eps: float = 0.05,
        max_iter: int = 80,
        cost_metric: str = "cosine",
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.max_iter = max_iter
        self.cost_metric = cost_metric

        # Learnable temperature / scaling parameter for similarity conversion
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    @property
    def tau(self) -> float:
        return float(torch.exp(self.log_tau).item())

    def pairwise_distance(
        self,
        facets_a: torch.Tensor,
        facets_b: torch.Tensor,
        weights_a: Optional[torch.Tensor] = None,
        weights_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Wasserstein distance between two multi-facet documents or queries.
        Args:
            facets_a: Facet vectors for document A [K_a, D] or [B, K_a, D].
            facets_b: Facet vectors for document B [K_b, D] or [B, K_b, D].
            weights_a: Prior weights for A [K_a] or [B, K_a].
            weights_b: Prior weights for B [K_b] or [B, K_b].
        Returns:
            dist: Wasserstein distance.
            plan: Optimal transport plan matrix.
        """
        cost = compute_cost_matrix(facets_a, facets_b, metric=self.cost_metric)
        dist, plan = sinkhorn_knopp(
            cost=cost,
            a=weights_a,
            b=weights_b,
            eps=self.eps,
            max_iter=self.max_iter,
        )
        return dist, plan

    def forward(
        self,
        facets_a: torch.Tensor,
        facets_b: torch.Tensor,
        weights_a: Optional[torch.Tensor] = None,
        weights_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Wasserstein distance and optimal transport plan."""
        return self.pairwise_distance(facets_a, facets_b, weights_a, weights_b)

    def similarity(
        self,
        facets_a: torch.Tensor,
        facets_b: torch.Tensor,
        weights_a: Optional[torch.Tensor] = None,
        weights_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert Wasserstein distance into a normalized similarity score in [0, 1]:
            Sim(A, B) = exp(-W(A, B) / tau)
        """
        dist, _ = self.pairwise_distance(facets_a, facets_b, weights_a, weights_b)
        return torch.exp(-dist / (self.tau + 1e-6))

    def align_query_to_corpus(
        self,
        query_facets: torch.Tensor,
        corpus_facets: torch.Tensor,
        query_weights: Optional[torch.Tensor] = None,
        corpus_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch align a query (multi-facet or tokens [K_q, D]) against all corpus documents [N, K_d, D].
        Args:
            query_facets: [K_q, D].
            corpus_facets: [N, K_d, D].
            query_weights: [K_q] or None.
            corpus_weights: [N, K_d] or None.
        Returns:
            distances: [N] Wasserstein distances to each document.
            plans: [N, K_q, K_d] transport plans showing which query facets map to which doc facets.
        """
        N = corpus_facets.shape[0]
        K_q = query_facets.shape[0]
        K_d = corpus_facets.shape[1]

        # Expand query across corpus batch
        q_exp = query_facets.unsqueeze(0).expand(N, K_q, -1)  # [N, K_q, D]
        cost = compute_cost_matrix(q_exp, corpus_facets, metric=self.cost_metric)  # [N, K_q, K_d]

        a = query_weights.unsqueeze(0).expand(N, -1) if query_weights is not None else None
        b = corpus_weights

        dist, plans = sinkhorn_knopp(
            cost=cost,
            a=a,
            b=b,
            eps=self.eps,
            max_iter=self.max_iter,
        )
        return dist, plans
