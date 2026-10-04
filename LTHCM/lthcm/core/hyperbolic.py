"""
Poincaré Ball Hyperbolic Geometry & Hierarchical Embeddings
============================================================
Implements the Poincaré Ball Model of Hyperbolic Space (B^D_c) with constant
negative curvature -c (c > 0):
    B^D_c = { x in R^D : c * ||x||^2 < 1 }

Hyperbolic space expands exponentially, embedding trees and hierarchical DAGs
with O(log N) distortion instead of Euclidean O(N) or O(b^h) exponential crowding.

Core Riemannian Operations:
  1. Geodesic Distance:
         d_B(x, y) = (1 / sqrt(c)) * arcosh(1 + 2c * ||x - y||^2 / ((1 - c ||x||^2) * (1 - c ||y||^2)))
  2. Möbius Vector Addition:
         x (+) y = ((1 + 2c <x, y> + c ||y||^2) x + (1 - c ||x||^2) y) / (1 + 2c <x, y> + c^2 ||x||^2 ||y||^2)
  3. Exponential & Logarithmic Maps (Origin-centered):
         exp_0(v) = tanh(sqrt(c) ||v||) * (v / (sqrt(c) ||v||))
         log_0(x) = artanh(sqrt(c) ||x||) * (x / (sqrt(c) ||x||))
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoincareBall(nn.Module):
    """
    Riemannian Manifold operations on the Poincaré Ball of constant curvature -c.
    """

    def __init__(self, c: float = 1.0, eps: float = 1e-5, learnable_c: bool = False):
        super().__init__()
        self.eps = eps
        if learnable_c:
            # Parameterize c = softplus(raw_c) to guarantee c > 0
            self.raw_c = nn.Parameter(torch.tensor(math.log(math.expm1(c))))
        else:
            self.register_buffer("c", torch.tensor(c, dtype=torch.float32))
            self.raw_c = None

    @property
    def curvature(self) -> torch.Tensor:
        if self.raw_c is not None:
            return F.softplus(self.raw_c) + 1e-4
        return self.c

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points onto the open Poincaré ball: c * ||x||^2 <= 1 - eps.
        """
        c = self.curvature.to(x.device)
        sqrt_c = torch.sqrt(c)
        max_norm = (1.0 - self.eps) / sqrt_c
        norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x * scale

    def conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Conformal metric factor lambda_x^c = 2 / (1 - c * ||x||^2).
        """
        c = self.curvature.to(x.device)
        x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
        return 2.0 / (1.0 - c * x2).clamp_min(self.eps)

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at origin: Maps Euclidean tangent vector v in T_0 B -> Point in B^D_c.
        """
        c = self.curvature.to(v.device)
        sqrt_c = torch.sqrt(c)
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        # tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
        factor = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return self.project(v * factor)

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at origin: Maps Point in B^D_c -> Euclidean tangent vector in T_0 B.
        """
        c = self.curvature.to(x.device)
        sqrt_c = torch.sqrt(c)
        x_proj = self.project(x)
        x_norm = torch.norm(x_proj, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
        scaled_norm = (sqrt_c * x_norm).clamp_max(1.0 - self.eps)
        factor = torch.atanh(scaled_norm) / (sqrt_c * x_norm)
        return x_proj * factor

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Möbius vector addition x (+) y on the Poincaré ball.
        """
        c = self.curvature.to(x.device)
        x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
        y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)

        num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
        denom = 1.0 + 2.0 * c * xy + (c ** 2) * x2 * y2
        return self.project(num / denom.clamp_min(self.eps))

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Poincaré geodesic distance between point sets x and y.
        """
        c = self.curvature.to(x.device)
        sqrt_c = torch.sqrt(c)
        x_p = self.project(x)
        y_p = self.project(y)

        x2 = torch.sum(x_p ** 2, dim=-1)
        y2 = torch.sum(y_p ** 2, dim=-1)
        diff2 = torch.sum((x_p - y_p) ** 2, dim=-1)
        diff2 = torch.clamp(diff2, min=0.0)
        denom = (1.0 - c * x2) * (1.0 - c * y2)
        alpha = 1.0 + 2.0 * c * diff2 / denom.clamp_min(self.eps)
        alpha = alpha.clamp_min(1.0)
        dist = torch.acosh(alpha) / sqrt_c
        dist = torch.where(diff2 < 1e-12, torch.zeros_like(dist), dist)
        return dist

    def pairwise_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Pairwise Poincaré distance matrix between x [..., N, D] and y [..., M, D].
        """
        c = self.curvature.to(x.device)
        sqrt_c = torch.sqrt(c)
        x_p = self.project(x)
        y_p = self.project(y)

        x2 = torch.sum(x_p ** 2, dim=-1, keepdim=True)  # [..., N, 1]
        y2 = torch.sum(y_p ** 2, dim=-1, keepdim=True).transpose(-1, -2)  # [..., 1, M]
        xy = torch.matmul(x_p, y_p.transpose(-1, -2))  # [..., N, M]
        diff2 = torch.clamp(x2 + y2 - 2 * xy, min=0.0)

        denom = (1.0 - c * x2) * (1.0 - c * y2)
        alpha = 1.0 + 2.0 * c * diff2 / denom.clamp_min(self.eps)
        alpha = alpha.clamp_min(1.0)
        dist = torch.acosh(alpha) / sqrt_c
        dist = torch.where(diff2 < 1e-12, torch.zeros_like(dist), dist)
        return dist

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Convert Poincaré distance into similarity score: Sim(x, y) = 1 / (1 + d_B(x, y)).
        """
        d = self.dist(x, y)
        return 1.0 / (1.0 + d)


class HyperbolicHierarchicalStore(nn.Module):
    """
    Hyperbolic Memory Store for hierarchical knowledge graphs and taxonomies.
    Nodes closer to the origin represent root/abstract concepts;
    nodes near the boundary represent fine-grained leaf entities.
    """

    def __init__(self, d_model: int = 64, curvature: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.manifold = PoincareBall(c=curvature)
        self.documents: List[Dict[str, Any]] = []
        self.register_buffer("embeddings", torch.empty(0, d_model))

    def add_concept(
        self,
        doc_id: str,
        content: str,
        embedding: torch.Tensor,
        is_hyperbolic: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Index a concept into hyperbolic memory.
        If is_hyperbolic is False, maps Euclidean embedding through expmap0().
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        if not is_hyperbolic:
            hyp_emb = self.manifold.expmap0(embedding)
        else:
            hyp_emb = self.manifold.project(embedding)

        self.documents.append({
            "idx": len(self.documents),
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
        })
        self.embeddings = torch.cat([self.embeddings, hyp_emb.to(self.embeddings.device)], dim=0)

    def retrieve(self, query: torch.Tensor, top_k: int = 5, is_hyperbolic: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve nearest hierarchical concepts using hyperbolic geodesic distance.
        """
        if len(self.documents) == 0:
            return []

        if query.dim() == 1:
            query = query.unsqueeze(0)

        if not is_hyperbolic:
            q_hyp = self.manifold.expmap0(query.to(self.embeddings.device))
        else:
            q_hyp = self.manifold.project(query.to(self.embeddings.device))

        distances = self.manifold.pairwise_dist(q_hyp, self.embeddings).squeeze(0)  # [N]
        k = min(top_k, len(self.documents))
        top_dists, top_indices = torch.topk(distances, k=k, largest=False)

        results = []
        for rank, (dist, idx) in enumerate(zip(top_dists.tolist(), top_indices.tolist())):
            doc = self.documents[idx]
            hyp_vec = self.embeddings[idx]
            radius = float(torch.norm(hyp_vec, p=2).item())
            results.append({
                "rank": rank + 1,
                "hyperbolic_distance": float(dist),
                "similarity": 1.0 / (1.0 + float(dist)),
                "doc_id": doc["id"],
                "content": doc["content"],
                "radius": radius,  # Radius indicates hierarchical depth (small = abstract, large = specific)
                "metadata": doc["metadata"],
            })
        return results
