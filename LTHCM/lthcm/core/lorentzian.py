"""
Lorentzian Pseudo-Riemannian Spacetime & Causal Light-Cone Attention
====================================================================
Implements Minkowski spacetime geometry with metric tensor eta = diag(-1, 1, 1, ...).
Enforces physical and logical causality via Past/Future Light-Cone horizons,
forbidding space-like separated or anachronistic facts from contaminating the reasoning state.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LorentzianMetric:
    """
    Utility class for Minkowski spacetime calculations.
    Signature: (-, +, +, ..., +) where dim 0 is Temporal/Entailment coordinate.
    """

    @staticmethod
    def inner_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Minkowski inner product: <u, v>_L = -u_0 * v_0 + sum_{i=1}^{D-1} u_i * v_i
        """
        temporal = u[..., :1] * v[..., :1]
        spatial = torch.sum(u[..., 1:] * v[..., 1:], dim=-1, keepdim=True)
        return -temporal + spatial

    @staticmethod
    def interval(u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute spacetime interval ds^2 = -(dt)^2 + ||dx||^2 between two events.
        Args:
            u: [..., D] (e.g. Query / Observer)
            v: [..., D] (e.g. Memory / Target)
        Returns:
            ds2: Spacetime interval (negative = timelike, zero = lightlike, positive = spacelike)
            dt: Temporal difference (u_t - v_t)
            dx_norm: Spatial Euclidean distance ||u_x - v_x||
        """
        dt = u[..., :1] - v[..., :1]
        dx = u[..., 1:] - v[..., 1:]
        dx_norm_sq = torch.sum(dx ** 2, dim=-1, keepdim=True)
        dt_sq = dt ** 2
        ds2 = -dt_sq + dx_norm_sq
        return ds2.squeeze(-1), dt.squeeze(-1), torch.sqrt(torch.clamp(dx_norm_sq.squeeze(-1), min=1e-8))


class LorentzianCausalAttention(nn.Module):
    """
    Lorentzian Causal Attention Layer.

    Projects embeddings into (1 + D_spatial) Minkowski spacetime.
    Attention weights are computed using proper time along timelike geodesics
    and strictly zeroed outside the Past Light Cone.
    """

    def __init__(self, d_model: int, temperature: float = 1.0, c: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.c = nn.Parameter(torch.tensor(c)) # Causal propagation velocity

        # Project spatial dimensions (1:), preserving temporal coordinate (0)
        self.q_spatial = nn.Linear(d_model - 1, d_model - 1, bias=False)
        self.k_spatial = nn.Linear(d_model - 1, d_model - 1, bias=False)
        nn.init.orthogonal_(self.q_spatial.weight)
        nn.init.orthogonal_(self.k_spatial.weight)

        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def compute_causal_mask(
        self,
        q_coords: torch.Tensor,
        k_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute spacetime interval and past light-cone mask.
        Args:
            q_coords: [B, N_q, D]
            k_coords: [B, N_k, D]
        Returns:
            interval: [B, N_q, N_k]
            past_light_cone: [B, N_q, N_k] boolean mask (True if k is in past light cone of q)
        """
        q_t, q_x = q_coords[..., :1], q_coords[..., 1:]
        k_t, k_x = k_coords[..., :1], k_coords[..., 1:]

        # Broadcast differences: [B, N_q, N_k, 1]
        dt = q_t.unsqueeze(2) - k_t.unsqueeze(1)
        dx = q_x.unsqueeze(2) - k_x.unsqueeze(1)

        dx_sq = torch.sum(dx ** 2, dim=-1) # [B, N_q, N_k]
        dt_sq = (dt.squeeze(-1)) ** 2      # [B, N_q, N_k]

        # ds^2 = - (c * dt)^2 + ||dx||^2
        c_eff = torch.clamp(torch.abs(self.c), min=1e-3)
        ds2 = -(c_eff ** 2) * dt_sq + dx_sq

        # In past light cone iff:
        # 1. dt > 0 (k preceded q chronologically / logically)
        # 2. ds2 <= 0 (causally reachable, timelike or lightlike)
        in_past = dt.squeeze(-1) > 0
        is_timelike = ds2 <= 0
        past_light_cone = in_past & is_timelike

        return ds2, past_light_cone

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        custom_timestamps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, N_q, D]
            key: [B, N_k, D]
            value: [B, N_k, D]
            custom_timestamps: Optional explicit time coordinates [B, N_k, 1]
        Returns:
            out: Contextual representation [B, N_q, D]
            attn_weights: Causal attention distribution [B, N_q, N_k]
            intervals: Spacetime intervals [B, N_q, N_k]
        """
        q_t, q_x = query[..., :1], query[..., 1:]
        k_t, k_x = key[..., :1], key[..., 1:]

        if custom_timestamps is not None:
            k_t = custom_timestamps

        Q = torch.cat([q_t, self.q_spatial(q_x)], dim=-1)
        K = torch.cat([k_t, self.k_spatial(k_x)], dim=-1)
        V = self.v_proj(value)

        ds2, causal_mask = self.compute_causal_mask(Q, K)

        # For timelike events (ds2 <= 0), proper time distance tau = sqrt(-ds2)
        # Events closer in proper time receive higher attention.
        proper_time = torch.sqrt(torch.clamp(-ds2, min=1e-8))
        logits = -proper_time / (self.temperature * (self.d_model ** 0.25))

        # Space-like separated or future events are strictly unreachable
        logits = logits.masked_fill(~causal_mask, float("-inf"))

        weights = F.softmax(logits, dim=-1)
        # If nothing is in the light cone for a query, output zeros
        weights = torch.nan_to_num(weights, nan=0.0)

        out = torch.matmul(weights, V)
        out = self.out_proj(out)

        return out, weights, ds2
