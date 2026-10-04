"""
Riemannian Langevin Dynamics for Memory Lifecycle & Emergent Forgetting
========================================================================
Implements curvature-aware thermodynamic memory consolidation and emergent forgetting
based on Riemannian Stochastic Gradient Langevin Dynamics (Riemannian SGLD).

Mathematical Formulation:
  1. Memory State Manifold (M, G):
     Fast weights M_t ∈ R^{B × H × D_v × D_k} live on a Riemannian manifold
     equipped with metric tensor G(M) induced by the empirical key covariance:
       G_t = (1 - ρ) G_{t-1} + ρ (k_t k_t^T + λ_0 I)

  2. Continuous Ito SDE on Riemannian Manifold:
     dM_t = -η G_t^{-1} ∇_M L_{surprise} dt + √(2η / β) G_t^{-1/2} dW_t - γ λ_{prior} M_t dt
     where:
       - ∇_M L_{surprise} is the surprise error gradient (v_pred - v_t) ⊗ k_t^T
       - G_t^{-1} ∇_M L is the natural gradient (curvature-corrected step)
       - β = 1 / T_{temp} is the thermodynamic inverse temperature
       - dW_t ~ N(0, I) is standard Brownian motion in ambient space
       - G_t^{-1/2} dW_t generates anisotropic thermal diffusion

  3. Emergent Thermodynamic Forgetting:
     - High-curvature directions (frequently referenced facts, λ_max(G) >> 1):
         Diffusion covariance G^{-1} << 1 => Thermal noise is suppressed.
         High-priority knowledge is stably preserved against degradation.
     - Low-curvature directions (stale or redundant facts, λ_min(G) ≈ λ_0):
         Diffusion covariance G^{-1} ≈ 1 / λ_0 >> 1 => Thermal drift dominates.
         Unused memory traces naturally evaporate without artificial heuristic TTL.

Reference: Tier 3 #12 (Riemannian Langevin Dynamics) in impro.md.
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RiemannianMetricTensor(nn.Module):
    """
    Online Riemannian metric tensor estimator G_t over key covariance manifolds.
    Computes natural gradients and samples anisotropic Brownian noise.
    """

    def __init__(
        self,
        d_key: int,
        damping: float = 0.1,
        momentum: float = 0.95,
    ):
        super().__init__()
        self.d_key = d_key
        self.damping = damping
        self.momentum = momentum
        self.register_buffer("G", None, persistent=False)

    def init_metric(self, batch_size: int, num_heads: int, device: torch.device):
        """Initializes metric tensor as a scaled identity matrix."""
        eye = torch.eye(self.d_key, device=device).unsqueeze(0).unsqueeze(0)
        self.G = eye.repeat(batch_size, num_heads, 1, 1) * (1.0 + self.damping)

    def update(self, k_norm: torch.Tensor):
        """
        Rank-1 Bayesian update of metric tensor from normalized incoming key:
          G_t = momentum * G_{t-1} + (1 - momentum) * (k_t k_t^T + damping * I)
        Args:
            k_norm: [B, H, D_k, 1]
        """
        B, H, D_k, _ = k_norm.shape
        if self.G is None or self.G.shape[0] != B or self.G.shape[1] != H:
            self.init_metric(B, H, k_norm.device)

        k_cov = torch.matmul(k_norm, k_norm.transpose(-1, -2)) # [B, H, D_k, D_k]
        eye = torch.eye(D_k, device=k_norm.device).unsqueeze(0).unsqueeze(0)
        target_cov = k_cov + self.damping * eye
        self.G = self.momentum * self.G + (1.0 - self.momentum) * target_cov

    def compute_cholesky(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Cholesky factor L (G = L L^T) and its inverse L^{-1}.
        Returns:
            L: [B, H, D_k, D_k]
            L_inv: [B, H, D_k, D_k]
        """
        # Ensure positive-definiteness via diagonal jitter
        eye = torch.eye(self.d_key, device=self.G.device).unsqueeze(0).unsqueeze(0)
        G_reg = self.G + 1e-5 * eye
        L = torch.linalg.cholesky(G_reg)
        L_inv = torch.linalg.inv(L)
        return L, L_inv

    def natural_gradient(self, grad: torch.Tensor, L_inv: torch.Tensor) -> torch.Tensor:
        """
        Computes natural gradient step: grad @ G^{-1} = grad @ (L_inv^T L_inv).
        Args:
            grad: [B, H, D_v, D_k]
            L_inv: [B, H, D_k, D_k]
        Returns:
            nat_grad: [B, H, D_v, D_k]
        """
        G_inv = torch.matmul(L_inv.transpose(-1, -2), L_inv)
        return torch.matmul(grad, G_inv)

    def sample_anisotropic_noise(
        self,
        shape: Tuple[int, ...],
        temperature: float,
        lr: float,
        L_inv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Samples anisotropic Riemannian Brownian noise:
          ΔM_noise = √(2 * lr * temperature) * ξ @ L_inv
        where ξ ~ N(0, I). Covariance: E[ΔM^T ΔM] ∝ L_inv^T L_inv = G^{-1}.
        """
        xi = torch.randn(shape, device=self.G.device, dtype=self.G.dtype)
        scale = math.sqrt(max(2.0 * lr * temperature, 0.0))
        # Project noise along inverse Riemannian metric
        noise = torch.matmul(xi, L_inv)
        return scale * noise

    def get_spectrum(self) -> Tuple[float, float]:
        """Returns (min_eigenvalue, max_eigenvalue) of the metric tensor."""
        if self.G is None:
            return 1.0, 1.0
        eigvals = torch.linalg.eigvalsh(self.G)
        return float(eigvals.min().item()), float(eigvals.max().item())


class RiemannianLangevinMemory(nn.Module):
    """
    Curvature-Aware Associative Fast-Weight Neural Memory.
    Governed by Riemannian Langevin Stochastic Differential Equations.
    """

    def __init__(
        self,
        d_key: int,
        d_val: int,
        num_heads: int = 4,
        temperature: float = 0.01,
        base_lr: float = 0.05,
        damping: float = 0.1,
        prior_decay: float = 0.001,
    ):
        super().__init__()
        self.d_key = d_key
        self.d_val = d_val
        self.num_heads = num_heads
        self.head_dim_k = d_key // num_heads
        self.head_dim_v = d_val // num_heads
        self.temperature = temperature
        self.base_lr = base_lr
        self.prior_decay = prior_decay

        # Linear projections (initialized to identity for zero-shot associative fidelity)
        self.q_proj = nn.Linear(d_key, d_key, bias=False)
        self.k_proj = nn.Linear(d_key, d_key, bias=False)
        self.v_proj = nn.Linear(d_val, d_val, bias=False)
        self.out_proj = nn.Linear(d_val, d_val, bias=False)
        nn.init.eye_(self.q_proj.weight)
        nn.init.eye_(self.k_proj.weight)
        nn.init.eye_(self.v_proj.weight)
        nn.init.eye_(self.out_proj.weight)

        # Riemannian metric tensor estimator
        self.metric = RiemannianMetricTensor(d_key=self.head_dim_k, damping=damping)

        # Persistent base memory priors
        self.base_memory = nn.Parameter(
            torch.zeros(num_heads, self.head_dim_v, self.head_dim_k)
        )

        # Active fast-weight state buffer
        self.register_buffer("M", None, persistent=False)

    def init_state(self, batch_size: int, device: torch.device):
        """Initializes fast-weight state and Riemannian metric tensor."""
        self.M = self.base_memory.unsqueeze(0).repeat(batch_size, 1, 1, 1).clone()
        self.metric.init_metric(batch_size, self.num_heads, device)

    def retrieve(self, q: torch.Tensor) -> torch.Tensor:
        """
        Query current test-time memory state.
        Args:
            q: [B, D_key] or [B, N, D_key]
        Returns:
            v_out: [B, D_val] or [B, N, D_val]
        """
        if self.M is None:
            self.init_state(q.shape[0], q.device)

        is_seq = q.dim() == 3
        if is_seq:
            B, N, _ = q.shape
            q_flat = q.reshape(B * N, -1)
        else:
            q_flat = q

        B_cur = q_flat.shape[0]
        q_proj = self.q_proj(q_flat)
        q_heads = q_proj.view(B_cur, self.num_heads, self.head_dim_k).unsqueeze(-1)
        q_heads = F.normalize(q_heads, p=2, dim=2)

        if self.M.shape[0] != B_cur:
            M_expanded = self.M.repeat_interleave(B_cur // self.M.shape[0], dim=0)
        else:
            M_expanded = self.M

        v_read = torch.matmul(M_expanded, q_heads).squeeze(-1) # [B, H, D_v]
        v_out = self.out_proj(v_read.view(B_cur, self.d_val))

        if is_seq:
            v_out = v_out.view(B, N, self.d_val)
        return v_out

    def step(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute one Riemannian Langevin update step:
          1. Evaluate surprise error
          2. Update Riemannian metric tensor G_t
          3. Compute natural gradient step via Cholesky decomposition
          4. Sample anisotropic thermal noise G_t^{-1/2} ξ
          5. Update fast weights M_t
        """
        B = k.shape[0]
        if self.M is None or self.M.shape[0] != B:
            self.init_state(B, k.device)

        # Project inputs
        k_p = self.k_proj(k).view(B, self.num_heads, self.head_dim_k)
        v_p = self.v_proj(v).view(B, self.num_heads, self.head_dim_v)

        k_norm = F.normalize(k_p, p=2, dim=-1).unsqueeze(-1) # [B, H, D_k, 1]
        v_target = v_p.unsqueeze(-1) # [B, H, D_v, 1]

        # 1. Forward read from fast weights: [B, H, D_v, 1]
        v_pred = torch.matmul(self.M, k_norm)

        # 2. Surprise error
        error = v_pred - v_target
        surprise_loss = 0.5 * torch.sum(error ** 2, dim=[1, 2, 3])

        # 3. Euclidean gradient of surprise: grad = error @ k_norm^T
        euc_grad = torch.matmul(error, k_norm.transpose(-1, -2))

        # 4. Bayesian metric tensor update
        self.metric.update(k_norm)

        # 5. Cholesky factorization: G = L L^T
        L, L_inv = self.metric.compute_cholesky()

        # 6. Natural gradient step: nat_grad = euc_grad @ G^{-1}
        nat_grad = self.metric.natural_gradient(euc_grad, L_inv)

        # 7. Anisotropic thermal noise: scale * ξ @ L_inv
        thermal_noise = self.metric.sample_anisotropic_noise(
            shape=self.M.shape,
            temperature=self.temperature,
            lr=self.base_lr,
            L_inv=L_inv,
        )

        # 8. Complete Riemannian Langevin SDE integration:
        #    M_t = M_{t-1} - η * nat_grad + thermal_noise - γ * prior_decay * M_{t-1}
        self.M = (
            (1.0 - self.prior_decay) * self.M
            - self.base_lr * nat_grad
            + thermal_noise
        )

        # Diagnostics
        min_eig, max_eig = self.metric.get_spectrum()
        diagnostics = {
            "surprise_loss": float(surprise_loss.mean().item()),
            "curvature_condition_number": float(max_eig / max(min_eig, 1e-6)),
            "min_curvature": min_eig,
            "max_curvature": max_eig,
            "thermal_noise_norm": float(torch.norm(thermal_noise).item()),
            "natural_grad_norm": float(torch.norm(nat_grad).item()),
            "effective_temperature": self.temperature,
        }

        v_out = self.out_proj(v_pred.squeeze(-1).view(B, self.d_val))
        return v_out, surprise_loss, diagnostics
