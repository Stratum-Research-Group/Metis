"""
Titans & Test-Time Training (TTT) Associative Neural Memory
===========================================================
Based on Google Research (Behrouz et al., 2024 / 2025: "Titans: Learning to Memorize at Test Time").
Treats memory as an active neural network whose parameters update at inference time
driven by the gradient of surprise.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from lthcm.core.riemannian_langevin import RiemannianMetricTensor


class TitansNeuralMemory(nn.Module):
    """
    Test-Time Training (TTT) Associative Neural Memory Module.

    Instead of a static KV-cache or fixed-vector cell state, this module maintains
    an associative fast-weight matrix M_t that updates during inference.

    Dynamics:
      1. Retrieval: v_pred = M_{t-1} @ k_t
      2. Surprise: L_surprise = 0.5 * ||v_pred - v_t||^2
      3. Surprise Gradient: grad = (v_pred - v_t) (x) k_t^T
      4. Momentum: S_t = beta * S_{t-1} + (1 - beta) * grad
      5. Fast-weight update: M_t = (1 - alpha_t) * M_{t-1} - eta_t * S_t
    """

    def __init__(
        self,
        d_key: int,
        d_val: int,
        num_heads: int = 4,
        momentum_beta: float = 0.9,
        base_lr: float = 0.1,
        use_riemannian_langevin: bool = False,
        langevin_temperature: float = 0.01,
    ):
        super().__init__()
        self.d_key = d_key
        self.d_val = d_val
        self.num_heads = num_heads
        self.head_dim_k = d_key // num_heads
        self.head_dim_v = d_val // num_heads
        self.momentum_beta = momentum_beta
        self.base_lr = base_lr
        self.use_riemannian_langevin = use_riemannian_langevin
        self.langevin_temperature = langevin_temperature
        self.riemannian_metric = (
            RiemannianMetricTensor(d_key=self.head_dim_k, damping=0.1)
            if use_riemannian_langevin
            else None
        )

        # Projections for inputs
        self.q_proj = nn.Linear(d_key, d_key, bias=False)
        self.k_proj = nn.Linear(d_key, d_key, bias=False)
        self.v_proj = nn.Linear(d_val, d_val, bias=False)
        self.out_proj = nn.Linear(d_val, d_val, bias=False)

        # Dynamic gating projections (learning rate eta and forget gate alpha)
        self.gate_proj = nn.Linear(d_key + d_val, 2 * num_heads)

        # Persistent base parameters (initial memory priors)
        self.base_memory = nn.Parameter(
            torch.zeros(num_heads, self.head_dim_v, self.head_dim_k)
        )
        nn.init.kaiming_uniform_(self.base_memory, a=0.01)

        # Active test-time states (per-batch fast weights and momentum)
        self.register_buffer("M", None, persistent=False)
        self.register_buffer("S", None, persistent=False)

    def init_state(self, batch_size: int, device: torch.device):
        """Initialize or reset the fast-weight memory state."""
        # [B, H, D_v, D_k]
        self.M = self.base_memory.unsqueeze(0).repeat(batch_size, 1, 1, 1).clone()
        self.S = torch.zeros_like(self.M)

    def retrieve(self, q: torch.Tensor) -> torch.Tensor:
        """
        Query current test-time memory without updating state.
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
        # Reshape to multi-head: [B, H, D_k, 1]
        q_heads = q_proj.view(B_cur, self.num_heads, self.head_dim_k).unsqueeze(-1)
        q_heads = F.normalize(q_heads, p=2, dim=2)

        # Read: M @ q -> [B, H, D_v, 1]
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step of Test-Time Training: Read, evaluate surprise, and update memory.
        Args:
            k: Key tensor [B, D_key]
            v: Target value tensor [B, D_val]
        Returns:
            retrieved_v: [B, D_val]
            surprise_loss: Scalar surprise metric [B]
        """
        B = k.shape[0]
        if self.M is None or self.M.shape[0] != B:
            self.init_state(B, k.device)

        # Project key and value
        k_p = self.k_proj(k).view(B, self.num_heads, self.head_dim_k)
        v_p = self.v_proj(v).view(B, self.num_heads, self.head_dim_v)

        k_norm = F.normalize(k_p, p=2, dim=-1).unsqueeze(-1) # [B, H, D_k, 1]
        v_target = v_p.unsqueeze(-1) # [B, H, D_v, 1]

        # 1. Forward read from current fast weights: [B, H, D_v, 1]
        v_pred = torch.matmul(self.M, k_norm)

        # 2. Compute surprise error: e = v_pred - v_target
        error = v_pred - v_target
        surprise_loss = 0.5 * torch.sum(error ** 2, dim=[1, 2, 3])

        # 3. Dynamic gating: eta (learning rate) and alpha (forget factor)
        gate_in = torch.cat([k, v], dim=-1)
        gates = self.gate_proj(gate_in).view(B, self.num_heads, 2)
        eta = torch.sigmoid(gates[..., 0]).unsqueeze(-1).unsqueeze(-1) * self.base_lr
        alpha = torch.sigmoid(gates[..., 1]).unsqueeze(-1).unsqueeze(-1) * 0.1

        # 4. Gradient of surprise: grad = error @ k_norm^T -> [B, H, D_v, D_k]
        grad = torch.matmul(error, k_norm.transpose(-1, -2))

        # 5. Momentum update: S_t = beta * S_{t-1} + (1 - beta) * grad
        self.S = self.momentum_beta * self.S + (1.0 - self.momentum_beta) * grad

        # 6. Test-time weight update: standard or Riemannian Langevin dynamics
        if self.use_riemannian_langevin and self.riemannian_metric is not None:
            self.riemannian_metric.update(k_norm)
            _, L_inv = self.riemannian_metric.compute_cholesky()
            nat_S = self.riemannian_metric.natural_gradient(self.S, L_inv)
            noise = self.riemannian_metric.sample_anisotropic_noise(
                shape=self.M.shape,
                temperature=self.langevin_temperature,
                lr=self.base_lr,
                L_inv=L_inv,
            )
            self.M = (1.0 - alpha * 0.05) * self.M - eta * nat_S + noise
        else:
            self.M = (1.0 - alpha) * self.M - eta * self.S

        # Output projection
        v_out = self.out_proj(v_pred.squeeze(-1).view(B, self.d_val))
        return v_out, surprise_loss

    def forward(
        self,
        seq_k: torch.Tensor,
        seq_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process an entire sequence step-by-step, updating the neural memory dynamically.
        Args:
            seq_k: [B, SeqLen, D_key]
            seq_v: [B, SeqLen, D_val]
        Returns:
            outputs: [B, SeqLen, D_val]
            surprises: [B, SeqLen]
        """
        B, T, _ = seq_k.shape
        self.init_state(B, seq_k.device)

        outputs = []
        surprises = []
        for t in range(T):
            kt = seq_k[:, t, :]
            vt = seq_v[:, t, :]
            out_t, surp_t = self.step(kt, vt)
            outputs.append(out_t)
            surprises.append(surp_t)

        return torch.stack(outputs, dim=1), torch.stack(surprises, dim=1)
