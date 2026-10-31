"""Second-order matrix preconditioner optimizer (Shampoo) and AdamW baseline.

Implements Distributed Shampoo (arXiv:2002.09018, Google Research scalable_shampoo)
for matrix parameters (e.g. LoRA projection adapters W_A and W_B).
Maintains Kronecker-factored left and right covariance accumulators:
    L_t = beta2 * L_{t-1} + (1 - beta2) * G_t G_t^T + eps * I
    R_t = beta2 * R_{t-1} + (1 - beta2) * G_t^T G_t + eps * I
and computes matrix inverse 4th roots periodically to precondition gradients:
    G~_t = L_t^{-1/4} * G_t * R_t^{-1/4}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ShampooConfig:
    """Hyperparameters for Shampoo second-order matrix preconditioning."""
    lr: float = 1e-3
    beta1: float = 0.9          # First-order momentum coefficient
    beta2: float = 0.99         # Covariance accumulator decay
    epsilon: float = 1e-4       # Tikhonov regularization floor
    weight_decay: float = 1e-4  # L2 weight decay / decoupled regularizer
    precondition_freq: int = 10 # Frequency of computing matrix inverse roots
    max_preconditioner_dim: int = 1024 # Block limit for high-dimensional matrices
    clip_grad_norm: float = 1.0 # Gradient clipping threshold


def matrix_inverse_fourth_root(
    matrix: np.ndarray,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Computes M^{-1/4} for a symmetric positive semidefinite matrix M.
    
    Uses symmetric eigendecomposition: M = Q * diag(lambda) * Q^T
    M^{-1/4} = Q * diag((max(lambda, eps))^{-1/4}) * Q^T.
    """
    # Ensure symmetry
    matrix_sym = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(matrix_sym)
    # Numerical floor
    evals_clamped = np.maximum(evals, epsilon)
    evals_inv_root = np.power(evals_clamped, -0.25)
    inv_fourth = (evecs * evals_inv_root) @ evecs.T
    return 0.5 * (inv_fourth + inv_fourth.T)


class ShampooOptimizer:
    """Second-order preconditioned optimizer for tensor / matrix parameters.
    
    Operates on collections of parameter matrices (e.g. LoRA adapter matrices).
    Supports 1D vectors (using standard scalar diagonal Adagrad/AdamW preconditioner)
    and 2D matrices (using Kronecker left and right covariance preconditioners).
    """

    def __init__(
        self,
        params: List[np.ndarray],
        config: Optional[ShampooConfig] = None,
    ):
        self.params = params
        self.config = config or ShampooConfig()
        self.step_count: int = 0

        # State dicts per parameter index
        self.momentum: List[np.ndarray] = [np.zeros_like(p) for p in params]
        self.left_cov: List[Optional[np.ndarray]] = []
        self.right_cov: List[Optional[np.ndarray]] = []
        self.inv_left: List[Optional[np.ndarray]] = []
        self.inv_right: List[Optional[np.ndarray]] = []
        self.vector_cov: List[Optional[np.ndarray]] = []

        for p in params:
            if p.ndim == 2:
                d_out, d_in = p.shape
                self.left_cov.append(np.zeros((d_out, d_out), dtype=p.dtype))
                self.right_cov.append(np.zeros((d_in, d_in), dtype=p.dtype))
                self.inv_left.append(np.eye(d_out, dtype=p.dtype))
                self.inv_right.append(np.eye(d_in, dtype=p.dtype))
                self.vector_cov.append(None)
            elif p.ndim == 1:
                self.left_cov.append(None)
                self.right_cov.append(None)
                self.inv_left.append(None)
                self.inv_right.append(None)
                self.vector_cov.append(np.zeros_like(p))
            else:
                # Higher-dimensional tensors flattened to 2D
                d_out = p.shape[0]
                d_in = int(np.prod(p.shape[1:]))
                self.left_cov.append(np.zeros((d_out, d_out), dtype=p.dtype))
                self.right_cov.append(np.zeros((d_in, d_in), dtype=p.dtype))
                self.inv_left.append(np.eye(d_out, dtype=p.dtype))
                self.inv_right.append(np.eye(d_in, dtype=p.dtype))
                self.vector_cov.append(None)

    def step(self, grads: List[np.ndarray]) -> None:
        """Execute one optimization step given parameter gradients."""
        if len(grads) != len(self.params):
            raise ValueError(f"Gradient count {len(grads)} does not match parameter count {len(self.params)}")

        self.step_count += 1
        cfg = self.config
        update_preconditioners = (self.step_count % cfg.precondition_freq == 0) or (self.step_count == 1)

        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue

            orig_shape = p.shape
            if p.ndim == 1:
                # 1D vector fallback: standard adaptive second moment
                vcov = self.vector_cov[i]
                assert vcov is not None
                vcov[:] = cfg.beta2 * vcov + (1.0 - cfg.beta2) * (g * g)
                bias_correction2 = 1.0 - (cfg.beta2 ** self.step_count)
                denom = np.sqrt(vcov / max(bias_correction2, 1e-8)) + cfg.epsilon
                preconditioned_g = g / denom
            else:
                g_2d = g.reshape(p.shape[0], -1)
                l_cov = self.left_cov[i]
                r_cov = self.right_cov[i]
                assert l_cov is not None and r_cov is not None

                # Update covariance statistics
                l_cov[:] = cfg.beta2 * l_cov + (1.0 - cfg.beta2) * (g_2d @ g_2d.T)
                r_cov[:] = cfg.beta2 * r_cov + (1.0 - cfg.beta2) * (g_2d.T @ g_2d)

                if update_preconditioners:
                    # Invert 4th root
                    self.inv_left[i] = matrix_inverse_fourth_root(l_cov, epsilon=cfg.epsilon)
                    self.inv_right[i] = matrix_inverse_fourth_root(r_cov, epsilon=cfg.epsilon)

                inv_l = self.inv_left[i]
                inv_r = self.inv_right[i]
                assert inv_l is not None and inv_r is not None

                # G~ = L^{-1/4} * G * R^{-1/4}
                preconditioned_g_2d = inv_l @ g_2d @ inv_r
                preconditioned_g = preconditioned_g_2d.reshape(orig_shape)

            # Gradient clipping
            if cfg.clip_grad_norm > 0.0:
                g_norm = float(np.linalg.norm(preconditioned_g))
                if g_norm > cfg.clip_grad_norm:
                    preconditioned_g = preconditioned_g * (cfg.clip_grad_norm / max(g_norm, 1e-8))

            # Weight decay (decoupled)
            if cfg.weight_decay > 0.0:
                p -= cfg.lr * cfg.weight_decay * p

            # First-order momentum update
            m = self.momentum[i]
            m[:] = cfg.beta1 * m + (1.0 - cfg.beta1) * preconditioned_g
            bias_correction1 = 1.0 - (cfg.beta1 ** self.step_count)
            step_update = m / max(bias_correction1, 1e-8)

            # Apply parameter update
            p -= cfg.lr * step_update

    def zero_grad(self) -> None:
        """Reset internal accumulator state (noop if caller supplies fresh grads)."""
        pass


class AdamWOptimizer:
    """Standard AdamW reference optimizer for matched comparative benchmarking."""

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 1e-4,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.step_count = 0

        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        self.step_count += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue

            if self.weight_decay > 0.0:
                p -= self.lr * self.weight_decay * p

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)

            bc1 = 1.0 - (self.beta1 ** self.step_count)
            bc2 = 1.0 - (self.beta2 ** self.step_count)

            m_hat = self.m[i] / max(bc1, 1e-8)
            v_hat = self.v[i] / max(bc2, 1e-8)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


def benchmark_shampoo_vs_adamw(
    dim_out: int = 16,
    dim_in: int = 32,
    num_steps: int = 40,
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs a controlled convergence benchmark of Shampoo vs AdamW on an ill-conditioned matrix problem.
    
    Objective: minimize 0.5 * ||W @ X - Y||_F^2 where X has decaying singular values (high condition number).
    """
    rng = np.random.RandomState(seed)

    # Ill-conditioned input matrix X
    U, _ = np.linalg.qr(rng.randn(dim_in, dim_in))
    V, _ = np.linalg.qr(rng.randn(10, 10))
    # Decay singular values exponentially
    singulars = np.logspace(0, -3, min(dim_in, 10))
    S = np.zeros((dim_in, 10))
    np.fill_diagonal(S, singulars)
    X = U[:, :dim_in] @ S @ V.T # shape (dim_in, 10)

    # Gold target W_gold
    W_gold = rng.randn(dim_out, dim_in)
    Y = W_gold @ X # shape (dim_out, 10)

    # Initial weights
    W0 = rng.randn(dim_out, dim_in) * 0.1

    # Run AdamW
    W_adam = W0.copy()
    opt_adam = AdamWOptimizer([W_adam], lr=0.01, weight_decay=0.0)
    losses_adam: List[float] = []

    for _ in range(num_steps):
        diff = W_adam @ X - Y
        loss = 0.5 * float(np.sum(diff ** 2))
        losses_adam.append(loss)
        grad = diff @ X.T
        opt_adam.step([grad])

    # Run Shampoo
    W_shampoo = W0.copy()
    cfg = ShampooConfig(lr=0.01, precondition_freq=5, weight_decay=0.0, epsilon=1e-4)
    opt_shampoo = ShampooOptimizer([W_shampoo], config=cfg)
    losses_shampoo: List[float] = []

    for _ in range(num_steps):
        diff = W_shampoo @ X - Y
        loss = 0.5 * float(np.sum(diff ** 2))
        losses_shampoo.append(loss)
        grad = diff @ X.T
        opt_shampoo.step([grad])

    return {
        "num_steps": num_steps,
        "adamw_initial_loss": round(losses_adam[0], 6),
        "adamw_final_loss": round(losses_adam[-1], 6),
        "shampoo_initial_loss": round(losses_shampoo[0], 6),
        "shampoo_final_loss": round(losses_shampoo[-1], 6),
        "shampoo_relative_gain_percent": round(
            100.0 * (losses_adam[-1] - losses_shampoo[-1]) / max(losses_adam[-1], 1e-9), 2
        ),
        "losses_adamw": [round(l, 4) for l in losses_adam],
        "losses_shampoo": [round(l, 4) for l in losses_shampoo],
    }
