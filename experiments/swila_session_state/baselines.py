"""Baselines for streaming session state comparison (Section 6).

Implements:
- B1: K independent narrow single-expert delta rule heads
- B2: One wide single-expert delta rule
- B3: Mean / EMA pooled value vector
- B4: Recency window with nearest-key lookup matching byte budget
- B5: Exact key-value store over all writes
- O1: Oracle projection ceiling
- C0: Random guess
"""

from __future__ import annotations

import math
import random
import torch
import torch.nn.functional as F


class B1NarrowHeads:
    """B1: K independent narrow single-expert delta rules.
    
    Each head j has state S_j of shape [D_v, D_k] with independent key projections.
    Total floats = K * D_v * D_k.
    """
    def __init__(self, K: int, D_k: int, D_v: int, beta: float = 0.5):
        self.K = K
        self.D_k = D_k
        self.D_v = D_v
        self.beta = beta
        # S: [K, D_v, D_k]
        self.S = torch.zeros(K, D_v, D_k, dtype=torch.float32)

    def write(self, k_heads: torch.Tensor, v: torch.Tensor):
        """Write key [K, D_k] and value [D_v]."""
        # k_heads: [K, D_k], v: [D_v]
        for j in range(self.K):
            k_j = k_heads[j]  # [D_k]
            pred_j = torch.matmul(self.S[j], k_j)  # [D_v]
            u_j = v - pred_j
            self.S[j] = self.S[j] + self.beta * torch.outer(u_j, k_j)

    def query(self, q_heads: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Query with q_heads [K, D_k], return average readout [D_v]."""
        readouts = []
        for j in range(self.K):
            q_j = q_heads[j] * scale
            v_hat_j = torch.matmul(self.S[j], q_j)
            readouts.append(v_hat_j)
        return torch.stack(readouts).mean(dim=0)


class B2WideState:
    """B2: One wide single-expert delta rule.
    
    Key dim D_k' = K * D_k, State S of shape [D_v, D_k'].
    Total floats = D_v * (K * D_k).
    """
    def __init__(self, D_k_prime: int, D_v: int, beta: float = 0.5):
        self.D_k_prime = D_k_prime
        self.D_v = D_v
        self.beta = beta
        self.S = torch.zeros(D_v, D_k_prime, dtype=torch.float32)

    def write(self, k_wide: torch.Tensor, v: torch.Tensor):
        """Write key [D_k_prime] and value [D_v]."""
        pred = torch.matmul(self.S, k_wide)
        u = v - pred
        self.S = self.S + self.beta * torch.outer(u, k_wide)

    def query(self, q_wide: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Query with q_wide [D_k_prime], return [D_v]."""
        return torch.matmul(self.S, q_wide * scale)


class B3MeanPooled:
    """B3: Running Mean / EMA pooled value vector."""
    def __init__(self, D_v: int, decay: float = 0.95):
        self.D_v = D_v
        self.decay = decay
        self.s = torch.zeros(D_v, dtype=torch.float32)
        self.count = 0

    def write(self, v: torch.Tensor):
        if self.count == 0:
            self.s = v.clone()
        else:
            self.s = self.decay * self.s + (1.0 - self.decay) * v
        self.count += 1

    def query(self) -> torch.Tensor:
        return self.s.clone()


class B4RecencyWindow:
    """B4: Recency window with nearest-key lookup matching S1 state float budget."""
    def __init__(self, total_state_floats: int, D_k: int, D_v: int):
        self.capacity = max(1, total_state_floats // (D_k + D_v))
        self.D_k = D_k
        self.D_v = D_v
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def write(self, k: torch.Tensor, v: torch.Tensor):
        self.keys.append(k.clone())
        self.values.append(v.clone())
        if len(self.keys) > self.capacity:
            self.keys.pop(0)
            self.values.pop(0)

    def query(self, q: torch.Tensor) -> torch.Tensor:
        if not self.keys:
            return torch.zeros(self.D_v, dtype=torch.float32)
        
        q_norm = F.normalize(q, dim=0)
        k_stack = torch.stack(self.keys)  # [N, D_k]
        k_norm = F.normalize(k_stack, dim=1)
        sims = torch.matmul(k_norm, q_norm)
        best_idx = int(torch.argmax(sims).item())
        return self.values[best_idx].clone()


class B5ExactKVStore:
    """B5: Exact key-value store over all historical writes (unbounded reference)."""
    def __init__(self, D_v: int):
        self.D_v = D_v
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def write(self, k: torch.Tensor, v: torch.Tensor):
        self.keys.append(k.clone())
        self.values.append(v.clone())

    def query(self, q: torch.Tensor) -> torch.Tensor:
        if not self.keys:
            return torch.zeros(self.D_v, dtype=torch.float32)
        q_norm = F.normalize(q, dim=0)
        k_stack = torch.stack(self.keys)
        k_norm = F.normalize(k_stack, dim=1)
        sims = torch.matmul(k_norm, q_norm)
        best_idx = int(torch.argmax(sims).item())
        return self.values[best_idx].clone()
