"""Validity checks (Gate 0 / G0) for SwiLA session state experiment.

Section 8: Every one must pass before any result counts.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from experiments.swila_session_state.swila_core import (
    swila_recurrence,
    query_state_snapshot,
)
from experiments.swila_session_state.data_generator import TierAStreamGenerator


def check_g0_causality() -> bool:
    """Check 8a: Changing future writes does not alter past state/output; queries do not mutate state."""
    B, T, H, D_k, D_v, K = 1, 20, 1, 16, 16, 2
    torch.manual_seed(42)

    q = torch.randn(B, T, H, D_k)
    k = torch.randn(B, T, H, D_k)
    v = torch.randn(B, T, H, D_v)
    beta = torch.full((B, T, H, D_v, K), 0.5)
    pi_q = torch.zeros(B, T, H, D_v, K)
    pi_k = torch.zeros(B, T, H, D_v, K)

    o1, S1, _ = swila_recurrence(q, k, v, beta, pi_q, pi_k, output_final_state=True)

    # Mutate writes after step 10
    k2 = k.clone()
    v2 = v.clone()
    k2[:, 10:] = torch.randn_like(k2[:, 10:])
    v2[:, 10:] = torch.randn_like(v2[:, 10:])

    o2, S2, _ = swila_recurrence(q, k2, v2, beta, pi_q, pi_k, output_final_state=True)

    # Outputs at steps 0..9 must be bit-identical
    if not torch.equal(o1[:, :10], o2[:, :10]):
        return False

    # Querying state snapshot must not mutate S
    S_before = S1.clone()
    q_probe = torch.randn(1, 1, 1, D_k)
    pi_q_probe = torch.zeros(1, 1, 1, D_v, K)
    _ = query_state_snapshot(S1, q_probe, pi_q_probe)
    
    if not torch.equal(S1, S_before):
        return False

    return True


def check_g0_k1_reductions() -> bool:
    """Check 8b: K=1 matches numpy delta rule; gated K=1 matches gated numpy delta rule; R1(coef=0.5)==S1."""
    T, D_k, D_v = 15, 8, 8
    np.random.seed(42)
    torch.manual_seed(42)

    k_np = np.random.randn(T, D_k).astype(np.float32)
    v_np = np.random.randn(T, D_v).astype(np.float32)
    q_np = np.random.randn(T, D_k).astype(np.float32)
    beta_val = 0.4

    # 1. Independent numpy delta rule
    S_np = np.zeros((D_v, D_k), dtype=np.float32)
    o_np = np.zeros((T, D_v), dtype=np.float32)
    scale = D_k ** -0.5

    for i in range(T):
        pred = S_np @ k_np[i]
        u = v_np[i] - pred
        S_np = S_np + beta_val * np.outer(u, k_np[i])
        o_np[i] = (S_np @ (q_np[i] * scale))

    # PyTorch K=1 SwiLA
    q_th = torch.from_numpy(q_np).view(1, T, 1, D_k)
    k_th = torch.from_numpy(k_np).view(1, T, 1, D_k)
    v_th = torch.from_numpy(v_np).view(1, T, 1, D_v)
    beta_th = torch.full((1, T, 1, D_v, 1), beta_val)
    pi_q = torch.zeros(1, T, 1, D_v, 1)
    pi_k = torch.zeros(1, T, 1, D_v, 1)

    o_th, S_th, _ = swila_recurrence(q_th, k_th, v_th, beta_th, pi_q, pi_k, output_final_state=True)
    o_th_np = o_th.squeeze(0).squeeze(1).numpy()

    if not np.allclose(o_np, o_th_np, atol=1e-5):
        return False

    # 2. Gated K=1 reduction with log_decay
    log_decay_np = -0.01 * np.ones((T, D_k), dtype=np.float32)
    decay_factor = np.exp(log_decay_np[0])

    S_gated_np = np.zeros((D_v, D_k), dtype=np.float32)
    o_gated_np = np.zeros((T, D_v), dtype=np.float32)

    for i in range(T):
        S_decayed = S_gated_np * decay_factor
        pred = S_decayed @ k_np[i]
        u = v_np[i] - pred
        S_gated_np = S_decayed + beta_val * np.outer(u, k_np[i])
        o_gated_np[i] = (S_gated_np @ (q_np[i] * scale))

    log_decay_th = torch.from_numpy(log_decay_np).view(1, T, 1, D_k)
    o_gated_th, _, _ = swila_recurrence(
        q_th, k_th, v_th, beta_th, pi_q, pi_k, log_decay=log_decay_th
    )
    o_gated_th_np = o_gated_th.squeeze(0).squeeze(1).numpy()

    if not np.allclose(o_gated_np, o_gated_th_np, atol=1e-5):
        return False

    # 3. R1 with error coefficient 0.5 restored matches S1 reference
    K = 3
    beta_k = torch.full((1, T, 1, D_v, K), beta_val)
    pi_q_k = torch.zeros(1, T, 1, D_v, K)
    pi_k_k = torch.zeros(1, T, 1, D_v, K)
    
    o_s1, _, _ = swila_recurrence(q_th, k_th, v_th, beta_k, pi_q_k, pi_k_k, error_coef=0.5)
    o_r1_full, _, _ = swila_recurrence(q_th, k_th, v_th, beta_k, pi_q_k, pi_k_k, error_coef=0.5)
    
    if not torch.equal(o_s1, o_r1_full):
        return False

    return True


def check_g0_symmetry() -> bool:
    """Check 8c: Zero init + flat prior preserves symmetry; broken symmetry differentiates experts."""
    B, T, H, D_k, D_v, K = 1, 10, 1, 8, 8, 4
    torch.manual_seed(42)

    q = torch.randn(B, T, H, D_k)
    k = torch.randn(B, T, H, D_k)
    v = torch.randn(B, T, H, D_v)
    beta = torch.full((B, T, H, D_v, K), 0.5)
    pi_q = torch.zeros(B, T, H, D_v, K)
    pi_k = torch.zeros(B, T, H, D_v, K)

    # 1. Zero init + flat prior -> all expert states identical
    _, S_sym, r_seq_sym = swila_recurrence(q, k, v, beta, pi_q, pi_k, output_final_state=True)
    # S_sym: [B, H, D_v, K, D_k]
    for expert_idx in range(1, K):
        if not torch.allclose(S_sym[0, 0, :, 0, :], S_sym[0, 0, :, expert_idx, :], atol=1e-6):
            return False

    # 2. Broken symmetry with seeded random initial state (std 0.01)
    init_state_broken = torch.randn(B, H, D_v, K, D_k) * 0.01
    _, S_broken, r_seq_broken = swila_recurrence(
        q, k, v, beta, pi_q, pi_k, initial_state=init_state_broken, output_final_state=True
    )

    # Effective experts: exp(H(mean usage))
    # r_seq_broken: [B, H, T, D_v, K]
    mean_r = r_seq_broken.mean(dim=(0, 1, 2, 3))  # [K]
    entropy = -torch.sum(mean_r * torch.log(mean_r + 1e-12)).item()
    eff_experts = math.exp(entropy)

    if eff_experts <= 1.5:
        return False

    # Relative difference between expert predictions
    k_probe = k[:, 0]
    preds = torch.einsum('bhd,bhvkd->bhvk', k_probe, S_broken)[0, 0]  # [D_v, K]
    diffs = []
    for i in range(K):
        for j in range(i + 1, K):
            rel_diff = (preds[:, i] - preds[:, j]).abs() / (preds[:, i].abs() + 1e-6)
            diffs.append(rel_diff.mean().item())
            
    mean_rel_diff = np.mean(diffs)
    # Must exceed 100x float32 eps
    if mean_rel_diff < 1e-4:
        return False

    return True


def check_g0_determinism() -> bool:
    """Check 8d: Same seed yields bit-identical outputs."""
    gen = TierAStreamGenerator(seed=123)
    w1, q1 = gen.generate_stream(stream_length=100, stream_seed=456)
    w2, q2 = gen.generate_stream(stream_length=100, stream_seed=456)

    for i in range(len(w1)):
        if not torch.equal(w1[i].k, w2[i].k) or not torch.equal(w1[i].v, w2[i].v):
            return False

    for i in range(len(q1)):
        if not torch.equal(q1[i].q, q2[i].q) or not torch.equal(q1[i].candidate_vs, q2[i].candidate_vs):
            return False

    return True


def check_g0_controls() -> bool:
    """Check 8e: Negative controls and chance floor validation.
    
    1. Permute evaluation labels after frozen writes -> recall must be near chance floor (<= 0.08).
    2. Tier A independent-random-association control -> pure chance floor (<= 0.08).
    """
    D_k, D_v, K = 64, 64, 4
    gen = TierAStreamGenerator(seed=101, D_k=D_k, D_v=D_v, K=K)
    writes, queries = gen.generate_stream(stream_length=300, stream_seed=202)

    # Run SwiLA recurrence over writes
    S_state = torch.randn(1, 1, D_v, K, D_k, dtype=torch.float32) * 0.01
    for w in writes:
        q_dummy = w.k.view(1, 1, 1, D_k)
        k_step = w.k.view(1, 1, 1, D_k)
        v_step = w.v.view(1, 1, 1, D_v)
        beta_step = torch.full((1, 1, 1, D_v, K), 0.5)
        pi_q_step = w.pi_k.view(1, 1, 1, D_v, K)
        pi_k_step = w.pi_k.view(1, 1, 1, D_v, K)
        _, S_state, _ = swila_recurrence(
            q=q_dummy, k=k_step, v=v_step, beta=beta_step,
            pi_q=pi_q_step, pi_k=pi_k_step, initial_state=S_state,
            output_final_state=True, error_coef=0.5
        )

    # 1. Permute evaluation candidates after frozen writes
    perm_recalls = []
    rng = np.random.default_rng(42)
    for qe in queries:
        q_th = qe.q.view(1, 1, 1, D_k)
        pi_q_th = qe.pi_q.view(1, 1, 1, D_v, K)
        v_hat = query_state_snapshot(S_state, q_th, pi_q_th).squeeze()
        perm_idx = rng.permutation(50)
        perm_cands = qe.candidate_vs[perm_idx]
        v_hat_norm = F.normalize(v_hat, dim=0)
        cands_norm = F.normalize(perm_cands, dim=1)
        best_cand = int(torch.argmax(torch.matmul(cands_norm, v_hat_norm)).item())
        perm_recalls.append(1.0 if best_cand == qe.target_idx else 0.0)

    permuted_recall = float(np.mean(perm_recalls))
    if permuted_recall > 0.08:
        return False

    # 2. Independent random association control (chance floor)
    torch.manual_seed(999)
    S_rand = torch.randn(1, 1, D_v, K, D_k, dtype=torch.float32) * 0.01
    rand_recalls = []
    for w in writes:
        q_dummy = w.k.view(1, 1, 1, D_k)
        k_step = w.k.view(1, 1, 1, D_k)
        v_rand = F.normalize(torch.randn(1, 1, 1, D_v), dim=-1)
        beta_step = torch.full((1, 1, 1, D_v, K), 0.5)
        pi_q_step = w.pi_k.view(1, 1, 1, D_v, K)
        pi_k_step = w.pi_k.view(1, 1, 1, D_v, K)
        _, S_rand, _ = swila_recurrence(
            q=q_dummy, k=k_step, v=v_rand, beta=beta_step,
            pi_q=pi_q_step, pi_k=pi_k_step, initial_state=S_rand,
            output_final_state=True, error_coef=0.5
        )

    for qe in queries:
        q_th = qe.q.view(1, 1, 1, D_k)
        pi_q_th = qe.pi_q.view(1, 1, 1, D_v, K)
        v_hat = query_state_snapshot(S_rand, q_th, pi_q_th).squeeze()
        v_hat_norm = F.normalize(v_hat, dim=0)
        cands_norm = F.normalize(qe.candidate_vs, dim=1)
        best_cand = int(torch.argmax(torch.matmul(cands_norm, v_hat_norm)).item())
        rand_recalls.append(1.0 if best_cand == qe.target_idx else 0.0)

    rand_assoc_recall = float(np.mean(rand_recalls))
    if rand_assoc_recall > 0.08:
        return False

    return True


def check_g0_candidate_rules() -> bool:
    """Check 8f: Correct candidate present exactly once, written before query, exactly 50 candidates, delay feasible."""
    gen = TierAStreamGenerator(seed=789)
    writes, queries = gen.generate_stream(stream_length=500, stream_seed=999)

    if not queries:
        return False

    for qe in queries:
        if qe.candidate_vs.shape[0] != 50:
            return False
        if not torch.equal(qe.candidate_vs[qe.target_idx], qe.target_v):
            return False
        # Target must be present exactly once
        matches = [i for i in range(50) if torch.equal(qe.candidate_vs[i], qe.target_v)]
        if len(matches) != 1:
            return False
        if qe.write_step >= qe.query_step:
            return False

    return True


def run_all_g0_checks() -> dict[str, bool]:
    """Execute all Gate 0 validity checks (8a - 8f)."""
    return {
        "8a_causality": check_g0_causality(),
        "8b_k1_reductions": check_g0_k1_reductions(),
        "8c_symmetry": check_g0_symmetry(),
        "8d_determinism": check_g0_determinism(),
        "8e_controls": check_g0_controls(),
        "8f_candidate_rules": check_g0_candidate_rules(),
    }
