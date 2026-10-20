"""SwiLA recurrent ops and reference implementations.

Pinned to commit 4b43bb5c9d3e76659e5f07740c15b046cddb581b
(github.com/lindermanlab/switching-linear-attention).
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def apply_log_decay(
    state: torch.Tensor,
    log_decay_i: torch.Tensor,
) -> torch.Tensor:
    """Apply multiplicative ``exp(log_decay_i)`` without expanding over D_v."""
    expected = (state.shape[0], state.shape[1], state.shape[-1])
    if tuple(log_decay_i.shape) != expected:
        raise ValueError(f"log_decay_i must have shape {expected}, got {tuple(log_decay_i.shape)}")
    decay_i = log_decay_i.exp()
    return state * decay_i[:, :, None, None, :]


def log_temporal_query_mix(
    log_prior: torch.Tensor,
    log_reset: torch.Tensor,
    gate_logit: torch.Tensor,
) -> torch.Tensor:
    """Log of the affine reset mixture, in any common log coordinate system."""
    zero = torch.zeros_like(gate_logit)
    carry = log_prior - torch.maximum(gate_logit, zero)
    reset = log_reset + torch.minimum(gate_logit, zero)
    maximum = torch.maximum(carry, reset)
    difference = carry - reset
    same_infinity = torch.isnan(difference)
    safe_difference = torch.where(same_infinity, torch.zeros_like(difference), difference)
    correction = (
        torch.log1p(torch.exp(-safe_difference.abs()))
        - torch.log1p(torch.exp(-gate_logit.abs()))
    )
    return torch.where(same_infinity, maximum, maximum + correction)


class ExactInitialTemporalPriorMix(torch.autograd.Function):
    """Stable exact reset mix with an analytic zero-prior backward."""

    @staticmethod
    def forward(
        ctx,
        initial: torch.Tensor,
        gate_logit: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        positive_initial = initial > 0
        safe_initial = torch.where(
            positive_initial,
            initial,
            torch.ones_like(initial),
        )
        log_initial = torch.where(
            positive_initial,
            torch.log(safe_initial),
            torch.full_like(initial, -float("inf")),
        )
        log_reset = F.log_softmax(router_logits, dim=-1)
        log_gate = F.logsigmoid(gate_logit)
        log_one_minus_gate = F.logsigmoid(-gate_logit)
        log_probability = torch.logaddexp(
            log_one_minus_gate + log_initial,
            log_gate + log_reset,
        )
        ctx.save_for_backward(
            initial,
            log_initial,
            log_reset,
            log_probability,
            log_gate,
            log_one_minus_gate,
        )
        return torch.exp(log_probability), log_probability

    @staticmethod
    def backward(
        ctx,
        d_probability: torch.Tensor | None,
        d_log_probability: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            initial,
            log_initial,
            log_reset,
            log_probability,
            log_gate,
            log_one_minus_gate,
        ) = ctx.saved_tensors
        if d_probability is None:
            d_probability = torch.zeros_like(log_probability)
        if d_log_probability is None:
            d_log_probability = torch.zeros_like(log_probability)

        reset = torch.exp(log_reset)
        gate = torch.exp(log_gate)
        one_minus_gate = torch.exp(log_one_minus_gate)
        reset_branch_weight = torch.exp(log_gate + log_reset - log_probability)
        initial_branch_weight = torch.exp(log_one_minus_gate + log_initial - log_probability)

        d_initial = d_probability * one_minus_gate

        nonzero_log_gradient = d_log_probability != 0
        safe_abs_log_gradient = torch.where(
            nonzero_log_gradient,
            d_log_probability.abs(),
            torch.ones_like(d_log_probability),
        )
        log_abs_initial_gradient = (
            torch.log(safe_abs_log_gradient.double())
            + (log_one_minus_gate - log_probability).double()
        )
        abs_initial_gradient = torch.exp(log_abs_initial_gradient).to(initial.dtype)
        signed_initial_gradient = torch.where(
            d_log_probability < 0,
            -abs_initial_gradient,
            abs_initial_gradient,
        )
        d_initial = d_initial + torch.where(
            nonzero_log_gradient,
            signed_initial_gradient,
            torch.zeros_like(signed_initial_gradient),
        )

        d_log_reset = (
            d_log_probability * reset_branch_weight
            + d_probability * gate * reset
        )
        d_router_logits = d_log_reset - reset * d_log_reset.sum(
            dim=-1,
            keepdim=True,
        )

        d_gate_log = d_log_probability * (
            reset_branch_weight * one_minus_gate
            - initial_branch_weight * gate
        )
        d_gate_probability = d_probability * (
            gate * one_minus_gate * (reset - initial)
        )
        d_gate_logit = (d_gate_log + d_gate_probability).sum(
            dim=-1,
            keepdim=True,
        )
        return d_initial, d_gate_logit, d_router_logits


def mix_exact_initial_temporal_prior(
    initial: torch.Tensor,
    gate_logit: torch.Tensor,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ExactInitialTemporalPriorMix.apply(
        initial,
        gate_logit,
        router_logits,
    )


def swila_recurrence_non_temporal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    error_coef: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Non-temporal SwiLA recurrence.
    
    If error_coef == 0.0, this implements the R1 fixed random routing baseline.
    """
    output_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = beta.to(torch.float32)
    pi_q = pi_q.to(torch.float32)
    pi_k = pi_k.to(torch.float32)
    initial_state = initial_state.to(torch.float32) if initial_state is not None else None

    B, T, H, D_k = q.shape
    D_v = v.shape[-1]
    K = pi_q.shape[-1]
    q = q.transpose(1, 2).contiguous() * scale
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    pi_q = pi_q.transpose(1, 2).contiguous()
    pi_k = pi_k.transpose(1, 2).contiguous()

    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = q.new_zeros(B, H, D_v, K, D_k)

    o = q.new_zeros(B, H, T, D_v)
    r_seq = pi_k.new_zeros(B, H, T, D_v, K)

    for i in range(T):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        beta_i = beta[:, :, i]
        pi_q_i = pi_q[:, :, i]
        pi_k_i = pi_k[:, :, i]

        pred = torch.einsum('bhd,bhvkd->bhvk', k_i, S)
        u = v_i.unsqueeze(-1) - pred
        
        if error_coef > 0.0:
            r_i = F.softmax(pi_k_i - error_coef * u.square(), dim=-1)
        else:
            r_i = F.softmax(pi_k_i, dim=-1)

        S = S + beta_i.unsqueeze(-1) * torch.einsum(
            'bhd,bhvk->bhvkd', k_i, r_i * u
        )

        qS = torch.einsum('bhd,bhvkd->bhvk', q_i, S)
        pi_q_star = F.softmax(pi_q_i, dim=-1)
        o_i = (pi_q_star * qS).sum(dim=-1)

        o[:, :, i] = o_i
        r_seq[:, :, i] = r_i

    o = o.transpose(1, 2).contiguous()
    r_seq = r_seq.transpose(1, 2).contiguous()
    final_state = S if output_final_state else None
    return o.to(output_dtype), final_state, r_seq


def swila_recurrence_non_temporal_gated(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    log_decay: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    error_coef: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    output_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = beta.to(torch.float32)
    pi_q = pi_q.to(torch.float32)
    pi_k = pi_k.to(torch.float32)
    log_decay = log_decay.to(torch.float32)
    initial_state = initial_state.to(torch.float32) if initial_state is not None else None

    B, T, H, D_k = q.shape
    D_v = v.shape[-1]
    K = pi_q.shape[-1]
    q = q.transpose(1, 2).contiguous() * scale
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    pi_q = pi_q.transpose(1, 2).contiguous()
    pi_k = pi_k.transpose(1, 2).contiguous()
    log_decay = log_decay.transpose(1, 2).contiguous()

    if initial_state is not None:
        S = initial_state.clone()
    else:
        S = q.new_zeros(B, H, D_v, K, D_k)

    o = q.new_zeros(B, H, T, D_v)
    r_seq = pi_k.new_zeros(B, H, T, D_v, K)

    for i in range(T):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        beta_i = beta[:, :, i]
        pi_q_i = pi_q[:, :, i]
        pi_k_i = pi_k[:, :, i]
        log_decay_i = log_decay[:, :, i]

        prediction_state = apply_log_decay(S, log_decay_i)
        pred = torch.einsum('bhd,bhvkd->bhvk', k_i, prediction_state)
        u = v_i.unsqueeze(-1) - pred
        
        if error_coef > 0.0:
            r_i = F.softmax(pi_k_i - error_coef * u.square(), dim=-1)
        else:
            r_i = F.softmax(pi_k_i, dim=-1)

        S_prior = prediction_state
        S = S_prior + beta_i.unsqueeze(-1) * torch.einsum(
            'bhd,bhvk->bhvkd', k_i, r_i * u
        )

        qS = torch.einsum('bhd,bhvkd->bhvk', q_i, S)
        pi_q_star = F.softmax(pi_q_i, dim=-1)
        o_i = (pi_q_star * qS).sum(dim=-1)

        o[:, :, i] = o_i
        r_seq[:, :, i] = r_i

    o = o.transpose(1, 2).contiguous()
    r_seq = r_seq.transpose(1, 2).contiguous()
    final_state = S if output_final_state else None
    return o.to(output_dtype), final_state, r_seq


def swila_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    pi_q: torch.Tensor,
    pi_k: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    log_decay: torch.Tensor | None = None,
    error_coef: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Unified recurrent runner for S1 (SwiLA) and R1 (Fixed Random Routing)."""
    D_k = q.shape[-1]
    if log_decay is not None and tuple(log_decay.shape) != tuple(q.shape):
        raise ValueError(f"log_decay must have shape {tuple(q.shape)}, got {tuple(log_decay.shape)}")
    if scale is None:
        scale = D_k ** -0.5

    if log_decay is None:
        return swila_recurrence_non_temporal(
            q=q,
            k=k,
            v=v,
            beta=beta,
            pi_q=pi_q,
            pi_k=pi_k,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            error_coef=error_coef,
        )
    return swila_recurrence_non_temporal_gated(
        q=q,
        k=k,
        v=v,
        beta=beta,
        pi_q=pi_q,
        pi_k=pi_k,
        log_decay=log_decay,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        error_coef=error_coef,
    )


def query_state_snapshot(
    S: torch.Tensor,
    q: torch.Tensor,
    pi_q: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Readout v_hat from a frozen state snapshot S without mutating S.
    
    Args:
        S: State tensor [B, H, D_v, K, D_k]
        q: Single query tensor [B, 1, H, D_k]
        pi_q: Query router prior [B, 1, H, D_v, K]
        scale: Scale factor (default 1.0 or D_k^-0.5)
    Returns:
        v_hat: [B, D_v] readout vector
    """
    D_k = q.shape[-1]
    if scale is None:
        scale = 1.0  # as specified in section 4 when q is L2 normalized
    
    q_scaled = q.transpose(1, 2).contiguous() * scale  # [B, H, 1, D_k]
    q_i = q_scaled[:, :, 0]                           # [B, H, D_k]
    pi_q_i = pi_q.transpose(1, 2)[:, :, 0]            # [B, H, D_v, K]
    
    qS = torch.einsum('bhd,bhvkd->bhvk', q_i, S)
    pi_q_star = F.softmax(pi_q_i, dim=-1)
    o_i = (pi_q_star * qS).sum(dim=-1)                 # [B, H, D_v]
    
    # For H=1, squeeze head
    return o_i.squeeze(1)                              # [B, D_v]
