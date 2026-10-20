"""Data generator for Tier A synthetic regime switching streams.

Generates interleaved topic streams, query schedules, and candidate sets
adhering strictly to Sections 4 and 5 of the experiment plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F


StreamMix = Literal["balanced", "skewed", "bursty"]


@dataclass
class WriteEvent:
    step: int
    topic_id: int
    k: torch.Tensor             # [D_k]
    v: torch.Tensor             # [D_v]
    k_wide: torch.Tensor        # [K * D_k] for B2
    k_heads: torch.Tensor       # [K, D_k] for B1
    pi_k: torch.Tensor          # [D_v, K] router prior


@dataclass
class QueryEvent:
    query_step: int
    write_step: int
    topic_id: int
    delay: int
    q: torch.Tensor             # [D_k]
    q_wide: torch.Tensor        # [K * D_k] for B2
    q_heads: torch.Tensor       # [K, D_k] for B1
    pi_q: torch.Tensor          # [D_v, K] query prior
    target_v: torch.Tensor      # [D_v]
    candidate_vs: torch.Tensor  # [50, D_v]
    target_idx: int = 0         # Randomized target position in candidate_vs [0..49]
    is_unwritten_control: bool = False


class TierAStreamGenerator:
    """Generates synthetic streams with M regime-switching linear maps."""
    def __init__(
        self,
        M: int = 4,
        D_k: int = 64,
        D_v: int = 64,
        K: int = 4,
        seed: int = 42,
        noise_std: float = 0.05,
        paraphrase_noise_std: float = 0.1,
    ):
        self.M = M
        self.D_k = D_k
        self.D_v = D_v
        self.K = K
        self.seed = seed
        self.noise_std = noise_std
        self.paraphrase_noise_std = paraphrase_noise_std

        # Initialize topic linear maps
        rng = torch.Generator().manual_seed(seed)
        self.topic_maps = [
            torch.randn(D_v, D_k, generator=rng) / math.sqrt(D_k)
            for _ in range(M)
        ]
        
        # Router projection matrices for pi_k and pi_q
        self.W_pi_k = torch.randn(D_k, K, generator=rng) / math.sqrt(D_k)
        self.W_pi_q = torch.randn(D_k, K, generator=rng) / math.sqrt(D_k)

        # Independent key projections for B1 heads
        self.B1_projections = [
            torch.randn(D_k, D_k, generator=rng) / math.sqrt(D_k)
            for _ in range(K)
        ]
        
        # Wide projection for B2
        self.B2_proj = torch.randn(K * D_k, D_k, generator=rng) / math.sqrt(D_k)

    def generate_stream(
        self,
        stream_length: int = 1200,
        mix: StreamMix = "skewed",
        delays: list[int] = [10, 100, 1000],
        stream_seed: int = 1001,
        queries_per_condition: int = 10,
    ) -> tuple[list[WriteEvent], list[QueryEvent]]:
        """Generate write stream and valid scheduled queries."""
        rng = np.random.default_rng(stream_seed)
        torch_rng = torch.Generator().manual_seed(stream_seed)

        # Topic distribution
        if mix == "balanced":
            topic_probs = [1.0 / self.M] * self.M
        elif mix == "skewed":
            # 70% majority, 20% second, 7% minority 1, 3% minority 2
            topic_probs = [0.70, 0.20, 0.07, 0.03]
        elif mix == "bursty":
            # Will be handled via stateful Markov transitions
            topic_probs = [0.70, 0.20, 0.07, 0.03]

        writes: list[WriteEvent] = []
        current_topic = 0
        burst_remaining = 0

        for t in range(stream_length):
            if mix == "bursty":
                if burst_remaining <= 0:
                    current_topic = int(rng.choice(self.M, p=topic_probs))
                    # majority topics have long runs, minority short bursts
                    burst_remaining = rng.integers(15, 30) if current_topic == 0 else rng.integers(2, 6)
                burst_remaining -= 1
                topic_id = current_topic
            else:
                topic_id = int(rng.choice(self.M, p=topic_probs))

            # Base key k ~ N(0, I), L2 normalized
            raw_k = torch.randn(self.D_k, generator=torch_rng)
            k = F.normalize(raw_k, dim=0)

            # Value v = W_topic * k + noise
            W_m = self.topic_maps[topic_id]
            noise = torch.randn(self.D_v, generator=torch_rng) * self.noise_std
            v = torch.matmul(W_m, k) + noise

            # Multi-head keys for B1
            k_heads = torch.stack([
                F.normalize(torch.matmul(P, k), dim=0)
                for P in self.B1_projections
            ])

            # Wide key for B2
            raw_wide = torch.matmul(self.B2_proj, k)
            k_wide = F.normalize(raw_wide, dim=0)

            # Prior pi_k: [D_v, K] broadcasted from k * W_pi_k
            pi_k_vec = torch.matmul(k, self.W_pi_k)  # [K]
            pi_k = pi_k_vec.unsqueeze(0).expand(self.D_v, self.K).clone()

            writes.append(
                WriteEvent(
                    step=t,
                    topic_id=topic_id,
                    k=k,
                    v=v,
                    k_wide=k_wide,
                    k_heads=k_heads,
                    pi_k=pi_k,
                )
            )

        # Generate Query events satisfying candidate and delay feasibility rules (Section 4 & 8f)
        queries: list[QueryEvent] = []

        # Find candidates for each delay and topic
        for delay in delays:
            for topic_id in range(self.M):
                # Eligible writes: must be written such that write_step + delay < stream_length
                # and write_step >= 50 so that at least 50 historical candidates exist before query_step
                eligible_writes = [
                    w for w in writes
                    if w.topic_id == topic_id
                    and (w.step + delay) < stream_length
                    and (w.step + delay) >= 50
                ]
                
                if not eligible_writes:
                    continue

                # Sample up to queries_per_condition
                chosen_writes = rng.choice(
                    eligible_writes,
                    size=min(len(eligible_writes), queries_per_condition),
                    replace=False,
                )

                for w in chosen_writes:
                    query_step = w.step + delay
                    
                    # Paraphrased query key
                    para_noise = torch.randn(self.D_k, generator=torch_rng) * self.paraphrase_noise_std
                    q = F.normalize(w.k + para_noise, dim=0)

                    # B1 query keys
                    q_heads = torch.stack([
                        F.normalize(torch.matmul(P, q), dim=0)
                        for P in self.B1_projections
                    ])

                    # B2 wide query key
                    raw_q_wide = torch.matmul(self.B2_proj, q)
                    q_wide = F.normalize(raw_q_wide, dim=0)

                    # Router prior pi_q
                    pi_q_vec = torch.matmul(q, self.W_pi_q)
                    pi_q = pi_q_vec.unsqueeze(0).expand(self.D_v, self.K).clone()

                    # Candidate set of 50 chunks: target placed at randomized index in [0..49]
                    past_candidates = [
                        pw.v for pw in writes
                        if pw.step < query_step and pw.step != w.step
                    ]
                    
                    distractor_indices = rng.choice(
                        len(past_candidates),
                        size=min(49, len(past_candidates)),
                        replace=False,
                    )
                    distractors = [past_candidates[i] for i in distractor_indices]
                    
                    # Ensure no exact-collision distractors with w.v
                    target_norm = F.normalize(w.v, dim=0)
                    filtered_distractors = []
                    for d in distractors:
                        d_norm = F.normalize(d, dim=0)
                        if torch.dot(target_norm, d_norm).item() < 0.9999:
                            filtered_distractors.append(d)
                        else:
                            # Re-sample a non-colliding distractor if possible
                            alt_cands = [pc for pc in past_candidates if pc not in distractors and torch.dot(target_norm, F.normalize(pc, dim=0)).item() < 0.9999]
                            if alt_cands:
                                filtered_distractors.append(alt_cands[0])
                            else:
                                filtered_distractors.append(d)
                    
                    target_idx = int(rng.integers(0, 50))
                    all_cands = filtered_distractors[:target_idx] + [w.v] + filtered_distractors[target_idx:]
                    candidate_vs = torch.stack(all_cands)  # [50, D_v]

                    queries.append(
                        QueryEvent(
                            query_step=query_step,
                            write_step=w.step,
                            topic_id=topic_id,
                            delay=delay,
                            q=q,
                            q_wide=q_wide,
                            q_heads=q_heads,
                            pi_q=pi_q,
                            target_v=w.v,
                            candidate_vs=candidate_vs,
                            target_idx=target_idx,
                        )
                    )

        # Sort queries by query_step
        queries.sort(key=lambda qe: qe.query_step)
        return writes, queries
