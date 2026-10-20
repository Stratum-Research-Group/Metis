"""Stage 1 Execution Runner for SwiLA-style streaming session state experiment.

Implements:
1. Gate 0 (G0) validity checks verification.
2. Tier A experimental sweep across streams, seeds, conditions and arms.
3. Paired bootstrap statistical decision gating (G1).
4. Results emission into structured JSON and comprehensive report.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn.functional as F

from experiments.swila_session_state.validity_checks import run_all_g0_checks
from experiments.swila_session_state.data_generator import (
    TierAStreamGenerator,
    WriteEvent,
    QueryEvent,
    StreamMix,
)
from experiments.swila_session_state.swila_core import (
    swila_recurrence,
    query_state_snapshot,
)
from experiments.swila_session_state.baselines import (
    B1NarrowHeads,
    B2WideState,
    B3MeanPooled,
    B4RecencyWindow,
    B5ExactKVStore,
)
from experiments.swila_session_state.evaluator import (
    paired_bootstrap_ci,
    classify_decision,
)


def score_recall_at_1(v_hat: torch.Tensor, candidate_vs: torch.Tensor, target_idx: int = 0) -> float:
    """Score whether nearest candidate (cosine similarity) to v_hat is the true chunk at target_idx."""
    v_hat_1d = v_hat.squeeze()
    v_hat_norm = F.normalize(v_hat_1d, dim=0)
    cands_norm = F.normalize(candidate_vs, dim=1)
    cos_sims = torch.matmul(cands_norm, v_hat_norm)
    best_cand_idx = int(torch.argmax(cos_sims).item())
    return 1.0 if best_cand_idx == target_idx else 0.0


def run_single_stream_evaluation(
    writes: list[WriteEvent],
    queries: list[QueryEvent],
    D_k: int = 64,
    D_v: int = 64,
    K: int = 4,
    beta: float = 0.5,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    """Run online streaming simulation for all arms over a single stream."""
    stream_length = len(writes)

    # Initialize models
    # S1: SwiLA (with small random initial state for symmetry breaking)
    # State shape: [1, 1, D_v, K, D_k]
    S1_state = torch.randn(1, 1, D_v, K, D_k, dtype=torch.float32) * 0.01

    # R1: Fixed random routing (coef = 0.0)
    R1_state = torch.randn(1, 1, D_v, K, D_k, dtype=torch.float32) * 0.01

    # Baselines
    b1 = B1NarrowHeads(K=K, D_k=D_k, D_v=D_v, beta=beta)
    b2 = B2WideState(D_k_prime=K * D_k, D_v=D_v, beta=beta)
    b3 = B3MeanPooled(D_v=D_v)
    total_floats = D_v * K * D_k
    b4 = B4RecencyWindow(total_state_floats=total_floats, D_k=D_k, D_v=D_v)
    b5 = B5ExactKVStore(D_v=D_v)

    # Group queries by step for online execution
    queries_by_step: dict[int, list[QueryEvent]] = {}
    for qe in queries:
        queries_by_step.setdefault(qe.query_step, []).append(qe)

    scores: dict[str, list[dict[str, Any]]] = {
        "S1": [],
        "R1": [],
        "B1": [],
        "B2": [],
        "B3": [],
        "B4": [],
        "B5": [],
        "O1": [],
        "C0": [],
    }

    # Tracking for mechanism diagnostics & collisions
    o1_collisions = 0
    s1_usages: list[torch.Tensor] = []
    s1_argmax_experts: list[int] = []
    write_topics: list[int] = []

    # Step-by-step stream execution
    for t, w in enumerate(writes):
        # 1. Answer any queries scheduled for step t BEFORE write t updates the state
        if t in queries_by_step:
            for qe in queries_by_step[t]:
                # Prepare query tensors
                q_th = qe.q.view(1, 1, 1, D_k)
                pi_q_th = qe.pi_q.view(1, 1, 1, D_v, K)

                # S1 Readout
                v_hat_s1 = query_state_snapshot(S1_state, q_th, pi_q_th)

                # R1 Readout
                v_hat_r1 = query_state_snapshot(R1_state, q_th, pi_q_th)

                # B1 Readout
                v_hat_b1 = b1.query(qe.q_heads)

                # B2 Readout
                v_hat_b2 = b2.query(qe.q_wide)

                # B3 Readout
                v_hat_b3 = b3.query()

                # B4 Readout
                v_hat_b4 = b4.query(qe.q)

                # B5 Readout
                v_hat_b5 = b5.query(qe.q)

                # O1 Oracle
                v_hat_o1 = qe.target_v.clone()

                # Collision check for O1: distractor cosine similarity with target_v
                target_norm = F.normalize(qe.target_v, dim=0)
                cand_norms = F.normalize(qe.candidate_vs, dim=1)
                sims = torch.matmul(cand_norms, target_norm)
                # Count if any non-target candidate has cosine > 0.999
                for c_idx, s in enumerate(sims):
                    if c_idx != qe.target_idx and s.item() > 0.999:
                        o1_collisions += 1

                # Score each arm with position-agnostic target_idx
                rec_meta = {
                    "topic_id": qe.topic_id,
                    "delay": qe.delay,
                    "write_step": qe.write_step,
                    "query_step": qe.query_step,
                }

                s1_acc = score_recall_at_1(v_hat_s1, qe.candidate_vs, target_idx=qe.target_idx)
                r1_acc = score_recall_at_1(v_hat_r1, qe.candidate_vs, target_idx=qe.target_idx)
                b1_acc = score_recall_at_1(v_hat_b1, qe.candidate_vs, target_idx=qe.target_idx)
                b2_acc = score_recall_at_1(v_hat_b2, qe.candidate_vs, target_idx=qe.target_idx)
                b3_acc = score_recall_at_1(v_hat_b3, qe.candidate_vs, target_idx=qe.target_idx)
                b4_acc = score_recall_at_1(v_hat_b4, qe.candidate_vs, target_idx=qe.target_idx)
                b5_acc = score_recall_at_1(v_hat_b5, qe.candidate_vs, target_idx=qe.target_idx)
                o1_acc = score_recall_at_1(v_hat_o1, qe.candidate_vs, target_idx=qe.target_idx)
                c0_acc = 1.0 if np.random.randint(0, 50) == qe.target_idx else 0.0

                scores["S1"].append({**rec_meta, "acc": s1_acc})
                scores["R1"].append({**rec_meta, "acc": r1_acc})
                scores["B1"].append({**rec_meta, "acc": b1_acc})
                scores["B2"].append({**rec_meta, "acc": b2_acc})
                scores["B3"].append({**rec_meta, "acc": b3_acc})
                scores["B4"].append({**rec_meta, "acc": b4_acc})
                scores["B5"].append({**rec_meta, "acc": b5_acc})
                scores["O1"].append({**rec_meta, "acc": o1_acc})
                scores["C0"].append({**rec_meta, "acc": c0_acc})

        # 2. Update states with write t
        # S1 update
        q_dummy = w.k.view(1, 1, 1, D_k)
        k_step = w.k.view(1, 1, 1, D_k)
        v_step = w.v.view(1, 1, 1, D_v)
        beta_step = torch.full((1, 1, 1, D_v, K), beta)
        pi_q_step = w.pi_k.view(1, 1, 1, D_v, K)
        pi_k_step = w.pi_k.view(1, 1, 1, D_v, K)

        _, S1_state, r_step_s1 = swila_recurrence(
            q=q_dummy,
            k=k_step,
            v=v_step,
            beta=beta_step,
            pi_q=pi_q_step,
            pi_k=pi_k_step,
            initial_state=S1_state,
            output_final_state=True,
            error_coef=0.5,
        )

        # Track usage across D_v
        # r_step_s1: [1, 1, 1, D_v, K]
        mean_r_step = r_step_s1[0, 0, 0].mean(dim=0)  # [K]
        s1_usages.append(mean_r_step)
        s1_argmax_experts.append(int(mean_r_step.argmax().item()))
        write_topics.append(w.topic_id)

        # R1 update (error_coef = 0.0)
        _, R1_state, _ = swila_recurrence(
            q=q_dummy,
            k=k_step,
            v=v_step,
            beta=beta_step,
            pi_q=pi_q_step,
            pi_k=pi_k_step,
            initial_state=R1_state,
            output_final_state=True,
            error_coef=0.0,
        )

        # Baselines updates
        b1.write(w.k_heads, w.v)
        b2.write(w.k_wide, w.v)
        b3.write(w.v)
        b4.write(w.k, w.v)
        b5.write(w.k, w.v)

    # Compute mechanism diagnostics
    stacked_usage = torch.stack(s1_usages)  # [T, K]
    overall_mean_usage = stacked_usage.mean(dim=0).numpy()  # [K]
    entropy = float(-np.sum(overall_mean_usage * np.log(overall_mean_usage + 1e-12)))
    effective_experts = float(math.exp(entropy))
    dead_expert_fraction = float(np.mean(overall_mean_usage < 0.01))

    from sklearn.metrics import normalized_mutual_info_score
    topic_nmi = float(normalized_mutual_info_score(write_topics, s1_argmax_experts))

    diagnostics = {
        "effective_experts": effective_experts,
        "dead_expert_fraction": dead_expert_fraction,
        "topic_nmi": topic_nmi,
        "o1_collisions": o1_collisions,
    }

    return scores, diagnostics


def aggregate_stream_metrics(scores: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    """Compute primary metric (minority recall@1 at delays 100 and 1000) and secondary metrics."""
    res = {}
    for arm, arm_scores in scores.items():
        if not arm_scores:
            continue
        
        # Primary metric: Macro over minority topics (topics 2 and 3) at delays 100 and 1000
        minority_queries = [
            s["acc"] for s in arm_scores
            if s["topic_id"] in (2, 3) and s["delay"] in (100, 1000)
        ]
        minority_acc = float(np.mean(minority_queries)) if minority_queries else 0.0

        # Majority topic metric (topic 0)
        majority_queries = [
            s["acc"] for s in arm_scores
            if s["topic_id"] == 0
        ]
        majority_acc = float(np.mean(majority_queries)) if majority_queries else 0.0

        # Overall recall
        overall_acc = float(np.mean([s["acc"] for s in arm_scores]))

        # Recall by delay
        delay_accs = {}
        for d in (10, 100, 1000):
            d_q = [s["acc"] for s in arm_scores if s["delay"] == d]
            delay_accs[f"delay_{d}"] = float(np.mean(d_q)) if d_q else 0.0

        res[arm] = {
            "primary_minority_recall": minority_acc,
            "majority_recall": majority_acc,
            "overall_recall": overall_acc,
            **delay_accs,
        }
    return res


def run_stage1_experiment(
    n_seeds: int = 20,
    stream_length: int = 1200,
    mixes: list[StreamMix] = ["skewed", "balanced", "bursty"],
    output_dir: str = "experiments/swila_session_state/results",
) -> dict[str, Any]:
    """Execute complete Stage 1 experiment."""
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    print("=" * 80)
    print("METIS EXPERIMENT: SWILA-STYLE STREAMING SESSION STATE (STAGE 1)")
    print("=" * 80)

    # 1. Run G0 Validity Checks
    print("\n[Stage 1.1] Executing Gate 0 (G0) Pre-Registration Validity Checks...")
    g0_results = run_all_g0_checks()
    for name, passed in g0_results.items():
        print(f"  - {name}: {'PASS' if passed else 'FAIL'}")

    if not all(g0_results.values()):
        print("\nFATAL: Gate 0 validity check failed! Stopping experiment per protocol Section 10.")
        return {"g0_passed": False, "g0_results": g0_results}

    print("\n[Gate 0 Passed] All causality, reduction, symmetry, determinism, and candidate invariants verified.")

    # 2. Pilot Run (1 seed)
    print("\n[Stage 1.2] Executing Single-Seed Pilot Run (seed 42)...")
    gen_pilot = TierAStreamGenerator(seed=42)
    pilot_writes, pilot_queries = gen_pilot.generate_stream(
        stream_length=stream_length, mix="skewed", stream_seed=42
    )
    pilot_scores, pilot_diag = run_single_stream_evaluation(pilot_writes, pilot_queries)
    pilot_metrics = aggregate_stream_metrics(pilot_scores)
    print("Pilot Primary Metric (Minority Recall@1 at Delay 100 & 1000):")
    for arm, m in pilot_metrics.items():
        print(f"  {arm:>4}: minority={m['primary_minority_recall']:.3f} | majority={m['majority_recall']:.3f} | overall={m['overall_recall']:.3f}")
    print(f"Pilot Diagnostics: eff_experts={pilot_diag['effective_experts']:.2f}, NMI={pilot_diag['topic_nmi']:.3f}, collisions={pilot_diag['o1_collisions']}")

    # 3. Full Tier A Sweep
    print(f"\n[Stage 1.3] Executing Tier A Sweep across {len(mixes)} stream mixes x {n_seeds} seeds...")
    sweep_results: dict[str, list[dict[str, Any]]] = {mix: [] for mix in mixes}
    sweep_diagnostics: dict[str, list[dict[str, float]]] = {mix: [] for mix in mixes}

    total_writes_executed = 0
    total_queries_executed = 0
    t_sweep_start = time.time()

    for mix in mixes:
        print(f"\n  Running Stream Mix: {mix.upper()} (seeds 1001..{1000 + n_seeds})...")
        for s_idx in range(n_seeds):
            seed = 1001 + s_idx
            gen = TierAStreamGenerator(seed=seed)
            writes, queries = gen.generate_stream(
                stream_length=stream_length, mix=mix, stream_seed=seed
            )
            raw_scores, raw_diag = run_single_stream_evaluation(writes, queries)
            metrics = aggregate_stream_metrics(raw_scores)
            sweep_results[mix].append(metrics)
            sweep_diagnostics[mix].append(raw_diag)
            total_writes_executed += len(writes)
            total_queries_executed += len(queries)
            print(".", end="", flush=True)
        print(" Done.")

    sweep_duration = time.time() - t_sweep_start
    per_write_us = (sweep_duration / total_writes_executed) * 1e6 if total_writes_executed else 0.0
    per_query_us = (sweep_duration / total_queries_executed) * 1e6 if total_queries_executed else 0.0

    # 4. Statistical Evaluation & Gate 1 Gating
    print("\n[Stage 1.4] Statistical Evaluation & Gate 1 Decision Gating...")
    gate_evaluations: dict[str, Any] = {}

    import resource
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for mix, seed_metrics in sweep_results.items():
        seed_diags = sweep_diagnostics[mix]

        # Arrays of primary metric across seeds
        arm_primary_arrays = {
            arm: [float(m[arm]["primary_minority_recall"]) for m in seed_metrics]
            for arm in ["S1", "R1", "B1", "B2", "B3", "B4", "B5", "O1", "C0"]
        }

        s1_primary = np.array(arm_primary_arrays["S1"])
        b1_primary = np.array(arm_primary_arrays["B1"])
        b2_primary = np.array(arm_primary_arrays["B2"])
        r1_primary = np.array(arm_primary_arrays["R1"])
        b4_primary = np.array(arm_primary_arrays["B4"])

        s1_majority = np.array([m["S1"]["majority_recall"] for m in seed_metrics])
        r1_majority = np.array([m["R1"]["majority_recall"] for m in seed_metrics])

        # Paired comparisons against S1
        comp_b1 = paired_bootstrap_ci(s1_primary - b1_primary)
        comp_b2 = paired_bootstrap_ci(s1_primary - b2_primary)
        comp_r1 = paired_bootstrap_ci(s1_primary - r1_primary)
        comp_b4 = paired_bootstrap_ci(s1_primary - b4_primary)

        # Majority drop
        majority_drop = float(np.mean(r1_majority - s1_majority))

        gate_evaluations[mix] = {
            "S1_vs_B1": {
                "est": comp_b1[0], "lo": comp_b1[1], "hi": comp_b1[2],
                "decision": classify_decision(*comp_b1)
            },
            "S1_vs_B2": {
                "est": comp_b2[0], "lo": comp_b2[1], "hi": comp_b2[2],
                "decision": classify_decision(*comp_b2)
            },
            "S1_vs_R1": {
                "est": comp_r1[0], "lo": comp_r1[1], "hi": comp_r1[2],
                "decision": classify_decision(*comp_r1)
            },
            "S1_vs_B4": {
                "est": comp_b4[0], "lo": comp_b4[1], "hi": comp_b4[2],
                "decision": classify_decision(*comp_b4)
            },
            "majority_drop": majority_drop,
            "mean_primary": {
                arm: float(np.mean(arm_primary_arrays[arm]))
                for arm in arm_primary_arrays
            },
            "per_seed_primary": arm_primary_arrays,
            "mechanism_diagnostics": {
                "effective_experts": float(np.mean([d["effective_experts"] for d in seed_diags])),
                "dead_expert_fraction": float(np.mean([d["dead_expert_fraction"] for d in seed_diags])),
                "topic_nmi": float(np.mean([d["topic_nmi"] for d in seed_diags])),
                "o1_collisions_total": int(np.sum([d["o1_collisions"] for d in seed_diags])),
            },
            "footprint": {
                "state_only_bytes": 65536,
                "candidate_table_bytes": stream_length * 64 * 4,
                "total_resident_bytes": 65536 + (stream_length * 64 * 4),
            }
        }

    total_time = time.time() - start_time

    # Construct complete results bundle
    bundle = {
        "metadata": {
            "pinned_metis_commit": "509d70cceb770b8b227ec99ec1ba0d2de7c8cd67",
            "pinned_swila_commit": "4b43bb5c9d3e76659e5f07740c15b046cddb581b",
            "date": "2026-10-20",
            "runtime_seconds": round(total_time, 2),
            "n_seeds": n_seeds,
            "stream_length": stream_length,
            "mixes": mixes,
            "hyperparameters": {
                "K": 4,
                "beta": 0.5,
                "tau": 1.0,
                "D_k": 64,
                "D_v": 64,
                "candidates_N": 50,
                "symmetry_breaking": "gaussian_init_std_0.01",
            },
            "profiling": {
                "per_write_us": round(per_write_us, 2),
                "per_query_us": round(per_query_us, 2),
                "peak_ram_mb": round(peak_rss_kb / 1024.0, 2),
            }
        },
        "g0_checks": g0_results,
        "pilot_metrics": pilot_metrics,
        "pilot_diagnostics": pilot_diag,
        "gate_evaluations": gate_evaluations,
    }

    # Write results.json
    results_path = os.path.join(output_dir, "tier_a_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print(f"\n[Completed in {total_time:.1f}s] Results saved to {results_path}")
    return bundle


if __name__ == "__main__":
    run_stage1_experiment()

