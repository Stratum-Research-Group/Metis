"""Analysis script for Phase 2 (Real Extractor Noise) Pilot.

Compares:
- Phase 2 Real Extractor Arms (B, C, D, F) vs Arm A (Text Baseline)
- Phase 2 Real Extractor Arms vs Phase 1 Synthetic 30% Corrupted Arms (matched on conf_s001-conf_s040)
- Detailed extraction error categorization and fidelity breakdown.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def bootstrap_story_paired_diff(
    story_scores_arm1: Dict[str, float],
    story_scores_arm2: Dict[str, float],
    num_resamples: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes mean paired difference (Arm1 - Arm2) and 95% bootstrap CI at STORY level."""
    story_ids = sorted(list(set(story_scores_arm1.keys()).intersection(story_scores_arm2.keys())))
    n = len(story_ids)
    if n == 0:
        return 0.0, 0.0, 0.0

    paired_diffs = [story_scores_arm1[sid] - story_scores_arm2[sid] for sid in story_ids]
    mean_diff = sum(paired_diffs) / n

    rng = random.Random(seed)
    boot_means: List[float] = []
    for _ in range(num_resamples):
        sample = [rng.choice(paired_diffs) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    ci_lower = boot_means[int(0.025 * num_resamples)]
    ci_upper = boot_means[int(0.975 * num_resamples)]
    return mean_diff, ci_lower, ci_upper


def classify_comparison(ci_lower: float, ci_upper: float, delta: float = 0.03) -> str:
    """Classifies outcome per Section 8:
    better: CI entirely above +delta
    worse: CI entirely below -delta
    equivalent: CI entirely inside [-delta, +delta]
    inconclusive: anything else
    """
    if ci_lower > delta:
        return "better"
    if ci_upper < -delta:
        return "worse"
    if ci_lower >= -delta and ci_upper <= delta:
        return "equivalent"
    return "inconclusive"


def analyze():
    base_dir = Path(__file__).parent
    phase2_res_path = base_dir / "phase2_results.jsonl"
    phase2_ext_path = base_dir / "phase2_extractions.jsonl"
    conf_res_path = base_dir / "confirmatory_results.jsonl"

    # 1. Load Phase 2 extractions
    extractions = []
    with open(phase2_ext_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                extractions.append(json.loads(line))

    fidelities = [e["metrics"]["overall_fidelity"] for e in extractions]
    perfect_count = sum(1 for f in fidelities if f >= 0.999)
    missing_ev_total = sum(len(e["metrics"].get("missing_events", [])) for e in extractions)
    hallucinated_ev_total = sum(len(e["metrics"].get("hallucinated_events", [])) for e in extractions)
    time_mismatches_total = sum(len(e["metrics"].get("timestamp_mismatches", [])) for e in extractions)
    dur_mismatches_total = sum(len(e["metrics"].get("duration_mismatches", [])) for e in extractions)

    # 2. Load Phase 2 results
    p2_story_arm_q: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(dict)  # (story_id, arm) -> q_type -> is_correct
    p2_story_arm_acc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    p2_arm_qtype_scores: Dict[Tuple[str, str], List[int]] = defaultdict(list)  # (arm, q_type) -> [0/1]

    pilot_story_ids = set()
    with open(phase2_res_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["story_id"]
            pilot_story_ids.add(sid)
            arm = rec["arm"]
            qtype = rec["qtype"]
            corr = 1 if rec["correct"] else 0
            p2_story_arm_acc[(sid, arm)].append(corr)
            p2_arm_qtype_scores[(arm, qtype)].append(corr)

    # 3. Load Confirmatory Phase 1 results for the matched 40 stories
    p1_matched_arm_acc: Dict[Tuple[str, float], Dict[str, float]] = defaultdict(dict)  # (arm, corrupt_rate) -> sid -> mean_acc
    p1_matched_arm_qtype: Dict[Tuple[str, float, str], List[int]] = defaultdict(list)  # (arm, corrupt_rate, qtype) -> [0/1]

    with open(conf_res_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["story_id"]
            if sid not in pilot_story_ids:
                continue
            arm = rec["arm"]
            crate = rec["level"]
            qtype = rec["qtype"]
            corr = 1 if rec["correct"] else 0
            if sid not in p1_matched_arm_acc[(arm, crate)]:
                p1_matched_arm_acc[(arm, crate)][sid] = []
            p1_matched_arm_acc[(arm, crate)][sid].append(corr)
            p1_matched_arm_qtype[(arm, crate, qtype)].append(corr)

    # Average story scores for Phase 1
    p1_story_means: Dict[Tuple[str, float], Dict[str, float]] = {}
    for key, sid_map in p1_matched_arm_acc.items():
        p1_story_means[key] = {sid: sum(vals) / len(vals) for sid, vals in sid_map.items()}

    # Average story scores for Phase 2
    p2_story_means: Dict[str, Dict[str, float]] = defaultdict(dict)  # arm -> sid -> mean_acc
    for (sid, arm), vals in p2_story_arm_acc.items():
        p2_story_means[arm][sid] = sum(vals) / len(vals)

    # Arm A from Phase 1 is our baseline (Text Only, intact)
    arm_a_story_means = p1_story_means[("A", 0.0)]
    arm_a_acc = sum(arm_a_story_means.values()) / len(arm_a_story_means)

    print(f"================================================================================")
    print(f"PHASE 2 REAL EXTRACTOR NOISE PILOT ANALYSIS (N = {len(pilot_story_ids)} stories, 640 evals)")
    print(f"================================================================================")
    print(f"\n--- EXTRACTION FIDELITY SUMMARY ---")
    print(f"Stories analyzed: {len(extractions)}")
    print(f"Mean graph fidelity: {sum(fidelities)/len(fidelities):.4f} ({min(fidelities)*100:.1f}% - {max(fidelities)*100:.1f}%)")
    print(f"Perfect extractions (100%): {perfect_count}/{len(extractions)} ({perfect_count/len(extractions)*100:.1f}%)")
    print(f"Total missing events (omissions): {missing_ev_total}")
    print(f"Total hallucinated events: {hallucinated_ev_total}")
    print(f"Total timestamp mismatches: {time_mismatches_total}")
    print(f"Total duration mismatches: {dur_mismatches_total}")

    print(f"\n--- ACCURACY BY ARM ---")
    print(f"{'Arm':<35} | {'Acc (Phase 2)':<15} | {'Phase 1 (Synthetic 30%)':<22} | {'Phase 1 (Intact)':<15}")
    print(f"{'-'*35}-|-{'-'*15}-|-{'-'*22}-|-{'-'*15}")

    p2_overall_acc = {}
    for arm in ["B", "C", "D", "F"]:
        p2_acc = sum(p2_story_means[arm].values()) / len(p2_story_means[arm])
        p2_overall_acc[arm] = p2_acc
        
        p1_synth_acc = sum(p1_story_means.get((arm, 0.30), {}).values()) / len(pilot_story_ids) if (arm, 0.30) in p1_story_means else 0.0
        p1_intact_acc = sum(p1_story_means.get((arm, 0.0), {}).values()) / len(pilot_story_ids) if (arm, 0.0) in p1_story_means else 0.0
        
        arm_name = {
            "B": "Arm B (Graph Only)",
            "C": "Arm C (Text + Graph)",
            "D": "Arm D (Reconcile)",
            "F": "Arm F (Chronological Control)",
        }[arm]
        print(f"{arm_name:<35} | {p2_acc*100:>6.2f}%         | {p1_synth_acc*100:>6.2f}%               | {p1_intact_acc*100:>6.2f}%")

    print(f"{'Arm A (Text Only Baseline)':<35} | {'[Matched P1]':<15} | {'N/A':<22} | {arm_a_acc*100:>6.2f}%")

    print(f"\n--- HYPOTHESES TESTING UNDER REAL EXTRACTION NOISE (Story-Level Bootstrap N=10,000, delta=0.03) ---")
    
    comparisons = [
        ("H1' (Arm B Extracted vs Arm A Baseline)", p2_story_means["B"], arm_a_story_means),
        ("H2' (Arm C Text+Extracted vs Arm A Baseline)", p2_story_means["C"], arm_a_story_means),
        ("H3' (Arm C Text+Extracted vs Arm B Extracted)", p2_story_means["C"], p2_story_means["B"]),
        ("H4' (Arm D Reconcile vs Arm F Matched Control)", p2_story_means["D"], p2_story_means["F"]),
        ("Synthetic vs Real: Arm B (Real Ext) vs Arm B (Synth 30%)", p2_story_means["B"], p1_story_means[("B", 0.30)]),
        ("Synthetic vs Real: Arm C (Real Ext) vs Arm C (Synth 30%)", p2_story_means["C"], p1_story_means[("C", 0.30)]),
        ("Synthetic vs Real: Arm D (Real Ext) vs Arm D (Synth 30%)", p2_story_means["D"], p1_story_means[("D", 0.30)]),
    ]

    results_table = []
    for label, arm1_dict, arm2_dict in comparisons:
        diff, lo, hi = bootstrap_story_paired_diff(arm1_dict, arm2_dict)
        verdict = classify_comparison(lo, hi, delta=0.03)
        results_table.append({
            "hypothesis": label,
            "diff": diff,
            "ci_lower": lo,
            "ci_upper": hi,
            "verdict": verdict,
        })
        print(f"{label}:")
        print(f"  Diff = {diff:+.4f} ({diff*100:+.2f} pp), 95% CI [{lo:+.4f}, {hi:+.4f}], Verdict: {verdict.upper()}")

    print(f"\n--- QUESTION TYPE ACCURACY BREAKDOWN ---")
    qtypes = ["Q1", "Q2", "Q3", "Q4"]
    qtype_labels = {
        "Q1": "Q1 (Ordering)",
        "Q2": "Q2 (Duration Arithmetic)",
        "Q3": "Q3 (Counterfactual DAG)",
        "Q4": "Q4 (Conditional Rule)",
    }
    print(f"{'Question Type':<28} | {'Arm A':<8} | {'Arm B (Ext)':<12} | {'Arm C (T+G)':<12} | {'Arm D (Rec)':<12} | {'Arm F (Ctrl)':<12}")
    print(f"{'-'*28}-|-{'-'*8}-|-{'-'*12}-|-{'-'*12}-|-{'-'*12}-|-{'-'*12}")
    
    qtype_breakdown = {}
    for qt in qtypes:
        acc_a = sum(p1_matched_arm_qtype[("A", 0.0, qt)]) / len(p1_matched_arm_qtype[("A", 0.0, qt)])
        acc_b = sum(p2_arm_qtype_scores[("B", qt)]) / len(p2_arm_qtype_scores[("B", qt)])
        acc_c = sum(p2_arm_qtype_scores[("C", qt)]) / len(p2_arm_qtype_scores[("C", qt)])
        acc_d = sum(p2_arm_qtype_scores[("D", qt)]) / len(p2_arm_qtype_scores[("D", qt)])
        acc_f = sum(p2_arm_qtype_scores[("F", qt)]) / len(p2_arm_qtype_scores[("F", qt)])
        qtype_breakdown[qt] = {
            "A": acc_a,
            "B": acc_b,
            "C": acc_c,
            "D": acc_d,
            "F": acc_f,
        }
        lbl = qtype_labels[qt]
        print(f"{lbl:<28} | {acc_a*100:>6.1f}% | {acc_b*100:>10.1f}% | {acc_c*100:>10.1f}% | {acc_d*100:>10.1f}% | {acc_f*100:>10.1f}%")

    # Save detailed summary
    out_summary = {
        "num_stories": len(pilot_story_ids),
        "total_evals": len(pilot_story_ids) * 16,
        "extraction_metrics": {
            "mean_fidelity": sum(fidelities) / len(fidelities),
            "min_fidelity": min(fidelities),
            "max_fidelity": max(fidelities),
            "perfect_extractions_pct": perfect_count / len(extractions),
            "total_missing_events": missing_ev_total,
            "total_hallucinated_events": hallucinated_ev_total,
            "total_timestamp_mismatches": time_mismatches_total,
            "total_duration_mismatches": dur_mismatches_total,
        },
        "overall_accuracy": {
            "Arm_A_baseline": arm_a_acc,
            "Arm_B_extracted": p2_overall_acc["B"],
            "Arm_C_text_plus_extracted": p2_overall_acc["C"],
            "Arm_D_reconcile": p2_overall_acc["D"],
            "Arm_F_matched_control": p2_overall_acc["F"],
        },
        "statistical_tests": results_table,
        "question_type_breakdown": qtype_breakdown,
    }

    out_json_path = base_dir / "phase2_pilot_analysis.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\nWrote full analysis to {out_json_path}")


if __name__ == "__main__":
    analyze()
