"""Analysis script for Phase 2 (Real Extractor Noise) Pilot.

Compares:
- Phase 2 Real Extractor Arms (B, C, D, F) vs Arm A (Text Baseline)
- Phase 2 Real Extractor Arms vs Phase 1 Synthetic 30% Corrupted Arms (matched on conf_s001-conf_s040)
- Detailed extraction error categorization and fidelity breakdown.
"""

from __future__ import annotations

import argparse
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


def analyze(
    results_path: Path | None = None,
    extractions_path: Path | None = None,
    output_path: Path | None = None,
    max_stories: int | None = None,
):
    base_dir = Path(__file__).parent
    phase2_res_path = results_path or (base_dir / "phase2_results.jsonl")
    phase2_ext_path = extractions_path or (base_dir / "phase2_extractions.jsonl")
    conf_res_path = base_dir / "confirmatory_results.jsonl"

    # 1. Identify stories available in Phase 2 results
    available_story_ids = set()
    with open(phase2_res_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            available_story_ids.add(rec["story_id"])

    # If max_stories is specified, limit to the first N sorted story IDs
    if max_stories:
        sorted_ids = sorted(list(available_story_ids))[:max_stories]
        target_story_ids = set(sorted_ids)
    else:
        target_story_ids = available_story_ids

    # 2. Load Phase 2 extractions for target stories
    extractions = []
    with open(phase2_ext_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["story_id"] in target_story_ids:
                    extractions.append(rec)

    fidelities = [e["metrics"]["overall_fidelity"] for e in extractions]
    perfect_count = sum(1 for f in fidelities if f >= 0.999)
    missing_ev_total = sum(len(e["metrics"].get("missing_events", [])) for e in extractions)
    hallucinated_ev_total = sum(len(e["metrics"].get("hallucinated_events", [])) for e in extractions)
    time_mismatches_total = sum(len(e["metrics"].get("timestamp_mismatches", [])) for e in extractions)
    dur_mismatches_total = sum(len(e["metrics"].get("duration_mismatches", [])) for e in extractions)

    # 3. Load Phase 2 results
    p2_story_arm_acc: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    p2_arm_qtype_scores: Dict[Tuple[str, str], List[int]] = defaultdict(list)  # (arm, q_type) -> [0/1]
    finish_reasons: Dict[str, int] = defaultdict(int)
    arm_finish_reasons: Dict[Tuple[str, str], int] = defaultdict(int)

    story_ids_with_results = set()
    with open(phase2_res_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["story_id"]
            if sid not in target_story_ids:
                continue
            story_ids_with_results.add(sid)
            arm = rec["arm"]
            qtype = rec["qtype"]
            corr = 1 if rec["correct"] else 0
            p2_story_arm_acc[(sid, arm)].append(corr)
            p2_arm_qtype_scores[(arm, qtype)].append(corr)
            fr = rec.get("finish_reason", "unknown")
            finish_reasons[fr] += 1
            arm_finish_reasons[(arm, fr)] += 1

    # 4. Load Confirmatory Phase 1 results for the matched stories
    p1_matched_arm_acc: Dict[Tuple[str, float], Dict[str, float]] = defaultdict(dict)  # (arm, corrupt_rate) -> sid -> mean_acc
    p1_matched_arm_qtype: Dict[Tuple[str, float, str], List[int]] = defaultdict(list)  # (arm, corrupt_rate, qtype) -> [0/1]

    with open(conf_res_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["story_id"]
            if sid not in story_ids_with_results:
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
    arm_a_story_means = p1_story_means.get(("A", 0.0), {})
    arm_a_acc = sum(arm_a_story_means.values()) / len(arm_a_story_means) if arm_a_story_means else 0.0

    num_stories = len(story_ids_with_results)
    total_evals = sum(len(vals) for vals in p2_story_arm_acc.values())

    print(f"================================================================================")
    print(f"PHASE 2 REAL EXTRACTOR NOISE ANALYSIS (N = {num_stories} stories, {total_evals} evals)")
    print(f"================================================================================")
    print(f"\n--- EXTRACTION FIDELITY SUMMARY ---")
    print(f"Stories analyzed: {len(extractions)}")
    if extractions:
        print(f"Mean graph fidelity: {sum(fidelities)/len(fidelities):.4f} ({min(fidelities)*100:.1f}% - {max(fidelities)*100:.1f}%)")
        print(f"Perfect extractions (100%): {perfect_count}/{len(extractions)} ({perfect_count/len(extractions)*100:.1f}%)")
        print(f"Total missing events (omissions): {missing_ev_total}")
        print(f"Total hallucinated events: {hallucinated_ev_total}")
        print(f"Total timestamp mismatches: {time_mismatches_total}")
        print(f"Total duration mismatches: {dur_mismatches_total}")

    print(f"\n--- GENERATION FINISH REASONS (TELEMETRY) ---")
    for fr, cnt in sorted(finish_reasons.items()):
        pct = (cnt / total_evals * 100) if total_evals > 0 else 0.0
        print(f"  finish_reason '{fr}': {cnt}/{total_evals} ({pct:.1f}%)")
    for arm in ["B", "C", "D", "F"]:
        arm_total = sum(arm_finish_reasons[(arm, fr)] for fr in finish_reasons.keys())
        arm_fr_str = ", ".join([f"{fr}: {arm_finish_reasons[(arm, fr)]}" for fr in sorted(finish_reasons.keys())])
        print(f"  Arm {arm} (N={arm_total}): {arm_fr_str}")

    print(f"\n--- ACCURACY BY ARM ---")
    print(f"{'Arm':<35} | {'Acc (Phase 2)':<15} | {'Phase 1 (Synthetic 30%)':<22} | {'Phase 1 (Intact)':<15}")
    print(f"{'-'*35}-|-{'-'*15}-|-{'-'*22}-|-{'-'*15}")

    p2_overall_acc = {}
    for arm in ["B", "C", "D", "F"]:
        arm_scores = p2_story_means.get(arm, {})
        p2_acc = sum(arm_scores.values()) / len(arm_scores) if arm_scores else 0.0
        p2_overall_acc[arm] = p2_acc

        p1_synth_acc = sum(p1_story_means.get((arm, 0.30), {}).values()) / num_stories if (arm, 0.30) in p1_story_means and num_stories > 0 else 0.0
        p1_intact_acc = sum(p1_story_means.get((arm, 0.0), {}).values()) / num_stories if (arm, 0.0) in p1_story_means and num_stories > 0 else 0.0

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
        ("H1' (Arm B Extracted vs Arm A Baseline)", p2_story_means.get("B", {}), arm_a_story_means),
        ("H2' (Arm C Text+Extracted vs Arm A Baseline)", p2_story_means.get("C", {}), arm_a_story_means),
        ("H3' (Arm C Text+Extracted vs Arm B Extracted)", p2_story_means.get("C", {}), p2_story_means.get("B", {})),
        ("H4' (Arm D Reconcile vs Arm F Matched Control)", p2_story_means.get("D", {}), p2_story_means.get("F", {})),
        ("Synthetic vs Real: Arm B (Real Ext) vs Arm B (Synth 30%)", p2_story_means.get("B", {}), p1_story_means.get(("B", 0.30), {})),
        ("Synthetic vs Real: Arm C (Real Ext) vs Arm C (Synth 30%)", p2_story_means.get("C", {}), p1_story_means.get(("C", 0.30), {})),
        ("Synthetic vs Real: Arm D (Real Ext) vs Arm D (Synth 30%)", p2_story_means.get("D", {}), p1_story_means.get(("D", 0.30), {})),
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
        p1_a_scores = p1_matched_arm_qtype.get(("A", 0.0, qt), [])
        acc_a = sum(p1_a_scores) / len(p1_a_scores) if p1_a_scores else 0.0
        acc_b = sum(p2_arm_qtype_scores[("B", qt)]) / len(p2_arm_qtype_scores[("B", qt)]) if p2_arm_qtype_scores[("B", qt)] else 0.0
        acc_c = sum(p2_arm_qtype_scores[("C", qt)]) / len(p2_arm_qtype_scores[("C", qt)]) if p2_arm_qtype_scores[("C", qt)] else 0.0
        acc_d = sum(p2_arm_qtype_scores[("D", qt)]) / len(p2_arm_qtype_scores[("D", qt)]) if p2_arm_qtype_scores[("D", qt)] else 0.0
        acc_f = sum(p2_arm_qtype_scores[("F", qt)]) / len(p2_arm_qtype_scores[("F", qt)]) if p2_arm_qtype_scores[("F", qt)] else 0.0
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
        "num_stories": num_stories,
        "total_evals": total_evals,
        "extraction_metrics": {
            "mean_fidelity": sum(fidelities) / len(fidelities) if fidelities else 0.0,
            "min_fidelity": min(fidelities) if fidelities else 0.0,
            "max_fidelity": max(fidelities) if fidelities else 0.0,
            "perfect_extractions_pct": (perfect_count / len(extractions)) if extractions else 0.0,
            "total_missing_events": missing_ev_total,
            "total_hallucinated_events": hallucinated_ev_total,
            "total_timestamp_mismatches": time_mismatches_total,
            "total_duration_mismatches": dur_mismatches_total,
        },
        "overall_accuracy": {
            "Arm_A_baseline": arm_a_acc,
            "Arm_B_extracted": p2_overall_acc.get("B", 0.0),
            "Arm_C_text_plus_extracted": p2_overall_acc.get("C", 0.0),
            "Arm_D_reconcile": p2_overall_acc.get("D", 0.0),
            "Arm_F_matched_control": p2_overall_acc.get("F", 0.0),
        },
        "statistical_tests": results_table,
        "question_type_breakdown": qtype_breakdown,
    }

    out_json_path = output_path or (base_dir / ("phase2_pilot_analysis.json" if num_stories <= 40 else "phase2_confirmatory_analysis.json"))
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\nWrote full analysis to {out_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 2 results")
    parser.add_argument("--results", type=Path, default=None, help="Path to phase2_results.jsonl")
    parser.add_argument("--extractions", type=Path, default=None, help="Path to phase2_extractions.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Path to output JSON")
    parser.add_argument("--stories", type=int, default=None, help="Limit analysis to first N stories")
    args = parser.parse_args()

    analyze(
        results_path=args.results,
        extractions_path=args.extractions,
        output_path=args.output,
        max_stories=args.stories,
    )


if __name__ == "__main__":
    main()
