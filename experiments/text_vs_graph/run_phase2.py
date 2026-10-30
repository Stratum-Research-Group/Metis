"""
Phase 2 Execution Script: Real Extractor Noise Evaluation.

Per test plan Section 5 & 7:
Evaluates arms B, C, D, and F using graphs extracted by the local model directly
from raw prose narrative. Measures real extraction error distribution and
evaluates whether live extractor artifacts replicate Phase 1 synthetic findings.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.text_vs_graph.extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    analyze_extraction_fidelity,
)
from experiments.text_vs_graph.generator import generate_story
from experiments.text_vs_graph.harness import call_local_hestia_chat
from experiments.text_vs_graph.models import Story
from experiments.text_vs_graph.runner import run_single_arm_question

logger = logging.getLogger(__name__)


def extract_story_graph(
    story: Story,
    model: str = "gemma-4-26b-a4b-abliterated",
    endpoint: str = "http://127.0.0.1:8080/v1/chat/completions",
) -> Dict[str, Any]:
    """Execute 1 extraction call to parse story prose into structured fact graph."""
    user_prompt = f"Extract all structured facts from this text:\n\n{story.prose}"
    resp = call_local_hestia_chat(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        endpoint=endpoint,
        temperature=0.0,
        max_tokens=768,
        stop=["<|turn>", "<|turn_end|>", "<end_of_turn>"],
    )
    return resp


def run_phase2(
    start_seed: int = 4001,
    num_stories: int = 300,
    results_path: str = "experiments/text_vs_graph/phase2_results.jsonl",
    summary_path: str = "experiments/text_vs_graph/phase2_summary.json",
    extractions_path: str = "experiments/text_vs_graph/phase2_extractions.jsonl",
    model: str = "gemma-4-26b-a4b-abliterated",
    endpoint: str = "http://127.0.0.1:8080/v1/chat/completions",
) -> Dict[str, Any]:
    """Execute Phase 2 real extraction study."""
    out_file = Path(results_path)
    ext_file = Path(extractions_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume capability: check completed story IDs
    completed_stories = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_stories.add(rec.get("story_id"))
                    except Exception:
                        pass

    cached_extractions = {}
    if ext_file.exists():
        with open(ext_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        cached_extractions[rec["story_id"]] = rec
                    except Exception:
                        pass

    print(f"=== Starting Phase 2 Real Extractor Noise Evaluation ===")
    print(f"Target: {num_stories} stories (seeds {start_seed}..{start_seed + num_stories - 1})")
    print(f"Completed stories so far: {len(completed_stories)}")
    print(f"Cached extractions: {len(cached_extractions)}")

    arms = ["B", "C", "D", "F"]
    t0_all = time.perf_counter()
    total_evals = 0
    extraction_metrics_list = []

    for idx in range(num_stories):
        seed = start_seed + idx
        story_id = f"conf_s{idx+1:03d}" if start_seed == 4001 else f"story_{seed}"

        # 1. Generate story
        story = generate_story(story_id=story_id, seed=seed)

        # 2. Extract or retrieve cached extraction
        if story_id in cached_extractions:
            ext_record = cached_extractions[story_id]
            extracted_graph = ext_record["extracted_graph"]
            fidelity_metrics = ext_record["metrics"]
        else:
            t_ext0 = time.perf_counter()
            resp_ext = extract_story_graph(story, model=model, endpoint=endpoint)
            extracted_graph = resp_ext["content"]
            dur_ext = time.perf_counter() - t_ext0

            # Analyze extraction fidelity against gold graph
            metrics_obj = analyze_extraction_fidelity(story.graph_block, extracted_graph)
            fidelity_metrics = {
                "gold_events": metrics_obj.gold_event_count,
                "extracted_events": metrics_obj.extracted_event_count,
                "missing_events": metrics_obj.missing_events,
                "hallucinated_events": metrics_obj.hallucinated_events,
                "timestamp_mismatches": metrics_obj.timestamp_mismatches,
                "duration_mismatches": metrics_obj.duration_mismatches,
                "overall_fidelity": metrics_obj.overall_fidelity,
                "tokens_in": resp_ext["tokens_in"],
                "tokens_out": resp_ext["tokens_out"],
                "seconds": round(dur_ext, 4),
            }

            ext_record = {
                "story_id": story_id,
                "seed": seed,
                "extracted_graph": extracted_graph,
                "metrics": fidelity_metrics,
            }
            with open(ext_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(ext_record) + "\n")
            cached_extractions[story_id] = ext_record

        extraction_metrics_list.append(fidelity_metrics)

        if story_id in completed_stories:
            continue

        print(f"[{idx+1}/{num_stories}] Evaluating {story_id} (extraction fidelity: {fidelity_metrics['overall_fidelity']:.2%})...")

        # 3. Evaluate arms B, C, D, F across all 4 questions
        story_results = []
        for q in story.questions:
            for arm in arms:
                res = run_single_arm_question(
                    story=story,
                    question=q,
                    arm=arm,
                    level=0.0,
                    corrupted_graph=extracted_graph,
                    model=model,
                    endpoint=endpoint,
                    seed=seed,
                )
                res["phase"] = 2
                res["level_label"] = "natural_extraction"
                res["extraction_fidelity"] = fidelity_metrics["overall_fidelity"]
                story_results.append(res)
                total_evals += 1

        # Atomically append to results file
        with open(out_file, "a", encoding="utf-8") as f:
            for r in story_results:
                f.write(json.dumps(r) + "\n")

        completed_stories.add(story_id)

    total_time = time.perf_counter() - t0_all
    print(f"\nPhase 2 Complete! Evaluated {total_evals} question calls across {len(completed_stories)} stories in {total_time:.2f}s.")

    # Write summary
    avg_fidelity = (
        sum(m["overall_fidelity"] for m in extraction_metrics_list) / len(extraction_metrics_list)
        if extraction_metrics_list
        else 1.0
    )
    summary_data = {
        "num_stories": len(completed_stories),
        "total_evals": total_evals,
        "avg_extraction_fidelity": round(avg_fidelity, 4),
        "total_seconds": round(total_time, 2),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 2 real extraction evaluation")
    parser.add_argument("--stories", type=int, default=5, help="Number of stories to evaluate")
    parser.add_argument("--start-seed", type=int, default=4001, help="Starting seed")
    parser.add_argument("--results", type=str, default="experiments/text_vs_graph/phase2_results.jsonl", help="Path to results file")
    parser.add_argument("--summary", type=str, default="experiments/text_vs_graph/phase2_summary.json", help="Path to summary file")
    parser.add_argument("--extractions", type=str, default="experiments/text_vs_graph/phase2_extractions.jsonl", help="Path to extractions file")
    args = parser.parse_args()

    run_phase2(
        start_seed=args.start_seed,
        num_stories=args.stories,
        results_path=args.results,
        summary_path=args.summary,
        extractions_path=args.extractions,
    )
