"""Execution engine for Smoke Test, Pilot, and Confirmatory runs.

Per test plan Section 7 & 13:
Executes Arms A, B, C, D, F (and optional E) across corruption levels.
Records JSONL item results and timing/token metrics.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from experiments.text_vs_graph.corruptor import corrupt_story_graph
from experiments.text_vs_graph.generator import generate_story
from experiments.text_vs_graph.harness import (
    call_local_hestia_chat,
    format_user_prompt,
    parse_model_answer,
    score_answer,
)
from experiments.text_vs_graph.models import Question, Story


def run_single_arm_question(
    story: Story,
    question: Question,
    arm: str,
    level: float,
    corrupted_graph: str,
    model: str = "gemma-4-26b-a4b-abliterated",
    endpoint: str = "http://127.0.0.1:8080/v1/chat/completions",
    seed: int = 42,
) -> Dict[str, Any]:
    """Runs a single question for a given arm and corruption level."""
    t0 = time.perf_counter()
    pass1_raw = None
    pass1_tokens_in = 0
    pass1_tokens_out = 0
    
    # Pass 1 for 2-pass arms (D and F)
    if arm in ("D", "F"):
        sys_p1, user_p1 = format_user_prompt(story, question, arm, corrupted_graph, pass_number=1)
        resp1 = call_local_hestia_chat(
            system_prompt=sys_p1,
            user_prompt=user_p1,
            model=model,
            endpoint=endpoint,
            temperature=0.0,
            max_tokens=250,
        )
        pass1_raw = resp1["content"]
        pass1_tokens_in = resp1["tokens_in"]
        pass1_tokens_out = resp1["tokens_out"]

    # Final answer pass (Pass 2 for D/F, Pass 1 for A/B/C/E)
    sys_p2, user_p2 = format_user_prompt(
        story, question, arm, corrupted_graph, pass_number=2, pass1_notes=pass1_raw
    )
    resp2 = call_local_hestia_chat(
        system_prompt=sys_p2,
        user_prompt=user_p2,
        model=model,
        endpoint=endpoint,
        temperature=0.0,
        max_tokens=512,
    )
    
    elapsed = time.perf_counter() - t0
    raw_answer_text = resp2["content"]
    finish_reason = resp2.get("finish_reason", "unknown")
    pass1_finish_reason = resp1.get("finish_reason") if arm in ("D", "F") else None
    parse_ok, parsed_val = parse_model_answer(raw_answer_text, question.answer_format)
    is_correct = score_answer(parsed_val, question.gold_answer, question.answer_format) if parse_ok else False
    
    total_tokens_in = pass1_tokens_in + resp2["tokens_in"]
    total_tokens_out = pass1_tokens_out + resp2["tokens_out"]
    
    return {
        "story_id": story.story_id,
        "qid": question.qid,
        "qtype": question.qtype,
        "arm": arm,
        "level": level,
        "pass1_raw": pass1_raw,
        "raw": raw_answer_text,
        "finish_reason": finish_reason,
        "pass1_finish_reason": pass1_finish_reason,
        "parsed": parsed_val,
        "gold": question.gold_answer,
        "correct": is_correct,
        "parse_ok": parse_ok,
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "seconds": round(elapsed, 4),
        "seed": seed,
    }


def run_smoke_test(
    num_stories: int = 5,
    level: float = 0.30,
    model: str = "gemma-4-26b-a4b-abliterated",
    output_path: str = "experiments/text_vs_graph/smoke_results.jsonl",
) -> List[Dict[str, Any]]:
    """Runs Stage 2 Smoke Test: 5 stories = 20 questions, level 30%, arms A, B, C, D, F (140 calls)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results: List[Dict[str, Any]] = []
    
    print(f"[Smoke Test] Generating {num_stories} stories at level {level*100:.0f}% noise...")
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for s_idx in range(num_stories):
            story_seed = 1000 + s_idx
            story = generate_story(f"smoke_story_{s_idx+1:02d}", num_events=8, seed=story_seed)
            corruption = corrupt_story_graph(story, level=level, seed=story_seed + 99)
            
            for q in story.questions:
                # 1. Arm A (Text only, corruption does not apply)
                res_a = run_single_arm_question(
                    story, q, arm="A", level=0.0, corrupted_graph=story.graph_block, model=model, seed=story_seed
                )
                results.append(res_a)
                f_out.write(json.dumps(res_a) + "\n")
                f_out.flush()
                
                # 2. Arms B, C, D, F at level (e.g. 0.30)
                for arm in ["B", "C", "D", "F"]:
                    res = run_single_arm_question(
                        story, q, arm=arm, level=level, corrupted_graph=corruption.corrupted_graph_block, model=model, seed=story_seed
                    )
                    results.append(res)
                    f_out.write(json.dumps(res) + "\n")
                    f_out.flush()
                    
                print(f"  Processed {q.qid} (5 arms complete)")
                
    print(f"[Smoke Test] Completed {len(results)} total calls. Results saved to {output_path}")
    return results
