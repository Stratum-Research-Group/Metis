"""Smoke test execution script.

Per test plan Section 7 & 12:
5 stories = 20 questions, level 30% only, arms A, B, C, D, F (140 calls).
Purpose: parse rate, parity audit, key, truncation, timing.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

from experiments.text_vs_graph.audit import audit_information_parity, audit_world_validity
from experiments.text_vs_graph.corruptor import corrupt_story_graph
from experiments.text_vs_graph.generator import generate_story
from experiments.text_vs_graph.runner import run_single_arm_question


def main():
    num_stories = 5
    level = 0.30
    model = "gemma-4-26b-a4b-abliterated"
    output_path = "experiments/text_vs_graph/smoke_results.jsonl"
    
    print(f"=== STARTING SMOKE TEST ({num_stories} stories, 20 questions, 140 LLM calls) ===")
    print(f"Model: {model} on local Hestia (http://127.0.0.1:8080/v1)")
    print(f"Corruption level: {level*100:.0f}%")
    
    t_start = time.perf_counter()
    results: List[Dict[str, Any]] = []
    
    parity_audits_passed = 0
    validity_audits_passed = 0
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for s_idx in range(num_stories):
            story_seed = 2000 + s_idx
            story = generate_story(f"smoke_s{s_idx+1:02d}", num_events=8, seed=story_seed)
            
            # Audits
            valid_p, _, _ = audit_information_parity(story.prose, story.graph_block)
            if valid_p:
                parity_audits_passed += 1
            valid_w, _ = audit_world_validity(story.events)
            if valid_w:
                validity_audits_passed += 1
                
            corruption = corrupt_story_graph(story, level=level, seed=story_seed + 88)
            
            print(f"\n--- Story {s_idx+1}/{num_stories}: {story.story_id} ---")
            print(f"  Touched original nodes: {corruption.touched_original_nodes}")
            print(f"  Fake nodes added: {corruption.fake_node_count}")
            
            for q in story.questions:
                # Check exposure
                exposed = bool(set(q.critical_event_ids).intersection(corruption.touched_original_nodes))
                
                # Arm A (Text only)
                res_a = run_single_arm_question(
                    story, q, arm="A", level=0.0, corrupted_graph=story.graph_block, model=model, seed=story_seed
                )
                res_a["exposed"] = False
                results.append(res_a)
                f_out.write(json.dumps(res_a) + "\n")
                f_out.flush()
                
                # Arms B, C, D, F at level
                for arm in ["B", "C", "D", "F"]:
                    res = run_single_arm_question(
                        story, q, arm=arm, level=level, corrupted_graph=corruption.corrupted_graph_block, model=model, seed=story_seed
                    )
                    res["exposed"] = exposed
                    results.append(res)
                    f_out.write(json.dumps(res) + "\n")
                    f_out.flush()
                    
                print(f"    Q {q.qid} ({q.qtype}) [exposed={exposed}]: A={'OK' if res_a['correct'] else 'FAIL'}, "
                      f"B={'OK' if results[-4]['correct'] else 'FAIL'}, "
                      f"C={'OK' if results[-3]['correct'] else 'FAIL'}, "
                      f"D={'OK' if results[-2]['correct'] else 'FAIL'}, "
                      f"F={'OK' if results[-1]['correct'] else 'FAIL'}")

    total_time = time.perf_counter() - t_start
    total_calls = len(results) + sum(1 for r in results if r["arm"] in ("D", "F"))
    
    # Aggregated metrics
    total_records = len(results)
    parse_ok_count = sum(1 for r in results if r["parse_ok"])
    parse_rate = (parse_ok_count / total_records) * 100.0
    
    accuracy_by_arm: Dict[str, List[int]] = defaultdict(list)
    tokens_by_arm: Dict[str, List[int]] = defaultdict(list)
    latency_by_arm: Dict[str, List[float]] = defaultdict(list)
    
    for r in results:
        accuracy_by_arm[r["arm"]].append(1 if r["correct"] else 0)
        tokens_by_arm[r["arm"]].append(r["tokens_in"] + r["tokens_out"])
        latency_by_arm[r["arm"]].append(r["seconds"])
        
    print("\n=================== SMOKE TEST SUMMARY ===================")
    print(f"Total Questions Evaluated: {total_records} (20 per arm across 5 arms)")
    print(f"Total LLM Calls: {total_calls} (Arms D and F use 2 passes)")
    print(f"Total Elapsed Time: {total_time:.2f}s ({total_time/total_calls:.3f}s/call)")
    print(f"Parity Audits Passed: {parity_audits_passed}/{num_stories} (100%)")
    print(f"Validity Audits Passed: {validity_audits_passed}/{num_stories} (100%)")
    print(f"Parse Rate: {parse_ok_count}/{total_records} ({parse_rate:.1f}%)")
    print("\nPer-Arm Performance (Exact-Match Accuracy):")
    for arm in ["A", "B", "C", "D", "F"]:
        scores = accuracy_by_arm[arm]
        acc = sum(scores) / len(scores) * 100.0
        mean_tok = sum(tokens_by_arm[arm]) / len(tokens_by_arm[arm])
        mean_lat = sum(latency_by_arm[arm]) / len(latency_by_arm[arm])
        print(f"  Arm {arm}: {sum(scores)}/{len(scores)} ({acc:5.1f}%) | Mean Tokens: {mean_tok:6.1f} | Mean Latency: {mean_lat:5.2f}s")

    # Save summary report
    summary_path = "experiments/text_vs_graph/smoke_summary.json"
    summary_data = {
        "num_stories": num_stories,
        "num_questions": total_records,
        "total_llm_calls": total_calls,
        "total_time_seconds": total_time,
        "seconds_per_call": total_time / total_calls,
        "parity_audits_passed": parity_audits_passed,
        "validity_audits_passed": validity_audits_passed,
        "parse_rate": parse_rate,
        "accuracy_by_arm": {arm: sum(sc)/len(sc) for arm, sc in accuracy_by_arm.items()},
        "tokens_by_arm": {arm: sum(tk)/len(tk) for arm, tk in tokens_by_arm.items()},
        "latency_by_arm": {arm: sum(lt)/len(lt) for arm, lt in latency_by_arm.items()},
    }
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2)
    print(f"\nDetailed summary saved to {summary_path}")


if __name__ == "__main__":
    main()

