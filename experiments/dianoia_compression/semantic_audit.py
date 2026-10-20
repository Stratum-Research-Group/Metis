"""Semantic Auditor for Dianoia Question-Conditioned Compression Experiment.

Implements Section 7 (Semantic Audit) of metis-dianoia-question-conditioned-compression-experiment-spec-2026-10-20-531c7bb-29a18f8e.txt:
1. Two distinct audit targets:
   (i) Context Preservation (A1, A2, A3 against frozen A0 context):
       For every gold fact in retrieved input -> faithful / distorted / contradicted / dropped / uncertain.
       Gate 2 floor: >= 95% faithful preservation.
   (ii) Final-Answer Sufficiency (all arms):
       Count unsupported claims against that arm's own context. Catches evidence laundering.
2. Fixed prespecified random samples:
   - 30 Dev items for rubric calibration.
   - 100 Test items drawn independently of outcomes, recorded before test execution.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

AUDIT_SYSTEM_PROMPT = """\
You are an expert, meticulous semantic fact auditor for knowledge retrieval systems.
Your task is to judge evidence preservation and answer sufficiency with strict fidelity.
Do NOT assume facts not explicitly stated. Adhere strictly to the rubric.\
"""

CONTEXT_AUDIT_PROMPT_TEMPLATE = """\
We are evaluating whether retrieved evidence was faithfully preserved after context compression.

GOLD SUPPORTING FACTS PRESENT IN RAW RETRIEVED EVIDENCE:
{gold_facts_formatted}

COMPRESSED CONTEXT (BRIEFING):
{compressed_context}

RUBRIC:
For each gold fact numbered above, determine its preservation status in the compressed context:
- "faithful": The core assertion, key entities, and relational claim of the fact are accurately represented without distortion or contradiction. (Note: Condensing text or omitting incidental parenthetical details/dates from a long compound sentence does NOT count as distortion if the primary assertion is faithfully preserved).
- "distorted": The primary claim, key entities, numbers, dates, or causal relations are altered, blurred, misattributed, or substantively weakened.
- "contradicted": The compressed context asserts the opposite or a contradictory claim.
- "dropped": The core claim of the fact is absent or omitted entirely from the compressed context.
- "uncertain": Ambiguous or impossible to determine with certainty.

Respond with a JSON object in exactly this format:
{{
  "facts": [
    {{
      "fact_index": 1,
      "fact_text": "<text>",
      "status": "faithful" | "distorted" | "contradicted" | "dropped" | "uncertain",
      "reason": "<short justification>"
    }}
  ]
}}
"""

ANSWER_SUFFICIENCY_PROMPT_TEMPLATE = """\
We are evaluating whether a generated answer is strictly supported by its context (checking for evidence laundering or unsupported claims).

QUESTION: {question}

ARM CONTEXT:
{arm_context}

GENERATED ANSWER:
{generated_answer}

RUBRIC:
Extract each factual claim asserted in the generated answer and determine if it is entailed by the arm context:
- If a claim is entailed by the context -> "supported": true
- If a claim introduces new facts not present in the context -> "supported": false

Respond with a JSON object in exactly this format:
{{
  "claims": [
    {{
      "claim": "<statement>",
      "supported": true | false,
      "reason": "<short justification>"
    }}
  ],
  "unsupported_count": <int>
}}
"""


def select_audit_sample_ids(
    all_item_ids: List[str],
    sample_size: int,
    seed: int = 42,
) -> List[str]:
    """Draw deterministic, outcome-independent random sample of item IDs."""
    rng = random.Random(seed)
    shuffled = list(all_item_ids)
    rng.shuffle(shuffled)
    return sorted(shuffled[:sample_size])


class SemanticAuditor:
    def __init__(
        self,
        model_name: str = "qwen2.5-14b-instruct",
        llm_base_url: str = "http://127.0.0.1:8080/v1",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.llm_base_url = llm_base_url
        self.api_key = (
            api_key
            or os.environ.get("HESTIA_SERVICE_TOKEN")
            or "0d4b83563dfab46fdec0d777728985eb1fd0bd0ba4c4a2b0"
        )
        self.client = OpenAI(base_url=self.llm_base_url, api_key=self.api_key)

    def _call_json_audit(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Call auditor LLM and parse JSON response."""
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        content = resp.choices[0].message.content or "{}"
        # Extract JSON block if wrapped in markdown
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw_output": content, "error": "json_parse_failed"}

    def audit_context_preservation(
        self,
        retrievable_facts: List[Dict[str, Any]],
        compressed_context: str,
    ) -> Dict[str, Any]:
        """Audit context preservation of retrievable gold facts."""
        if not retrievable_facts:
            return {
                "counts": {"faithful": 0, "distorted": 0, "contradicted": 0, "dropped": 0, "uncertain": 0},
                "total_facts": 0,
                "faithful_rate": 1.0,
                "facts": [],
                "details": [],
            }

        facts_formatted = "\n".join(
            f"{i+1}. [{f.get('title')}]: {f.get('sentence_text')}"
            for i, f in enumerate(retrievable_facts)
        )
        prompt = CONTEXT_AUDIT_PROMPT_TEMPLATE.format(
            gold_facts_formatted=facts_formatted,
            compressed_context=compressed_context,
        )

        res = self._call_json_audit(AUDIT_SYSTEM_PROMPT, prompt)
        facts = res.get("facts", [])

        # Categorize
        counts = {"faithful": 0, "distorted": 0, "contradicted": 0, "dropped": 0, "uncertain": 0}
        for f in facts:
            st = str(f.get("status", "uncertain")).lower()
            if st in counts:
                counts[st] += 1
            else:
                counts["uncertain"] += 1

        total = len(retrievable_facts)
        # Handle cases where LLM returned fewer/more facts
        eval_total = sum(counts.values()) or total
        faithful_rate = counts["faithful"] / eval_total if eval_total > 0 else 0.0

        return {
            "counts": counts,
            "total_facts": eval_total,
            "faithful_rate": round(faithful_rate, 4),
            "details": facts,
        }

    def audit_answer_sufficiency(
        self,
        question: str,
        arm_context: str,
        generated_answer: str,
    ) -> Dict[str, Any]:
        """Audit whether generated answer contains unsupported claims."""
        prompt = ANSWER_SUFFICIENCY_PROMPT_TEMPLATE.format(
            question=question,
            arm_context=arm_context,
            generated_answer=generated_answer,
        )

        res = self._call_json_audit(AUDIT_SYSTEM_PROMPT, prompt)
        claims = res.get("claims", [])
        unsupported = sum(1 for c in claims if not c.get("supported", True))
        return {
            "total_claims": len(claims),
            "unsupported_claims": unsupported,
            "is_fully_supported": (unsupported == 0),
            "details": claims,
        }


def run_sample_audit(
    split: str = "dev",
    sample_size: int = 30,
    results_dir: str = "experiments/dianoia_compression/results",
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute semantic audit on prespecified random sample."""
    results_file = Path(results_dir) / f"{split}_per_item_results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError(f"Missing results file: {results_file}. Run evaluator.py first.")

    with open(results_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    # Load gold_supporting_facts from source dataset
    dataset_file = Path("experiments/dianoia_compression/data") / (
        "dev_150.jsonl" if split == "dev" else "test_300.jsonl"
    )
    gold_facts_by_id = {}
    if dataset_file.exists():
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                gold_facts_by_id[d["id"]] = d.get("gold_supporting_facts", [])

    all_ids = [it["id"] for it in items]
    sample_ids = select_audit_sample_ids(all_ids, sample_size=min(sample_size, len(all_ids)), seed=seed)
    sample_id_set = set(sample_ids)
    sampled_items = [it for it in items if it["id"] in sample_id_set]

    print(f"[-] Running Semantic Audit on {len(sampled_items)} {split} items (Seed: {seed})...")
    auditor = SemanticAuditor()

    audit_records = []
    preservation_totals = {
        "A1": {"faithful": 0, "distorted": 0, "contradicted": 0, "dropped": 0, "uncertain": 0, "total": 0},
        "A2": {"faithful": 0, "distorted": 0, "contradicted": 0, "dropped": 0, "uncertain": 0, "total": 0},
        "A3": {"faithful": 0, "distorted": 0, "contradicted": 0, "dropped": 0, "uncertain": 0, "total": 0},
    }
    unsupported_claims_by_arm = {"A0": 0, "A1": 0, "A2": 0, "A3": 0}

    for idx, it in enumerate(sampled_items):
        q_id = it["id"]
        question = it["question"]
        ret_ceil = it.get("retrieval_ceiling", {})
        gold_facts = it.get("gold_supporting_facts") or gold_facts_by_id.get(q_id, [])

        # Filter only retrievable facts (present in retrieved input)
        # A fact is retrievable if it is not in uncovered_facts
        uncovered_texts = {uf.get("sentence_text") for uf in ret_ceil.get("uncovered_facts", [])}
        retrievable_facts = [f for f in gold_facts if f.get("sentence_text") not in uncovered_texts]

        item_audit = {"id": q_id, "question": question, "retrievable_facts_count": len(retrievable_facts)}

        # Context Preservation: A1, A2, A3
        for arm_name, brief_key in [("A1", "briefing"), ("A2", "briefing"), ("A3", "context")]:
            context_text = it[arm_name].get(brief_key, "")
            res_p = auditor.audit_context_preservation(retrievable_facts, context_text)
            item_audit[f"{arm_name}_preservation"] = res_p
            for k, v in res_p["counts"].items():
                preservation_totals[arm_name][k] += v
            preservation_totals[arm_name]["total"] += res_p["total_facts"]

        # Final-Answer Sufficiency: A0, A1, A2, A3
        for arm_name, ctx_key in [("A0", "context"), ("A1", "briefing"), ("A2", "briefing"), ("A3", "context")]:
            ctx = it[arm_name].get(ctx_key, "")
            ans = it[arm_name].get("answer", "")
            res_s = auditor.audit_answer_sufficiency(question, ctx, ans)
            item_audit[f"{arm_name}_sufficiency"] = res_s
            unsupported_claims_by_arm[arm_name] += res_s["unsupported_claims"]

        audit_records.append(item_audit)
        if (idx + 1) % 5 == 0 or (idx + 1) == len(sampled_items):
            print(f"[{idx + 1:2d}/{len(sampled_items)}] Audited.")

    # Compute overall rates
    summary = {
        "split": split,
        "sample_size": len(sampled_items),
        "seed": seed,
        "sample_ids": sample_ids,
        "context_preservation": {},
        "unsupported_claims_count": unsupported_claims_by_arm,
    }

    for arm_name in ["A1", "A2", "A3"]:
        tot = preservation_totals[arm_name]["total"]
        f_count = preservation_totals[arm_name]["faithful"]
        f_rate = f_count / tot if tot > 0 else 0.0
        summary["context_preservation"][arm_name] = {
            "counts": preservation_totals[arm_name],
            "faithful_rate": round(f_rate, 4),
            "gate_2_pass": bool(f_rate >= 0.95),
        }

    audit_out_file = Path(results_dir) / f"{split}_semantic_audit_results.json"
    with open(audit_out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "items": audit_records}, f, indent=2)

    print(f"\n[+] Semantic Audit Completed for {split}!")
    print(f"    - Output: {audit_out_file}")
    for arm_name in ["A1", "A2", "A3"]:
        cp = summary["context_preservation"][arm_name]
        print(f"    - {arm_name} Faithful Preservation: {cp['faithful_rate']*100:.2f}% ({cp['counts']['faithful']}/{cp['counts']['total']}) | Gate >=95%: {'PASS' if cp['gate_2_pass'] else 'FAIL'}")
    print(f"    - Unsupported Claims: {unsupported_claims_by_arm}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Semantic Audit.")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_sample_audit(split=args.split, sample_size=args.sample_size, seed=args.seed)
