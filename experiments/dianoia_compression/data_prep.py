"""Data preparation and deterministic chunk ID mapping for HotpotQA.

Implements Section 5 of metis-dianoia-question-conditioned-compression-experiment-spec-2026-10-20-531c7bb-29a18f8e.txt:
1. Extract 150 Dev questions from HotpotQA distractor validation split.
2. Extract 300 Test questions strictly disjoint in gold supporting document titles and questions.
3. Deterministic mapping of gold supporting facts (title + sentence_id) to LTHCM chunk IDs.
4. Gate 0 Coverage check: verify fraction of questions with full coverage >= 95%.
5. Export dev_150.jsonl, test_300.jsonl, and mapping_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from datasets import load_dataset


def sanitize_title_to_chunk_id(title: str) -> str:
    """Generate deterministic, valid LTHCM chunk ID for a document title."""
    h = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    clean_title = re.sub(r"[^a-zA-Z0-9_]", "_", title).strip("_")
    clean_title = clean_title[:32] if clean_title else "doc"
    return f"hp_{h}_{clean_title}"


def prepare_dataset_splits(
    dev_size: int = 150,
    test_size: int = 300,
    output_dir: str = "experiments/dianoia_compression/data",
    coverage_floor: float = 0.95,
) -> Dict[str, Any]:
    """Extract disjoint Dev/Test splits, build chunk mappings, and verify invariants."""
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")

    print(f"[-] Loading HotpotQA validation split (distractor)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", cache_dir=cache_dir)
    total_available = len(ds)
    print(f"[-] Loaded {total_available} questions from cache.")

    # 1. Select Dev Items (first dev_size questions)
    dev_raw = []
    dev_gold_titles: Set[str] = set()
    dev_questions_normalized: Set[str] = set()

    for i in range(dev_size):
        ex = ds[i]
        q_norm = ex["question"].strip().lower()
        titles = set(ex["supporting_facts"]["title"])
        dev_raw.append(ex)
        dev_gold_titles.update(titles)
        dev_questions_normalized.add(q_norm)

    print(f"[-] Extracted Dev split: {len(dev_raw)} items, {len(dev_gold_titles)} unique gold document titles.")

    # 2. Select Test Items (disjoint from Dev gold titles & questions)
    test_raw = []
    test_gold_titles: Set[str] = set()
    skipped_title_overlap = 0

    for i in range(dev_size, total_available):
        ex = ds[i]
        q_norm = ex["question"].strip().lower()
        if q_norm in dev_questions_normalized:
            continue
        titles = set(ex["supporting_facts"]["title"])
        if not titles.isdisjoint(dev_gold_titles):
            skipped_title_overlap += 1
            continue

        test_raw.append(ex)
        test_gold_titles.update(titles)
        if len(test_raw) == test_size:
            break

    print(f"[-] Extracted Test split: {len(test_raw)} items, {len(test_gold_titles)} unique gold document titles.")
    print(f"[-] Skipped {skipped_title_overlap} questions due to gold title overlap with Dev.")

    # Invariant: Disjoint sets
    overlap = dev_gold_titles.intersection(test_gold_titles)
    if overlap:
        raise ValueError(f"FATAL: Dev and Test share {len(overlap)} gold supporting document titles: {overlap}")

    # 3. Process items into structured format with deterministic chunk mappings
    def process_split(raw_items: List[Dict[str, Any]], split_name: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        processed_items = []
        total_gold_facts = 0
        covered_gold_facts = 0
        fully_covered_questions = 0

        for ex in raw_items:
            q_id = str(ex["id"])
            question = str(ex["question"]).strip()
            answer = str(ex["answer"]).strip()
            q_type = str(ex.get("type", "bridge")).strip().lower()
            q_level = str(ex.get("level", "medium")).strip().lower()

            ctx_titles = ex["context"]["title"]
            ctx_sentences = ex["context"]["sentences"]

            # Build candidate chunks from the 10 context articles
            chunks: List[Dict[str, Any]] = []
            title_to_chunk_id: Dict[str, str] = {}

            for title, sentences in zip(ctx_titles, ctx_sentences):
                chunk_id = sanitize_title_to_chunk_id(title)
                title_to_chunk_id[title] = chunk_id
                body = " ".join(sentences).strip()
                content = f"{title}. {body}" if not body.startswith(title) else body
                chunks.append({
                    "chunk_id": chunk_id,
                    "title": title,
                    "content": content,
                    "sentence_count": len(sentences),
                    "char_count": len(content),
                })

            # Map gold supporting facts to chunk IDs
            gold_titles = ex["supporting_facts"]["title"]
            gold_sent_ids = ex["supporting_facts"]["sent_id"]
            mapped_facts = []
            item_fully_covered = True

            for g_title, g_sent_idx in zip(gold_titles, gold_sent_ids):
                total_gold_facts += 1
                fact_covered = False
                target_sentence = ""

                # Find the sentence in the context
                if g_title in ctx_titles:
                    t_idx = ctx_titles.index(g_title)
                    sents = ctx_sentences[t_idx]
                    if g_sent_idx < len(sents):
                        target_sentence = sents[g_sent_idx].strip()

                # Find which chunks contain the target sentence
                covering_chunks = []
                for chk in chunks:
                    if chk["title"] == g_title and target_sentence and target_sentence in chk["content"]:
                        covering_chunks.append(chk["chunk_id"])

                if covering_chunks:
                    fact_covered = True
                    covered_gold_facts += 1
                else:
                    item_fully_covered = False

                mapped_facts.append({
                    "title": g_title,
                    "sent_id": g_sent_idx,
                    "sentence_text": target_sentence,
                    "covering_chunk_ids": covering_chunks,
                    "is_covered": fact_covered,
                })

            if item_fully_covered:
                fully_covered_questions += 1

            processed_items.append({
                "id": q_id,
                "split": split_name,
                "question": question,
                "answer": answer,
                "type": q_type,
                "level": q_level,
                "gold_supporting_facts": mapped_facts,
                "gold_chunk_ids": sorted(list({cid for f in mapped_facts for cid in f["covering_chunk_ids"]})),
                "context_chunks": chunks,
            })

        fact_coverage_pct = (covered_gold_facts / total_gold_facts) if total_gold_facts else 0.0
        question_coverage_pct = (fully_covered_questions / len(raw_items)) if raw_items else 0.0

        stats = {
            "split": split_name,
            "total_questions": len(raw_items),
            "fully_covered_questions": fully_covered_questions,
            "question_coverage_rate": round(question_coverage_pct, 4),
            "total_gold_facts": total_gold_facts,
            "covered_gold_facts": covered_gold_facts,
            "fact_coverage_rate": round(fact_coverage_pct, 4),
        }

        return processed_items, stats

    print("[-] Processing Dev items and mapping chunk IDs...")
    dev_processed, dev_stats = process_split(dev_raw, "dev")
    print(f"    Dev Question Coverage: {dev_stats['question_coverage_rate'] * 100:.2f}% ({dev_stats['fully_covered_questions']}/{dev_stats['total_questions']})")
    print(f"    Dev Fact Coverage:     {dev_stats['fact_coverage_rate'] * 100:.2f}% ({dev_stats['covered_gold_facts']}/{dev_stats['total_gold_facts']})")

    print("[-] Processing Test items and mapping chunk IDs...")
    test_processed, test_stats = process_split(test_raw, "test")
    print(f"    Test Question Coverage: {test_stats['question_coverage_rate'] * 100:.2f}% ({test_stats['fully_covered_questions']}/{test_stats['total_questions']})")
    print(f"    Test Fact Coverage:     {test_stats['fact_coverage_rate'] * 100:.2f}% ({test_stats['covered_gold_facts']}/{test_stats['total_gold_facts']})")

    # Gate 0 Coverage Floor Invariant Verification (Section 5)
    if dev_stats["question_coverage_rate"] < coverage_floor:
        raise ValueError(f"Gate 0 Failure: Dev question coverage {dev_stats['question_coverage_rate']} < floor {coverage_floor}")
    if test_stats["question_coverage_rate"] < coverage_floor:
        raise ValueError(f"Gate 0 Failure: Test question coverage {test_stats['question_coverage_rate']} < floor {coverage_floor}")

    # Export to JSONL and Manifest
    dev_path = os.path.join(output_dir, "dev_150.jsonl")
    with open(dev_path, "w", encoding="utf-8") as f:
        for item in dev_processed:
            f.write(json.dumps(item) + "\n")

    test_path = os.path.join(output_dir, "test_300.jsonl")
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_processed:
            f.write(json.dumps(item) + "\n")

    manifest = {
        "dataset": "hotpotqa/hotpot_qa",
        "split_source": "validation_distractor",
        "coverage_floor_required": coverage_floor,
        "dev_stats": dev_stats,
        "test_stats": test_stats,
        "split_disjointness": {
            "dev_gold_titles_count": len(dev_gold_titles),
            "test_gold_titles_count": len(test_gold_titles),
            "overlap_count": len(overlap),
            "disjoint_verified": len(overlap) == 0,
        },
        "files": {
            "dev_jsonl": dev_path,
            "test_jsonl": test_path,
        }
    }

    manifest_path = os.path.join(output_dir, "mapping_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[+] Splits successfully prepared and verified:")
    print(f"    Dev:      {dev_path}")
    print(f"    Test:     {test_path}")
    print(f"    Manifest: {manifest_path}")

    return manifest


if __name__ == "__main__":
    prepare_dataset_splits()
