"""Frozen walk generator for Dianoia Question-Conditioned Compression Experiment.

Implements Phase 1 of metis-dianoia-question-conditioned-compression-experiment-spec-2026-10-20-531c7bb-29a18f8e.txt:
1. Builds frozen LTHCM index snapshot over all candidate context chunks in the split.
2. Computes and records deterministic corpus_snapshot_hash.
3. Runs WalkController(model='qwen2.5-14b-instruct', max_steps=6, top_k=5, temp=0.2, top_p=0.9, skip_synthesis=True).
4. Records per-question visited passages (node IDs, content, order), seeded nodes, path, latency, tokens,
   and computes the retrieval ceiling (fraction of retrievable gold supporting facts).
5. Exports frozen manifest to visited_passages_{split}.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from dianoia.embedding import TextEmbedder
from dianoia.walk import WalkController
from lthcm.engine.pipeline import LTHCMEngine
from lthcm.graph.build_edges import build_edges


def compute_corpus_hash(chunks: Dict[str, Dict[str, Any]]) -> str:
    """Compute deterministic SHA-256 hash over all chunks in corpus."""
    sorted_items = sorted(chunks.items(), key=lambda x: x[0])
    hasher = hashlib.sha256()
    for cid, c in sorted_items:
        hasher.update(cid.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(c["content"].encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def build_split_lthcm_index(
    chunks: Dict[str, Dict[str, Any]],
    embedder: TextEmbedder,
    verbose: bool = True,
) -> tuple[LTHCMEngine, int, str]:
    """Index all candidate chunks and construct wormhole graph edges."""
    corpus_hash = compute_corpus_hash(chunks)
    if verbose:
        print(f"[-] Building LTHCM index for {len(chunks)} chunks (Corpus Hash: {corpus_hash[:16]}...)...")

    engine = LTHCMEngine(d_model=embedder.d_model, num_facets=1)
    docs = [{"id": cid, "content": c["content"], "title": c["title"]} for cid, c in chunks.items()]

    t0 = time.time()
    for d in docs:
        emb = embedder.encode_single(d["content"])
        facet_vectors = emb.unsqueeze(0)
        engine.index_document(
            doc_id=d["id"],
            content=d["content"],
            facet_vectors=facet_vectors,
            timestamp=100.0,
        )
    t_index = time.time() - t0
    if verbose:
        print(f"[-] Ingested {len(docs)} documents into superposition store in {t_index:.2f}s.")

    # Build wormhole edges
    t0 = time.time()
    corpus_docs = [{"id": d["id"], "text": d["content"], "title": d["title"]} for d in docs]
    corpus_embs = embedder.encode([d["content"] for d in docs])
    doc_embs = corpus_embs.cpu().numpy() if hasattr(corpus_embs, "cpu") else corpus_embs
    edges = build_edges(docs=corpus_docs, doc_embeddings=doc_embs)
    t_edges = time.time() - t0

    edge_count = 0
    for src_id, dst_id, rel_type, perm in edges:
        if src_id in engine.wormhole.node_embeddings and dst_id in engine.wormhole.node_embeddings:
            engine.wormhole.add_wormhole(
                source_id=src_id,
                target_id=dst_id,
                relation_type=rel_type,
                permeability=perm,
            )
            edge_count += 1

    if verbose:
        print(f"[-] Built & added {edge_count} wormhole edges in {t_edges:.2f}s.")

    return engine, edge_count, corpus_hash


def compute_item_retrieval_ceiling(
    gold_supporting_facts: List[Dict[str, Any]],
    visited_chunk_ids: Set[str],
) -> Dict[str, Any]:
    """Compute whether all gold facts are covered by visited chunks under one-to-many mapping."""
    total_facts = len(gold_supporting_facts)
    covered_facts = 0
    uncovered = []

    for fact in gold_supporting_facts:
        covering = set(fact.get("covering_chunk_ids", []))
        # A fact is covered if at least one of its covering chunks was visited
        if covering.intersection(visited_chunk_ids):
            covered_facts += 1
        else:
            uncovered.append({
                "title": fact.get("title"),
                "sent_id": fact.get("sent_id"),
                "sentence_text": fact.get("sentence_text"),
                "covering_chunk_ids": list(covering),
            })

    is_retrievable = (covered_facts == total_facts and total_facts > 0)
    return {
        "retrievable": is_retrievable,
        "covered_facts_count": covered_facts,
        "total_facts_count": total_facts,
        "fact_recall": covered_facts / total_facts if total_facts > 0 else 0.0,
        "uncovered_facts": uncovered,
    }


def run_frozen_walks(
    split: str = "dev",
    data_dir: str = "experiments/dianoia_compression/data",
    model_name: str = "qwen2.5-14b-instruct",
    llm_base_url: str = "http://127.0.0.1:8080/v1",
    max_steps: int = 6,
    top_k: int = 5,
    temperature: float = 0.2,
    top_p: float = 0.9,
    limit: int | None = None,
) -> Dict[str, Any]:
    """Execute frozen walks for a given split and export results."""
    split_file = Path(data_dir) / f"{split}_{150 if split == 'dev' else 300}.jsonl"
    if not split_file.exists():
        # Fallback to any file matching {split}_*.jsonl
        candidates = list(Path(data_dir).glob(f"{split}_*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No split file found for {split} in {data_dir}")
        split_file = candidates[0]

    out_file = Path(data_dir) / f"visited_passages_{split}.jsonl"
    manifest_file = Path(data_dir) / f"walk_manifest_{split}.json"

    print(f"[-] Loading split items from {split_file}...")
    with open(split_file, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    if limit is not None:
        items = items[:limit]
        print(f"[-] Limiting run to first {len(items)} items.")

    # Collect unique chunks
    all_chunks: Dict[str, Dict[str, Any]] = {}
    for it in items:
        for c in it["context_chunks"]:
            cid = c["chunk_id"]
            if cid not in all_chunks:
                all_chunks[cid] = c

    # Initialize embedder & build index
    embedder = TextEmbedder(device=torch.device("cpu"))
    engine, edge_count, corpus_hash = build_split_lthcm_index(all_chunks, embedder)

    # Initialize WalkController
    controller = WalkController(
        engine=engine,
        embedder=embedder,
        llm_base_url=llm_base_url,
        model_name=model_name,
        fallback_model_name=None,
        max_steps=max_steps,
        top_k=top_k,
        temperature=temperature,
        top_p=top_p,
        use_stochastic_resonance=False,
    )

    print(f"\n[-] Starting frozen walks on {len(items)} questions ({split} split)...")
    completed_records = {}
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        completed_records[rec["id"]] = rec
                    except json.JSONDecodeError:
                        pass
        if completed_records:
            print(f"[-] Found {len(completed_records)} already completed records. Resuming...")

    records = []
    total_retrievable = 0
    total_facts = 0
    total_covered_facts = 0
    total_walk_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    t_start_all = time.time()

    with open(out_file, "a", encoding="utf-8") as out_f:
        for idx, it in enumerate(items):
            q_id = it["id"]
            question = it["question"]
            gold_answer = it["answer"]
            gold_facts = it.get("gold_supporting_facts", [])
            gold_chunk_ids = it.get("gold_chunk_ids", [])

            if q_id in completed_records:
                rec = completed_records[q_id]
                records.append(rec)
                ceil_res = rec["retrieval_ceiling"]
                if ceil_res["retrievable"]:
                    total_retrievable += 1
                total_facts += ceil_res["total_facts_count"]
                total_covered_facts += ceil_res["covered_facts_count"]
                for k in total_walk_tokens:
                    total_walk_tokens[k] += rec.get("walk_tokens", {}).get(k, 0)
                continue

            t0 = time.time()
            # Deliberative walk without answer synthesis (synthesis is unconfounded across arms)
            walk_res = controller.walk(question, skip_synthesis=True)
            latency_s = time.time() - t0

            visited_passages = walk_res["visited_passages"]
            visited_chunk_ids = {p["node_id"] for p in visited_passages}
            ceil_res = compute_item_retrieval_ceiling(gold_facts, visited_chunk_ids)

            if ceil_res["retrievable"]:
                total_retrievable += 1
            total_facts += ceil_res["total_facts_count"]
            total_covered_facts += ceil_res["covered_facts_count"]

            tokens = walk_res.get("tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            for k in total_walk_tokens:
                total_walk_tokens[k] += tokens.get(k, 0)

            rec = {
                "id": q_id,
                "split": split,
                "question": question,
                "gold_answer": gold_answer,
                "gold_supporting_facts": gold_facts,
                "gold_chunk_ids": gold_chunk_ids,
                "seeded_nodes": walk_res["seeded_nodes"],
                "visited_passages": visited_passages,
                "walk_path": walk_res["path"],
                "steps_used": walk_res["steps_used"],
                "walk_latency_s": round(latency_s, 4),
                "walk_tokens": tokens,
                "retrieval_ceiling": ceil_res,
            }
            records.append(rec)
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

            if (idx + 1) % 10 == 0 or (idx + 1) == len(items):
                elapsed = time.time() - t_start_all
                rate = (idx + 1) / elapsed
                rem = (len(items) - (idx + 1)) / rate if rate > 0 else 0
                pct_ret = (total_retrievable / (idx + 1)) * 100
                print(
                    f"[{idx + 1:3d}/{len(items)}] Retrievable: {pct_ret:.1f}% | "
                    f"Avg Latency: {elapsed / (idx + 1):.2f}s | "
                    f"Elapsed: {elapsed:.1f}s | Est Rem: {rem:.1f}s"
                )

    total_time = time.time() - t_start_all
    retrieval_ceiling_pct = (total_retrievable / len(items)) * 100 if items else 0.0
    fact_coverage_pct = (total_covered_facts / total_facts) * 100 if total_facts > 0 else 0.0

    manifest = {
        "split": split,
        "num_items": len(items),
        "corpus_snapshot_hash": corpus_hash,
        "total_chunks": len(all_chunks),
        "wormhole_edges": edge_count,
        "controller_model": model_name,
        "max_steps": max_steps,
        "top_k": top_k,
        "temperature": temperature,
        "top_p": top_p,
        "total_wall_time_s": round(total_time, 2),
        "mean_item_latency_s": round(total_time / len(items), 4) if items else 0.0,
        "total_walk_tokens": total_walk_tokens,
        "retrieval_ceiling_questions": total_retrievable,
        "retrieval_ceiling_pct": round(retrieval_ceiling_pct, 2),
        "total_facts": total_facts,
        "total_covered_facts": total_covered_facts,
        "fact_coverage_pct": round(fact_coverage_pct, 2),
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[+] Completed frozen walks for {split}!")
    print(f"    - Output: {out_file}")
    print(f"    - Manifest: {manifest_file}")
    print(f"    - Retrieval Ceiling (full gold facts covered): {retrieval_ceiling_pct:.2f}% ({total_retrievable}/{len(items)})")
    print(f"    - Fact Recall Ceiling: {fact_coverage_pct:.2f}% ({total_covered_facts}/{total_facts})")
    print(f"    - Total Walk Tokens: {total_walk_tokens['total_tokens']:,}")
    print(f"    - Total Wall Time: {total_time:.1f}s (mean {total_time/len(items):.2f}s/item)")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate frozen graph walks.")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"], help="Dataset split to walk.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items for test/dry-run.")
    args = parser.parse_args()

    run_frozen_walks(split=args.split, limit=args.limit)
