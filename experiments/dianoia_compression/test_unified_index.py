"""Test unified Dev LTHCM index building and walk."""

import json
import os
import sys
import time
import hashlib
import pickle
import torch

from dianoia.embedding import TextEmbedder
from dianoia.walk import WalkController
from lthcm.engine.pipeline import LTHCMEngine
from lthcm.graph.build_edges import build_edges

def main():
    dev_path = "experiments/dianoia_compression/data/dev_150.jsonl"
    with open(dev_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    print(f"[-] Loaded {len(items)} Dev items.")

    # Collect all unique context chunks
    all_chunks = {}
    for item in items:
        for c in item["context_chunks"]:
            cid = c["chunk_id"]
            if cid not in all_chunks:
                all_chunks[cid] = c

    print(f"[-] Total unique chunks across Dev items: {len(all_chunks)}")

    # Initialize embedder
    embedder = TextEmbedder()

    # Build LTHCM engine
    engine = LTHCMEngine(d_model=embedder.d_model, num_facets=1)
    
    docs = [{"id": cid, "content": c["content"], "title": c["title"]} for cid, c in all_chunks.items()]

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
    t1 = time.time()
    print(f"[-] Indexed {len(docs)} documents in {t1 - t0:.2f}s.")

    # Build edges
    t0 = time.time()
    corpus_docs = [{"id": d["id"], "text": d["content"], "title": d["title"]} for d in docs]
    corpus_embs = embedder.encode([d["content"] for d in docs])
    doc_embs = corpus_embs.cpu().numpy() if hasattr(corpus_embs, "cpu") else corpus_embs
    edges = build_edges(docs=corpus_docs, doc_embeddings=doc_embs)
    t1 = time.time()
    print(f"[-] Built {len(edges)} edges in {t1 - t0:.2f}s.")

    added_edges = 0
    for src_id, dst_id, rel_type, perm in edges:
        if src_id in engine.wormhole.node_embeddings and dst_id in engine.wormhole.node_embeddings:
            engine.wormhole.add_wormhole(
                source_id=src_id,
                target_id=dst_id,
                relation_type=rel_type,
                permeability=perm,
            )
            added_edges += 1
    print(f"[-] Added {added_edges} wormhole edges to engine.")

    # Test WalkController on first 2 questions
    controller = WalkController(
        engine=engine,
        embedder=embedder,
        llm_base_url="http://127.0.0.1:8080/v1",
        model_name="qwen2.5-14b-instruct",
        fallback_model_name=None,
        max_steps=6,
        top_k=5,
        temperature=0.2,
        top_p=0.9,
        use_stochastic_resonance=False,
    )

    for i in range(2):
        q_item = items[i]
        print(f"\n[-] Testing Question {i+1}: {q_item['question']}")
        print(f"    Gold chunk IDs: {q_item['gold_chunk_ids']}")
        t_start = time.time()
        res = controller.walk(q_item["question"])
        t_end = time.time()
        visited_ids = [p["node_id"] for p in res["visited_passages"]]
        gold_covered = [gid in visited_ids for gid in q_item["gold_chunk_ids"]]
        print(f"    Walk time: {t_end - t_start:.2f}s | Steps: {res['steps_used']}")
        print(f"    Seeded: {res['seeded_nodes']}")
        print(f"    Visited ({len(visited_ids)}): {visited_ids}")
        print(f"    Gold covered: {gold_covered} ({sum(gold_covered)}/{len(gold_covered)})")
        print(f"    Answer: {res['answer'][:120]}...")

if __name__ == "__main__":
    main()
