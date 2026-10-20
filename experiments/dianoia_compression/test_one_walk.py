"""Quick test of LTHCM indexing and WalkController on 1 Dev question."""

import os
import sys
import time
import json
import torch

from dianoia.embedding import TextEmbedder
from dianoia.walk import WalkController
from lthcm.engine.pipeline import LTHCMEngine
from lthcm.graph.build_edges import build_edges

def main():
    dev_path = "experiments/dianoia_compression/data/dev_150.jsonl"
    with open(dev_path, "r", encoding="utf-8") as f:
        first_item = json.loads(f.readline())

    print(f"[-] Question: {first_item['question']}")
    print(f"[-] Gold answer: {first_item['answer']}")
    print(f"[-] Gold chunk IDs: {first_item['gold_chunk_ids']}")

    # 1. Initialize embedder
    print("[-] Initializing TextEmbedder...")
    embedder = TextEmbedder()

    # 2. Build LTHCM engine for this question's 10 chunks (as a fast unit test)
    engine = LTHCMEngine(d_model=embedder.d_model, num_facets=1)
    
    docs = []
    for c in first_item["context_chunks"]:
        docs.append({"id": c["chunk_id"], "content": c["content"], "title": c["title"]})

    print(f"[-] Indexing {len(docs)} documents into LTHCM...")
    for d in docs:
        emb = embedder.encode_single(d["content"])
        facet_vectors = emb.unsqueeze(0)
        engine.index_document(
            doc_id=d["id"],
            content=d["content"],
            facet_vectors=facet_vectors,
            timestamp=time.time(),
        )

    # Build edges
    corpus_docs = [{"id": d["id"], "text": d["content"], "title": d["title"]} for d in docs]
    corpus_embs = embedder.encode([d["content"] for d in docs])
    doc_embs = corpus_embs.cpu().numpy() if hasattr(corpus_embs, "cpu") else corpus_embs
    edges = build_edges(docs=corpus_docs, doc_embeddings=doc_embs)
    print(f"[-] Built {len(edges)} wormhole edges.")

    for src_id, dst_id, rel_type, perm in edges:
        if src_id in engine.wormhole.node_embeddings and dst_id in engine.wormhole.node_embeddings:
            engine.wormhole.add_wormhole(
                source_id=src_id,
                target_id=dst_id,
                relation_type=rel_type,
                permeability=perm,
            )

    # 3. Initialize WalkController
    controller = WalkController(
        engine=engine,
        embedder=embedder,
        llm_base_url="http://127.0.0.1:8080/v1",
        model_name="qwen2.5-14b-instruct",
        fallback_model_name=None,
        max_steps=6,
        top_k=3,
        temperature=0.2,
        top_p=0.9,
        use_stochastic_resonance=False, # Use superposition for predictable retrieval
    )

    print("[-] Running walk...")
    t0 = time.time()
    res = controller.walk(first_item["question"])
    t1 = time.time()

    print(f"[-] Walk completed in {t1 - t0:.2f}s")
    print(f"[-] Steps used: {res['steps_used']}")
    print(f"[-] Seeded nodes: {res['seeded_nodes']}")
    print(f"[-] Visited passages ({len(res['visited_passages'])}):")
    for p in res["visited_passages"]:
        print(f"    - [{p['node_id']}]: {p['content'][:80]}...")
    print(f"[-] Synthesized answer: {res['answer']}")

if __name__ == "__main__":
    main()
