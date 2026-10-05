"""
Dianoia CLI — demo interface for the deliberative walk pipeline.

Single-shot: one question per invocation.
Accepts EITHER --docs <dir> (builds a fresh LTHCM index) OR --index <path>
(pre-built index).

Exit code 0 on success, 1 on error.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dianoia")


def _setup_logging(verbose: bool = False):
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_documents(docs_dir: str) -> List[Dict[str, str]]:
    """
    Load text documents from a directory.

    Supports .txt and .md files. Fails gracefully with a clear message
    on an empty directory or unsupported files.
    """
    docs_path = Path(docs_dir)
    if not docs_path.is_dir():
        print(f"Error: '{docs_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    supported_exts = {".txt", ".md"}
    documents = []

    for fpath in sorted(docs_path.iterdir()):
        if fpath.is_file() and fpath.suffix.lower() in supported_exts:
            try:
                content = fpath.read_text(encoding="utf-8")
                if content.strip():
                    documents.append({
                        "id": fpath.stem,
                        "content": content.strip(),
                    })
            except Exception as e:
                logger.warning("Skipping '%s': %s", fpath, e)

    if not documents:
        print(
            f"Error: no supported documents (.txt, .md) found in '{docs_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    return documents


def _build_fresh_index(
    documents: List[Dict[str, str]],
    embedder,
) -> "LTHCMEngine":
    """
    Build a fresh LTHCMEngine index from a list of documents.
    """
    from lthcm.core.sheaf import SheafContradictionFilter
    from lthcm.engine.pipeline import LTHCMEngine
    from lthcm.graph.build_edges import build_edges

    engine = LTHCMEngine(d_model=embedder.d_model, num_facets=1)

    # Wire NLI model per LTHCM benchmark convention
    engine.sheaf_filter = SheafContradictionFilter(
        d_model=embedder.d_model,
        contradiction_threshold=2.5,
        nli_model_name="cross-encoder/nli-deberta-v3-xsmall",
    ).to(engine.device)

    print(f"Indexing {len(documents)} document(s)...")

    for doc in documents:
        emb = embedder.encode_single(doc["content"])
        facet_vectors = emb.unsqueeze(0)  # [1, d_model] for num_facets=1
        engine.index_document(
            doc_id=doc["id"],
            content=doc["content"],
            facet_vectors=facet_vectors,
        )

    # Build wormhole edges from corpus
    corpus_docs = [{"id": d["id"], "text": d["content"], "title": d["id"]} for d in documents]
    corpus_embs = embedder.encode([d["content"] for d in documents])
    doc_embs = corpus_embs.cpu().numpy() if hasattr(corpus_embs, "cpu") else corpus_embs

    try:
        edges = build_edges(
            docs=corpus_docs,
            doc_embeddings=doc_embs,
        )
    except Exception as e:
        print(f"Error: edge building failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not edges:
        print(
            f"Error: edge building produced 0 edges for {len(documents)} document(s). "
            "Deliberative walk requires an edge structure to navigate.",
            file=sys.stderr,
        )
        sys.exit(1)

    edge_count = 0
    for edge in edges:
        src_id, dst_id, rel_type, perm = edge
        if (
            src_id in engine.wormhole.node_embeddings
            and dst_id in engine.wormhole.node_embeddings
        ):
            engine.wormhole.add_wormhole(
                source_id=src_id,
                target_id=dst_id,
                relation_type=rel_type,
                permeability=perm,
            )
            edge_count += 1

    if edge_count == 0:
        print(
            "Error: no valid edges could be added to the wormhole index.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Built {edge_count} wormhole edge(s).")
    return engine


def _save_index(engine: "LTHCMEngine", path: str):
    """Serialize engine state to disk."""
    with open(path, "wb") as f:
        pickle.dump(engine, f)
    print(f"Index saved to {path}")


def _load_index(path: str, allow_trusted_pickle: bool = False) -> "LTHCMEngine":
    """
    Load a pre-built LTHCMEngine index from disk.

    Fails gracefully on corrupt, mismatched, or unauthorized untrusted index files.
    """
    if not allow_trusted_pickle and not os.environ.get("DIANOIA_ALLOW_TRUSTED_PICKLE"):
        print(
            f"Error: untrusted pickle index loading refused for '{path}'. "
            "Pass --allow-trusted-pickle or set DIANOIA_ALLOW_TRUSTED_PICKLE=1 to authorize.",
            file=sys.stderr,
        )
        sys.exit(1)

    index_path = Path(path)
    if not index_path.is_file():
        print(f"Error: index file '{path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        if index_path.stat().st_size > 2 * 1024 * 1024 * 1024:
            print(f"Error: index file '{path}' exceeds 2GB maximum limit.", file=sys.stderr)
            sys.exit(1)
    except Exception:
        pass

    try:
        with open(path, "rb") as f:
            engine = pickle.load(f)
    except Exception as e:
        print(
            f"Error: failed to load index from '{path}'. "
            f"File may be corrupt or version-mismatched: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    from lthcm.engine.pipeline import LTHCMEngine
    if not isinstance(engine, LTHCMEngine):
        print(
            f"Error: loaded object is not an LTHCMEngine "
            f"(got {type(engine).__name__}).",
            file=sys.stderr,
        )
        sys.exit(1)

    if not hasattr(engine, "sheaf_filter") or engine.sheaf_filter is None:
        print(
            f"Error: index '{path}' is missing a sheaf_filter. "
            "Deliberative walk requires a SheafContradictionFilter for answer verification.",
            file=sys.stderr,
        )
        sys.exit(1)

    engine.sheaf_filter.nli_model_name = "cross-encoder/nli-deberta-v3-xsmall"
    engine.sheaf_filter.threshold = 2.5
    engine.sheaf_filter._nli_model = None
    nli_msg = (
        f"Applied verified NLI defaults to index: "
        f"model='{engine.sheaf_filter.nli_model_name}', "
        f"threshold={engine.sheaf_filter.threshold}"
    )
    print(nli_msg)
    logger.info(nli_msg)

    # Verify loaded index has edges
    edge_count = 0
    if hasattr(engine, "wormhole") and hasattr(engine.wormhole, "graph"):
        if hasattr(engine.wormhole.graph, "number_of_edges"):
            edge_count = engine.wormhole.graph.number_of_edges()
        elif hasattr(engine.wormhole.graph, "edges"):
            edge_count = len(engine.wormhole.graph.edges)
    if edge_count == 0:
        print(
            f"Error: index '{path}' contains 0 graph edges. "
            "Deliberative walk requires an edge structure to navigate.",
            file=sys.stderr,
        )
        sys.exit(1)

    return engine


def _print_walk_result(result: Dict[str, Any]):
    """Pretty-print the walk result step by step."""
    print("\n" + "=" * 70)
    print(" DIANOIA — DELIBERATIVE WALK")
    print("=" * 70)

    # Initial walk path (if re-walked)
    if result.get("initial_walk_path"):
        print("\n--- Initial Walk Path (before conflict resolution) ---")
        for i, step in enumerate(result["initial_walk_path"], 1):
            status = " [PARSE FAILED]" if step.get("parse_failed") else ""
            print(f"  Step {i}: {step['action']}{status}")
            print(f"    Node: {step['node']}")
            if step.get("reason"):
                print(f"    Reason: {step['reason']}")

    # Walk path
    header = "Re-Walk Path" if result.get("initial_walk_path") else "Walk Path"
    print(f"\n--- {header} ---")
    for i, step in enumerate(result["walk_path"], 1):
        status = " [PARSE FAILED]" if step.get("parse_failed") else ""
        print(f"  Step {i}: {step['action']}{status}")
        print(f"    Node: {step['node']}")
        if step.get("reason"):
            print(f"    Reason: {step['reason']}")

    # Verification report
    print("\n--- Verification Report ---")
    v = result["verification"]
    print(f"  {v['report']}")
    if v.get("rewalked"):
        print("  (Re-walk was triggered by conflicts)")
    if v.get("rewalk_failed"):
        print(f"  (Re-walk attempt failed: {v.get('rewalk_error', 'unspecified error')})")
    if v.get("conflict_detail"):
        print(f"  Conflict Detail:\n    {v['conflict_detail']}")

    # Final answer
    print("\n--- Final Answer ---")
    print(result["final_answer"])

    # Metadata
    print("\n--- Metadata ---")
    print(f"  Model: {result['model']}")
    print(f"  Fallback used: {result['fallback_used']}")
    print(f"  Steps used: {result['steps_used']}")
    print(f"  Seed nodes: {result['seeded_nodes']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        prog="dianoia",
        description=(
            "Dianoia — deliberative cognition over LTHCM topological memory. "
            "LTHCM remembers, Dianoia thinks."
        ),
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--docs",
        metavar="DIR",
        help="Directory of text documents to build a fresh LTHCM index from.",
    )
    source_group.add_argument(
        "--index",
        metavar="PATH",
        help="Path to a pre-built LTHCM index file.",
    )

    parser.add_argument(
        "question",
        help="The question to answer.",
    )

    parser.add_argument(
        "--save-index",
        metavar="PATH",
        help="Save the built index to this path (only with --docs).",
    )
    parser.add_argument(
        "--llm-url",
        default="http://localhost:8080/v1",
        help="Base URL for the llama.cpp OpenAI-compatible endpoint (default: http://localhost:8080/v1).",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-7b-instruct",
        help="Model name for the llama.cpp server (default: qwen2.5-7b-instruct).",
    )
    parser.add_argument(
        "--fallback-model",
        default="llama-3.1-8b-instruct",
        help="Fallback model name (default: llama-3.1-8b-instruct).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum walk steps (default: 8).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of seed nodes from retrieval (default: 5).",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Path for JSONL walk log (default: ~/.dianoia/walks.jsonl).",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for text embedding (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--allow-trusted-pickle",
        action="store_true",
        default=False,
        help="Explicitly permit loading index using local trusted pickle archive.",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    # Ensure LTHCM is importable (submodule or monorepo sibling must be on sys.path)
    lthcm_dir = Path(__file__).resolve().parent.parent / "LTHCM"
    if not lthcm_dir.is_dir():
        lthcm_dir = Path(__file__).resolve().parent.parent.parent / "LTHCM"
    lthcm_path = str(lthcm_dir)
    if lthcm_path not in sys.path:
        sys.path.insert(0, lthcm_path)

    # Import after path setup
    from dianoia.embedding import TextEmbedder
    from dianoia.walk import WalkController

    # Initialize embedder
    print("Loading embedding model...")
    embedder = TextEmbedder(model_name=args.embedding_model)

    # Build or load the LTHCM engine
    if args.docs:
        documents = _load_documents(args.docs)
        engine = _build_fresh_index(documents, embedder)
        if args.save_index:
            _save_index(engine, args.save_index)
    else:
        engine = _load_index(args.index, allow_trusted_pickle=args.allow_trusted_pickle)

    # Initialize the walk controller
    controller = WalkController(
        engine=engine,
        embedder=embedder,
        llm_base_url=args.llm_url,
        model_name=args.model,
        max_steps=args.max_steps,
        top_k=args.top_k,
        fallback_model_name=args.fallback_model,
        log_path=args.log_path,
    )

    # Run the walk
    print(f"\nQuestion: {args.question}")
    print("Starting deliberative walk...\n")

    try:
        result = controller.run(args.question)
    except Exception as e:
        logger.error("Walk failed: %s", e, exc_info=True)
        print(f"\nError: walk failed: {e}", file=sys.stderr)
        sys.exit(1)

    _print_walk_result(result)
    sys.exit(0)


if __name__ == "__main__":
    main()
