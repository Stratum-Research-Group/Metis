"""
Unified Deterministic Retriever Interface for LTHCM
===================================================
Provides a shared, deterministic retriever interface for both production library callers
and benchmark evaluation suites:
  - index(corpus, embeddings=None, edges=None)
  - search(query, top_k=5, config=None) -> RetrievalResult

Guarantees exact parity between the public API and benchmark evaluation paths.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
import numpy as np
import torch
import torch.nn.functional as F

from lthcm.core.wormhole import WormholeGraphIndex
from lthcm.graph.build_edges import (
    build_edges,
    generate_random_edges,
    generate_oracle_edges,
)
from lthcm.graph.learned_edge_scorer import LearnedEdgeScorer, FixedWeightEdgeScorer


@dataclass
class RetrievalConfig:
    """
    Configuration for deterministic LTHCM retrieval.
    All experimental modules default to read-only or explicit flags.
    """
    mode: str = "graph_mined"  # "flat", "flat_2step", "graph_mined", "graph_learned", "graph_fixed_weights", "graph_random", "graph_oracle"
    top_k: int = 5
    graph_method: str = "title_mention+ner"
    max_hops: int = 2
    hop_decay: float = 0.75
    min_permeability: float = 0.35
    bridge_bonus_weight: float = 0.35
    max_out: int = 4
    use_cross_encoder_rerank: bool = False
    reranker_top_k: int = 5
    reranker_model: Optional[Any] = None
    enforce_causality: bool = False
    enforce_sheaf_verification: bool = False
    use_stochastic_resonance: bool = False
    use_fisher_rao: bool = False
    update_online_memory: bool = False  # Strictly False by default for read-only inference
    audit_completeness: bool = False
    seed: int = 42


@dataclass
class RetrievalResult:
    """
    Deterministic ranked retrieval output with stage traces and metrics.
    """
    ranked_doc_ids: List[str]
    ranked_titles: List[str]
    scores: List[float]
    results: List[Dict[str, Any]]
    stage_traces: Dict[str, Any] = field(default_factory=dict)


class LTHCMRetriever:
    """
    Unified LTHCM Retriever.
    Provides identical ranking and candidate generation for library users and benchmarks.
    """

    def __init__(
        self,
        encoder: Optional[Any] = None,
        d_model: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        engine: Optional[Any] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = encoder
        self.engine = engine

        if d_model is not None:
            self.d_model = d_model
        elif encoder is not None and hasattr(encoder, "get_sentence_embedding_dimension"):
            self.d_model = encoder.get_sentence_embedding_dimension()
        elif encoder is not None and hasattr(encoder, "get_embedding_dimension"):
            self.d_model = encoder.get_embedding_dimension()
        elif engine is not None and hasattr(engine, "d_model"):
            self.d_model = engine.d_model
        else:
            self.d_model = 384  # Default MiniLM dimension

        self.wormhole = WormholeGraphIndex(d_model=self.d_model)
        self.docs: List[Dict[str, Any]] = []
        self.doc_map: Dict[str, Dict[str, Any]] = {}
        self.doc_embeddings: Optional[np.ndarray] = None
        self.doc_titles: List[str] = []
        self.edges: List[Tuple[str, str, str, float]] = []

    def index(
        self,
        corpus: List[Dict[str, Any]],
        embeddings: Optional[Union[np.ndarray, torch.Tensor]] = None,
        edges: Optional[List[Tuple[str, str, str, float]]] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        """
        Index a corpus of documents with text, titles, and optional graph edges.

        Args:
            corpus: List of dicts, each with 'id' (or 'doc_id'), 'text' (or 'paragraph'/'content'), and optional 'title'.
            embeddings: Optional precomputed embeddings [N, D]. If None and encoder available, encodes corpus text.
            edges: Optional precomputed or mined edges (src, dst, rel, permeability).
            config: Optional RetrievalConfig for edge mining.
        """
        self.docs = []
        self.doc_map = {}
        self.doc_titles = []

        for idx, d in enumerate(corpus):
            did = str(d.get("id", d.get("doc_id", f"doc_{idx}")))
            title = str(d.get("title", did)).strip()
            text = str(d.get("text", d.get("paragraph", d.get("content", "")))).strip()
            item = {"id": did, "title": title, "text": text}
            self.docs.append(item)
            self.doc_map[did] = item
            self.doc_titles.append(title)

        n_docs = len(self.docs)
        if n_docs == 0:
            return

        # Embeddings
        if embeddings is not None:
            if isinstance(embeddings, torch.Tensor):
                self.doc_embeddings = embeddings.detach().cpu().numpy().astype("float32")
            else:
                self.doc_embeddings = np.asarray(embeddings, dtype="float32")
        elif self.encoder is not None:
            texts = [f"{d['title']}. {d['text']}".strip() for d in self.docs]
            self.doc_embeddings = self.encoder.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).astype("float32")
        else:
            # Fallback zero/identity embedding
            self.doc_embeddings = np.zeros((n_docs, self.d_model), dtype="float32")

        # Normalize embeddings
        norms = np.linalg.norm(self.doc_embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self.doc_embeddings = self.doc_embeddings / norms

        # Build wormhole index
        self.wormhole = WormholeGraphIndex(d_model=self.d_model)
        for i, d in enumerate(self.docs):
            emb_t = torch.from_numpy(self.doc_embeddings[i])
            self.wormhole.add_node(d["id"], emb_t, d["text"], cluster=d.get("title", "default"))

        # Edges
        cfg = config or RetrievalConfig()
        if edges is not None:
            self.edges = edges
        elif cfg.mode == "graph_mined":
            self.edges = build_edges(self.docs, method=cfg.graph_method, max_out=cfg.max_out)
        elif cfg.mode == "graph_learned":
            self.edges = build_edges(self.docs, method="learned", max_out=cfg.max_out, doc_embeddings=self.doc_embeddings)
        elif cfg.mode == "graph_fixed_weights":
            self.edges = build_edges(self.docs, method="fixed_weights", max_out=cfg.max_out, doc_embeddings=self.doc_embeddings)
        else:
            self.edges = []

        for src, dst, rel, perm in self.edges:
            self.wormhole.add_wormhole(src, dst, rel, permeability=perm)

    def search(
        self,
        query: Union[str, torch.Tensor, np.ndarray],
        top_k: Optional[int] = None,
        config: Optional[RetrievalConfig] = None,
        gold_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> RetrievalResult:
        """
        Execute deterministic search across the indexed corpus.

        Args:
            query: Query text string or query vector.
            top_k: Number of items to return (overrides config.top_k).
            config: RetrievalConfig specifying retrieval arm and modules.
            gold_pairs: Optional gold pairs strictly for Arm 4 (oracle upper bound).
        """
        cfg = config or RetrievalConfig()
        k = top_k if top_k is not None else cfg.top_k
        t0 = time.perf_counter()
        traces = {"mode": cfg.mode, "top_k": k}

        n_docs = len(self.docs)
        if n_docs == 0:
            return RetrievalResult(ranked_doc_ids=[], ranked_titles=[], scores=[], results=[], stage_traces=traces)

        # 1. Encode query vector
        query_str = query if isinstance(query, str) else ""
        if isinstance(query, str):
            if self.encoder is None:
                raise ValueError("Retriever has no encoder configured to encode text query.")
            q_emb = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].astype("float32")
        elif isinstance(query, torch.Tensor):
            q_emb = query.detach().cpu().numpy().flatten().astype("float32")
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 1e-12:
                q_emb = q_emb / q_norm
        else:
            q_emb = np.asarray(query, dtype="float32").flatten()
            q_norm = np.linalg.norm(q_emb)
            if q_norm > 1e-12:
                q_emb = q_emb / q_norm

        # 2. Stage 1: Base Inner-Product Similarity
        base_sims = np.dot(self.doc_embeddings, q_emb)
        ranked_seed_indices = np.argsort(base_sims, kind="stable")[::-1]

        traces["stage1_latency_s"] = time.perf_counter() - t0
        traces["top_seed_id"] = self.docs[ranked_seed_indices[0]]["id"]
        traces["top_seed_sim"] = float(base_sims[ranked_seed_indices[0]])

        # 3. Handle Arm Execution Modes
        if cfg.mode in {"flat", "flat_1shot"}:
            final_indices = ranked_seed_indices
            final_scores = base_sims[final_indices]

        elif cfg.mode == "flat_2step":
            # 2-step iterative retrieval: top-1 is preserved; second hop queried with top-1 text appended
            top1_idx = ranked_seed_indices[0]
            if self.encoder is not None and query_str:
                step2_query = f"{query_str} {self.docs[top1_idx]['text']}"
                q2_emb = self.encoder.encode([step2_query], normalize_embeddings=True, show_progress_bar=False)[0].astype("float32")
                sims_2step = np.dot(self.doc_embeddings, q2_emb)
                sims_2step_masked = sims_2step.copy()
                sims_2step_masked[top1_idx] = -1e9
                step2_indices = np.argsort(sims_2step_masked, kind="stable")[::-1]
                final_indices = [top1_idx] + [i for i in step2_indices if i != top1_idx]
                final_scores = [float(base_sims[top1_idx])] + [float(sims_2step[i]) for i in final_indices[1:]]
            else:
                final_indices = ranked_seed_indices
                final_scores = base_sims[final_indices]

        elif cfg.mode in {"graph_mined", "graph_learned", "graph_fixed_weights", "graph_random", "graph_oracle"}:
            # Build mode-specific edges if needed
            active_edges = self.edges
            if cfg.mode == "graph_random":
                active_edges = generate_random_edges(self.docs, self.edges, seed=cfg.seed)
            elif cfg.mode == "graph_oracle":
                active_edges = generate_oracle_edges(gold_pairs or [], permeability=1.0)
            elif cfg.mode == "graph_learned" and not any(r.startswith("learned_") for _, _, r, _ in self.edges):
                active_edges = build_edges(self.docs, method="learned", max_out=cfg.max_out, doc_embeddings=self.doc_embeddings)
            elif cfg.mode == "graph_fixed_weights" and not any(r.startswith("fixed_") for _, _, r, _ in self.edges):
                active_edges = build_edges(self.docs, method="fixed_weights", max_out=cfg.max_out, doc_embeddings=self.doc_embeddings)

            # Build temporary index if active edges differ from indexed edges
            if active_edges != self.edges:
                wh_idx = WormholeGraphIndex(d_model=self.d_model)
                for i, d in enumerate(self.docs):
                    emb_t = torch.from_numpy(self.doc_embeddings[i])
                    wh_idx.add_node(d["id"], emb_t, d["text"])
                for src, dst, rel, perm in active_edges:
                    wh_idx.add_wormhole(src, dst, rel, permeability=perm)
            else:
                wh_idx = self.wormhole

            # Wormhole bridge traversal from top seed
            top_seed_idx = ranked_seed_indices[0]
            top_seed_id = self.docs[top_seed_idx]["id"]
            top_seed_sim = float(base_sims[top_seed_idx])

            bridges = wh_idx.get_bridges(top_seed_id, min_permeability=cfg.min_permeability)
            traces["bridges_traversed"] = len(bridges)

            combined_scores = base_sims.copy()
            id_to_idx = {d["id"]: i for i, d in enumerate(self.docs)}
            for b in bridges:
                target_id = b["target"]
                if target_id in id_to_idx:
                    t_idx = id_to_idx[target_id]
                    t_emb = self.doc_embeddings[t_idx]
                    bridge_sim = float(np.dot(q_emb, t_emb))
                    perm = b["permeability"]
                    bridge_bonus = cfg.bridge_bonus_weight * top_seed_sim * perm * max(0.0, bridge_sim)
                    combined_scores[t_idx] += bridge_bonus

            final_indices = np.argsort(combined_scores, kind="stable")[::-1]
            final_scores = combined_scores[final_indices]

        else:
            final_indices = ranked_seed_indices
            final_scores = base_sims[final_indices]

        ranked_titles = [self.doc_titles[i] for i in final_indices]
        ranked_doc_ids = [self.docs[i]["id"] for i in final_indices]

        # 4. Optional Cross-Encoder Reranking
        if cfg.use_cross_encoder_rerank and cfg.reranker_model is not None and query_str:
            t_rerank0 = time.perf_counter()
            top_rerank_k = min(len(ranked_titles), cfg.reranker_top_k)
            cand_titles = ranked_titles[:top_rerank_k]
            cand_paras = [self.doc_map[self.docs[final_indices[i]]["id"]]["text"] for i in range(top_rerank_k)]
            cand_ids = ranked_doc_ids[:top_rerank_k]

            pairs = [(query_str, f"{t}. {p}".strip()) for t, p in zip(cand_titles, cand_paras)]
            ce_scores = cfg.reranker_model.predict(pairs)

            ce_order = np.argsort(ce_scores)[::-1]
            reranked_ids = [cand_ids[i] for i in ce_order] + ranked_doc_ids[top_rerank_k:]
            reranked_titles = [cand_titles[i] for i in ce_order] + ranked_titles[top_rerank_k:]
            reranked_scores = [float(ce_scores[i]) for i in ce_order] + [float(s) for s in final_scores[top_rerank_k:]]

            ranked_doc_ids = reranked_ids
            ranked_titles = reranked_titles
            final_scores = reranked_scores
            traces["rerank_latency_s"] = time.perf_counter() - t_rerank0

        # Assemble results
        results = []
        for i in range(min(k, len(ranked_doc_ids))):
            did = ranked_doc_ids[i]
            d = self.doc_map[did]
            score_val = float(final_scores[i])
            results.append({
                "doc_id": did,
                "title": d["title"],
                "content": d["text"],
                "score": score_val,
                "rank": i + 1,
            })

        traces["total_latency_s"] = time.perf_counter() - t0

        return RetrievalResult(
            ranked_doc_ids=ranked_doc_ids[:k],
            ranked_titles=ranked_titles[:k],
            scores=[float(s) for s in final_scores[:k]],
            results=results,
            stage_traces=traces,
        )
