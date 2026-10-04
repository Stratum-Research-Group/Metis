"""
Cross-Encoder Late-Interaction Reranking Stage (Tier R2)
=========================================================
Implements the cross-encoder / late-interaction reranking stage described in Tier R2
of improv2.md. Inserts between first-stage retrieval (FAISS/BM25) and topological/geometric
stages (Wormholes, Fisher-Rao, Sheaves).

Enables the rigorous 2x2 ablation grid:
               | no rerank       | + rerank
  -------------+-----------------+-----------------
  flat         | Flat FAISS      | Flat + CrossEncoder
  + LTHCM      | LTHCM Mined     | LTHCM Mined + CrossEncoder

Provides:
  - Vectorized and batch-cached pair scoring
  - Sigmoid calibration: sigma(s) in [0, 1] for coherent downstream manifold blending
  - Fast run-level reranking for BEIR / HotpotQA benchmarks
"""

import math
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import torch


class CrossEncoderReranker:
    """
    Cross-Encoder scoring module for candidate reranking.
    Encodes query-document pairs jointly: s(q, d) = w^T BERT([CLS] q [SEP] d).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        cache_scores: bool = True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.cache_scores = cache_scores
        self._model = None
        self._score_cache: Dict[Tuple[str, str], float] = {}

    @property
    def model(self):
        """Lazy loader for sentence-transformers CrossEncoder."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            if hasattr(self._model, "model") and hasattr(self._model.model, "eval"):
                self._model.model.eval()
        return self._model

    def predict_pairs(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 64,
        apply_sigmoid: bool = False,
    ) -> np.ndarray:
        """
        Computes relevance logits or probabilities for query-document text pairs.
        Uses in-memory caching to avoid redundant forward passes.
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        scores = np.zeros(len(pairs), dtype=np.float32)
        uncached_indices = []
        uncached_pairs = []

        if self.cache_scores:
            for idx, p in enumerate(pairs):
                if p in self._score_cache:
                    scores[idx] = self._score_cache[p]
                else:
                    uncached_indices.append(idx)
                    uncached_pairs.append(p)
        else:
            uncached_indices = list(range(len(pairs)))
            uncached_pairs = pairs

        if uncached_pairs:
            preds = self.model.predict(
                uncached_pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            # Ensure 1D float array
            preds = np.asarray(preds, dtype=np.float32).reshape(-1)

            for idx, pred_val in zip(uncached_indices, preds):
                raw_score = float(pred_val)
                scores[idx] = raw_score
                if self.cache_scores:
                    self._score_cache[pairs[idx]] = raw_score

        if apply_sigmoid:
            # Numerically stable sigmoid: 1 / (1 + exp(-x))
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -20.0, 20.0)))

        return scores

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        batch_size: int = 32,
        apply_sigmoid: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate documents against a query string.
        Each candidate dictionary should contain text in 'text', 'content', or 'paragraph'.
        """
        if not candidates:
            return []

        pairs = []
        for c in candidates:
            text = c.get("text", c.get("content", c.get("paragraph", "")))
            pairs.append((query, str(text)))

        scores = self.predict_pairs(pairs, batch_size=batch_size, apply_sigmoid=apply_sigmoid)

        reranked = []
        for c, s in zip(candidates, scores):
            c_copy = dict(c)
            score_val = float(s)
            c_copy["rerank_score"] = score_val
            c_copy["score"] = score_val
            reranked.append(c_copy)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def rerank_run(
        self,
        query_texts: Dict[str, str],
        candidate_runs: Dict[str, Dict[str, float]],
        corpus_texts: Dict[str, str],
        top_k: int = 100,
        batch_size: int = 64,
        apply_sigmoid: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Reranks top-k candidates in a standard TREC/BEIR run dict {qid: {did: score}}.
        Preserves candidates outside the top-k with their original relative ordering.
        """
        reranked_run: Dict[str, Dict[str, float]] = {}

        # Collect all query-doc pairs across all queries
        all_pairs = []
        pair_mapping = []  # List of (qid, did)

        for qid in sorted(candidate_runs.keys()):
            did_scores = candidate_runs[qid]
            q_text = query_texts.get(qid, "")
            # Sort dids by initial score with deterministic tie-break
            sorted_dids = sorted(did_scores.items(), key=lambda x: (-x[1], x[0]))
            top_dids = [d for d, _ in sorted_dids[:top_k]]

            for did in top_dids:
                d_text = corpus_texts.get(did, "")
                all_pairs.append((q_text, d_text))
                pair_mapping.append((qid, did))

        # Predict in large efficient batches
        all_scores = self.predict_pairs(all_pairs, batch_size=batch_size, apply_sigmoid=apply_sigmoid)

        # Re-assemble runs
        score_by_q_d: Dict[str, Dict[str, float]] = {}
        for (qid, did), score in zip(pair_mapping, all_scores):
            score_by_q_d.setdefault(qid, {})[did] = float(score)

        for qid in sorted(candidate_runs.keys()):
            did_scores = candidate_runs[qid]
            sorted_dids = sorted(did_scores.items(), key=lambda x: (-x[1], x[0]))
            top_dids = [d for d, _ in sorted_dids[:top_k]]
            remainder_dids = [d for d, _ in sorted_dids[top_k:]]

            reranked_items = []
            if qid in score_by_q_d:
                for did in top_dids:
                    reranked_items.append((did, score_by_q_d[qid].get(did, did_scores[did])))

            reranked_items.sort(key=lambda x: (-x[1], x[0]))

            # Assign monotonic fallback scores for remainder
            min_top_score = reranked_items[-1][1] if reranked_items else 0.0
            for idx, did in enumerate(remainder_dids):
                reranked_items.append((did, min_top_score - 1.0 - idx * 0.01))

            reranked_run[qid] = {d: s for d, s in reranked_items}

        return reranked_run

    def clear_cache(self) -> None:
        """Clears score cache."""
        self._score_cache.clear()
