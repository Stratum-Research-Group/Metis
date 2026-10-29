"""
ScaNN Approximate Nearest Neighbor (ANN) Index Adapter
======================================================
Provides a robust, typed interface wrapping Google Research's ScaNN library
(Anisotropic Vector Quantization & SOAR, ICML 2020 / NeurIPS 2023).

Enables sub-linear approximate nearest neighbor retrieval over LTHCM embeddings,
supporting:
  - Tree-AH anisotropic quantization (score_ah) with reordering
  - Brute force exact scoring for small collections (< 200 items)
  - Docid tracking with native string identifiers
  - Dynamic mutations: upsert and delete
  - Filesystem serialization and deserialization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    import scann
    HAS_SCANN = True
except ImportError:
    scann = None
    HAS_SCANN = False


@dataclass
class ScannConfig:
    """
    Configuration parameters for ScaNN index construction and search.
    """
    distance_measure: str = "dot_product"  # "dot_product" or "squared_l2"
    num_neighbors: int = 10
    num_leaves: Optional[int] = None
    num_leaves_to_search: Optional[int] = None
    training_sample_size: Optional[int] = None
    dimensions_per_block: int = 2
    anisotropic_quantization_threshold: float = 0.2
    reorder_num_neighbors: Optional[int] = None
    brute_force: bool = False


class ScannIndex:
    """
    Production-grade ScaNN vector index adapter for LTHCM.
    Wraps scann_ops_pybind with input normalization, docid tracking, and lifecycle helpers.
    """

    def __init__(self, config: Optional[ScannConfig] = None):
        if not HAS_SCANN:
            raise ImportError(
                "scann is not installed in the active environment. "
                "Install via 'pip install scann' to use ScannIndex."
            )
        self.config = config or ScannConfig()
        self.searcher: Optional[Any] = None
        self._docids: List[str] = []
        self._docid_set: set[str] = set()
        self._d_model: Optional[int] = None

    @property
    def is_built(self) -> bool:
        return self.searcher is not None

    @property
    def size(self) -> int:
        return len(self._docids)

    @property
    def d_model(self) -> Optional[int]:
        return self._d_model

    def build(
        self,
        embeddings: Union[np.ndarray, List[List[float]]],
        doc_ids: Optional[List[str]] = None,
    ) -> "ScannIndex":
        """
        Build ScaNN index over the provided embeddings.

        Args:
            embeddings: Float32 array of shape [N, D].
            doc_ids: Optional list of N unique string document identifiers.
        """
        db = np.asarray(embeddings, dtype=np.float32)
        if db.ndim != 2:
            raise ValueError(f"Embeddings must be 2-dimensional [N, D], got shape {db.shape}")

        n_samples, d_model = db.shape
        if n_samples == 0:
            raise ValueError("Cannot build ScannIndex on empty embeddings.")

        self._d_model = d_model

        # Document IDs
        if doc_ids is not None:
            if len(doc_ids) != n_samples:
                raise ValueError(
                    f"Length mismatch: {len(doc_ids)} doc_ids provided for {n_samples} embeddings."
                )
            self._docids = [str(did) for did in doc_ids]
        else:
            self._docids = [f"doc_{i}" for i in range(n_samples)]

        self._docid_set = set(self._docids)

        # Normalize embeddings if using dot product (cosine similarity)
        if self.config.distance_measure == "dot_product":
            norms = np.linalg.norm(db, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            db = (db / norms).astype(np.float32)

        # ScaNN Builder
        k = min(self.config.num_neighbors, n_samples)
        builder = scann.scann_ops_pybind.builder(db, k, self.config.distance_measure)

        # If dataset is small (< 200 items) or brute force requested, use brute force scoring
        if self.config.brute_force or n_samples < 200:
            self.searcher = builder.score_brute_force().build(docids=self._docids)
            return self

        # Otherwise configure tree AH (Anisotropic Vector Quantization)
        num_leaves = self.config.num_leaves or max(int(np.sqrt(n_samples)), 4)
        num_leaves = min(num_leaves, n_samples // 4)
        leaves_to_search = self.config.num_leaves_to_search or max(num_leaves // 4, 2)
        sample_size = self.config.training_sample_size or min(n_samples, 100000)

        builder = builder.tree(
            num_leaves=num_leaves,
            num_leaves_to_search=leaves_to_search,
            training_sample_size=sample_size,
        )

        dim_per_block = self.config.dimensions_per_block
        if d_model % dim_per_block != 0:
            # Fall back to 1 or 2 if not divisible
            dim_per_block = 1 if d_model % 2 != 0 else 2

        builder = builder.score_ah(
            dimensions_per_block=dim_per_block,
            anisotropic_quantization_threshold=self.config.anisotropic_quantization_threshold,
        )

        if self.config.reorder_num_neighbors:
            reorder_k = min(self.config.reorder_num_neighbors, n_samples)
            builder = builder.reorder(reorder_k)

        self.searcher = builder.build(docids=self._docids)
        return self

    def search(
        self,
        query: Union[np.ndarray, List[float]],
        top_k: Optional[int] = None,
        leaves_to_search: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Search top-k nearest neighbors for a single query.

        Args:
            query: Float32 vector of shape [D].
            top_k: Optional override for number of neighbors.
            leaves_to_search: Optional runtime override for partitions to search.

        Returns:
            Tuple of (doc_ids, scores).
        """
        if not self.is_built:
            raise RuntimeError("ScannIndex is not built. Call build() first.")

        q = np.asarray(query, dtype=np.float32).flatten()
        if q.shape[0] != self._d_model:
            raise ValueError(f"Query dimension {q.shape[0]} does not match index dimension {self._d_model}")

        if self.config.distance_measure == "dot_product":
            q_norm = np.linalg.norm(q)
            if q_norm > 1e-12:
                q = (q / q_norm).astype(np.float32)

        k = top_k or self.config.num_neighbors
        k = min(k, len(self._docids))
        if k == 0:
            return [], np.empty(0, dtype=np.float32)

        kwargs: Dict[str, Any] = {"final_num_neighbors": k}
        if leaves_to_search is not None:
            kwargs["leaves_to_search"] = leaves_to_search

        doc_ids, distances = self.searcher.search(q, **kwargs)
        # Ensure returned doc_ids are Python strings
        cleaned_ids = [did.decode("utf-8") if isinstance(did, bytes) else str(did) for did in doc_ids]
        return cleaned_ids, np.asarray(distances, dtype=np.float32)

    def search_batched(
        self,
        queries: Union[np.ndarray, List[List[float]]],
        top_k: Optional[int] = None,
        leaves_to_search: Optional[int] = None,
    ) -> Tuple[List[List[str]], np.ndarray]:
        """
        Search top-k nearest neighbors for a batch of queries.

        Args:
            queries: Float32 array of shape [B, D].
            top_k: Optional override for number of neighbors.
            leaves_to_search: Optional runtime override for partitions to search.

        Returns:
            Tuple of (list_of_doc_id_lists, 2D_scores_array).
        """
        if not self.is_built:
            raise RuntimeError("ScannIndex is not built. Call build() first.")

        qs = np.asarray(queries, dtype=np.float32)
        if qs.ndim != 2 or qs.shape[1] != self._d_model:
            raise ValueError(f"Queries must be [B, {self._d_model}], got shape {qs.shape}")

        if self.config.distance_measure == "dot_product":
            norms = np.linalg.norm(qs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            qs = (qs / norms).astype(np.float32)

        k = top_k or self.config.num_neighbors
        k = min(k, len(self._docids))
        if k == 0:
            return [[] for _ in range(len(qs))], np.empty((len(qs), 0), dtype=np.float32)

        kwargs: Dict[str, Any] = {"final_num_neighbors": k}
        if leaves_to_search is not None:
            kwargs["leaves_to_search"] = leaves_to_search

        batch_ids, batch_dists = self.searcher.search_batched(qs, **kwargs)
        cleaned_batch_ids: List[List[str]] = []
        for id_list in batch_ids:
            cleaned_batch_ids.append(
                [did.decode("utf-8") if isinstance(did, bytes) else str(did) for did in id_list]
            )

        return cleaned_batch_ids, np.asarray(batch_dists, dtype=np.float32)

    def upsert(self, doc_id: str, vector: Union[np.ndarray, List[float]]) -> None:
        """
        Insert or update a document vector in the active index.
        """
        if not self.is_built:
            raise RuntimeError("ScannIndex is not built. Call build() first.")

        v = np.asarray(vector, dtype=np.float32).flatten()
        if v.shape[0] != self._d_model:
            raise ValueError(f"Vector dimension {v.shape[0]} does not match index dimension {self._d_model}")

        if self.config.distance_measure == "dot_product":
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-12:
                v = (v / v_norm).astype(np.float32)

        doc_id_str = str(doc_id)
        self.searcher.upsert(doc_id_str, v)

        # ScaNN pybind directly mutates the docids list passed at build time
        if doc_id_str not in self._docid_set:
            self._docid_set.add(doc_id_str)
            if doc_id_str not in self._docids:
                self._docids.append(doc_id_str)

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document identifier from the active index.
        Returns True if the document was present and deleted, False otherwise.
        """
        if not self.is_built:
            raise RuntimeError("ScannIndex is not built. Call build() first.")

        doc_id_str = str(doc_id)
        if doc_id_str not in self._docid_set:
            return False

        self.searcher.delete(doc_id_str)
        self._docid_set.remove(doc_id_str)
        # In case pybind didn't strip it from the list
        self._docids = [did for did in self._docids if did != doc_id_str]
        return True

    def serialize(self, artifacts_dir: Union[str, Path]) -> Path:
        """
        Serialize index artifacts and metadata to directory.
        """
        import json
        if not self.is_built:
            raise RuntimeError("ScannIndex is not built. Call build() first.")

        out_path = Path(artifacts_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.searcher.serialize(str(out_path))

        # Persist index metadata (d_model, docids, config)
        meta = {
            "d_model": self._d_model,
            "docids": [did.decode("utf-8") if isinstance(did, bytes) else str(did) for did in self._docids],
            "distance_measure": self.config.distance_measure,
            "brute_force": self.config.brute_force,
        }
        with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return out_path

    @classmethod
    def load(
        cls,
        artifacts_dir: Union[str, Path],
        config: Optional[ScannConfig] = None,
    ) -> "ScannIndex":
        """
        Load a serialized ScaNN index and metadata from directory.
        """
        import json
        if not HAS_SCANN:
            raise ImportError("scann is not installed. Install via 'pip install scann'.")

        path = Path(artifacts_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"ScaNN artifacts directory not found: {path}")

        instance = cls(config=config)
        instance.searcher = scann.scann_ops_pybind.load_searcher(str(path))

        meta_path = path / "metadata.json"
        if meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                instance._d_model = meta.get("d_model")
                instance._docids = meta.get("docids", [])
                instance._docid_set = set(instance._docids)
                if "distance_measure" in meta:
                    instance.config.distance_measure = meta["distance_measure"]
            except Exception as e:
                logger.warning(f"Failed to load ScannIndex metadata from {meta_path}: {e}")

        return instance
