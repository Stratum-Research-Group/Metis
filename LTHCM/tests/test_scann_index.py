"""
Unit Tests for ScannIndex & ANN Retrieval
=========================================
Verifies:
  1. Construction, building, and searching with ScannIndex.
  2. Sub-linear Tree-AH mode vs brute force exact mode.
  3. Docid resolution, upsert, and delete operations.
  4. Index serialization and deserialization.
  5. LTHCMRetriever integration with ann_backend="scann".
"""

import tempfile
import shutil
import unittest
import numpy as np

import pytest
pytest.importorskip("scann", reason="scann not installed in environment")

from lthcm.core.scann_index import ScannConfig, ScannIndex
from lthcm.engine.retriever import LTHCMRetriever, RetrievalConfig


class TestScannIndex(unittest.TestCase):

    def setUp(self):
        self.d_model = 32
        self.n_docs = 300
        rng = np.random.default_rng(42)
        raw_embs = rng.normal(size=(self.n_docs, self.d_model)).astype("float32")
        self.embeddings = raw_embs / np.linalg.norm(raw_embs, axis=1, keepdims=True)
        self.doc_ids = [f"doc_{i}" for i in range(self.n_docs)]
        self.query = self.embeddings[0]

    def test_brute_force_exact_match(self):
        """Verify brute force mode finds exact top-1 match for identical vector."""
        cfg = ScannConfig(distance_measure="dot_product", brute_force=True)
        index = ScannIndex(cfg).build(self.embeddings, doc_ids=self.doc_ids)

        self.assertTrue(index.is_built)
        self.assertEqual(index.size, self.n_docs)
        self.assertEqual(index.d_model, self.d_model)

        ids, scores = index.search(self.query, top_k=5)
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[0], "doc_0")
        self.assertAlmostEqual(scores[0], 1.0, places=4)

    def test_tree_ah_quantized_search(self):
        """Verify Tree-AH search retrieves high-recall matches."""
        cfg = ScannConfig(
            distance_measure="dot_product",
            num_leaves=10,
            num_leaves_to_search=8,
            training_sample_size=self.n_docs,
            dimensions_per_block=2,
            anisotropic_quantization_threshold=0.2,
            reorder_num_neighbors=30,
            brute_force=False,
        )
        index = ScannIndex(cfg).build(self.embeddings, doc_ids=self.doc_ids)

        ids, scores = index.search(self.query, top_k=10)
        self.assertIn("doc_0", ids)
        self.assertGreater(scores[0], 0.95)

    def test_batched_search(self):
        """Verify batched search returns matching rows and dimensions."""
        index = ScannIndex(ScannConfig(brute_force=True)).build(self.embeddings, doc_ids=self.doc_ids)
        queries = self.embeddings[:3]

        batch_ids, batch_scores = index.search_batched(queries, top_k=4)
        self.assertEqual(len(batch_ids), 3)
        self.assertEqual(batch_scores.shape, (3, 4))
        for i in range(3):
            self.assertEqual(batch_ids[i][0], f"doc_{i}")

    def test_upsert_and_delete(self):
        """Verify dynamic upsert and delete mutate active index."""
        index = ScannIndex(ScannConfig(brute_force=True)).build(self.embeddings, doc_ids=self.doc_ids)

        new_vec = np.ones(self.d_model, dtype=np.float32)
        new_vec /= np.linalg.norm(new_vec)

        # Upsert
        index.upsert("doc_mutation_test", new_vec)
        self.assertEqual(index.size, self.n_docs + 1)

        ids, _ = index.search(new_vec, top_k=1)
        self.assertEqual(ids[0], "doc_mutation_test")

        # Delete
        deleted = index.delete("doc_mutation_test")
        self.assertTrue(deleted)
        self.assertEqual(index.size, self.n_docs)

        ids_after, _ = index.search(new_vec, top_k=1)
        self.assertNotEqual(ids_after[0], "doc_mutation_test")

    def test_serialize_and_load(self):
        """Verify serialization and deserialization of index artifacts."""
        index = ScannIndex(ScannConfig(brute_force=True)).build(self.embeddings, doc_ids=self.doc_ids)
        temp_dir = tempfile.mkdtemp(prefix="scann_test_")
        try:
            index.serialize(temp_dir)
            loaded = ScannIndex.load(temp_dir)
            self.assertTrue(loaded.is_built)

            ids, _ = loaded.search(self.query, top_k=3)
            self.assertEqual(ids[0], "doc_0")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_retriever_parity_with_scann_backend(self):
        """Verify LTHCMRetriever operates with ann_backend='scann'."""
        corpus = [
            {"id": f"doc_{i}", "title": f"Doc {i}", "text": f"Content of document {i}."}
            for i in range(len(self.embeddings))
        ]
        retriever = LTHCMRetriever(d_model=self.d_model)
        cfg = RetrievalConfig(ann_backend="scann", top_k=5)
        retriever.index(corpus, embeddings=self.embeddings, config=cfg)

        res = retriever.search(self.query, config=cfg)
        self.assertEqual(len(res.ranked_doc_ids), 5)
        self.assertEqual(res.stage_traces.get("ann_backend"), "scann")
        self.assertIn("doc_0", res.ranked_doc_ids[:3])


if __name__ == "__main__":
    unittest.main()
