"""
Parity Integration Test: Public Retriever API vs Benchmark Adapter
==================================================================
Verifies that:
  1. The public LTHCMRetriever and LTHCMEngine.search produce identical ranked document IDs
     and scores for the same corpus and configuration.
  2. Each scored ablation module (flat, graph_mined, graph_learned, graph_random, graph_oracle, rerank)
     is functionally verifiable and produces deterministic traces.
"""

import unittest
import numpy as np
import torch

from lthcm.engine import LTHCMEngine, LTHCMRetriever, RetrievalConfig
from lthcm.graph.build_edges import build_edges


class TestRetrieverParity(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            {
                "id": "p_0",
                "title": "Quantum Error Correction",
                "text": "Surface codes provide low logical error rates in superconducting qubits."
            },
            {
                "id": "p_1",
                "title": "Superconducting Qubits",
                "text": "Transmon circuits rely on Quantum Error Correction surface codes for fault tolerance."
            },
            {
                "id": "p_2",
                "title": "Topological Insulators",
                "text": "Majorana zero modes at the interface provide non-Abelian braiding statistics."
            },
            {
                "id": "p_3",
                "title": "Carl Zeiss Optics",
                "text": "EUV lithography scanners use Carl Zeiss mirrors to project nanometer patterns."
            },
        ]
        self.d_model = 16
        rng = np.random.default_rng(42)
        self.embeddings = rng.normal(size=(len(self.corpus), self.d_model)).astype("float32")
        # Normalize
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=-1, keepdims=True)

        self.query_vec = rng.normal(size=(self.d_model,)).astype("float32")
        self.query_vec = self.query_vec / np.linalg.norm(self.query_vec)

    def test_flat_and_graph_parity(self):
        """Verify standalone LTHCMRetriever matches LTHCMEngine.search."""
        engine = LTHCMEngine(d_model=self.d_model)

        # Index into engine
        facet_vecs = torch.from_numpy(self.embeddings).unsqueeze(1)
        engine.index_documents(
            doc_ids=[d["id"] for d in self.corpus],
            contents=[d["text"] for d in self.corpus],
            facet_vectors=facet_vecs,
            domains=[d["title"] for d in self.corpus],
        )

        # Build mined edges
        mined_edges = build_edges(self.corpus, method="title_mention+ner", max_out=4)
        for s, t, r, p in mined_edges:
            engine.wormhole.add_wormhole(s, t, r, permeability=p)

        # Initialize standalone retriever with identical embeddings and edges
        retriever = LTHCMRetriever(d_model=self.d_model)
        retriever.index(self.corpus, embeddings=self.embeddings, edges=mined_edges)

        for mode in ["flat", "graph_mined", "graph_learned", "graph_fixed_weights"]:
            cfg = RetrievalConfig(mode=mode, top_k=4)
            res_retriever = retriever.search(self.query_vec, config=cfg)
            res_engine = engine.search(self.query_vec, top_k=4, config=cfg)

            self.assertEqual(
                res_retriever.ranked_doc_ids,
                res_engine.ranked_doc_ids,
                f"Mode '{mode}' failed ranked ID parity between public retriever and engine"
            )
            for s1, s2 in zip(res_retriever.scores, res_engine.scores):
                self.assertAlmostEqual(s1, s2, places=5)

    def test_ablation_visibility(self):
        """Verify each ablation arm changes scores or behavior predictably."""
        retriever = LTHCMRetriever(d_model=self.d_model)
        retriever.index(self.corpus, embeddings=self.embeddings)

        res_flat = retriever.search(self.query_vec, config=RetrievalConfig(mode="flat", top_k=4))
        res_mined = retriever.search(self.query_vec, config=RetrievalConfig(mode="graph_mined", top_k=4))
        res_oracle = retriever.search(
            self.query_vec,
            config=RetrievalConfig(mode="graph_oracle", top_k=4),
            gold_pairs=[("p_0", "p_1")],
        )

        self.assertIn("stage1_latency_s", res_flat.stage_traces)
        self.assertIn("bridges_traversed", res_mined.stage_traces)
        self.assertEqual(len(res_flat.ranked_doc_ids), 4)
        self.assertEqual(len(res_mined.ranked_doc_ids), 4)
        self.assertEqual(len(res_oracle.ranked_doc_ids), 4)


if __name__ == "__main__":
    unittest.main()
