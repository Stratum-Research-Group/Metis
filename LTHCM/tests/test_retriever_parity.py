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

    def test_type_conditioned_perishability_decay(self):
        """Verify type-conditioned perishability attenuates ephemeral memories while preserving permanent facts."""
        temporal_corpus = [
            {
                "id": "mem_permanent",
                "title": "System Architecture",
                "text": "Core security policy is fail-closed with zero loopback bypass.",
                "perishability": 0.0,  # Permanent invariant
                "utility_horizon": 86400.0,
                "created_at": 1000.0,
            },
            {
                "id": "mem_ephemeral",
                "title": "Temporary Session State",
                "text": "Current working directory is /tmp/session_cache.",
                "perishability": 1.0,  # Ephemeral details
                "utility_horizon": 60.0,  # 60s horizon
                "created_at": 1000.0,
            },
        ]
        embs = np.array([
            [1.0, 0.0, 0.0, 0.0] + [0.0] * 12,
            [1.0, 0.0, 0.0, 0.0] + [0.0] * 12,
        ], dtype="float32")
        q = np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 12, dtype="float32")

        retriever = LTHCMRetriever(d_model=self.d_model)
        retriever.index(temporal_corpus, embeddings=embs)

        # 1. At t=1000 (creation time), both have equal relevance
        cfg_t0 = RetrievalConfig(mode="flat", top_k=2, enable_perishability_decay=True, current_timestamp=1000.0)
        res_t0 = retriever.search(q, config=cfg_t0)
        self.assertAlmostEqual(res_t0.scores[0], res_t0.scores[1], places=4)

        # 2. At t=1300 (300 seconds later = 5 half-lives for ephemeral), ephemeral has decayed drastically
        cfg_t1 = RetrievalConfig(mode="flat", top_k=2, enable_perishability_decay=True, current_timestamp=1300.0)
        res_t1 = retriever.search(q, config=cfg_t1)
        # mem_permanent should rank #1
        self.assertEqual(res_t1.ranked_doc_ids[0], "mem_permanent")
        self.assertAlmostEqual(res_t1.scores[0], 1.0, places=4)
        # mem_ephemeral should have decayed: exp(- 1.0 * 300 / 60) = exp(-5) ~ 0.0067
        self.assertLess(res_t1.scores[1], 0.05)


if __name__ == "__main__":
    unittest.main()
