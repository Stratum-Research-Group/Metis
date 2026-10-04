"""
Unit tests for GraphRAG Community Detection & Sheaf Verification (R4)
=====================================================================
Verifies:
  1. Leiden community partitioning on modular graphs.
  2. Embedding similarity graph construction and partitioning.
  3. Single-doc and disconnected-node edge cases.
  4. Centroid-salience extractive community summarization with MMR.
  5. Community summarization caching.
  6. GraphRAGEvidenceBuilder multi-evidence synthesis.
  7. SheafContradictionFilter pure geometric vs NLI-grounded evaluation.
"""

import unittest
import numpy as np
import torch

from lthcm.graph.community import (
    LeidenCommunityDetector,
    CommunitySummarizer,
    GraphRAGEvidenceBuilder,
)
from lthcm.core.sheaf import SheafContradictionFilter, CellularSheaf


class TestCommunityGraphRAG(unittest.TestCase):

    def setUp(self):
        # Two thematic clusters:
        # Cluster A: Quantum computing
        # Cluster B: Renaissance art
        self.passages = [
            {
                "id": "qc_1",
                "text": "Superconducting qubits are artificial atoms formed by Josephson junctions. They operate at millikelvin temperatures.",
            },
            {
                "id": "qc_2",
                "text": "Surface codes provide fault tolerance for superconducting quantum processors. Physical qubits are coupled in a 2D lattice.",
            },
            {
                "id": "qc_3",
                "text": "Quantum error correction suppresses decoherence in transmon qubit architectures through stabilizer measurements.",
            },
            {
                "id": "art_1",
                "text": "Leonardo da Vinci painted the Mona Lisa in Florence during the Italian Renaissance. His sfumato technique softens outlines.",
            },
            {
                "id": "art_2",
                "text": "The Italian Renaissance witnessed groundbreaking masterpieces by Michelangelo, Raphael, and Leonardo in Rome and Florence.",
            },
            {
                "id": "art_3",
                "text": "Renaissance fresco painting required artists to apply pigments directly onto freshly laid wet lime plaster.",
            },
        ]

    def test_leiden_basic_clique_partition(self):
        """Verify Leiden partitions two distinct cliques into separate communities."""
        detector = LeidenCommunityDetector(seed=42)
        doc_ids = [p["id"] for p in self.passages]
        # Edges within cluster A (qc) and cluster B (art), with weak inter-cluster bridge
        edges = [
            ("qc_1", "qc_2", 1.0),
            ("qc_2", "qc_3", 1.0),
            ("qc_1", "qc_3", 1.0),
            ("art_1", "art_2", 1.0),
            ("art_2", "art_3", 1.0),
            ("art_1", "art_3", 1.0),
            ("qc_3", "art_1", 0.05),  # weak noise bridge
        ]
        comms = detector.detect_communities(doc_ids=doc_ids, docs=self.passages, edges=edges)
        self.assertGreaterEqual(len(comms), 2)
        # Check that qc docs are together and art docs are together
        c0_members = set(comms[0])
        c1_members = set(comms[1])
        qc_ids = {"qc_1", "qc_2", "qc_3"}
        art_ids = {"art_1", "art_2", "art_3"}
        self.assertTrue((qc_ids == c0_members and art_ids == c1_members) or
                        (qc_ids == c1_members and art_ids == c0_members))

    def test_leiden_disconnected_graph(self):
        """Verify detector handles isolated nodes gracefully."""
        detector = LeidenCommunityDetector(seed=42)
        doc_ids = ["d1", "d2", "d3"]
        docs = [
            {"id": "d1", "text": "Zebra safari migration in Serengeti grassland."},
            {"id": "d2", "text": "Quantum mechanics Schrödinger wavefunction eigenvalue."},
            {"id": "d3", "text": "Renaissance fresco painting sfumato Florence."},
        ]
        comms = detector.detect_communities(doc_ids=doc_ids, docs=docs)
        self.assertEqual(len(comms), 3)
        all_members = [item for sublist in comms.values() for item in sublist]
        self.assertEqual(set(all_members), set(doc_ids))

    def test_leiden_single_doc(self):
        """Verify single-doc input returns 1 community containing that doc."""
        detector = LeidenCommunityDetector(seed=42)
        comms = detector.detect_communities(doc_ids=["solo"], docs=[{"id": "solo", "text": "Solo text."}])
        self.assertEqual(len(comms), 1)
        self.assertEqual(comms[0], ["solo"])

    def test_leiden_embedding_similarity(self):
        """Verify community detection partitions nodes based on mock embedding similarity."""
        detector = LeidenCommunityDetector(similarity_threshold=0.3, seed=42)
        doc_ids = ["a1", "a2", "b1", "b2"]
        # Cluster A points along x-axis, Cluster B points along y-axis
        embs = np.array([
            [1.0, 0.05, 0.0],
            [0.95, 0.0, 0.0],
            [0.0, 1.0, 0.05],
            [0.05, 0.95, 0.0],
        ], dtype=np.float32)
        comms = detector.detect_communities(doc_ids=doc_ids, embeddings=embs)
        self.assertEqual(len(comms), 2)
        c0 = set(comms[0])
        c1 = set(comms[1])
        self.assertTrue(({"a1", "a2"} == c0 and {"b1", "b2"} == c1) or
                        ({"a1", "a2"} == c1 and {"b1", "b2"} == c0))

    def test_community_summarizer_single_doc(self):
        """Verify single doc summarization preserves core content."""
        summarizer = CommunitySummarizer(max_sentences=2)
        single_doc = [{"id": "doc_x", "text": "Single passage about astrophysics. Dark energy drives cosmic acceleration."}]
        summary = summarizer.summarize_community(community_id=0, docs=single_doc)
        self.assertEqual(summary["community_id"], 0)
        self.assertIn("Single passage", summary["summary"])
        self.assertEqual(summary["constituent_doc_ids"], ["doc_x"])

    def test_community_summarizer_centroid_and_caching(self):
        """Verify multi-document summarization selects salient sentences and caches results."""
        summarizer = CommunitySummarizer(max_sentences=2)
        docs = [
            {"id": "d1", "text": "Superconducting qubits are scalable. They are used in Google and IBM quantum processors."},
            {"id": "d2", "text": "Transmon qubits utilize Josephson junctions. Quantum error correction ensures fault tolerance."},
            {"id": "d3", "text": "Quantum computing holds immense promise for simulating molecular chemistry."},
        ]
        s1 = summarizer.summarize_community(community_id=1, docs=docs)
        self.assertIsNotNone(s1["summary"])
        self.assertLessEqual(len(s1["key_facts"]), 2)
        self.assertEqual(len(s1["constituent_doc_ids"]), 3)

        # Verify caching
        s2 = summarizer.summarize_community(community_id=1, docs=docs)
        self.assertEqual(s1["summary"], s2["summary"])

    def test_graphrag_evidence_builder(self):
        """Verify GraphRAGEvidenceBuilder groups passages and produces community summaries."""
        builder = GraphRAGEvidenceBuilder(similarity_threshold=0.25, max_sentences_per_community=2, seed=42)
        evidence_pkg = builder.build_evidence(passages=self.passages)
        self.assertIn("raw_concatenated", evidence_pkg)
        self.assertIn("concatenated_summary", evidence_pkg)
        self.assertGreaterEqual(evidence_pkg["num_communities"], 1)
        self.assertGreater(len(evidence_pkg["community_summaries"]), 0)

    def test_sheaf_pure_vs_nli_grounded(self):
        """Verify SheafContradictionFilter supports pure geometric Dirichlet energy without NLI."""
        d_model = 64
        filter_sheaf = SheafContradictionFilter(d_model=d_model, nli_model_name=None, contradiction_threshold=0.1)
        
        torch.manual_seed(42)
        # Random passage embeddings
        embs = torch.randn(3, d_model)
        passages = [
            {"doc_id": "p1", "content": "The earth orbits the sun."},
            {"doc_id": "p2", "content": "Planetary bodies follow elliptical orbits."},
            {"doc_id": "p3", "content": "Gravity governs orbital celestial mechanics."},
        ]

        # Pure geometric evaluation (no NLI)
        res_pure = filter_sheaf.evaluate_passages(embs, passages, use_nli=False)
        self.assertIn("total_obstruction", res_pure)
        self.assertIn("is_consistent", res_pure)
        self.assertIsInstance(res_pure["total_obstruction"], float)
        # Geometric Dirichlet energy is non-negative
        self.assertGreaterEqual(res_pure["total_obstruction"], 0.0)


if __name__ == "__main__":
    unittest.main()
