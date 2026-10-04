"""
Unit tests for Universal Corpus Edge Builder (lthcm.graph.build_edges)
====================================================================
Verifies:
  1. Title mention extraction without leakage.
  2. IDF-weighted entity co-occurrence edges.
  3. Max out-degree capping.
  4. Random control graph degree preservation.
  5. Post-hoc diagnostic statistics & gold-pair coverage.
"""

import unittest
from lthcm.graph.build_edges import (
    build_edges,
    compute_edge_statistics,
    generate_random_edges,
    generate_oracle_edges,
)


class TestBuildEdges(unittest.TestCase):

    def setUp(self):
        self.docs = [
            {
                "id": "doc_1",
                "title": "Quantum Error Correction",
                "text": "Surface codes provide low logical error rates in superconducting qubits.",
            },
            {
                "id": "doc_2",
                "title": "Superconducting Qubits",
                "text": "Transmon circuits rely on Quantum Error Correction surface codes for fault tolerance.",
            },
            {
                "id": "doc_3",
                "title": "Topological Insulators",
                "text": "Majorana zero modes at the interface provide non-Abelian braiding statistics.",
            },
            {
                "id": "doc_4",
                "title": "Classical Lithography",
                "text": "EUV scanners manufactured by Carl Zeiss SMT operate at high numerical aperture.",
            },
        ]

    def test_title_mention_edges(self):
        """Doc 2 mentions doc 1's title 'Quantum Error Correction' in text."""
        edges = build_edges(self.docs, method="title_mention_only", max_out=5)
        # Check that doc_2 -> doc_1 edge exists
        edge_map = {(src, dst): rel for src, dst, rel, _ in edges}
        self.assertIn(("doc_2", "doc_1"), edge_map)
        self.assertEqual(edge_map[("doc_2", "doc_1")], "title_mention")

    def test_shared_entity_edges(self):
        """Doc 1 and Doc 2 share entities like 'surface codes'."""
        edges = build_edges(self.docs, method="ner", max_out=5)
        edge_pairs = {(src, dst) for src, dst, _, _ in edges}
        self.assertTrue(("doc_1", "doc_2") in edge_pairs or ("doc_2", "doc_1") in edge_pairs)

    def test_out_degree_capping(self):
        """Verify out-degree is strictly capped at max_out."""
        max_out = 1
        edges = build_edges(self.docs, method="title_mention+ner", max_out=max_out)
        out_degrees = {}
        for src, _, _, _ in edges:
            out_degrees[src] = out_degrees.get(src, 0) + 1
        for src, deg in out_degrees.items():
            self.assertLessEqual(deg, max_out)

    def test_random_edges_degree_preservation(self):
        """Verify Arm 2 control matches empirical out-degree distribution."""
        edges = build_edges(self.docs, method="title_mention+ner", max_out=3)
        random_edges = generate_random_edges(self.docs, edges, seed=42)

        orig_out = {}
        for s, _, _, _ in edges:
            orig_out[s] = orig_out.get(s, 0) + 1

        rand_out = {}
        for s, _, _, _ in random_edges:
            rand_out[s] = rand_out.get(s, 0) + 1

        self.assertEqual(orig_out, rand_out)

    def test_oracle_edges_tagging(self):
        """Verify Arm 4 oracle edges are explicitly stamped as ORACLE_UPPER_BOUND."""
        gold = [("doc_1", "doc_2")]
        oracle = generate_oracle_edges(gold)
        for _, _, rel, p in oracle:
            self.assertEqual(rel, "ORACLE_UPPER_BOUND")
            self.assertEqual(p, 1.0)

    def test_edge_statistics_and_gold_coverage(self):
        """Verify diagnostic statistics calculation."""
        edges = [("doc_1", "doc_2", "rel", 0.85), ("doc_2", "doc_3", "rel", 0.70)]
        gold = [("doc_1", "doc_2"), ("doc_3", "doc_4")]
        stats = compute_edge_statistics(edges, total_docs=4, gold_pairs=gold)
        self.assertEqual(stats["total_edges"], 2)
        self.assertEqual(stats["total_docs"], 4)
        self.assertEqual(stats["gold_pairs_total"], 2)
        self.assertEqual(stats["gold_pairs_discovered"], 1)
        self.assertEqual(stats["gold_pair_coverage"], 0.5)

    def test_learned_edge_scorer_prediction_and_bounds(self):
        """Verify 6D feature extraction, calibrated link probability bounds, and JSON serialization."""
        import tempfile
        import os
        from lthcm.graph.learned_edge_scorer import LearnedEdgeScorer

        scorer = LearnedEdgeScorer()
        doc_i = {"id": "i", "title": "Source Paper", "text": "This paper investigates Superconducting Qubits in depth."}
        doc_j = {"id": "j", "title": "Superconducting Qubits", "text": "Qubits based on Josephson junctions."}

        # Feature extraction
        feat = scorer.extract_features(doc_i, doc_j, shared_entities={"qubits", "josephson"}, in_degree_j=2)
        self.assertEqual(feat.shape, (6,))
        self.assertEqual(feat[0], 1.0, "Title mention flag must be 1.0")
        self.assertLess(feat[4], 1.0, "Mention position must be < 1.0 for valid mention")

        # Probability bounds
        prob, breakdown = scorer.score_candidate(
            doc_i, doc_j, shared_entities={"qubits"}, idf_map={"qubits": 2.5}, in_degree_j=2
        )
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertGreater(prob, 0.75, "Strong title mention and shared entity must yield p > 0.75")
        self.assertIn("has_title_mention", breakdown)


        # Serialization test
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            scorer.save(tmp_path)
            loaded_scorer = LearnedEdgeScorer.load(tmp_path)
            prob_loaded, _ = loaded_scorer.score_candidate(
                doc_i, doc_j, shared_entities={"qubits"}, idf_map={"qubits": 2.5}, in_degree_j=2
            )
            self.assertAlmostEqual(prob, prob_loaded, places=6)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_build_edges_method_learned(self):
        """Verify build_edges with method='learned' produces valid, calibrated edges."""
        edges = build_edges(self.docs, method="learned", max_out=2, min_permeability=0.30)
        self.assertGreater(len(edges), 0)

        # Check edge structure
        edge_pairs = set()
        for src, dst, rel, p in edges:
            self.assertTrue(rel.startswith("learned_"))
            self.assertGreaterEqual(p, 0.30)
            self.assertLessEqual(p, 1.0)
            edge_pairs.add((src, dst))

        # Doc 1 and Doc 2 are mutual bridge documents
        self.assertTrue(("doc_1", "doc_2") in edge_pairs or ("doc_2", "doc_1") in edge_pairs)

        # Out-degree capping
        out_degs = {}
        for src, _, _, _ in edges:
            out_degs[src] = out_degs.get(src, 0) + 1
        for src, deg in out_degs.items():
            self.assertLessEqual(deg, 2)

    def test_learned_edge_scorer_calibration_from_corpus(self):
        """Verify supervised training on gold bridge pairs shifts weights toward gold features."""
        from lthcm.graph.learned_edge_scorer import LearnedEdgeScorer

        gold = [("doc_1", "doc_2")]
        calibrated_scorer = LearnedEdgeScorer.calibrate_from_corpus(self.docs, gold)
        self.assertEqual(len(calibrated_scorer.weights), 6)

        # Under calibrated scorer, gold bridge pair must achieve higher probability than non-gold
        prob_gold, _ = calibrated_scorer.score_candidate(self.docs[0], self.docs[1])
        prob_nongold, _ = calibrated_scorer.score_candidate(self.docs[0], self.docs[3])
        self.assertGreater(prob_gold, prob_nongold)




if __name__ == "__main__":
    unittest.main()


class TestCrossEncoderReranker(unittest.TestCase):
    """Unit tests for R2 CrossEncoderReranker (lthcm.core.cross_encoder)."""

    def test_predict_pairs_scores_and_bounds(self):
        """Verify predict_pairs returns calibrated sigmoid probabilities in [0, 1]."""
        from lthcm.core.cross_encoder import CrossEncoderReranker
        ce = CrossEncoderReranker()
        pairs = [
            ("What causes cancer?", "Cancer is caused by genetic mutations in DNA."),
            ("What causes cancer?", "The Eiffel Tower is located in Paris, France."),
        ]
        scores = ce.predict_pairs(pairs, apply_sigmoid=True)
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], 0.5, "Relevant pair must score > 0.5 after sigmoid")
        self.assertLess(scores[1], 0.5, "Irrelevant pair must score < 0.5 after sigmoid")
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_predict_pairs_caching(self):
        """Verify score cache avoids redundant forward passes."""
        from lthcm.core.cross_encoder import CrossEncoderReranker
        ce = CrossEncoderReranker(cache_scores=True)
        pair = ("test query", "test document text")
        s1 = ce.predict_pairs([pair], apply_sigmoid=False)
        self.assertEqual(len(ce._score_cache), 1)
        s2 = ce.predict_pairs([pair], apply_sigmoid=False)
        self.assertAlmostEqual(float(s1[0]), float(s2[0]), places=6)

    def test_predict_pairs_empty(self):
        """Verify empty input returns empty array."""
        from lthcm.core.cross_encoder import CrossEncoderReranker
        ce = CrossEncoderReranker()
        scores = ce.predict_pairs([])
        self.assertEqual(len(scores), 0)

    def test_rerank_ordering(self):
        """Verify rerank returns candidates sorted by cross-encoder score."""
        from lthcm.core.cross_encoder import CrossEncoderReranker
        ce = CrossEncoderReranker()
        candidates = [
            {"id": "irrel", "text": "The weather in Tokyo is warm in summer."},
            {"id": "rel", "text": "Photosynthesis converts sunlight into chemical energy in plants."},
        ]
        reranked = ce.rerank("How do plants produce energy?", candidates, apply_sigmoid=True)
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["id"], "rel", "Relevant doc must be ranked first")
        self.assertGreater(reranked[0]["rerank_score"], reranked[1]["rerank_score"])

    def test_rerank_run_preserves_all_candidates(self):
        """Verify rerank_run preserves candidates outside top_k with fallback scores."""
        from lthcm.core.cross_encoder import CrossEncoderReranker
        ce = CrossEncoderReranker()
        query_texts = {"q1": "genetic mutations"}
        corpus_texts = {"d1": "DNA mutations cause diseases", "d2": "Paris is a city", "d3": "RNA editing"}
        candidate_runs = {"q1": {"d1": 0.9, "d2": 0.5, "d3": 0.3}}
        reranked = ce.rerank_run(query_texts, candidate_runs, corpus_texts, top_k=2)
        self.assertIn("q1", reranked)
        self.assertEqual(len(reranked["q1"]), 3, "All 3 candidates must be preserved")

    def test_clear_cache(self):
        """Verify cache clears correctly."""
        from lthcm.core.cross_encoder import CrossEncoderReranker
        ce = CrossEncoderReranker(cache_scores=True)
        ce.predict_pairs([("q", "d")])
        self.assertGreater(len(ce._score_cache), 0)
        ce.clear_cache()
        self.assertEqual(len(ce._score_cache), 0)


class TestQueryExpansion(unittest.TestCase):
    """Unit tests for R3 HyDE + Query Decomposition (lthcm.core.query_expansion)."""

    def test_hyde_keyword_extraction(self):
        """Verify keyword extraction removes stopwords and question words."""
        from lthcm.core.query_expansion import HyDEExpander
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        hyde = HyDEExpander(enc)
        keywords = hyde._extract_keywords("What is the capital of France?")
        self.assertNotIn("what", keywords)
        self.assertNotIn("is", keywords)
        self.assertIn("capital", keywords)
        self.assertIn("france", keywords)

    def test_hyde_generates_hypothetical_docs(self):
        """Verify hypothetical document generation produces non-empty strings."""
        from lthcm.core.query_expansion import HyDEExpander
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        hyde = HyDEExpander(enc, num_expansions=3)
        docs = hyde.generate_hypothetical_docs("What causes genetic mutations?")
        self.assertEqual(len(docs), 3)
        for d in docs:
            self.assertGreater(len(d), 10)

    def test_hyde_embedding_is_normalized(self):
        """Verify HyDE-expanded embedding is unit-normalized."""
        import numpy as np
        from lthcm.core.query_expansion import HyDEExpander
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        hyde = HyDEExpander(enc, alpha=0.5)
        emb = hyde.expand_query_embedding("What is photosynthesis?")
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_query_decomposer_compound(self):
        """Verify compound question splitting."""
        from lthcm.core.query_expansion import QueryDecomposer
        dec = QueryDecomposer()
        parts = dec.decompose("What are the roles of DNA and RNA?")
        self.assertGreater(len(parts), 1)

    def test_query_decomposer_no_decomposition(self):
        """Verify simple questions return unchanged."""
        from lthcm.core.query_expansion import QueryDecomposer
        dec = QueryDecomposer()
        parts = dec.decompose("What is the capital of France?")
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], "What is the capital of France?")

    def test_multihop_expander_bridge_type(self):
        """Verify bridge-type routing generates sub-queries."""
        from lthcm.core.query_expansion import MultiHopQueryExpander
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        exp = MultiHopQueryExpander(enc)
        # A compound bridge question should decompose
        queries = exp.expand("What is the role of DNA and what does RNA do?", q_type="bridge")
        self.assertGreater(len(queries), 1, "Bridge compound question should decompose")

    def test_multihop_retrieval_scores_shape(self):
        """Verify multi-query retrieval returns correct shape."""
        import numpy as np
        from lthcm.core.query_expansion import MultiHopQueryExpander
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        exp = MultiHopQueryExpander(enc)
        docs = ["Photosynthesis converts sunlight.", "Paris is the capital of France.", "DNA stores genetic info."]
        doc_embs = enc.encode(docs, normalize_embeddings=True).astype("float32")
        scores = exp.multi_query_retrieval_scores("How do plants produce energy?", doc_embs, q_type="comparison")
        self.assertEqual(len(scores), 3)
        # Photosynthesis doc should score highest
        self.assertEqual(np.argmax(scores), 0, "Photosynthesis doc should be most relevant")

