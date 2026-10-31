"""
Unit and integration tests for Kybernetes Heuresis Scientific Discovery Adapter & CLI.
"""

import json
import tempfile
import unittest
from pathlib import Path
from click.testing import CliRunner

from kybernetes.adapters.heuresis import DiscoveryProposalRecord, HeuresisAdapter
from kybernetes.cli import cli


class TestHeuresisAdapter(unittest.TestCase):
    """Test HeuresisAdapter discovery search, writeup, crawling, and Dokime forward evaluation."""

    def setUp(self):
        self.adapter = HeuresisAdapter()

    def test_search_candidates_generates_top_combinations(self):
        candidates = self.adapter.search_candidates(top_k=5, beta=1.25, fringe_quota=0.15)
        self.assertEqual(len(candidates), 5)

        for c in candidates:
            self.assertIn("rank", c)
            self.assertIn("atoms", c)
            self.assertEqual(len(c["atoms"]), 3)
            self.assertIn("coherence_score", c)
            self.assertIn("availability_score", c)
            self.assertIn("discovery_score", c)
            self.assertIn("is_oov_synthesized", c)
            self.assertIn("obstruction_energy", c)

        # First candidate should have positive discovery score
        self.assertGreater(candidates[0]["discovery_score"], 0.0)

    def test_writeup_proposal_produces_falsifiable_format(self):
        candidates = self.adapter.search_candidates(top_k=1, beta=1.25)
        cand = candidates[0]

        record = self.adapter.writeup_proposal(cand)
        self.assertIsInstance(record, DiscoveryProposalRecord)
        self.assertTrue(record.written_up)
        self.assertIn("Core Research Question", record.writeup_text)
        self.assertIn("Falsifiable Hypothesis", record.writeup_text)
        self.assertIn("Failure Criterion", record.writeup_text)
        self.assertIn("Proposed Methodology & Experiment", record.writeup_text)
        self.assertIn("Conceptual Provenance", record.writeup_text)

    def test_crawl_and_index_forums_extracts_boundary_objections(self):
        sample_forums = [
            {
                "forum": "forum_neurips_01",
                "content": {
                    "title": "Quantum Sheaves for Graph Transformers",
                    "abstract": "We explore quantum sheaf cohomology on graph topologies.",
                    "primary_area": "Theory",
                },
                "details": {"directReplies": []},
                "reviews": [
                    {
                        "rating": "4: Ok, but missing empirical validation",
                        "comment": "The submission lacks a theoretical bridge between cellular_sheaves and graph_transformers.",
                    },
                    {
                        "rating": "7: Good paper",
                        "comment": "Interesting connection to cellular_sheaves.",
                    },
                ],
            }
        ]

        summary = self.adapter.crawl_and_index_forums(sample_forums)
        self.assertEqual(summary["forums_parsed"], 1)
        self.assertEqual(summary["contested_forums"], 1)
        self.assertGreaterEqual(summary["objections_extracted"], 1)
        self.assertGreaterEqual(summary["availability_tracked_pairs"], 1)

    def test_evaluate_forward_in_time_passes_gate(self):
        eval_res = self.adapter.evaluate_forward_in_time(top_k=5, beta=1.25)
        self.assertEqual(eval_res["total_queries"], 3)
        self.assertEqual(eval_res["recovery_rate"], 1.0)
        self.assertEqual(eval_res["hits"], 3)


class TestHeuresisCLI(unittest.TestCase):
    """Test click CLI invocation of kybernetes heuresis subcommands."""

    def setUp(self):
        self.runner = CliRunner()

    def test_cli_heuresis_search(self):
        result = self.runner.invoke(cli, ["heuresis", "search", "--top-k", "3"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Heuresis Discovery Search", result.output)
        self.assertIn("cellular_sheaves", result.output)

    def test_cli_heuresis_search_json(self):
        result = self.runner.invoke(cli, ["heuresis", "search", "--top-k", "2", "--json-out"])
        self.assertEqual(result.exit_code, 0)
        data = json.loads(result.output)
        self.assertEqual(len(data), 2)
        self.assertIn("discovery_score", data[0])

    def test_cli_heuresis_writeup(self):
        result = self.runner.invoke(cli, ["heuresis", "writeup", "--top-k", "1"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Unified Discovery:", result.output)
        self.assertIn("Falsifiable Hypothesis", result.output)
        self.assertIn("Failure Criterion", result.output)

    def test_cli_heuresis_crawl(self):
        sample_forums = [
            {
                "forum": "f1",
                "content": {"title": "Test", "abstract": "Test", "primary_area": "AI"},
                "reviews": [
                    {"rating": "4: borderline", "comment": "lacks a theoretical bridge between diffusion_guidance and cellular_sheaves"},
                    {"rating": "7: good paper", "comment": "interesting approach to cellular_sheaves"},
                ],
            }
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(sample_forums, tf)
            tf_path = tf.name

        try:
            result = self.runner.invoke(cli, ["heuresis", "crawl", "--input-file", tf_path])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Forums Parsed:              1", result.output)
            self.assertIn("Contested Boundary Forums:  1", result.output)
        finally:
            Path(tf_path).unlink(missing_ok=True)

    def test_cli_heuresis_eval(self):
        result = self.runner.invoke(cli, ["heuresis", "eval", "--top-k", "5"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Dokime Heuresis Forward-in-Time Evaluation", result.output)
        self.assertIn("Forward Recovery Gate:          PASS", result.output)


if __name__ == "__main__":
    unittest.main()
