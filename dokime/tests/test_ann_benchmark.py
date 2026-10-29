"""
Unit Tests for Dokime ANN Benchmark Suite
=========================================
Verifies:
  1. Benchmark configuration and synthetic vector generation.
  2. Exact baseline computation and recall metrics.
  3. ScaNN Tree-AH and Brute-Force evaluation paths.
  4. Formatted comparison table generation.
  5. CLI ann-benchmark execution.
"""

import tempfile
import shutil
import unittest
from pathlib import Path
import numpy as np

from dokime.tasks.ann_benchmark import (
    ANNBenchmarkConfig,
    ANNBenchmarkReport,
    ANNBenchmarkSuite,
)
from dokime.cli import main


class TestANNBenchmark(unittest.TestCase):

    def setUp(self):
        self.config = ANNBenchmarkConfig(
            num_documents=250,
            d_model=32,
            num_queries=15,
            top_k=5,
            seed=42,
        )

    def test_synthetic_data_generation(self):
        """Verify synthetic embeddings are correctly sized and normalized."""
        suite = ANNBenchmarkSuite(self.config)
        corpus, doc_ids, queries = suite.generate_synthetic_data()

        self.assertEqual(corpus.shape, (250, 32))
        self.assertEqual(len(doc_ids), 250)
        self.assertEqual(queries.shape, (15, 32))

        # Check normalization
        corpus_norms = np.linalg.norm(corpus, axis=1)
        np.testing.assert_allclose(corpus_norms, 1.0, atol=1e-5)
        queries_norms = np.linalg.norm(queries, axis=1)
        np.testing.assert_allclose(queries_norms, 1.0, atol=1e-5)

    def test_benchmark_suite_execution(self):
        """Verify benchmark runs and returns valid report schema with exact metrics."""
        suite = ANNBenchmarkSuite(self.config)
        report = suite.run()

        self.assertIsInstance(report, ANNBenchmarkReport)
        self.assertEqual(report.num_documents, 250)
        self.assertEqual(report.d_model, 32)
        self.assertEqual(report.top_k, 5)

        self.assertIn("exact", report.algorithms)
        exact_res = report.algorithms["exact"]
        self.assertEqual(exact_res.recall_at_1, 1.0)
        self.assertEqual(exact_res.recall_at_k, 1.0)
        self.assertEqual(exact_res.speedup_vs_exact, 1.0)
        self.assertGreater(exact_res.qps, 0.0)

        # Check ScaNN algorithms if installed
        if "scann_brute_force" in report.algorithms:
            bf_res = report.algorithms["scann_brute_force"]
            self.assertEqual(bf_res.recall_at_1, 1.0)
            self.assertEqual(bf_res.recall_at_k, 1.0)
            self.assertGreater(bf_res.speedup_vs_exact, 0.5)

        if "scann_tree_ah" in report.algorithms:
            ah_res = report.algorithms["scann_tree_ah"]
            self.assertGreaterEqual(ah_res.recall_at_1, 0.0)
            self.assertLessEqual(ah_res.recall_at_1, 1.0)
            self.assertGreater(ah_res.qps, 0.0)

        # Check table formatting
        table = report.format_table()
        self.assertIn("Exact Linear Dot-Product", table)
        self.assertIn("| Algorithm | Recall@1 | Recall@K |", table)

    def test_cli_ann_benchmark(self):
        """Verify dokime ann-benchmark CLI invocation and JSON persistence."""
        tmp_dir = tempfile.mkdtemp(prefix="dokime_ann_test_")
        out_file = Path(tmp_dir) / "ann_report.json"
        try:
            exit_code = main([
                "ann-benchmark",
                "--num-docs", "150",
                "--dim", "16",
                "--num-queries", "10",
                "--top-k", "4",
                "--output", str(out_file),
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.is_file())
            content = out_file.read_text(encoding="utf-8")
            self.assertIn("algorithms", content)
            self.assertIn("Exact Linear Dot-Product", content)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
