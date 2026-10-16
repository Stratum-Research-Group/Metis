"""
Unit tests for Real Benchmark Modules and Significance Testing
==============================================================
Verifies execution integrity and schema of real benchmark runners
in smoke configuration.
"""

import tempfile
import shutil
import unittest
import pytest

pytest.importorskip("faiss", reason="faiss-cpu not installed (optional benchmark extra)")

from lthcm.benchmarks.real.utils.significance import paired_bootstrap_test
from lthcm.benchmarks.real.hierarchy.wordnet_eval import WordNetHierarchyBenchmark
from lthcm.benchmarks.real.niah.ruler_eval import RULERBenchmark
from lthcm.benchmarks.real.multihop.musique_eval import MuSiQueRealBenchmark
from lthcm.benchmarks.real.multihop.hotpot_real_eval import HotpotQARealBenchmark
from lthcm.benchmarks.real.multihop.twowiki_eval import TwoWikiRealBenchmark
from lthcm.benchmarks.real.contradiction.vitaminc_eval import VitaminCRealBenchmark
from lthcm.benchmarks.real.ir.scifact_eval import SciFactRealBenchmark


class TestRealBenchmarks(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="lthcm_test_runs_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_paired_bootstrap_significance(self):
        """Verify paired bootstrap calculation on deterministic score arrays."""
        scores_a = [0.9, 0.85, 0.88, 0.92, 0.87, 0.91, 0.89, 0.93]
        scores_b = [0.6, 0.65, 0.62, 0.68, 0.61, 0.64, 0.63, 0.67]
        res = paired_bootstrap_test(scores_a, scores_b, num_samples=1000, confidence_level=0.95, seed=42)
        
        self.assertIn("p_value", res)
        self.assertIn("ci_lower", res)
        self.assertIn("ci_upper", res)
        self.assertTrue(res["is_significant"])
        self.assertGreater(res["delta"], 0.20)
        self.assertGreater(res["ci_lower"], 0.0)

    def test_wordnet_hierarchy_smoke(self):
        """Verify WordNet hyperbolic hierarchy benchmark executes and returns valid schema."""
        bench = WordNetHierarchyBenchmark(d_model=8)
        res = bench.run_evaluation(max_nodes=25, output_dir=self.test_dir)
        self.assertIn("poincare_hyperbolic", res)
        self.assertIn("euclidean", res)
        self.assertGreaterEqual(res["poincare_hyperbolic"]["map"], 0.0)
        self.assertLessEqual(res["poincare_hyperbolic"]["map"], 1.0)
        self.assertIn("significance_test", res)

    def test_ruler_smoke(self):
        """Verify RULER long-context benchmark executes and returns valid schema."""
        bench = RULERBenchmark()
        res = bench.run_evaluation(num_distractors=15, num_needles=3, output_dir=self.test_dir)
        self.assertIn("flat_dense_baseline", res)
        self.assertIn("lthcm_topological", res)
        self.assertGreaterEqual(res["lthcm_topological"]["recall@1"], 0.0)
        self.assertLessEqual(res["lthcm_topological"]["recall@1"], 1.0)

    def test_musique_smoke(self):
        """Verify MuSiQue benchmark executes and returns valid schema."""
        bench = MuSiQueRealBenchmark()
        res = bench.run_evaluation(max_questions=2, output_dir=self.test_dir, allow_synthetic_fallback=True)
        self.assertIn("summary", res)
        self.assertIn("flat_1shot", res["summary"])
        self.assertIn("flat_2step", res["summary"])
        self.assertIn("lthcm_mined", res["summary"])

    def test_hotpot_smoke(self):
        """Verify HotpotQA benchmark executes and returns valid schema."""
        bench = HotpotQARealBenchmark(allow_synthetic_fallback=True)
        res = bench.run_evaluation(max_questions=2, output_dir=self.test_dir)
        self.assertIn("summary", res)
        self.assertIn("by_question_type", res)
        self.assertIn("flat_1shot", res["summary"])
        self.assertIn("lthcm_mined", res["summary"])
        self.assertIn("significance_mined_vs_random_joint@2", res)

    def test_twowiki_smoke(self):
        """Verify 2WikiMultiHopQA benchmark executes and returns valid schema."""
        bench = TwoWikiRealBenchmark()
        res = bench.run_evaluation(max_questions=2, output_dir=self.test_dir, force_synthetic=True)
        self.assertIn("summary", res)
        self.assertIn("flat_1shot", res["summary"])
        self.assertIn("lthcm_mined", res["summary"])

    def test_vitaminc_smoke(self):
        """Verify VitaminC contradiction verification benchmark executes and returns valid schema."""
        bench = VitaminCRealBenchmark()
        res = bench.run_evaluation(max_pairs=4, output_dir=self.test_dir, allow_synthetic_fallback=True)
        self.assertIn("3_way_classification", res)
        self.assertIn("2_way_classification", res)
        c3 = res["3_way_classification"]
        self.assertIn("flat_cosine_baseline_a", c3)
        self.assertIn("raw_deberta_nli_baseline_b", c3)
        self.assertIn("pure_untrained_sheaf_arm_c1", c3)
        self.assertIn("nli_grounded_sheaf_arm_c2", c3)

    def test_scifact_smoke(self):
        """Verify SciFact real benchmark executes without crash and returns valid schema."""
        bench = SciFactRealBenchmark()
        res = bench.run_evaluation(max_queries=2, output_dir=self.test_dir)
        self.assertIn("summary", res)
        self.assertIn("bm25_lexical", res["summary"])
        self.assertIn("flat_dense", res["summary"])
        self.assertIn("lthcm_flat", res["summary"])
        self.assertIn("lthcm_graph", res["summary"])
        self.assertIn("lthcm_oracle_graph", res["summary"])
        self.assertIn("significance_mined_vs_random_ndcg@10", res)


if __name__ == "__main__":
    unittest.main()
