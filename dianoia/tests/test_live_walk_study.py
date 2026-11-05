"""Unit and integration tests for Anadrome Milestone 3.2 Live Walk Test Study."""

from __future__ import annotations

import unittest
from dianoia.anadrome.study import (
    AnadromeLiveStudy,
    AnadromeStudyReport,
    FROZEN_BENCHMARK_CORPUS,
    FROZEN_TEST_QUESTIONS,
    PREDECLARED_SEEDS,
)


class TestAnadromeLiveStudy(unittest.TestCase):
    """Test harness and evaluation criteria for Anadrome Milestone 3.2."""

    def setUp(self):
        try:
            import torch
            if torch is None:
                self.skipTest("torch not installed")
        except ImportError:
            self.skipTest("torch not installed")
        self.study = AnadromeLiveStudy(seeds=[42])

    def test_frozen_dataset_and_graph_integrity(self):
        """Verify leakage discipline, unique QIDs, and valid graph structure."""
        self.assertGreaterEqual(len(FROZEN_TEST_QUESTIONS), 10)
        qids = [q["qid"] for q in FROZEN_TEST_QUESTIONS]
        self.assertEqual(len(qids), len(set(qids)), "QIDs must be unique")

        # Check question families
        families = {q["family"] for q in FROZEN_TEST_QUESTIONS}
        self.assertIn("multi_hop_entity", families)
        self.assertIn("single_hop_precision", families)
        self.assertIn("contradiction_resolution", families)
        self.assertIn("temporal_reasoning", families)

        # Check graph nodes exist
        for q in FROZEN_TEST_QUESTIONS:
            self.assertIn(q["start_node"], self.study.node_metadata)
            self.assertGreater(len(q["gold_key_terms"]), 0)

        self.assertEqual(len(PREDECLARED_SEEDS), 3)

    def test_single_query_execution_baseline_and_candidate(self):
        """Verify individual arm query execution and result contracts."""
        q = FROZEN_TEST_QUESTIONS[0]  # Q01 astronomy multihop
        res_baseline = self.study.run_arm_query(q, "baseline_llm", seed=42)

        self.assertEqual(res_baseline.policy_name, "baseline_llm")
        self.assertEqual(res_baseline.qid, q["qid"])
        self.assertTrue(res_baseline.correct)
        self.assertGreater(res_baseline.steps, 0)
        self.assertFalse(res_baseline.hard_error)

        res_greedy = self.study.run_arm_query(q, "greedy_similarity", seed=42)
        self.assertEqual(res_greedy.policy_name, "greedy_similarity")
        self.assertGreater(res_greedy.steps, 0)

    def test_sheaf_contradiction_backtrack_rejects_conflicts(self):
        """Verify SheafContradictionBacktrackPolicy avoids hard error on contradictory graphs."""
        contradiction_qs = [q for q in FROZEN_TEST_QUESTIONS if q.get("has_contradiction")]
        self.assertGreater(len(contradiction_qs), 0)

        for q in contradiction_qs:
            res_sheaf = self.study.run_arm_query(q, "sheaf_contradiction_backtrack", seed=42)
            self.assertFalse(res_sheaf.hard_error, f"Policy failed to prevent hard error on {q['qid']}")

    def test_full_study_execution_and_dokime_bootstrap(self):
        """Execute full study over single seed and verify statistical output."""
        report = self.study.execute_study()

        self.assertIsInstance(report, AnadromeStudyReport)
        self.assertEqual(report.total_questions, len(FROZEN_TEST_QUESTIONS))
        self.assertIn("baseline_llm", report.baseline_stats.policy_name)
        self.assertEqual(report.baseline_stats.total_runs, len(FROZEN_TEST_QUESTIONS))

        # Check all candidate arms exist
        expected_candidates = [
            "greedy_similarity",
            "early_exit_answer",
            "sheaf_contradiction_backtrack",
            "stochastic_softmax",
        ]
        for cand in expected_candidates:
            self.assertIn(cand, report.candidate_stats)
            self.assertIn(cand, report.paired_comparisons)
            comp = report.paired_comparisons[cand]
            self.assertIn("promotion_verdict", comp)
            self.assertIn(comp["promotion_verdict"], ["PASS", "NO_EVIDENCE", "REGRESSION"])

        # Check markdown report formatting
        md_text = self.study.generate_markdown_report(report)
        self.assertIn("# Anadrome Milestone 3.2: Frozen Live Walk Test Report", md_text)
        self.assertIn("Dokime 10,000 Paired Bootstrap Resamples", md_text)
        self.assertIn("Decision Gate Recommendation", md_text)
