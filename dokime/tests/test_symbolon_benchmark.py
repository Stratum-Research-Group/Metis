"""
Unit and Integration Tests for Dokime Symbolon Discrete Code Retrieval Benchmark
================================================================================
Verifies:
  1. SymbolonBenchmarkTask dataset loading and Query protocol compliance.
  2. Document ID extraction from various LLM response formats.
  3. Detailed scoring (gold recall, contradiction retention).
  4. 4-arm comparative benchmark execution (Baseline, Symbolon Alone, Uncapped Union, Capped RRF).
  5. Markdown reporting and CLI subcommand execution.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dokime.cli import main
from dokime.tasks.protocol import Query
from dokime.tasks.registry import get_task
from dokime.tasks.symbolon_benchmark import (
    SymbolonBenchmarkReport,
    SymbolonBenchmarkTask,
    SymbolonRetrievalBenchmarkSuite,
    extract_retrieved_doc_ids,
)


class TestSymbolonBenchmarkTask(unittest.TestCase):
    """Test suite for SymbolonBenchmarkTask plugin and scoring logic."""

    def setUp(self):
        self.task = SymbolonBenchmarkTask()

    def test_registry_discovery(self):
        """Verify symbolon_code_retrieval is registered and loadable via registry."""
        task = get_task("symbolon_code_retrieval")
        self.assertIsInstance(task, SymbolonBenchmarkTask)
        self.assertEqual(task.task_id, "symbolon_code_retrieval")

    def test_dataset_loading(self):
        """Verify queries load properly from symbolon_suite.json."""
        queries = self.task.load()
        self.assertEqual(len(queries), 8)
        for q in queries:
            self.assertIsInstance(q, Query)
            self.assertTrue(q.id)
            self.assertTrue(q.prompt)
            self.assertIn("gold_relevant", q.metadata)
            self.assertIn("contradiction_docs", q.metadata)
            self.assertGreater(len(q.metadata["gold_relevant"]), 0)

    def test_extract_retrieved_doc_ids_formats(self):
        """Verify extraction across JSON array, markdown codeblock, and token list."""
        # JSON array
        r1 = extract_retrieved_doc_ids('["doc_sci_01", "doc_sci_03_contradiction"]')
        self.assertEqual(r1, ["doc_sci_01", "doc_sci_03_contradiction"])

        # Markdown JSON code block
        r2 = extract_retrieved_doc_ids("```json\n[\"doc_geo_01\", \"doc_geo_02\"]\n```")
        self.assertEqual(r2, ["doc_geo_01", "doc_geo_02"])

        # Free text with doc tokens
        r3 = extract_retrieved_doc_ids("Retrieved doc_space_01 followed by doc_space_03_contradiction.")
        self.assertEqual(r3, ["doc_space_01", "doc_space_03_contradiction"])

        # Empty or whitespace
        self.assertEqual(extract_retrieved_doc_ids(""), [])
        self.assertEqual(extract_retrieved_doc_ids("   "), [])

    def test_score_and_score_detailed(self):
        """Verify recall and contradiction metrics computation."""
        q = Query(
            id="q_test",
            prompt="Test prompt",
            metadata={
                "gold_relevant": ["doc_sci_01"],
                "contradiction_docs": ["doc_sci_03_contradiction"],
            },
        )

        # Full hit
        resp_full = '["doc_sci_01", "doc_sci_03_contradiction"]'
        score = self.task.score(q, resp_full)
        self.assertEqual(score, 1.0)
        detailed = self.task.score_detailed(q, resp_full)
        self.assertEqual(detailed["gold_recall"], 1.0)
        self.assertEqual(detailed["contradiction_retention"], 1.0)
        self.assertTrue(detailed["hit_gold"])
        self.assertTrue(detailed["hit_contradiction"])

        # Gold miss, contradiction hit
        resp_miss = '["doc_sci_02_distractor", "doc_sci_03_contradiction"]'
        score_miss = self.task.score(q, resp_miss)
        self.assertEqual(score_miss, 0.0)
        detailed_miss = self.task.score_detailed(q, resp_miss)
        self.assertEqual(detailed_miss["gold_recall"], 0.0)
        self.assertEqual(detailed_miss["contradiction_retention"], 1.0)
        self.assertFalse(detailed_miss["hit_gold"])
        self.assertTrue(detailed_miss["hit_contradiction"])


class TestSymbolonRetrievalBenchmarkSuite(unittest.TestCase):
    """Test suite for the 4-arm comparative benchmark execution and reporting."""

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed; skipping Symbolon retrieval benchmark")

    def test_run_benchmark_mock_adapter(self):
        """Verify execution of all 4 comparative arms."""
        suite = SymbolonRetrievalBenchmarkSuite(top_k=3, level=4)
        report = suite.run_benchmark()

        self.assertIsInstance(report, SymbolonBenchmarkReport)
        self.assertEqual(report.top_k, 3)
        self.assertEqual(report.level_evaluated, 4)
        self.assertEqual(report.decision_verdict, "ARCHIVE_UNCAPPED_NO_LOSS_RECORDED")

        # Verify all 4 comparative arms
        expected_arms = {"baseline", "symbolon_alone", "uncapped_union", "capped_fusion"}
        self.assertEqual(set(report.arms.keys()), expected_arms)

        baseline = report.arms["baseline"]
        self.assertEqual(baseline.mean_recall, 1.0)
        self.assertEqual(baseline.candidate_expansion, 1.0)

        uncapped = report.arms["uncapped_union"]
        self.assertEqual(uncapped.mean_recall, 1.0)
        self.assertGreater(uncapped.candidate_expansion, 1.0)

        # Markdown formatting check
        md = report.to_markdown()
        self.assertIn("# Symbolon Retrieval Benchmark", md)
        self.assertIn("ARCHIVE_UNCAPPED_NO_LOSS_RECORDED", md)
        self.assertIn("Flat Dense Baseline (LTHCM Flat)", md)
        self.assertIn("Uncapped Union (Base + Symbolon)", md)
        self.assertIn("Capped RRF Fusion", md)

    def test_cli_symbolon_benchmark(self):
        """Verify dokime symbolon-benchmark CLI command execution and JSON export."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            exit_code = main(["symbolon-benchmark", "--top-k", "3", "--level", "4", "--output", str(tmp_path)])
            self.assertEqual(exit_code, 0)
            self.assertTrue(tmp_path.exists())

            with open(tmp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["decision_verdict"], "ARCHIVE_UNCAPPED_NO_LOSS_RECORDED")
            self.assertIn("arms", data)
            self.assertIn("baseline", data["arms"])
            self.assertIn("uncapped_union", data["arms"])
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
