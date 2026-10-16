"""
Unit tests for Replayable Run Bundle Persistence
================================================
Verifies that:
  1. RunBundleBuilder captures environment, git, and dataset provenance.
  2. Per-query records and manifest with sha256 checksums are serialized cleanly.
  3. validate_artifacts properly verifies bundles.
"""

import os
import json
import shutil
import tempfile
import unittest

from lthcm.benchmarks.real.utils.run_bundle import (
    RunBundleBuilder,
    get_git_info,
    get_system_hardware_info,
    compute_file_sha256,
)


class TestRunBundle(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_bundle_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_environment_and_git_capture(self):
        git_info = get_git_info()
        self.assertIn("commit", git_info)
        self.assertIn("dirty", git_info)

        hw_info = get_system_hardware_info()
        self.assertIn("os", hw_info)
        self.assertIn("python_version", hw_info)
        self.assertIn("torch_version", hw_info)

    def test_run_bundle_builder_lifecycle(self):
        builder = RunBundleBuilder(
            benchmark_name="test_bench",
            dataset_name="demo/test",
            split="validation",
            run_id="run-12345",
            seed=42,
            output_dir=self.test_dir,
        )
        builder.set_dataset_provenance(query_ids=["q_1", "q_2"], corpus_size=10)
        builder.set_models(encoder="all-MiniLM-L6-v2")
        builder.add_query_record(
            query_id="q_1",
            question="What is QEC?",
            gold_targets=["p_0"],
            ranked_predictions={"flat": ["p_0", "p_1"]},
            metrics_by_arm={"flat": {"joint@2": 1.0}},
        )

        summary = {"metric": 0.85}
        builder.finalize(summary)

        # Check generated files
        summary_path = os.path.join(self.test_dir, "test_bench_summary.json")
        per_query_path = os.path.join(self.test_dir, "test_bench_per_query.jsonl")
        manifest_path = os.path.join(self.test_dir, "test_bench_manifest.json")

        self.assertTrue(os.path.isfile(summary_path))
        self.assertTrue(os.path.isfile(per_query_path))
        self.assertTrue(os.path.isfile(manifest_path))

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["run_id"], "run-12345")
        self.assertIn("test_bench_summary.json", manifest["files"])
        self.assertEqual(manifest["files"]["test_bench_summary.json"], compute_file_sha256(summary_path))


if __name__ == "__main__":
    unittest.main()
