"""Unit and integration tests for Symbolon discrete code adapter and E1 diagnostic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from symbolon.codes import (
    ScopedCode,
    compute_scoped_code_jaccard,
    compute_scoped_code_overlap,
)
from symbolon.diagnostic import E1_PROBE_SET, run_e1_diagnostic
from symbolon.tokenizer_adapter import SymbolonTokenizerAdapter


class TestScopedCode(unittest.TestCase):
    """Test the expanded 6-tuple discrete code identity rule."""

    def test_scoped_code_identity_and_formatting(self):
        c1 = ScopedCode(
            checkpoint_id="xsum_v1",
            tokenizer_id="xsum_vqvae_48",
            level=2,
            window_idx=1,
            pos=5,
            code_int=123,
        )
        self.assertEqual(c1.format_key(), "xsum_v1:xsum_vqvae_48:lvl2:w1:p5:c123")
        self.assertEqual(str(c1), c1.format_key())

        # Test dictionary conversion
        c_dict = c1.to_dict()
        self.assertEqual(c_dict["level"], 2)
        self.assertEqual(c_dict["window_idx"], 1)
        self.assertEqual(c_dict["code_int"], 123)

    def test_strict_scope_inequality(self):
        """Cross-window or cross-level codes must NEVER be considered identical."""
        base = ScopedCode("xsum_v1", "xsum_vqvae_48", level=0, window_idx=0, pos=0, code_int=42)

        # Different window
        diff_window = ScopedCode("xsum_v1", "xsum_vqvae_48", level=0, window_idx=1, pos=0, code_int=42)
        self.assertNotEqual(base, diff_window)
        self.assertNotEqual(hash(base), hash(diff_window))

        # Different level
        diff_level = ScopedCode("xsum_v1", "xsum_vqvae_48", level=1, window_idx=0, pos=0, code_int=42)
        self.assertNotEqual(base, diff_level)

        # Different position
        diff_pos = ScopedCode("xsum_v1", "xsum_vqvae_48", level=0, window_idx=0, pos=1, code_int=42)
        self.assertNotEqual(base, diff_pos)

        # Identical scope and code
        identical = ScopedCode("xsum_v1", "xsum_vqvae_48", level=0, window_idx=0, pos=0, code_int=42)
        self.assertEqual(base, identical)
        self.assertEqual(hash(base), hash(identical))


class TestScopedCodeOverlap(unittest.TestCase):
    """Test scoped discrete code overlap and Jaccard metrics."""

    def setUp(self):
        self.code_a = ScopedCode("xsum_v1", "xsum_vqvae_48", level=0, window_idx=0, pos=0, code_int=10)
        self.code_b = ScopedCode("xsum_v1", "xsum_vqvae_48", level=0, window_idx=0, pos=1, code_int=20)
        self.code_c = ScopedCode("xsum_v1", "xsum_vqvae_48", level=0, window_idx=1, pos=0, code_int=30)
        self.code_d = ScopedCode("xsum_v1", "xsum_vqvae_48", level=1, window_idx=0, pos=0, code_int=10)

    def test_overlap_metric(self):
        query = [self.code_a, self.code_b]
        doc1 = [self.code_a, self.code_c]  # Only code_a matches
        doc2 = [self.code_c, self.code_d]  # Zero matches (code_d is level 1, code_a is level 0)

        self.assertEqual(compute_scoped_code_overlap(query, doc1), 1)
        self.assertEqual(compute_scoped_code_overlap(query, doc2), 0)

    def test_jaccard_metric(self):
        query = [self.code_a, self.code_b]
        doc = [self.code_b, self.code_c]
        # Intersection: {code_b} (size 1)
        # Union: {code_a, code_b, code_c} (size 3)
        self.assertAlmostEqual(compute_scoped_code_jaccard(query, doc), 1.0 / 3.0, places=4)

        # Empty sets
        self.assertEqual(compute_scoped_code_jaccard([], []), 0.0)


class TestSymbolonTokenizerAdapter(unittest.TestCase):
    """Test passage chunking, multi-window code extraction, and reconstruction."""

    def setUp(self):
        self.adapter = SymbolonTokenizerAdapter(use_mock=True)

    def test_multi_window_chunking(self):
        """Passages must be cleanly split into 48-token context windows."""
        # 100 words ~ 130 tokens -> 3 windows of 48 tokens
        passage = "The quick brown fox jumps over the lazy dog. " * 20
        windows = self.adapter.chunk_passage_tokens(passage)

        self.assertGreaterEqual(len(windows), 2)
        for w in windows:
            self.assertEqual(len(w), 48)

    def test_encode_passage_scoped_codes(self):
        """Encoding a multi-window passage assigns passage-relative window indexes."""
        passage = "Apollo 11 landed on the moon in 1969. " * 15
        codes = self.adapter.encode_passage(passage)

        self.assertTrue(codes)
        # Verify codes have valid window indices
        window_indices = {c.window_idx for c in codes}
        self.assertIn(0, window_indices)
        self.assertIn(1, window_indices)

        # Verify all 9 levels are represented
        levels_present = {c.level for c in codes}
        self.assertEqual(levels_present, set(range(9)))

    def test_reconstruct_window_previews(self):
        """Reconstructing a 48-token window returns previews for all 9 levels."""
        window_text = "The cellular sheaf Laplacian bounds topological drift in knowledge graphs."
        previews = self.adapter.reconstruct_window(window_text)

        self.assertEqual(len(previews), 9)
        for p in previews:
            self.assertTrue(isinstance(p, str))
            self.assertTrue(p)


class TestSymbolonDiagnosticE1(unittest.TestCase):
    """Test Experiment E1: Reconstruction-Fidelity Diagnostic execution."""

    def test_run_e1_diagnostic_and_report_generation(self):
        adapter = SymbolonTokenizerAdapter(use_mock=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "e1_report.md"

            report_data = run_e1_diagnostic(adapter=adapter, output_report_path=report_path)

            self.assertEqual(report_data["experiment"], "E1_reconstruction_fidelity")
            self.assertEqual(report_data["probes_evaluated"], len(E1_PROBE_SET))
            self.assertEqual(len(report_data["level_summary"]), 9)

            # Check that report file was created and contains key headers
            self.assertTrue(report_path.exists())
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("# Symbolon Experiment E1: Reconstruction-Fidelity Diagnostic Report", content)
            self.assertIn("probe_entities_geo", content)
            self.assertIn("probe_topological_math", content)
            self.assertIn("probe_numerical_data", content)
            self.assertIn("probe_scientific_biochem", content)


if __name__ == "__main__":
    unittest.main()
