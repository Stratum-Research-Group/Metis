"""Unit tests for the Dianoia CLI."""

import argparse
import os
import tempfile
import unittest

from dianoia.cli import _load_documents, _print_walk_result


class TestCLI(unittest.TestCase):
    def test_load_documents_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "doc1.txt")
            f2 = os.path.join(tmpdir, "doc2.md")
            f3 = os.path.join(tmpdir, "ignored.bin")

            with open(f1, "w", encoding="utf-8") as f:
                f.write("Content of text doc.")
            with open(f2, "w", encoding="utf-8") as f:
                f.write("# Markdown Document\nContent of markdown doc.")
            with open(f3, "wb") as f:
                f.write(b"\x00\x01\x02")

            docs = _load_documents(tmpdir)
            self.assertEqual(len(docs), 2)
            doc_ids = {d["id"] for d in docs}
            self.assertEqual(doc_ids, {"doc1", "doc2"})

    def test_load_documents_empty_dir_exits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(SystemExit) as cm:
                _load_documents(tmpdir)
            self.assertEqual(cm.exception.code, 1)

    def test_load_documents_non_dir_exits(self):
        with self.assertRaises(SystemExit) as cm:
            _load_documents("/non/existent/path/that/cannot/exist/12345")
        self.assertEqual(cm.exception.code, 1)

    def test_cli_argument_parser_mutual_exclusion(self):
        parser = argparse.ArgumentParser(prog="dianoia")
        source_group = parser.add_mutually_exclusive_group(required=True)
        source_group.add_argument("--docs", metavar="DIR")
        source_group.add_argument("--index", metavar="PATH")
        parser.add_argument("question")
        parser.add_argument("--max-steps", type=int, default=8)

        # Valid with --docs
        args1 = parser.parse_args(["--docs", "/path/to/docs", "What is memory?"])
        self.assertEqual(args1.docs, "/path/to/docs")
        self.assertEqual(args1.question, "What is memory?")
        self.assertEqual(args1.max_steps, 8)

        # Valid with --index
        args2 = parser.parse_args(["--index", "/path/to/index.pkl", "What is memory?"])
        self.assertEqual(args2.index, "/path/to/index.pkl")
        self.assertEqual(args2.question, "What is memory?")

        # Fails when both provided
        with self.assertRaises(SystemExit):
            parser.parse_args(["--docs", "/path", "--index", "/path", "Question"])

        # Fails when neither provided
        with self.assertRaises(SystemExit):
            parser.parse_args(["Question"])

    def test_print_walk_result_formatting(self):
        # Ensure pretty-printer runs cleanly without exception
        result = {
            "walk_path": [
                {"node": "doc_1", "action": "EXPAND", "reason": "Moving", "parse_failed": False},
                {"node": "doc_2", "action": "ANSWER", "reason": "Done", "parse_failed": False},
            ],
            "verification": {
                "report": "Verification consistent",
                "conflicts_found": False,
                "rewalked": False,
            },
            "final_answer": "Final synthesized answer.",
            "model": "qwen2.5-7b-instruct",
            "fallback_used": False,
            "steps_used": 2,
            "seeded_nodes": ["doc_1"],
        }
        # Should execute without error
        _print_walk_result(result)

    def test_load_index_overrides_nli_defaults(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from unittest.mock import MagicMock, patch
        from dianoia.cli import _load_index
        from lthcm.engine.pipeline import LTHCMEngine

        mock_engine = MagicMock()
        mock_engine.__class__ = LTHCMEngine
        mock_sheaf = MagicMock()
        mock_sheaf.nli_model_name = "stale-model"
        mock_sheaf.threshold = 0.1
        mock_sheaf._nli_model = MagicMock()
        mock_engine.sheaf_filter = mock_sheaf
        mock_engine.wormhole.graph.number_of_edges.return_value = 4

        with patch("pickle.load", return_value=mock_engine):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("builtins.open", MagicMock()):
                    loaded = _load_index("/fake/path.pkl", allow_trusted_pickle=True)
                    self.assertEqual(
                        loaded.sheaf_filter.nli_model_name,
                        "cross-encoder/nli-deberta-v3-xsmall",
                    )
                    self.assertEqual(loaded.sheaf_filter.threshold, 2.5)
                    self.assertIsNone(loaded.sheaf_filter._nli_model)

    def test_load_index_untrusted_pickle_refused(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from dianoia.cli import _load_index

        with self.assertRaises(SystemExit) as cm:
            _load_index("/fake/path.pkl", allow_trusted_pickle=False)
        self.assertEqual(cm.exception.code, 1)

    def test_load_index_missing_sheaf_filter_exits(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        from unittest.mock import MagicMock, patch
        from dianoia.cli import _load_index
        from lthcm.engine.pipeline import LTHCMEngine

        mock_engine = MagicMock()
        mock_engine.__class__ = LTHCMEngine
        mock_engine.sheaf_filter = None

        with patch("pickle.load", return_value=mock_engine):
            with patch("pathlib.Path.is_file", return_value=True):
                with patch("builtins.open", MagicMock()):
                    with self.assertRaises(SystemExit) as cm:
                        _load_index("/fake/path.pkl", allow_trusted_pickle=True)
                    self.assertEqual(cm.exception.code, 1)

    def test_print_walk_result_rewalk_failed_status(self):
        import io
        from contextlib import redirect_stdout
        from dianoia.cli import _print_walk_result

        result = {
            "walk_path": [
                {"node": "doc_1", "action": "EXPAND", "reason": "Moving", "parse_failed": False},
            ],
            "verification": {
                "report": "Conflicts found",
                "conflicts_found": True,
                "rewalked": False,
                "rewalk_failed": True,
                "rewalk_error": "Connection reset by peer",
            },
            "final_answer": "Final synthesized answer.",
            "model": "qwen2.5-7b-instruct",
            "fallback_used": False,
            "steps_used": 1,
            "seeded_nodes": ["doc_1"],
        }
        f = io.StringIO()
        with redirect_stdout(f):
            _print_walk_result(result)
        out = f.getvalue()
        self.assertIn("Re-walk attempt failed: Connection reset by peer", out)


if __name__ == "__main__":
    unittest.main()
