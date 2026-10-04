"""Unit tests for walk logging module."""

import json
import os
import tempfile
import unittest

from dianoia.log import SCHEMA_VERSION, append_walk_log, build_walk_record


class TestWalkLogging(unittest.TestCase):
    def test_schema_version(self):
        self.assertEqual(SCHEMA_VERSION, 1)

    def test_build_walk_record(self):
        record = build_walk_record(
            question="What is hippocampal replay?",
            seeded_nodes=["node_1", "node_2"],
            path=[
                {
                    "node": "node_1",
                    "action": "EXPAND",
                    "raw_output": "ACTION: EXPAND node_2\nREASON: Relevant",
                    "reason": "Relevant",
                    "parse_failed": False,
                },
                {
                    "node": "node_2",
                    "action": "ANSWER",
                    "raw_output": "ACTION: ANSWER\nREASON: Enough evidence",
                    "reason": "Enough evidence",
                    "parse_failed": False,
                },
            ],
            verification={
                "conflicts_found": False,
                "rewalked": False,
                "report": "Consistent",
                "conflict_detail": None,
            },
            final_answer="Hippocampal replay is...",
            max_steps=8,
            steps_used=2,
            model="qwen2.5-7b-instruct",
            fallback_used=False,
        )

        expected_keys = {
            "schema_version",
            "walk_id",
            "question",
            "seeded_nodes",
            "path",
            "verification",
            "final_answer",
            "max_steps",
            "steps_used",
            "model",
            "fallback_used",
        }
        self.assertEqual(set(record.keys()), expected_keys)
        self.assertEqual(record["schema_version"], 1)
        self.assertTrue(isinstance(record["walk_id"], str) and len(record["walk_id"]) > 0)
        self.assertEqual(record["max_steps"], 8)
        self.assertEqual(record["steps_used"], 2)
        self.assertFalse(record["fallback_used"])
        self.assertEqual(len(record["path"]), 2)

    def test_build_walk_record_with_initial_path(self):
        initial = [{"node": "n1", "action": "EXPAND", "raw_output": "...", "reason": "", "parse_failed": False}]
        record = build_walk_record(
            question="Question?",
            seeded_nodes=["n1"],
            path=[],
            verification={"conflicts_found": False, "rewalked": True, "report": "ok", "conflict_detail": None},
            final_answer="Answer",
            max_steps=8,
            steps_used=1,
            model="test-model",
            fallback_used=False,
            initial_path=initial,
        )
        self.assertIn("initial_path", record)
        self.assertEqual(record["initial_path"], initial)

    def test_append_walk_log_creates_directory_and_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "nested", "sub", "walks.jsonl")

            rec1 = build_walk_record(
                question="Question 1",
                seeded_nodes=["n1"],
                path=[],
                verification={"conflicts_found": False, "rewalked": False, "report": "ok", "conflict_detail": None},
                final_answer="Answer 1",
                max_steps=8,
                steps_used=1,
                model="test-model",
                fallback_used=False,
            )

            rec2 = build_walk_record(
                question="Question 2 with non-ASCII: café résumé",
                seeded_nodes=["n2"],
                path=[],
                verification={"conflicts_found": True, "rewalked": True, "report": "conflict", "conflict_detail": "detail"},
                final_answer="Answer 2",
                max_steps=8,
                steps_used=3,
                model="test-model",
                fallback_used=True,
            )

            written_path = append_walk_log(rec1, log_path=log_path)
            self.assertEqual(written_path, log_path)
            self.assertTrue(os.path.exists(log_path))

            append_walk_log(rec2, log_path=log_path)

            with open(log_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            self.assertEqual(len(lines), 2)
            parsed_1 = json.loads(lines[0])
            parsed_2 = json.loads(lines[1])
            self.assertEqual(parsed_1["question"], "Question 1")
            self.assertEqual(parsed_2["question"], "Question 2 with non-ASCII: café résumé")
            self.assertTrue(parsed_2["fallback_used"])


if __name__ == "__main__":
    unittest.main()
