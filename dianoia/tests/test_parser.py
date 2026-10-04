"""Unit tests for the controller output action parser."""

import unittest

from dianoia.walk import parse_controller_output


class TestActionParser(unittest.TestCase):
    def setUp(self):
        self.valid_neighbours = ["node_alpha", "node_beta", "node_gamma"]

    def test_valid_expand(self):
        raw = "ACTION: EXPAND node_alpha\nREASON: Node alpha discusses the mechanism."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "EXPAND")
        self.assertEqual(parsed.target, "node_alpha")
        self.assertEqual(parsed.reason, "Node alpha discusses the mechanism.")
        self.assertFalse(parsed.parse_failed)
        self.assertEqual(parsed.raw_output, raw)

    def test_expand_invalid_neighbour_triggers_policy(self):
        # Per spec: names a non-existent neighbour -> treat as BACKTRACK, parse_failed=True
        raw = "ACTION: EXPAND node_unknown\nREASON: Looks promising."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertIsNone(parsed.target)
        self.assertTrue(parsed.parse_failed)

    def test_valid_backtrack(self):
        raw = "ACTION: BACKTRACK\nREASON: Current node does not contain relevant facts."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertIsNone(parsed.target)
        self.assertFalse(parsed.parse_failed)
        self.assertEqual(parsed.reason, "Current node does not contain relevant facts.")

    def test_valid_answer(self):
        raw = "ACTION: ANSWER\nREASON: We have collected all required information."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "ANSWER")
        self.assertIsNone(parsed.target)
        self.assertFalse(parsed.parse_failed)
        self.assertEqual(parsed.reason, "We have collected all required information.")

    def test_multiple_action_lines_triggers_policy(self):
        # Per spec: lists multiple actions -> treat as BACKTRACK, parse_failed=True
        raw = (
            "ACTION: EXPAND node_alpha\n"
            "ACTION: ANSWER\n"
            "REASON: Conflicted"
        )
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)

    def test_missing_action_line_triggers_policy(self):
        raw = "I think we should look into node_alpha because it is relevant."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)
        self.assertEqual(parsed.reason, "")

    def test_unknown_action_triggers_policy(self):
        raw = "ACTION: SEARCH query\nREASON: Trying to search."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)

    def test_lowercase_action_rejected(self):
        # Round 3 Item 3: lowercase "action:" must be rejected
        raw = "action: EXPAND node_beta\nREASON: Lowercase should fail."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)
    def test_expand_with_trailing_answer_fails(self):
        # Item 6: "ACTION: EXPAND valid_id ANSWER" must fail
        raw = "ACTION: EXPAND node_alpha ANSWER\nREASON: Mixed actions."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)

    def test_expand_with_extra_trailing_tokens_fails(self):
        # Item 6: extra trailing tokens must fail
        raw = "ACTION: EXPAND node_alpha now\nREASON: Extra token."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)

    def test_backtrack_with_extra_trailing_tokens_fails(self):
        raw = "ACTION: BACKTRACK to previous\nREASON: Extra token."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)

    def test_answer_with_extra_trailing_tokens_fails(self):
        raw = "ACTION: ANSWER immediately\nREASON: Extra token."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
    def test_notaction_answer_fails(self):
        # Round 2 Item 3: "NOTACTION: ANSWER" must not pass
        raw = "NOTACTION: ANSWER\nREASON: Fake action line."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)

    def test_extra_prose_lines_fail(self):
        # Round 2 Item 3: extra prose lines must cause parse failure
        raw = "Let me think about this step.\nACTION: ANSWER\nREASON: Done."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)

    def test_missing_reason_line_fails(self):
        raw = "ACTION: ANSWER"
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)

    def test_reason_first_rejected(self):
        # Round 3 Item 3: REASON before ACTION must be rejected
        raw = "REASON: Done.\nACTION: ANSWER"
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)

    def test_trailing_blank_line_rejected(self):
        # Round 3 Item 3: trailing blank line must be rejected
        raw = "ACTION: ANSWER\nREASON: Information is complete.\n"
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)

    def test_blank_line_between_rejected(self):
        # Round 3 Item 3: blank line between ACTION and REASON must be rejected
        raw = "ACTION: ANSWER\n\nREASON: Information is complete."
        parsed = parse_controller_output(raw, self.valid_neighbours)
        self.assertEqual(parsed.action, "BACKTRACK")
        self.assertTrue(parsed.parse_failed)


if __name__ == "__main__":
    unittest.main()
