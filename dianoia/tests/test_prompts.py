"""Unit tests for prompt templates."""

import unittest

from dianoia.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    CONTROLLER_SYSTEM_PROMPT,
    REWALK_CONFLICT_PREFIX,
    STEP_PROMPT_TEMPLATE,
)


class TestPromptTemplates(unittest.TestCase):
    def test_system_prompt_contains_core_actions(self):
        self.assertIn("ACTION: EXPAND <neighbour_id>", CONTROLLER_SYSTEM_PROMPT)
        self.assertIn("ACTION: BACKTRACK", CONTROLLER_SYSTEM_PROMPT)
        self.assertIn("ACTION: ANSWER", CONTROLLER_SYSTEM_PROMPT)
        self.assertIn("REASON:", CONTROLLER_SYSTEM_PROMPT)

    def test_step_prompt_template_formatting(self):
        formatted = STEP_PROMPT_TEMPLATE.format(
            question="What is the capital of France?",
            current_node_id="doc_1",
            current_content="Paris is the capital of France.",
            neighbours_block="  [doc_2]: France is in Europe.",
            steps_remaining=5,
            visited_summary="  [doc_1]: Paris is...",
        )
        self.assertIn("What is the capital of France?", formatted)
        self.assertIn("doc_1", formatted)
        self.assertIn("Paris is the capital of France.", formatted)
        self.assertIn("Steps remaining: 5", formatted)

    def test_answer_prompt_template_formatting(self):
        formatted = ANSWER_PROMPT_TEMPLATE.format(
            question="What is X?",
            visited_passages="[node_1]: Content about X.",
        )
        self.assertIn("What is X?", formatted)
        self.assertIn("[node_1]: Content about X.", formatted)

    def test_rewalk_conflict_prefix_formatting(self):
        formatted = REWALK_CONFLICT_PREFIX.format(
            conflict_report="Answer asserted Y but node_1 stated not Y."
        )
        self.assertIn("Answer asserted Y but node_1 stated not Y.", formatted)
        self.assertIn("CONFLICT REPORT:", formatted)


    def test_synthesis_system_prompt_defined(self):
        from dianoia.prompts import SYNTHESIS_SYSTEM_PROMPT
        self.assertIn("synthesize a comprehensive", SYNTHESIS_SYSTEM_PROMPT)
        self.assertNotIn("ACTION: EXPAND", SYNTHESIS_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
