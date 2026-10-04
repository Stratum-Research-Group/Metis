"""Unit tests for the WalkController and deliberative walk pipeline."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

try:
    import torch
except ImportError:
    torch = None

from dianoia.walk import WalkController


class MockLLMResponse:
    def __init__(self, content: str):
        self.choices = [MagicMock(message=MagicMock(content=content))]


class TestWalkController(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("torch not installed")

        self.mock_engine = MagicMock()
        self.mock_embedder = MagicMock()
        self.mock_embedder.d_model = 64
        self.mock_embedder.encode_single.return_value = torch.zeros(64)
        self.mock_embedder.encode.return_value = torch.zeros(2, 64)

        # Mock engine graph / wormhole
        self.mock_engine.wormhole.get_bridges.return_value = []
        mock_graph = MagicMock()
        mock_graph.__contains__.side_effect = lambda k: k in ["node_1", "node_2"]
        mock_graph.successors.side_effect = lambda k: ["node_2"] if k == "node_1" else []
        mock_graph.number_of_edges.return_value = 2
        self.mock_engine.wormhole.graph = mock_graph
        self.mock_engine.wormhole.node_metadata = {
            "node_1": {"content": "Content of node 1."},
            "node_2": {"content": "Content of node 2."},
        }
        self.mock_engine.wormhole.node_embeddings = {
            "node_1": torch.zeros(64),
            "node_2": torch.zeros(64),
        }

        # Mock engine seeding
        self.mock_engine.stochastic_resonance.resonate_retrieval.return_value = [
            {"doc_id": "node_1", "content": "Content of node 1."}
        ]

        # Mock sheaf filter for verification
        self.mock_engine.sheaf_filter.evaluate_passages.return_value = {
            "is_consistent": True,
            "total_obstruction": 0.0,
            "conflicting_pairs": [],
        }

    def test_normal_walk_and_answer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "walks.jsonl")

            mock_client = MagicMock()
            # Step 1: EXPAND node_2
            # Step 2: ANSWER
            # Synthesis: Final answer text
            mock_client.chat.completions.create.side_effect = [
                MockLLMResponse("ACTION: EXPAND node_2\nREASON: Node 2 continues the topic."),
                MockLLMResponse("ACTION: ANSWER\nREASON: Information is complete."),
                MockLLMResponse("Synthesized answer based on node 1 and 2."),
            ]

            controller = WalkController(
                engine=self.mock_engine,
                embedder=self.mock_embedder,
                model_name="qwen-primary",
                fallback_model_name="llama-fallback",
                max_steps=5,
                top_k=2,
                log_path=log_path,
                client=mock_client,
            )

            result = controller.run("What is the connection?")

            self.assertEqual(result["final_answer"], "Synthesized answer based on node 1 and 2.")
            self.assertEqual(result["steps_used"], 2)
            self.assertFalse(result["fallback_used"])
            self.assertEqual(len(result["walk_path"]), 2)
            self.assertEqual(result["walk_path"][0]["action"], "EXPAND")
            self.assertEqual(result["walk_path"][1]["action"], "ANSWER")
            self.assertFalse(result["verification"]["conflicts_found"])
            self.assertFalse(result["verification"]["rewalked"])

            # Verify log was written
            self.assertTrue(os.path.exists(log_path))

    def test_parse_failure_policy_and_fallback_switch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "walks.jsonl")

            mock_client = MagicMock()
            # 3 consecutive parse failures
            # Followed by a valid ANSWER
            mock_client.chat.completions.create.side_effect = [
                MockLLMResponse("Malformed garbage 1"),
                MockLLMResponse("Malformed garbage 2"),
                MockLLMResponse("Malformed garbage 3"),
                MockLLMResponse("ACTION: ANSWER\nREASON: Finally answering."),
                MockLLMResponse("Synthesized final answer."),
            ]

            controller = WalkController(
                engine=self.mock_engine,
                embedder=self.mock_embedder,
                model_name="qwen-primary",
                fallback_model_name="llama-fallback",
                consecutive_failures_to_fallback=3,
                max_steps=6,
                log_path=log_path,
                client=mock_client,
            )

            result = controller.run("Test question")

            # Fallback should have been triggered after 3 failures
            self.assertTrue(result["fallback_used"])
            self.assertEqual(result["model"], "llama-fallback")
            # First 3 steps should be recorded as BACKTRACK with parse_failed=True
            self.assertTrue(result["walk_path"][0]["parse_failed"])
            self.assertEqual(result["walk_path"][0]["action"], "BACKTRACK")
            self.assertTrue(result["walk_path"][1]["parse_failed"])
            self.assertTrue(result["walk_path"][2]["parse_failed"])
            # Step 4 succeeded with ANSWER
            self.assertFalse(result["walk_path"][3]["parse_failed"])
            self.assertEqual(result["walk_path"][3]["action"], "ANSWER")

    def test_budget_exhaustion_forces_answer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "walks.jsonl")

            mock_client = MagicMock()
            # 3 steps of BACKTRACK, max_steps is 3
            mock_client.chat.completions.create.side_effect = [
                MockLLMResponse("ACTION: BACKTRACK\nREASON: Exploring"),
                MockLLMResponse("ACTION: BACKTRACK\nREASON: Exploring"),
                MockLLMResponse("ACTION: BACKTRACK\nREASON: Exploring"),
                MockLLMResponse("Answer from exhausted budget"),
            ]

            controller = WalkController(
                engine=self.mock_engine,
                embedder=self.mock_embedder,
                max_steps=3,
                log_path=log_path,
                client=mock_client,
            )

            result = controller.run("Question with budget limit")

            self.assertEqual(result["steps_used"], 3)
            # Budget exhausted forces ANSWER record
            self.assertEqual(result["walk_path"][-1]["action"], "ANSWER")
            self.assertIn("budget exhausted", result["walk_path"][-1]["raw_output"])

    def test_rewalk_on_conflict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "walks.jsonl")

            mock_client = MagicMock()
            # Walk 1: ANSWER -> synthesis
            # Walk 2 (re-walk): ANSWER -> synthesis
            mock_client.chat.completions.create.side_effect = [
                MockLLMResponse("ACTION: ANSWER\nREASON: First try"),
                MockLLMResponse("Contradictory answer"),
                MockLLMResponse("ACTION: ANSWER\nREASON: Second try after conflict"),
                MockLLMResponse("Corrected non-contradictory answer"),
            ]

            # Sheaf filter returns conflict on first call, no conflict on second call
            self.mock_engine.sheaf_filter.evaluate_passages.side_effect = [
                {
                    "is_consistent": False,
                    "total_obstruction": 1.2,
                    "conflicting_pairs": [
                        {
                            "doc_a": "__dianoia_answer__",
                            "doc_b": "node_1",
                            "content_a": "Contradictory answer",
                            "content_b": "Content of node 1.",
                            "obstruction_energy": 1.2,
                        }
                    ],
                },
                {
                    "is_consistent": True,
                    "total_obstruction": 0.0,
                    "conflicting_pairs": [],
                },
            ]

            controller = WalkController(
                engine=self.mock_engine,
                embedder=self.mock_embedder,
                max_steps=4,
                log_path=log_path,
                client=mock_client,
            )

            result = controller.run("Question about conflict")

            self.assertTrue(result["verification"]["rewalked"])
            self.assertFalse(result["verification"]["conflicts_found"])
            self.assertEqual(result["final_answer"], "Corrected non-contradictory answer")
            self.assertIsNotNone(result.get("initial_walk_path"))

            # Round 2 Items 4 & 5: Conflict provenance & shared walk_id linkage
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 2)
            import json
            rec1 = json.loads(lines[0])
            rec2 = json.loads(lines[1])

            # Item 5: Both records share the exact same walk_id
            self.assertEqual(rec1["walk_id"], rec2["walk_id"])
            self.assertEqual(rec1["walk_id"], result["walk_id"])

            # Item 4: Conflict provenance
            # Record 1 (initial): rewalked=False, carries first walk conflict detail
            self.assertTrue(rec1["verification"]["conflicts_found"])
            self.assertFalse(rec1["verification"]["rewalked"])
            self.assertIn("Answer conflicts with [node_1]", rec1["verification"]["conflict_detail"])

            # Record 2 (re-walk): rewalked=True, carries ITS OWN conflict detail (None)
            self.assertFalse(rec2["verification"]["conflicts_found"])
            self.assertTrue(rec2["verification"]["rewalked"])
            self.assertIsNone(rec2["verification"]["conflict_detail"])
            self.assertEqual(rec2["final_answer"], "Corrected non-contradictory answer")
            self.assertIn("initial_path", rec2)

    def test_zero_edges_fails_loudly(self):
        # Item 2: Zero-edge graph must fail loudly
        self.mock_engine.wormhole.graph.number_of_edges.return_value = 0
        controller = WalkController(
            engine=self.mock_engine,
            embedder=self.mock_embedder,
            client=MagicMock(),
        )
        with self.assertRaises(ValueError) as ctx:
            controller.run("Question?")
        self.assertIn("0 edges", str(ctx.exception))

    def test_synthesis_uses_synthesis_system_prompt(self):
        # Item 4: Synthesis must use SYNTHESIS_SYSTEM_PROMPT, not CONTROLLER_SYSTEM_PROMPT
        from dianoia.prompts import SYNTHESIS_SYSTEM_PROMPT

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            MockLLMResponse("ACTION: ANSWER\nREASON: Done."),
            MockLLMResponse("Final synthesized text."),
        ]
        controller = WalkController(
            engine=self.mock_engine,
            embedder=self.mock_embedder,
            client=mock_client,
        )
        controller.run("Question?")

        # Check call arguments of the 2nd call (synthesis)
        second_call_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
        system_msg = next(m["content"] for m in second_call_messages if m["role"] == "system")
        self.assertEqual(system_msg, SYNTHESIS_SYSTEM_PROMPT)

    def test_default_fallback_model_is_spec_value(self):
        # Item 5: Default fallback model should be llama-3.1-8b-instruct
        controller = WalkController(
            engine=self.mock_engine,
            embedder=self.mock_embedder,
            client=MagicMock(),
        )
        self.assertEqual(controller.fallback_model, "llama-3.1-8b-instruct")

    def test_verify_server_models_warns_when_fallback_missing(self):
        # Round 2 Item 2: warn if fallback model is missing from server
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "qwen2.5-7b-instruct"
        mock_client.models.list.return_value = [mock_model]

        with self.assertLogs("dianoia.walk", level="WARNING") as log_ctx:
            controller = WalkController(
                engine=self.mock_engine,
                embedder=self.mock_embedder,
                client=mock_client,
                fallback_model_name="llama-3.1-8b-instruct",
            )
        self.assertTrue(any("fallback model 'llama-3.1-8b-instruct' was not found" in m.lower() for m in log_ctx.output))

    def test_step_level_oom_triggers_fallback(self):
        # Round 2 Item 2: OOM during step execution triggers fallback and retries
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("CUDA out of memory: tried to allocate 2.0 GiB"),
            MockLLMResponse("ACTION: ANSWER\nREASON: Answered via fallback."),
            MockLLMResponse("Final answer."),
        ]
        controller = WalkController(
            engine=self.mock_engine,
            embedder=self.mock_embedder,
            client=mock_client,
            fallback_model_name="llama-3.1-8b-instruct",
        )
        result = controller.run("Question about OOM")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["model"], "llama-3.1-8b-instruct")

    def test_rewalk_failure_logs_honest_status(self):
        # Round 2 Item 4: If re-walk fails, record must reflect rewalk_failed=True, rewalked=False
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "walks.jsonl")
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [
                MockLLMResponse("ACTION: ANSWER\nREASON: Initial."),
                MockLLMResponse("Contradictory answer"),
                # Re-walk step fails with exception
                RuntimeError("LLM server connection disconnected unexpectedly"),
            ]

            controller = WalkController(
                engine=self.mock_engine,
                embedder=self.mock_embedder,
                max_steps=4,
                log_path=log_path,
                client=mock_client,
            )

            # Verification returns conflict first time
            controller._verify_answer = MagicMock(return_value={
                "conflicts_found": True,
                "report": "Initial contradiction found",
                "conflict_detail": "Detailed conflict info",
            })

            result = controller.run("Question with failing rewalk")
            self.assertFalse(result["verification"]["rewalked"])
            self.assertTrue(result["verification"]["rewalk_failed"])
            self.assertIn("Re-walk failed", result["verification"]["report"])

            # Verify the logged record
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            self.assertEqual(len(lines), 2)
            import json
            rec2 = json.loads(lines[1])
            self.assertTrue(rec2["verification"]["rewalk_failed"])
            self.assertFalse(rec2["verification"]["rewalked"])

    def test_backtrack_stack_semantics(self):
        # Round 3 Item 1: Test backtrack stack semantics
        # A -> B -> C -> BACKTRACK lands on B; then BACKTRACK lands on A;
        # then EXPAND D -> BACKTRACK lands on A; BACKTRACK at root stays at A.
        mock_graph = MagicMock()
        nodes = ["node_A", "node_B", "node_C", "node_D"]
        mock_graph.__contains__.side_effect = lambda k: k in nodes
        mock_graph.successors.side_effect = lambda k: {
            "node_A": ["node_B", "node_D"],
            "node_B": ["node_C"],
            "node_C": [],
            "node_D": [],
        }.get(k, [])
        mock_graph.number_of_edges.return_value = 3

        mock_engine = MagicMock()
        mock_engine.wormhole.get_bridges.return_value = []
        mock_engine.wormhole.graph = mock_graph
        mock_engine.wormhole.node_metadata = {n: {"content": f"Content of {n}"} for n in nodes}
        mock_engine.stochastic_resonance.resonate_retrieval.return_value = [
            {"doc_id": "node_A", "content": "Content of node_A"}
        ]
        mock_engine.sheaf_filter.evaluate_passages.return_value = {
            "is_consistent": True,
            "total_obstruction": 0.0,
            "conflicting_pairs": [],
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            MockLLMResponse("ACTION: EXPAND node_B\nREASON: Go to B"),
            MockLLMResponse("ACTION: EXPAND node_C\nREASON: Go to C"),
            MockLLMResponse("ACTION: BACKTRACK\nREASON: Back from C to B"),
            MockLLMResponse("ACTION: BACKTRACK\nREASON: Back from B to A"),
            MockLLMResponse("ACTION: EXPAND node_D\nREASON: Go to D"),
            MockLLMResponse("ACTION: BACKTRACK\nREASON: Back from D to A"),
            MockLLMResponse("ACTION: BACKTRACK\nREASON: Backtrack at root stays at A"),
            MockLLMResponse("ACTION: ANSWER\nREASON: Done"),
            MockLLMResponse("Final synthesized text."),
        ]

        controller = WalkController(
            engine=mock_engine,
            embedder=self.mock_embedder,
            max_steps=10,
            client=mock_client,
        )

        result = controller.run("Question about stack semantics?")
        path = result["walk_path"]

        # Step 1: at A, EXPAND node_B
        self.assertEqual(path[0]["node"], "node_A")
        self.assertEqual(path[0]["action"], "EXPAND")

        # Step 2: at B, EXPAND node_C
        self.assertEqual(path[1]["node"], "node_B")
        self.assertEqual(path[1]["action"], "EXPAND")

        # Step 3: at C, BACKTRACK -> must land on B (not A!)
        self.assertEqual(path[2]["node"], "node_C")
        self.assertEqual(path[2]["action"], "BACKTRACK")

        # Step 4: at B (verified!), BACKTRACK -> must land on A
        self.assertEqual(path[3]["node"], "node_B")
        self.assertEqual(path[3]["action"], "BACKTRACK")

        # Step 5: at A (verified!), EXPAND node_D
        self.assertEqual(path[4]["node"], "node_A")
        self.assertEqual(path[4]["action"], "EXPAND")

        # Step 6: at D, BACKTRACK -> must land on A
        self.assertEqual(path[5]["node"], "node_D")
        self.assertEqual(path[5]["action"], "BACKTRACK")

        # Step 7: at A, BACKTRACK at root -> remains at A
        self.assertEqual(path[6]["node"], "node_A")
        self.assertEqual(path[6]["action"], "BACKTRACK")

        # Step 8: at A, ANSWER
        self.assertEqual(path[7]["node"], "node_A")
        self.assertEqual(path[7]["action"], "ANSWER")

    def test_verify_server_models_warns_on_network_error(self):
        # Round 3 Item 2(a): warn loudly if /v1/models raises error
        mock_client = MagicMock()
        mock_client.models.list.side_effect = Exception("Connection refused by localhost:8080")

        with self.assertLogs("dianoia.walk", level="WARNING") as log_ctx:
            controller = WalkController(
                engine=self.mock_engine,
                embedder=self.mock_embedder,
                client=mock_client,
                fallback_model_name="llama-3.1-8b-instruct",
            )
        self.assertTrue(any("could not verify models at server" in m for m in log_ctx.output))

    def test_verify_server_models_warns_on_empty_list(self):
        # Round 3 Item 2(a): warn loudly if /v1/models returns empty list
        mock_client = MagicMock()
        mock_client.models.list.return_value = []

        with self.assertLogs("dianoia.walk", level="WARNING") as log_ctx:
            controller = WalkController(
                engine=self.mock_engine,
                embedder=self.mock_embedder,
                client=mock_client,
                fallback_model_name="llama-3.1-8b-instruct",
            )
        self.assertTrue(any("returned an empty models list" in m for m in log_ctx.output))

    def test_trigger_fallback_aborts_when_not_served(self):
        # Round 3 Item 2(b): After explicit warning that fallback model is not served,
        # _trigger_fallback must NOT switch; it must abort the walk with a clear message.
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "qwen2.5-7b-instruct"
        mock_client.models.list.return_value = [mock_model]
        mock_client.chat.completions.create.return_value = MockLLMResponse("garbage output")

        controller = WalkController(
            engine=self.mock_engine,
            embedder=self.mock_embedder,
            client=mock_client,
            fallback_model_name="llama-3.1-8b-instruct",
            consecutive_failures_to_fallback=1,
        )
        self.assertFalse(controller.fallback_available)

        with self.assertRaises(RuntimeError) as ctx:
            controller.run("Question triggering unserved fallback")
        self.assertIn("is not served on the LLM server. Aborting walk.", str(ctx.exception))

    def test_verify_server_models_warns_when_primary_missing(self):
        # Round 4 Item 2: warn loudly if primary deliberation model is missing from server
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "llama-3.1-8b-instruct"
        mock_client.models.list.return_value = [mock_model]

        with self.assertLogs("dianoia.walk", level="WARNING") as log_ctx:
            controller = WalkController(
                engine=self.mock_engine,
                embedder=self.mock_embedder,
                client=mock_client,
                model_name="qwen2.5-7b-instruct",
                fallback_model_name="llama-3.1-8b-instruct",
            )
        self.assertTrue(any("primary model 'qwen2.5-7b-instruct' was not found" in m.lower() for m in log_ctx.output))

    def test_verify_server_models_warns_when_client_lacks_models_list(self):
        # Round 4 Item 3: warn loudly if injected client lacks models.list method
        mock_client = object()

        with self.assertLogs("dianoia.walk", level="WARNING") as log_ctx:
            controller = WalkController(
                engine=self.mock_engine,
                embedder=self.mock_embedder,
                client=mock_client,
                model_name="qwen2.5-7b-instruct",
                fallback_model_name="llama-3.1-8b-instruct",
            )
        self.assertTrue(any("client has no models.list method" in m.lower() for m in log_ctx.output))


if __name__ == "__main__":
    unittest.main()
