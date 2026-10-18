"""Tests for Anadrome: Decision-tree continuation logging and offline replay engine."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

try:
    import torch
except ImportError:
    torch = None

from dianoia.anadrome.canonical import (
    canonical_json_dumps,
    compute_observation_hash,
)
from dianoia.anadrome.contracts import (
    DecisionObservation,
    EligibleAction,
    EnvironmentIdentity,
    RecordedTransition,
    ReplayTreeRecord,
)
from dianoia.anadrome.logger import AnadromeLogger
from dianoia.anadrome.probe_cache import AnswerProbeCache
from dianoia.anadrome.replay import AnadromeReplayEngine, EpisodeReplayResult
from dianoia.walk import WalkController


class MockLLMResponse:
    def __init__(self, content: str):
        self.choices = [MagicMock(message=MagicMock(content=content))]


class TestAnadromeCanonical(unittest.TestCase):
    """Test RFC 8785 canonical serialization and observation hashing."""

    def test_canonical_json_dumps_sorting_and_whitespace(self):
        d1 = {"b": 2, "a": 1, "nested": {"z": 10, "y": 20}}
        d2 = {"nested": {"y": 20, "z": 10}, "a": 1, "b": 2}
        s1 = canonical_json_dumps(d1)
        s2 = canonical_json_dumps(d2)

        self.assertEqual(s1, s2)
        # Check no whitespace around delimiters
        self.assertNotIn(" ", s1)
        self.assertNotIn("\n", s1)
        self.assertTrue(s1.startswith('{"a":1,"b":2,"nested":{"y":20,"z":10}}'))

    def test_float_stabilization_and_rounding(self):
        # 4-decimal float stabilization prevents platform drift
        d1 = {"score": 0.8123456}
        d2 = {"score": 0.8123499}
        self.assertEqual(canonical_json_dumps(d1), canonical_json_dumps(d2))
        self.assertEqual(compute_observation_hash(d1), compute_observation_hash(d2))

    def test_observation_hash_determinism(self):
        obj1 = {"nodes": ["b", "a"], "budget": 5, "score": 0.55551}
        obj2 = {"score": 0.55554, "budget": 5, "nodes": ["b", "a"]}
        hash1 = compute_observation_hash(obj1)
        hash2 = compute_observation_hash(obj2)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA-256 hex string


class TestAnadromeContracts(unittest.TestCase):
    """Test data contracts and automatic observation hash calculation."""

    def setUp(self):
        self.env = EnvironmentIdentity(
            corpus_snapshot_hash="snap_123",
            index_build_id="idx_456",
            embedder_id="bge-m3",
            controller_model_id="gemini-2.5-flash",
            sampling_params={"temperature": 0.0},
            prompt_template_version="v1",
            answer_generator_config={"model": "gemini-2.5-flash"},
            seeding_config={"top_k": 5},
            budget_config={"max_steps": 6},
        )

    def test_environment_identity(self):
        env_dict = self.env.to_dict()
        self.assertEqual(env_dict["corpus_snapshot_hash"], "snap_123")
        env_hash = self.env.compute_env_hash()
        self.assertTrue(isinstance(env_hash, str))
        self.assertEqual(len(env_hash), 64)

    def test_eligible_action_formatting(self):
        act_expand = EligibleAction(action_type="EXPAND", target_node_id="node_42", score=0.91234)
        self.assertEqual(act_expand.format_string(), "EXPAND node_42")
        self.assertEqual(act_expand.to_dict()["score"], 0.9123)

        act_answer = EligibleAction(action_type="ANSWER")
        self.assertEqual(act_answer.format_string(), "ANSWER")

        act_backtrack = EligibleAction(action_type="BACKTRACK")
        self.assertEqual(act_backtrack.format_string(), "BACKTRACK")

    def test_decision_observation_auto_hash(self):
        obs = DecisionObservation(
            env_hash=self.env.compute_env_hash(),
            current_node_id="node_1",
            current_content="Passage content here.",
            path_prefix=["node_0", "node_1"],
            visited_nodes=["node_1", "node_0", "node_1"],  # duplicates and unsorted
            remaining_budget=4,
            eligible_actions=[
                EligibleAction(action_type="EXPAND", target_node_id="node_2"),
                EligibleAction(action_type="ANSWER"),
            ],
        )
        self.assertTrue(obs.observation_hash)
        self.assertEqual(len(obs.observation_hash), 64)
        # Visited nodes normalized and sorted
        obs_dict = obs.to_dict()
        self.assertEqual(obs_dict["visited_nodes"], ["node_0", "node_1"])

    def test_replay_tree_record_roundtrip(self):
        obs = DecisionObservation(
            env_hash=self.env.compute_env_hash(),
            current_node_id="node_1",
            current_content="Test",
            path_prefix=["node_1"],
            visited_nodes=["node_1"],
            remaining_budget=5,
            eligible_actions=[EligibleAction(action_type="ANSWER")],
        )
        tr = RecordedTransition(
            step_index=0,
            observation_hash=obs.observation_hash,
            observation=obs,
            taken_action="ANSWER",
            raw_model_output="Reasoning: Done. Action: ANSWER",
            parsed_reason="Done.",
            parse_failed=False,
            next_node_id=None,
            answer_probe=None,
        )
        tree = ReplayTreeRecord(
            schema_version="1.0.0",
            tree_id="tree_001",
            walk_id="walk_001",
            question="What is Metis?",
            seeded_nodes=["node_1"],
            environment=self.env,
            is_shadow=False,
            parent_walk_id=None,
            branch_point_step=None,
            branch_point_node=None,
            transitions=[tr],
            final_answer="Metis is deliberative memory.",
            steps_used=1,
            terminal_status="ANSWER",
        )
        tree_dict = tree.to_dict()
        self.assertEqual(tree_dict["tree_id"], "tree_001")
        self.assertEqual(len(tree_dict["transitions"]), 1)
        self.assertEqual(tree_dict["transitions"][0]["taken_action"], "ANSWER")

        # Ensure json encodable
        dumped = json.dumps(tree_dict)
        loaded = json.loads(dumped)
        self.assertEqual(loaded["walk_id"], "walk_001")


class TestAnswerProbeCache(unittest.TestCase):
    """Test SQLite probe cache."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_cache.db")
        self.cache = AnswerProbeCache(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_put_and_get(self):
        state_id = "state_abc"
        model_id = "gemini-2.5-flash"
        prompt_hash = "prompt_xyz"

        # Initially absent
        self.assertIsNone(self.cache.get(state_id, model_id, prompt_hash))

        # Store entry
        self.cache.put(
            state_id_hash=state_id,
            model_id=model_id,
            prompt_hash=prompt_hash,
            draft_answer="Draft answer 1.",
            verified=True,
            conflict_detail=None,
        )

        res = self.cache.get(state_id, model_id, prompt_hash)
        self.assertIsNotNone(res)
        self.assertEqual(res["draft_answer"], "Draft answer 1.")
        self.assertTrue(res["verified"])
        self.assertIsNone(res["conflict_detail"])

    def test_overwrite_entry(self):
        state_id = "state_1"
        model_id = "model_1"
        prompt_hash = "hash_1"

        self.cache.put(state_id, model_id, prompt_hash, "Ans 1", False, "conflict A")
        self.cache.put(state_id, model_id, prompt_hash, "Ans 2", True, None)

        res = self.cache.get(state_id, model_id, prompt_hash)
        self.assertEqual(res["draft_answer"], "Ans 2")
        self.assertTrue(res["verified"])


class TestAnadromeLogger(unittest.TestCase):
    """Test JSONL tree logging and reading."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.temp_dir.name, "replay_trees.jsonl")
        self.logger = AnadromeLogger(log_path=self.log_path)
        self.env = EnvironmentIdentity(
            corpus_snapshot_hash="snap_1",
            index_build_id="idx_1",
            embedder_id="emb_1",
            controller_model_id="m1",
            sampling_params={},
            prompt_template_version="v1",
            answer_generator_config={},
            seeding_config={},
            budget_config={},
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_append_and_load_trees(self):
        tree = ReplayTreeRecord(
            schema_version="1.0.0",
            tree_id="tree_10",
            walk_id="walk_10",
            question="Question 10",
            seeded_nodes=["n1"],
            environment=self.env,
            is_shadow=False,
            parent_walk_id=None,
            branch_point_step=None,
            branch_point_node=None,
            transitions=[],
            final_answer="Answer 10",
            steps_used=1,
            terminal_status="ANSWER",
        )
        self.logger.append_tree(tree)
        trees = self.logger.load_all_trees()
        self.assertEqual(len(trees), 1)
        self.assertEqual(trees[0]["tree_id"], "tree_10")
        self.assertEqual(trees[0]["walk_id"], "walk_10")


class TestWalkControllerAnadromeIntegration(unittest.TestCase):
    """Integration test: WalkController logging continuation trees with probes and shadow branch."""

    def setUp(self):
        if torch is None:
            self.skipTest("torch not installed")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.walk_log = os.path.join(self.temp_dir.name, "walks.jsonl")
        self.replay_trees_log = os.path.join(self.temp_dir.name, "replay_trees.jsonl")
        self.probe_cache_db = os.path.join(self.temp_dir.name, "probes.db")

        self.mock_engine = MagicMock()
        self.mock_embedder = MagicMock()
        self.mock_embedder.d_model = 64
        self.mock_embedder.model_name = "test_embedder"
        self.mock_embedder.encode_single.return_value = torch.zeros(64)
        self.mock_embedder.encode.return_value = torch.zeros(3, 64)

        # Graph with node_1 connected to node_2 and node_3
        self.mock_engine.wormhole.get_bridges.return_value = []
        mock_graph = MagicMock()
        mock_graph.__contains__.side_effect = lambda k: k in ["node_1", "node_2", "node_3"]
        mock_graph.successors.side_effect = lambda k: ["node_2", "node_3"] if k == "node_1" else []
        mock_graph.number_of_edges.return_value = 2
        mock_graph.get_edge_data.return_value = {"weight": 0.5, "edge_type": "directed"}
        self.mock_engine.wormhole.graph = mock_graph
        self.mock_engine.wormhole.node_metadata = {
            "node_1": {"content": "Node 1 content."},
            "node_2": {"content": "Node 2 content."},
            "node_3": {"content": "Node 3 content."},
        }
        self.mock_engine.wormhole.node_embeddings = {
            "node_1": torch.zeros(64),
            "node_2": torch.zeros(64),
            "node_3": torch.zeros(64),
        }

        self.mock_engine.stochastic_resonance.resonate_retrieval.return_value = [
            {"doc_id": "node_1", "content": "Node 1 content."}
        ]
        self.mock_engine.sheaf_filter.evaluate_passages.return_value = {
            "is_consistent": True,
            "total_obstruction": 0.0,
            "conflicting_pairs": [],
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_walk_generates_replay_tree_with_shadow_expansion(self):
        mock_client = MagicMock()

        # Sequence:
        # Step 0: EXPAND node_2 (node_3 is alternate -> triggers shadow branch)
        # Step 1: EXPAND node_3
        # Step 2: ANSWER
        # For shadow branch: ANSWER
        # For answer synthesis calls: return synthesized text
        mock_client.chat.completions.create.side_effect = [
            # Step 0 deliberation
            MockLLMResponse("ACTION: EXPAND node_2\nREASON: Expand node 2."),
            # Shadow branch Step 0 deliberation
            MockLLMResponse("ACTION: ANSWER\nREASON: Shadow branch answer."),
            # Shadow branch answer synthesis
            MockLLMResponse("Shadow branch answer text."),
            # Step 1 deliberation
            MockLLMResponse("ACTION: EXPAND node_3\nREASON: Expand node 3."),
            # Step 1 intermediate probe synthesis
            MockLLMResponse("Intermediate probe answer text."),
            # Step 2 deliberation
            MockLLMResponse("ACTION: ANSWER\nREASON: Now answer."),
            # Final answer synthesis
            MockLLMResponse("Final verified answer text."),
        ]

        controller = WalkController(
            engine=self.mock_engine,
            embedder=self.mock_embedder,
            log_path=self.walk_log,
            enable_anadrome_logging=True,
            replay_tree_log_path=self.replay_trees_log,
            probe_cache_path=self.probe_cache_db,
            max_shadow_expansions=1,
            corpus_snapshot_hash="corpus_test_hash",
            index_build_id="idx_test_build",
            client=mock_client,
        )

        res = controller.run("What is in the graph?")

        # Check walk output
        self.assertIn("walk_id", res)
        self.assertEqual(res["final_answer"], "Final verified answer text.")

        # Read back replay trees
        logger = AnadromeLogger(log_path=self.replay_trees_log)
        trees = logger.load_all_trees()

        # Should contain primary tree and shadow branch tree
        self.assertGreaterEqual(len(trees), 2)

        primary_tree = next(t for t in trees if not t["is_shadow"])
        shadow_tree = next(t for t in trees if t["is_shadow"])

        # Check primary tree properties
        self.assertEqual(primary_tree["walk_id"], res["walk_id"])
        self.assertEqual(primary_tree["environment"]["corpus_snapshot_hash"], "corpus_test_hash")
        self.assertEqual(primary_tree["terminal_status"], "ANSWER")
        self.assertEqual(len(primary_tree["transitions"]), 3)

        # Check intermediate probe was recorded on step 1 (path depth >= 2)
        step_1_tr = primary_tree["transitions"][1]
        self.assertIsNotNone(step_1_tr["answer_probe"])
        self.assertEqual(step_1_tr["answer_probe"]["draft_answer"], "Intermediate probe answer text.")

        # Check shadow branch properties
        self.assertEqual(shadow_tree["parent_walk_id"], primary_tree["walk_id"])
        self.assertEqual(shadow_tree["branch_point_node"], "node_3")
        self.assertTrue(shadow_tree["is_shadow"])


class TestAnadromeReplayEngine(unittest.TestCase):
    """Test AnadromeReplayEngine, Zero Imputation Invariant, and Baseline Sanity Check."""

    def setUp(self):
        self.env = EnvironmentIdentity(
            corpus_snapshot_hash="snap_test",
            index_build_id="idx_test",
            embedder_id="embed_test",
            controller_model_id="gemini-2.5-flash",
            sampling_params={"temperature": 0.0},
            prompt_template_version="v1",
            answer_generator_config={},
            seeding_config={},
            budget_config={"max_steps": 5},
        )

        # Build synthetic tree with 2 steps:
        # Step 0: State A -> EXPAND node_2 -> Next: node_2
        # Step 1: State B -> ANSWER -> Final: "Synthesized Answer."
        obs_a = DecisionObservation(
            env_hash=self.env.compute_env_hash(),
            current_node_id="node_1",
            current_content="Content 1",
            path_prefix=["node_1"],
            visited_nodes=["node_1"],
            remaining_budget=5,
            eligible_actions=[
                EligibleAction(action_type="EXPAND", target_node_id="node_2"),
                EligibleAction(action_type="ANSWER"),
            ],
        )
        tr_0 = RecordedTransition(
            step_index=0,
            observation_hash=obs_a.observation_hash,
            observation=obs_a,
            taken_action="EXPAND node_2",
            raw_model_output="Reasoning: Move. Action: EXPAND node_2",
            parsed_reason="Move.",
            parse_failed=False,
            next_node_id="node_2",
            answer_probe={
                "draft_answer": "Draft answer at state A.",
                "verified": True,
                "conflict_detail": None,
            },
        )

        obs_b = DecisionObservation(
            env_hash=self.env.compute_env_hash(),
            current_node_id="node_2",
            current_content="Content 2",
            path_prefix=["node_1", "node_2"],
            visited_nodes=["node_1", "node_2"],
            remaining_budget=4,
            eligible_actions=[
                EligibleAction(action_type="BACKTRACK"),
                EligibleAction(action_type="ANSWER"),
            ],
        )
        tr_1 = RecordedTransition(
            step_index=1,
            observation_hash=obs_b.observation_hash,
            observation=obs_b,
            taken_action="ANSWER",
            raw_model_output="Reasoning: Complete. Action: ANSWER",
            parsed_reason="Complete.",
            parse_failed=False,
            next_node_id=None,
            answer_probe=None,
        )

        self.tree_record = ReplayTreeRecord(
            schema_version="1.0.0",
            tree_id="tree_test_1",
            walk_id="walk_test_1",
            question="What is testing?",
            seeded_nodes=["node_1"],
            environment=self.env,
            is_shadow=False,
            parent_walk_id=None,
            branch_point_step=None,
            branch_point_node=None,
            transitions=[tr_0, tr_1],
            final_answer="Synthesized Answer.",
            steps_used=2,
            terminal_status="ANSWER",
        )
        self.trees = [self.tree_record.to_dict()]
        self.engine = AnadromeReplayEngine(self.trees)

    def test_baseline_sanity_100_percent(self):
        """Baseline sanity check must succeed with 100% bit-exact reproduction."""
        success, report = self.engine.verify_baseline_sanity()
        self.assertTrue(success, f"Baseline sanity failed: {report}")
        self.assertIn("100.0%", report)

    def test_zero_imputation_unsupported_action(self):
        """
        Zero Imputation Invariant (Spec 4.2 & 4.7):
        If candidate policy takes an unrecorded action at state A (e.g. BACKTRACK),
        the episode is strictly unsupported with NO smoothing or extrapolation.
        """
        def candidate_policy(obs: dict) -> str:
            # Illegal/unrecorded action at Step 0
            return "BACKTRACK"

        res = self.engine.replay_tree(self.trees[0], candidate_policy)
        self.assertFalse(res.is_fully_supported)
        self.assertEqual(res.terminal_status, "UNSUPPORTED")
        self.assertEqual(len(res.steps), 1)
        self.assertFalse(res.steps[0].is_supported)
        self.assertIsNone(res.final_answer)

    def test_intermediate_answer_probe_support(self):
        """
        Candidate policy chooses ANSWER at Step 0 (State A):
        Because State A has a recorded answer probe, ANSWER is supported!
        """
        def early_answer_policy(obs: dict) -> str:
            return "ANSWER"

        res = self.engine.replay_tree(self.trees[0], early_answer_policy)
        self.assertTrue(res.is_fully_supported)
        self.assertEqual(res.terminal_status, "ANSWER")
        self.assertEqual(len(res.steps), 1)
        self.assertTrue(res.steps[0].is_supported)
        self.assertEqual(res.final_answer, "Draft answer at state A.")

    def test_policy_evaluation_coverage_metrics(self):
        """Evaluate policy over multiple trees and verify support coverage rates."""
        # Policy that matches baseline: 100% coverage
        def baseline_policy(obs: dict) -> str:
            if obs["current_node_id"] == "node_1":
                return "EXPAND node_2"
            return "ANSWER"

        metrics = self.engine.evaluate_policy(baseline_policy)
        self.assertEqual(metrics["total_episodes"], 1)
        self.assertEqual(metrics["supported_count"], 1)
        self.assertEqual(metrics["coverage_rate"], 1.0)

        # Policy that deviates: 0% coverage
        def unsupported_policy(obs: dict) -> str:
            return "EXPAND node_999"

        metrics_unsup = self.engine.evaluate_policy(unsupported_policy)
        self.assertEqual(metrics_unsup["total_episodes"], 1)
        self.assertEqual(metrics_unsup["supported_count"], 0)
        self.assertEqual(metrics_unsup["coverage_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
