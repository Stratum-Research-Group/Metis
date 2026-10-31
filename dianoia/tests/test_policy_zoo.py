"""Unit and integration tests for Anadrome Policy Zoo and Offline Evaluation Harness."""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from dianoia.anadrome.contracts import (
    DecisionObservation,
    EligibleAction,
    EnvironmentIdentity,
    RecordedTransition,
    ReplayTreeRecord,
)
from dianoia.anadrome.evaluator import (
    AnadromeEvaluator,
    PairedComparisonResult,
    PolicyEvaluationSummary,
)
from dianoia.anadrome.policies import (
    APPROVED_CONTROLLER_POLICIES,
    BaselineReplayPolicy,
    EarlyExitAnswerPolicy,
    GreedySimilarityPolicy,
    HighEntropyStochasticPolicy,
    KleroterionAdaptivePolicy,
    SheafContradictionBacktrackPolicy,
)


def create_mock_env() -> EnvironmentIdentity:
    return EnvironmentIdentity(
        corpus_snapshot_hash="snap_test",
        index_build_id="idx_test",
        embedder_id="bge-m3",
        controller_model_id="gemini-2.5-flash",
        sampling_params={"temperature": 0.0},
        prompt_template_version="v1",
        answer_generator_config={"model": "gemini-2.5-flash"},
        seeding_config={"top_k": 3},
        budget_config={"max_steps": 6},
    )


class TestAnadromePolicies(unittest.TestCase):
    """Test policy decision logic across candidate walk policies."""

    def setUp(self):
        self.env = create_mock_env()
        self.env_hash = self.env.compute_env_hash()

    def test_greedy_similarity_policy_selection(self):
        policy = GreedySimilarityPolicy(avoid_loops=True)
        obs = {
            "current_node_id": "doc_A",
            "current_content": "Passage text A",
            "path_prefix": ["doc_A"],
            "visited_nodes": ["doc_A"],
            "remaining_budget": 5,
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_B", "score": 0.65},
                {"action_type": "EXPAND", "target_node_id": "doc_C", "score": 0.92},
                {"action_type": "EXPAND", "target_node_id": "doc_A", "score": 0.99},  # already visited
                {"action_type": "ANSWER"},
            ],
            "observation_hash": "hash_1",
        }
        action = policy(obs)
        # Should select doc_C (highest score among unvisited)
        self.assertEqual(action, "EXPAND doc_C")

    def test_greedy_similarity_loop_avoidance_and_backtrack(self):
        policy = GreedySimilarityPolicy(avoid_loops=True, fallback_action="BACKTRACK")
        obs = {
            "current_node_id": "doc_B",
            "current_content": "Passage text B",
            "path_prefix": ["doc_A", "doc_B"],
            "visited_nodes": ["doc_A", "doc_B", "doc_C"],
            "remaining_budget": 4,
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_C", "score": 0.85},
                {"action_type": "BACKTRACK"},
                {"action_type": "ANSWER"},
            ],
            "observation_hash": "hash_2",
        }
        # doc_C is already visited, so no unvisited expand candidates exist; stack > 1 -> BACKTRACK
        action = policy(obs)
        self.assertEqual(action, "BACKTRACK")

    def test_greedy_similarity_budget_exhaustion(self):
        policy = GreedySimilarityPolicy()
        obs = {
            "current_node_id": "doc_B",
            "current_content": "Passage text B",
            "path_prefix": ["doc_A", "doc_B"],
            "visited_nodes": ["doc_A", "doc_B"],
            "remaining_budget": 1,  # Exhausted budget
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_Z", "score": 0.99},
                {"action_type": "ANSWER"},
            ],
            "observation_hash": "hash_3",
        }
        action = policy(obs)
        self.assertEqual(action, "ANSWER")

    def test_stochastic_softmax_policy_determinism(self):
        # Two policy calls with identical observation_hash and seed must return identical actions
        policy1 = HighEntropyStochasticPolicy(temperature=1.0, seed=42)
        policy2 = HighEntropyStochasticPolicy(temperature=1.0, seed=42)
        obs = {
            "current_node_id": "doc_A",
            "current_content": "Passage text",
            "path_prefix": ["doc_A"],
            "visited_nodes": ["doc_A"],
            "remaining_budget": 4,
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_1", "score": 0.4},
                {"action_type": "EXPAND", "target_node_id": "doc_2", "score": 0.7},
                {"action_type": "EXPAND", "target_node_id": "doc_3", "score": 0.5},
                {"action_type": "ANSWER"},
            ],
            "observation_hash": "fixed_obs_hash_12345",
        }
        action1 = policy1(obs)
        action2 = policy2(obs)
        self.assertEqual(action1, action2)
        self.assertIn(action1, ["EXPAND doc_1", "EXPAND doc_2", "EXPAND doc_3", "ANSWER"])

    def test_early_exit_answer_policy(self):
        policy = EarlyExitAnswerPolicy(min_visited_passages=2, exit_confidence_threshold=0.80)

        # Case 1: Visited < 2 -> continues expanding
        obs1 = {
            "current_node_id": "doc_A",
            "current_content": "Passage text",
            "path_prefix": ["doc_A"],
            "visited_nodes": ["doc_A"],
            "remaining_budget": 5,
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_B", "score": 0.60},
                {"action_type": "ANSWER"},
            ],
            "observation_hash": "hash_e1",
        }
        self.assertEqual(policy(obs1), "EXPAND doc_B")

        # Case 2: Visited >= 2 and top score < 0.80 -> early exit to ANSWER
        obs2 = {
            "current_node_id": "doc_B",
            "current_content": "Passage text 2",
            "path_prefix": ["doc_A", "doc_B"],
            "visited_nodes": ["doc_A", "doc_B"],
            "remaining_budget": 4,
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_C", "score": 0.65},  # < 0.80 threshold
                {"action_type": "ANSWER"},
            ],
            "observation_hash": "hash_e2",
        }
        self.assertEqual(policy(obs2), "ANSWER")

        # Case 3: Visited >= 2 but top score >= 0.80 -> continues expansion
        obs3 = {
            "current_node_id": "doc_B",
            "current_content": "Passage text 2",
            "path_prefix": ["doc_A", "doc_B"],
            "visited_nodes": ["doc_A", "doc_B"],
            "remaining_budget": 4,
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_C", "score": 0.95},  # >= 0.80
                {"action_type": "ANSWER"},
            ],
            "observation_hash": "hash_e3",
        }
        self.assertEqual(policy(obs3), "EXPAND doc_C")

    def test_sheaf_contradiction_backtrack_policy(self):
        policy = SheafContradictionBacktrackPolicy(conflict_score_threshold=0.5)

        # Case 1: Normal state -> greedy expansion
        obs_normal = {
            "current_node_id": "doc_B",
            "current_content": "Consistent content without conflict.",
            "path_prefix": ["doc_A", "doc_B"],
            "visited_nodes": ["doc_A", "doc_B"],
            "remaining_budget": 4,
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_C", "score": 0.88},
                {"action_type": "BACKTRACK"},
                {"action_type": "ANSWER"},
            ],
            "realization_ids": {"conflict_score": 0.1, "sheaf_obstruction": False},
            "observation_hash": "norm_hash",
        }
        self.assertEqual(policy(obs_normal), "EXPAND doc_C")

        # Case 2: Sheaf obstruction detected via realization_ids -> immediate BACKTRACK!
        obs_conflict_ids = {
            "current_node_id": "doc_B",
            "current_content": "Passage with hidden topological obstruction.",
            "path_prefix": ["doc_A", "doc_B"],
            "visited_nodes": ["doc_A", "doc_B"],
            "remaining_budget": 4,
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_C", "score": 0.99},
                {"action_type": "BACKTRACK"},
                {"action_type": "ANSWER"},
            ],
            "realization_ids": {"sheaf_obstruction": True},
            "observation_hash": "conflict_hash_1",
        }
        self.assertEqual(policy(obs_conflict_ids), "BACKTRACK")

        # Case 3: Conflict marker in content -> immediate BACKTRACK
        obs_conflict_content = {
            "current_node_id": "doc_B",
            "current_content": "[CONFLICT] Contradictory assertion on date of treaty.",
            "path_prefix": ["doc_A", "doc_B"],
            "visited_nodes": ["doc_A", "doc_B"],
            "remaining_budget": 4,
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_C", "score": 0.99},
                {"action_type": "BACKTRACK"},
                {"action_type": "ANSWER"},
            ],
            "realization_ids": {},
            "observation_hash": "conflict_hash_2",
        }
        self.assertEqual(policy(obs_conflict_content), "BACKTRACK")

    def test_kleroterion_adaptive_policy_decisions(self):
        policy = KleroterionAdaptivePolicy(
            proposal_count=4,
            temperature=1.0,
            adapt_temperature=True,
            target_entropy_ratio=0.60,
            controller_gain=0.50,
            seed=42,
        )
        self.assertIn("kleroterion", policy.name)
        self.assertIn("antithetic proposals", policy.description)

        obs = {
            "current_node_id": "doc_1",
            "current_content": "Evidence passage 1",
            "path_prefix": ["doc_1"],
            "visited_nodes": ["doc_1"],
            "remaining_budget": 5,
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_2", "score": 0.85},
                {"action_type": "EXPAND", "target_node_id": "doc_3", "score": 0.80},
                {"action_type": "ANSWER"},
            ],
            "observation_hash": "kler_hash_1",
        }
        action = policy(obs)
        self.assertIn(action, ["EXPAND doc_2", "EXPAND doc_3", "ANSWER"])
        self.assertGreater(policy.last_marginal_entropy, 0.0)
        self.assertGreaterEqual(policy.last_disagreement, 0.0)
        self.assertGreater(policy.current_temperature, 0.0)
        self.assertIsNotNone(policy.last_runner_up_action)
        self.assertIn(policy.last_runner_up_action, ["doc_2", "doc_3", "ANSWER"])

    def test_kleroterion_adaptive_temperature_feedback(self):
        policy = KleroterionAdaptivePolicy(
            proposal_count=4,
            temperature=1.0,
            adapt_temperature=True,
            target_entropy_ratio=0.30,
            controller_gain=0.50,
            seed=123,
        )
        initial_temp = policy.current_temperature

        obs = {
            "current_node_id": "doc_A",
            "current_content": "Passage text",
            "path_prefix": ["doc_A"],
            "visited_nodes": ["doc_A"],
            "remaining_budget": 4,
            "eligible_actions": [
                {"action_type": "EXPAND", "target_node_id": "doc_B", "score": 0.70},
                {"action_type": "EXPAND", "target_node_id": "doc_C", "score": 0.68},
                {"action_type": "EXPAND", "target_node_id": "doc_D", "score": 0.69},
                {"action_type": "ANSWER"},
            ],
            "observation_hash": "kler_hash_adapt",
        }
        policy(obs)
        # Because uniform-like distribution has high entropy compared to 0.30 target,
        # PID temperature controller should decrease temperature
        self.assertNotEqual(policy.current_temperature, initial_temp)
        self.assertLess(policy.current_temperature, initial_temp)


class TestAnadromeEvaluatorAndCoverageGate(unittest.TestCase):
    """Test policy evaluation harness, coverage gating, and Dokime bootstrap comparison."""

    def setUp(self):
        self.env = create_mock_env()
        self.env_hash = self.env.compute_env_hash()

        # Build 10 synthetic decision continuation trees
        self.trees: List[Dict[str, Any]] = []
        for i in range(10):
            q = f"Question {i} about entity {i % 3}"
            gold = f"Answer_{i}"

            obs0 = DecisionObservation(
                env_hash=self.env_hash,
                current_node_id=f"root_{i}",
                current_content=f"Root passage content for question {i}.",
                path_prefix=[f"root_{i}"],
                visited_nodes=[f"root_{i}"],
                remaining_budget=5,
                eligible_actions=[
                    EligibleAction("EXPAND", f"node_{i}_A", 0.90),
                    EligibleAction("EXPAND", f"node_{i}_B", 0.40),
                    EligibleAction("ANSWER"),
                ],
            )
            tr0 = RecordedTransition(
                step_index=0,
                observation_hash=obs0.observation_hash,
                observation=obs0,
                taken_action=f"EXPAND node_{i}_A",
                raw_model_output=f"EXPAND node_{i}_A",
                parsed_reason="Select highest similarity",
                parse_failed=False,
                next_node_id=f"node_{i}_A",
                answer_probe={"draft_answer": f"Early_Answer_{i}", "is_verified": False},
            )

            obs1 = DecisionObservation(
                env_hash=self.env_hash,
                current_node_id=f"node_{i}_A",
                current_content=f"Detailed passage A containing Answer_{i}.",
                path_prefix=[f"root_{i}", f"node_{i}_A"],
                visited_nodes=[f"root_{i}", f"node_{i}_A"],
                remaining_budget=4,
                eligible_actions=[
                    EligibleAction("EXPAND", f"node_{i}_C", 0.30),
                    EligibleAction("BACKTRACK"),
                    EligibleAction("ANSWER"),
                ],
            )
            tr1 = RecordedTransition(
                step_index=1,
                observation_hash=obs1.observation_hash,
                observation=obs1,
                taken_action="ANSWER",
                raw_model_output=f"ANSWER Answer_{i}",
                parsed_reason="Synthesize answer from evidence",
                parse_failed=False,
                next_node_id=None,
                answer_probe={"draft_answer": f"Answer_{i}", "is_verified": True},
            )

            tree_record = ReplayTreeRecord(
                schema_version="1.0.0",
                tree_id=f"tree_{i}",
                walk_id=f"walk_{i}",
                question=q,
                seeded_nodes=[f"root_{i}"],
                environment=self.env,
                is_shadow=False,
                parent_walk_id=None,
                branch_point_step=None,
                branch_point_node=None,
                transitions=[tr0, tr1],
                final_answer=f"Answer_{i}",
                steps_used=2,
                terminal_status="ANSWER",
            )
            d = tree_record.to_dict()
            d["ground_truth"] = gold
            self.trees.append(d)

    def test_evaluator_baseline_100_percent_coverage(self):
        evaluator = AnadromeEvaluator(self.trees, coverage_gate=0.80)
        baseline_policy = GreedySimilarityPolicy(min_score_threshold=0.50)

        summary = evaluator.evaluate_policy(baseline_policy)
        self.assertEqual(summary.total_episodes, 10)
        self.assertEqual(summary.supported_count, 10)
        self.assertEqual(summary.coverage_rate, 1.0)
        self.assertTrue(summary.passes_coverage_gate)
        self.assertEqual(summary.mean_correctness, 1.0)
        self.assertEqual(summary.hard_error_rate, 0.0)

    def test_evaluator_unsupported_coverage_gate_rejection(self):
        # A policy that chooses an unrecorded action on every step
        def wild_policy(obs: Dict[str, Any]) -> str:
            return "EXPAND unrecorded_ghost_node"

        evaluator = AnadromeEvaluator(self.trees, coverage_gate=0.80)
        summary = evaluator.evaluate_policy(wild_policy, policy_name="wild_policy")

        self.assertEqual(summary.total_episodes, 10)
        self.assertEqual(summary.supported_count, 0)
        self.assertEqual(summary.coverage_rate, 0.0)
        self.assertFalse(summary.passes_coverage_gate)

        comp = evaluator.compare_policies(
            candidate_policy=wild_policy,
            baseline_policy=GreedySimilarityPolicy(min_score_threshold=0.50),
            candidate_name="wild_policy",
            baseline_name="greedy_baseline",
        )
        self.assertEqual(comp.promotion_verdict, "FAILED_COVERAGE_GATE")
        self.assertIn("failed the >=", comp.verdict_explanation)

    def test_evaluator_early_exit_comparison_and_bootstrap(self):
        evaluator = AnadromeEvaluator(self.trees, coverage_gate=0.80)
        greedy_base = GreedySimilarityPolicy(min_score_threshold=0.50)
        early_exit_cand = EarlyExitAnswerPolicy(min_visited_passages=2, exit_confidence_threshold=0.80)

        comp = evaluator.compare_policies(
            candidate_policy=early_exit_cand,
            baseline_policy=greedy_base,
            candidate_name="early_exit",
            baseline_name="greedy",
        )
        self.assertEqual(comp.mutually_supported_count, 10)
        self.assertEqual(comp.mutual_coverage_rate, 1.0)
        self.assertTrue(comp.candidate_passes_gate)
        self.assertEqual(comp.baseline_mean, 1.0)
        self.assertEqual(comp.candidate_mean, 1.0)
        self.assertEqual(comp.delta_point_estimate, 0.0)
        # Identical performance -> CI is [0.0, 0.0] -> NO_EVIDENCE per Spec 6
        self.assertEqual(comp.promotion_verdict, "NO_EVIDENCE")
        self.assertIn("No evidence of a better policy", comp.verdict_explanation)

    def test_evaluator_zoo_evaluation(self):
        evaluator = AnadromeEvaluator(self.trees, coverage_gate=0.80)
        policies = [
            GreedySimilarityPolicy(min_score_threshold=0.50, name="greedy"),
            EarlyExitAnswerPolicy(min_visited_passages=2, name="early_exit"),
            SheafContradictionBacktrackPolicy(name="sheaf_backtrack"),
        ]
        base_policy = GreedySimilarityPolicy(min_score_threshold=0.50, name="baseline")


        zoo_res = evaluator.evaluate_zoo(policies, baseline_policy=base_policy)
        self.assertEqual(zoo_res["total_recorded_trees"], 10)
        self.assertEqual(zoo_res["coverage_gate"], 0.80)
        self.assertEqual(len(zoo_res["candidate_summaries"]), 3)
        self.assertEqual(len(zoo_res["paired_comparisons"]), 3)

        for comp in zoo_res["paired_comparisons"]:
            self.assertIn("candidate_name", comp)
            self.assertIn("promotion_verdict", comp)
            self.assertIn("delta_95_ci", comp)

    def test_baseline_replay_policy_reproduction(self):
        baseline = BaselineReplayPolicy(trees_or_map=self.trees)
        evaluator = AnadromeEvaluator(self.trees)
        summary = evaluator.evaluate_policy(baseline)
        self.assertEqual(summary.supported_count, 10)
        self.assertEqual(summary.coverage_rate, 1.0)
        self.assertEqual(summary.mean_correctness, 1.0)

    def test_evaluator_promotion_verdicts_pass_and_regression(self):
        evaluator = AnadromeEvaluator(self.trees, coverage_gate=0.80)

        # Baseline that gets 0% correctness (by answering "Wrong")
        def wrong_policy(obs: Dict[str, Any]) -> str:
            return "ANSWER"  # answers early on root before evidence -> gets wrong answer

        # Candidate that gets 100% correctness
        good_policy = GreedySimilarityPolicy(min_score_threshold=0.50)

        # Compare good candidate vs wrong baseline -> should PASS promotion
        pass_comp = evaluator.compare_policies(
            candidate_policy=good_policy,
            baseline_policy=wrong_policy,
            candidate_name="good_candidate",
            baseline_name="wrong_baseline",
        )
        self.assertEqual(pass_comp.promotion_verdict, "PASS")
        self.assertTrue(pass_comp.ci_excludes_zero)
        self.assertGreater(pass_comp.delta_point_estimate, 0.0)

        # Compare wrong candidate vs good baseline -> should be REGRESSION
        reg_comp = evaluator.compare_policies(
            candidate_policy=wrong_policy,
            baseline_policy=good_policy,
            candidate_name="wrong_candidate",
            baseline_name="good_baseline",
        )
        self.assertEqual(reg_comp.promotion_verdict, "REGRESSION")

    def test_policy_visible_view_isolation(self):
        # Spec 4.7: Policies cannot inspect future steps, gold labels, or sibling outcomes
        leaked_keys = []

        def spy_policy(obs: Dict[str, Any]) -> str:
            for forbidden in ["gold_answer", "gold_truth", "future_steps", "sibling_branches", "tree_id"]:
                if forbidden in obs:
                    leaked_keys.append(forbidden)
            return "ANSWER"

        evaluator = AnadromeEvaluator(self.trees)
        evaluator.evaluate_policy(spy_policy)
        self.assertEqual(leaked_keys, [], f"Policy visible view leaked forbidden keys: {leaked_keys}")


if __name__ == "__main__":
    unittest.main()

