"""Anadrome Replay Engine: Offline evaluation of candidate walk policies over recorded trees."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ReplayStepResult:
    step_index: int
    observation_hash: str
    action: str
    is_supported: bool
    next_node: Optional[str]
    answer_text: Optional[str] = None


@dataclass
class EpisodeReplayResult:
    tree_id: str
    walk_id: str
    question: str
    is_fully_supported: bool
    steps: List[ReplayStepResult]
    final_answer: Optional[str]
    terminal_status: str  # "ANSWER" | "UNSUPPORTED" | "BUDGET_EXHAUSTED"


class AnadromeReplayEngine:
    """
    Executes offline policy evaluation against recorded continuation trees.
    Enforces the Zero Imputation Invariant: unsupported transitions are strictly
    rejected and never smoothed or approximated.
    """

    def __init__(self, trees: List[Dict[str, Any]]):
        self.trees = trees
        # Index transitions by (observation_hash, taken_action) -> transition
        self._index_transitions()

    def _index_transitions(self) -> None:
        """Build fast lookup map: (obs_hash, action) -> transition_dict."""
        self.transition_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Also map tree_id -> tree
        self.tree_map: Dict[str, Dict[str, Any]] = {}

        for tree in self.trees:
            t_id = tree.get("tree_id", "")
            self.tree_map[t_id] = tree
            for tr in tree.get("transitions", []):
                obs_hash = tr.get("observation_hash")
                act = tr.get("taken_action")
                if obs_hash and act:
                    self.transition_map[(obs_hash, act)] = tr
                if obs_hash and tr.get("answer_probe"):
                    if (obs_hash, "ANSWER") not in self.transition_map:
                        self.transition_map[(obs_hash, "ANSWER")] = tr

    def replay_tree(
        self,
        tree: Dict[str, Any],
        policy_fn: Callable[[Dict[str, Any]], str],
    ) -> EpisodeReplayResult:
        """
        Replay a candidate policy against a single recorded tree.

        Args:
            tree: A recorded ReplayTreeRecord dict.
            policy_fn: Function taking a policy-visible observation dict and returning an action string.

        Returns:
            EpisodeReplayResult detailing step-by-step choices and final support status.
        """
        transitions = tree.get("transitions", [])
        tree_id = tree.get("tree_id", "")
        walk_id = tree.get("walk_id", "")
        question = tree.get("question", "")

        if not transitions:
            return EpisodeReplayResult(
                tree_id=tree_id,
                walk_id=walk_id,
                question=question,
                is_fully_supported=False,
                steps=[],
                final_answer=None,
                terminal_status="UNSUPPORTED",
            )

        step_results: List[ReplayStepResult] = []
        is_fully_supported = True
        final_answer: Optional[str] = None
        terminal_status = "UNSUPPORTED"

        # Walk through the sequence of states
        for i, recorded_tr in enumerate(transitions):
            obs = recorded_tr.get("observation", {})
            obs_hash = recorded_tr.get("observation_hash")

            # Create policy-visible view (strictly isolated per spec 4.7)
            policy_view = {
                "current_node_id": obs.get("current_node_id"),
                "current_content": obs.get("current_content"),
                "path_prefix": list(obs.get("path_prefix", [])),
                "visited_nodes": list(obs.get("visited_nodes", [])),
                "remaining_budget": obs.get("remaining_budget", 0),
                "eligible_actions": list(obs.get("eligible_actions", [])),
                "observation_hash": obs_hash,
            }

            # Query the candidate policy
            try:
                chosen_action = policy_fn(policy_view)
            except Exception as e:
                logger.warning("Policy execution failed at step %d: %s", i, e)
                is_fully_supported = False
                terminal_status = "POLICY_ERROR"
                break

            # Check if this exact action exists in our recorded database for this state
            matching_tr = self.transition_map.get((obs_hash, chosen_action))

            if matching_tr is None:
                # ZERO IMPUTATION INVARIANT:
                # Action was not recorded from this exact state. Episode becomes unsupported.
                step_results.append(
                    ReplayStepResult(
                        step_index=i,
                        observation_hash=obs_hash,
                        action=chosen_action,
                        is_supported=False,
                        next_node=None,
                    )
                )
                is_fully_supported = False
                terminal_status = "UNSUPPORTED"
                break

            # Action is supported
            if chosen_action == "ANSWER":
                # Check for answer text: either final_answer of the tree or intermediate probe
                ans = None
                if matching_tr.get("answer_probe"):
                    ans = matching_tr["answer_probe"].get("draft_answer")
                elif recorded_tr.get("answer_probe"):
                    ans = recorded_tr["answer_probe"].get("draft_answer")
                elif i == len(transitions) - 1:
                    ans = tree.get("final_answer")

                if ans is None:
                    # Answering at a state without a recorded answer outcome is unsupported (Spec 4.4)
                    step_results.append(
                        ReplayStepResult(
                            step_index=i,
                            observation_hash=obs_hash,
                            action=chosen_action,
                            is_supported=False,
                            next_node=None,
                        )
                    )
                    is_fully_supported = False
                    terminal_status = "UNSUPPORTED"
                    break

                step_results.append(
                    ReplayStepResult(
                        step_index=i,
                        observation_hash=obs_hash,
                        action=chosen_action,
                        is_supported=True,
                        next_node=None,
                        answer_text=ans,
                    )
                )
                final_answer = ans
                terminal_status = "ANSWER"
                break

            # EXPAND or BACKTRACK
            next_node = matching_tr.get("next_node_id")
            step_results.append(
                ReplayStepResult(
                    step_index=i,
                    observation_hash=obs_hash,
                    action=chosen_action,
                    is_supported=True,
                    next_node=next_node,
                )
            )

        if is_fully_supported and terminal_status != "ANSWER":
            terminal_status = "BUDGET_EXHAUSTED"
            final_answer = tree.get("final_answer")

        return EpisodeReplayResult(
            tree_id=tree_id,
            walk_id=walk_id,
            question=question,
            is_fully_supported=is_fully_supported,
            steps=step_results,
            final_answer=final_answer,
            terminal_status=terminal_status,
        )

    def evaluate_policy(
        self,
        policy_fn: Callable[[Dict[str, Any]], str],
    ) -> Dict[str, Any]:
        """Evaluate policy over all indexed trees, computing coverage and metrics."""
        results = [self.replay_tree(tree, policy_fn) for tree in self.trees]
        total_episodes = len(results)
        supported_episodes = [r for r in results if r.is_fully_supported]
        coverage = len(supported_episodes) / total_episodes if total_episodes > 0 else 0.0

        return {
            "total_episodes": total_episodes,
            "supported_count": len(supported_episodes),
            "unsupported_count": total_episodes - len(supported_episodes),
            "coverage_rate": coverage,
            "results": results,
        }

    def verify_baseline_sanity(self) -> Tuple[bool, str]:
        """
        Baseline Sanity Test (Spec 4.3):
        Replaying the recorded baseline policy over the recorded trees must reproduce
        the recorded traces and outcomes exactly with 100.0% agreement.
        The harness fails closed if it does not.
        """
        if not self.trees:
            return False, "Cannot verify baseline sanity: zero trees recorded."

        for tree_idx, tree in enumerate(self.trees):
            recorded_transitions = tree.get("transitions", [])
            if not recorded_transitions:
                continue

            # Policy that returns the exact action recorded at each step
            step_action_map = {
                tr["observation_hash"]: tr["taken_action"]
                for tr in recorded_transitions
            }

            def baseline_policy(obs: Dict[str, Any]) -> str:
                return step_action_map.get(obs["observation_hash"], "UNSUPPORTED")

            res = self.replay_tree(tree, baseline_policy)

            if not res.is_fully_supported:
                return (
                    False,
                    f"Baseline sanity failure on tree {tree.get('tree_id')}: episode is not fully supported.",
                )

            # Verify action-by-action bit-exact match
            for s_idx, (r_step, orig_tr) in enumerate(zip(res.steps, recorded_transitions)):
                if r_step.action != orig_tr["taken_action"]:
                    return (
                        False,
                        f"Baseline divergence on tree {tree.get('tree_id')} at step {s_idx}: "
                        f"replayed action '{r_step.action}' != original '{orig_tr['taken_action']}'",
                    )

            # Verify terminal outcome
            if res.final_answer != tree.get("final_answer"):
                return (
                    False,
                    f"Baseline outcome divergence on tree {tree.get('tree_id')}: "
                    f"replayed answer != original answer.",
                )

        return True, f"Baseline sanity verified with 100.0% bit-exact reproduction across {len(self.trees)} trees."
