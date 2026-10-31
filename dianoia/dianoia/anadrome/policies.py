"""Anadrome Policy Zoo: Candidate walk policies for offline replay and evaluation."""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
    from kleroterion import (
        ActionDecision,
        ActionID,
        ControllerState,
        KleroterionConfig,
        PolicyProfile,
        PolicyProfileType,
        RngState,
        RoutingRequest,
        SelectionMode,
    )
    try:
        from adapters.dianoia.adapter import DianoiaActionAdapter, DianoiaWalkAction
    except Exception:
        from kleroterion.adapters.dianoia.adapter import DianoiaActionAdapter, DianoiaWalkAction
    HAS_KLEROTERION = True
except Exception:
    np = None  # type: ignore
    HAS_KLEROTERION = False


class AnadromePolicy(ABC):
    """Abstract base class for all Anadrome walk policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the policy."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the policy strategy."""
        pass

    @abstractmethod
    def select_action(self, obs: Dict[str, Any]) -> str:
        """
        Select an action given a policy-visible observation view.

        Args:
            obs: Controller-visible observation dict per Spec 4.7.
                 Contains: current_node_id, current_content, path_prefix,
                 visited_nodes, remaining_budget, eligible_actions, observation_hash.

        Returns:
            Action string (e.g. "EXPAND <node_id>", "BACKTRACK", "ANSWER").
        """
        pass

    def __call__(self, obs: Dict[str, Any]) -> str:
        return self.select_action(obs)


class GreedySimilarityPolicy(AnadromePolicy):
    """
    Policy 1: Greedy Similarity EXPAND.
    Selects the EXPAND action with the highest embedding similarity / edge score
    among eligible actions, preferring unvisited nodes.
    Falls back to BACKTRACK (if stack depth > 1) or ANSWER when no expansion is viable.
    """

    def __init__(
        self,
        name: str = "greedy_similarity",
        avoid_loops: bool = True,
        min_score_threshold: Optional[float] = None,
        fallback_action: str = "ANSWER",
    ):
        self._name = name
        self.avoid_loops = avoid_loops
        self.min_score_threshold = min_score_threshold
        self.fallback_action = fallback_action

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Greedily selects highest-scoring EXPAND action among unvisited neighbors."

    def select_action(self, obs: Dict[str, Any]) -> str:
        eligible_actions = obs.get("eligible_actions", [])
        visited_nodes = set(obs.get("visited_nodes", []))
        path_prefix = obs.get("path_prefix", [])
        remaining_budget = obs.get("remaining_budget", 0)

        # Budget exhaustion check
        if remaining_budget <= 1:
            for act in eligible_actions:
                if act.get("action_type") == "ANSWER":
                    return "ANSWER"

        # Filter expand candidates
        expand_candidates = []
        for act in eligible_actions:
            if act.get("action_type") == "EXPAND" and act.get("target_node_id"):
                target = act["target_node_id"]
                score = act.get("score")
                if score is None:
                    score = 0.0

                if self.min_score_threshold is not None and score < self.min_score_threshold:
                    continue

                if self.avoid_loops and target in visited_nodes:
                    continue

                expand_candidates.append((score, target))

        if expand_candidates:
            # Sort descending by score, tie-break by target node id
            expand_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best_target = expand_candidates[0][1]
            return f"EXPAND {best_target}"

        # Fallback when no viable expand candidates exist
        if self.fallback_action == "ANSWER":
            for act in eligible_actions:
                if act.get("action_type") == "ANSWER":
                    return "ANSWER"
            if len(path_prefix) > 1:
                for act in eligible_actions:
                    if act.get("action_type") == "BACKTRACK":
                        return "BACKTRACK"
        else:
            if len(path_prefix) > 1:
                for act in eligible_actions:
                    if act.get("action_type") == "BACKTRACK":
                        return "BACKTRACK"
            for act in eligible_actions:
                if act.get("action_type") == "ANSWER":
                    return "ANSWER"

        return "ANSWER" if self.fallback_action == "ANSWER" else "BACKTRACK"



class HighEntropyStochasticPolicy(AnadromePolicy):
    """
    Policy 2: High-Entropy Stochastic Selection (Kleroterion-style).
    Applies temperature-scaled softmax over eligible action scores to sample
    the next action. Uses deterministic pseudo-random draws keyed on
    (seed, observation_hash) for exact replay reproducibility.
    """

    def __init__(
        self,
        name: str = "stochastic_softmax",
        temperature: float = 1.0,
        seed: int = 42,
        answer_base_score: float = 0.5,
        backtrack_base_score: float = 0.3,
    ):
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self._name = name
        self.temperature = temperature
        self.seed = seed
        self.answer_base_score = answer_base_score
        self.backtrack_base_score = backtrack_base_score

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Softmax sampling over candidate actions with temperature {self.temperature}."

    def select_action(self, obs: Dict[str, Any]) -> str:
        eligible_actions = obs.get("eligible_actions", [])
        if not eligible_actions:
            return "ANSWER"

        obs_hash = obs.get("observation_hash", "")
        path_prefix = obs.get("path_prefix", [])

        # Assign raw scores to each eligible action
        action_keys: List[str] = []
        action_scores: List[float] = []

        for act in eligible_actions:
            act_type = act.get("action_type")
            if act_type == "EXPAND" and act.get("target_node_id"):
                action_keys.append(f"EXPAND {act['target_node_id']}")
                score = act.get("score")
                action_scores.append(float(score) if score is not None else 0.0)
            elif act_type == "ANSWER":
                action_keys.append("ANSWER")
                action_scores.append(self.answer_base_score)
            elif act_type == "BACKTRACK":
                if len(path_prefix) > 1:
                    action_keys.append("BACKTRACK")
                    action_scores.append(self.backtrack_base_score)

        if not action_keys:
            return "ANSWER"

        # Temperature-scaled softmax
        scaled_logits = [s / self.temperature for s in action_scores]
        max_logit = max(scaled_logits)
        exp_logits = [math.exp(l - max_logit) for l in scaled_logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        # Deterministic pseudo-random draw keyed on seed and observation hash
        hasher = hashlib.sha256()
        hasher.update(f"{self.seed}:{obs_hash}".encode("utf-8"))
        rand_int = int.from_bytes(hasher.digest()[:8], byteorder="big")
        rand_uniform = rand_int / (2**64 - 1)

        # Sample from categorical distribution
        cumulative = 0.0
        for action_key, prob in zip(action_keys, probs):
            cumulative += prob
            if rand_uniform <= cumulative:
                return action_key

        return action_keys[-1]


class EarlyExitAnswerPolicy(AnadromePolicy):
    """
    Policy 3: Early-Exit Answer Threshold.
    Monitors accumulated evidence (visited nodes count, step depth, or probe confidence).
    Triggers ANSWER early if visited evidence exceeds threshold, preventing over-walking.
    Otherwise delegates to greedy expansion.
    """

    def __init__(
        self,
        name: str = "early_exit_answer",
        min_visited_passages: int = 2,
        exit_confidence_threshold: float = 0.85,
    ):
        self._name = name
        self.min_visited_passages = min_visited_passages
        self.exit_confidence_threshold = exit_confidence_threshold

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return (
            f"Triggers early ANSWER once >= {self.min_visited_passages} passages visited "
            f"or top expansion score < {self.exit_confidence_threshold}."
        )

    def select_action(self, obs: Dict[str, Any]) -> str:
        visited_nodes = obs.get("visited_nodes", [])
        eligible_actions = obs.get("eligible_actions", [])
        remaining_budget = obs.get("remaining_budget", 0)

        # Check if ANSWER is eligible
        answer_available = any(act.get("action_type") == "ANSWER" for act in eligible_actions)

        # Collect expand candidates
        expand_candidates = []
        for act in eligible_actions:
            if act.get("action_type") == "EXPAND" and act.get("target_node_id"):
                score = act.get("score")
                expand_candidates.append((float(score) if score is not None else 0.0, act["target_node_id"]))

        # Early exit condition: sufficient passages visited AND (no strong expand target or low budget)
        if answer_available and len(visited_nodes) >= self.min_visited_passages:
            if not expand_candidates:
                return "ANSWER"
            top_score = max(c[0] for c in expand_candidates)
            if top_score < self.exit_confidence_threshold or remaining_budget <= 2:
                return "ANSWER"

        # Otherwise greedy expansion
        if expand_candidates:
            expand_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return f"EXPAND {expand_candidates[0][1]}"

        # Fallback
        if answer_available:
            return "ANSWER"
        return "BACKTRACK"


class SheafContradictionBacktrackPolicy(AnadromePolicy):
    """
    Policy 4: Sheaf Contradiction Backtrack.
    Inspects observation metadata and content for topological contradiction / obstruction
    detected by cellular sheaf Laplacian analysis.
    If contradiction/conflict is detected and stack depth > 1, immediately forces BACKTRACK
    to prune the contradictory branch rather than continuing expansion.
    """

    def __init__(
        self,
        name: str = "sheaf_contradiction_backtrack",
        conflict_score_threshold: float = 0.5,
    ):
        self._name = name
        self.conflict_score_threshold = conflict_score_threshold

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Forces immediate BACKTRACK upon detecting cellular sheaf obstruction/conflict."

    def select_action(self, obs: Dict[str, Any]) -> str:
        path_prefix = obs.get("path_prefix", [])
        eligible_actions = obs.get("eligible_actions", [])
        current_content = obs.get("current_content", "")
        realization_ids = obs.get("realization_ids", {})

        backtrack_available = any(act.get("action_type") == "BACKTRACK" for act in eligible_actions)

        # Check for contradiction signals:
        has_conflict = False
        if realization_ids.get("sheaf_obstruction") or realization_ids.get("has_conflict"):
            has_conflict = True
        elif realization_ids.get("conflict_score", 0.0) >= self.conflict_score_threshold:
            has_conflict = True
        elif "[CONFLICT]" in current_content or "CONTRADICTION DETECTED" in current_content:
            has_conflict = True

        # If obstruction detected and we can backtrack, prune immediately!
        if has_conflict and backtrack_available and len(path_prefix) > 1:
            return "BACKTRACK"

        # Otherwise, greedy expansion on unvisited nodes
        visited_nodes = set(obs.get("visited_nodes", []))
        expand_candidates = []
        for act in eligible_actions:
            if act.get("action_type") == "EXPAND" and act.get("target_node_id"):
                target = act["target_node_id"]
                if target not in visited_nodes:
                    score = act.get("score")
                    expand_candidates.append((float(score) if score is not None else 0.0, target))

        if expand_candidates:
            expand_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return f"EXPAND {expand_candidates[0][1]}"

        # Fallback to ANSWER if available
        if any(act.get("action_type") == "ANSWER" for act in eligible_actions):
            return "ANSWER"

        return "BACKTRACK" if len(path_prefix) > 1 and backtrack_available else "ANSWER"


class BaselineReplayPolicy(AnadromePolicy):
    """
    Baseline Recorded Policy.
    Replays the exact action taken in the original walk log at each decision point.
    Used for baseline sanity checks and delta comparisons.
    """

    def __init__(
        self,
        trees_or_map: Optional[Sequence[Dict[str, Any]] | Dict[str, str]] = None,
        name: str = "baseline_recorded",
    ):
        self._name = name
        self.step_action_map: Dict[str, str] = {}
        if isinstance(trees_or_map, dict):
            self.step_action_map = dict(trees_or_map)
        elif isinstance(trees_or_map, (list, tuple)):
            for tree in trees_or_map:
                for tr in tree.get("transitions", []):
                    h = tr.get("observation_hash")
                    act = tr.get("taken_action")
                    if h and act:
                        self.step_action_map[h] = act

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Replays exact actions recorded in the baseline reference tree."

    def select_action(self, obs: Dict[str, Any]) -> str:
        obs_hash = obs.get("observation_hash", "")
        return self.step_action_map.get(obs_hash, "ANSWER")


class KleroterionAdaptivePolicy(AnadromePolicy):
    """
    Policy: Kleroterion Adaptive Entropy Feedback & Epistemic Disagreement.
    Integrates Kleroterion's dimension-scaled masked-softmax proposal engine with:
      - Antithetic Gaussian feature perturbation proposals (proposal_count >= 2)
      - Marginal entropy feedback loop (H(pi_bar)) with PID log-temperature adaptation
      - Jensen-Shannon proposal disagreement tracking:
            J = H(pi_bar) - (1/K) * sum(H(pi_k))
        measuring epistemic uncertainty across candidate actions.
    """

    def __init__(
        self,
        name: str = "kleroterion_adaptive",
        temperature: float = 1.0,
        adapt_temperature: bool = True,
        target_entropy_ratio: float = 0.60,
        controller_gain: float = 0.50,
        proposal_count: int = 4,
        entropy_deadband: float = 0.05,
        seed: int = 42,
        answer_base_score: float = 0.5,
        backtrack_base_score: float = 0.3,
        disagreement_shadow_threshold: float = 0.30,
        disagreement_backtrack_threshold: float = 0.85,
    ):
        if not HAS_KLEROTERION:
            raise RuntimeError("Kleroterion submodule or dependencies not available.")

        self._name = name
        self.temperature = max(0.01, float(temperature))
        self.adapt_temperature = adapt_temperature
        self.target_entropy_ratio = target_entropy_ratio
        self.controller_gain = controller_gain
        self.proposal_count = proposal_count
        self.entropy_deadband = entropy_deadband
        self.seed = seed
        self.answer_base_score = answer_base_score
        self.backtrack_base_score = backtrack_base_score
        self.disagreement_shadow_threshold = disagreement_shadow_threshold
        self.disagreement_backtrack_threshold = disagreement_backtrack_threshold

        # Initialize Kleroterion controller state and RNG
        self.config = KleroterionConfig(
            proposal_count=proposal_count,
            noise_rms_budget=0.10,
            noise_hard_radius=0.20,
            temperature_initial=self.temperature,
            temperature_bounds=(0.05, 10.0),
            adapt_temperature=adapt_temperature,
            entropy_target=target_entropy_ratio,
            entropy_ema_alpha=0.20,
            controller_gain=controller_gain,
            entropy_deadband=entropy_deadband,
            selection_mode=SelectionMode.SAMPLE,
        )
        self.config.validate()
        self.controller_state = ControllerState(log_temperature=math.log(self.temperature))
        self.rng_state = RngState.from_seed(seed)
        self.executed_request_ids: Set[str] = set()

        self.last_decision: Optional[ActionDecision] = None
        self.last_runner_up_action: Optional[str] = None
        self._step_counter = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return (
            "Kleroterion adaptive entropy feedback policy with dimension-scaled masked softmax, "
            f"antithetic proposals (K={self.proposal_count}), and Jensen-Shannon epistemic disagreement tracking."
        )

    @property
    def last_disagreement(self) -> float:
        """Jensen-Shannon epistemic proposal disagreement J = H_marg - H_cond."""
        return float(self.last_decision.proposal_disagreement) if self.last_decision else 0.0

    @property
    def last_marginal_entropy(self) -> float:
        """Marginal distribution entropy H(pi_bar)."""
        return float(self.last_decision.marginal_entropy) if self.last_decision else 0.0

    @property
    def last_conditional_entropy(self) -> float:
        """Expected conditional entropy under perturbations H_cond."""
        return float(self.last_decision.conditional_entropy) if self.last_decision else 0.0

    @property
    def current_temperature(self) -> float:
        """Active sampling temperature from controller state."""
        return math.exp(self.controller_state.log_temperature)

    def select_action(self, obs: Dict[str, Any]) -> str:
        eligible_actions = obs.get("eligible_actions", [])
        if not eligible_actions:
            return "ANSWER"

        self._step_counter += 1
        path_prefix = obs.get("path_prefix", [])
        remaining_budget = obs.get("remaining_budget", 1)
        obs_hash = obs.get("observation_hash") or hashlib.sha256(str(obs.get("current_node_id")).encode()).hexdigest()[:16]

        # 1. Map eligible actions to DianoiaWalkAction and candidate scores
        d_actions: List[DianoiaWalkAction] = []
        action_scores: List[float] = []

        for act in eligible_actions:
            act_type = act.get("action_type")
            if act_type == "EXPAND" and act.get("target_node_id"):
                d_actions.append(DianoiaWalkAction(action_type="EXPAND", target=act["target_node_id"]))
                raw_score = act.get("score")
                action_scores.append(float(raw_score) if raw_score is not None else 0.5)
            elif act_type == "BACKTRACK":
                if len(path_prefix) > 1:
                    d_actions.append(DianoiaWalkAction(action_type="BACKTRACK"))
                    action_scores.append(self.backtrack_base_score)
            elif act_type == "ANSWER":
                d_actions.append(DianoiaWalkAction(action_type="ANSWER"))
                action_scores.append(self.answer_base_score)

        if not d_actions:
            return "ANSWER"

        # 2. Build 1D policy feature vector
        scores_arr = np.array(action_scores, dtype=np.float64)
        policy_features = np.array([
            float(remaining_budget) / 10.0,
            float(len(path_prefix)) / 5.0,
            float(np.max(scores_arr)),
            float(np.mean(scores_arr)),
        ], dtype=np.float64)

        # 3. Assemble adapter callbacks
        def scorer_fn(req: RoutingRequest, x: np.ndarray, actions: List[ActionID]) -> np.ndarray:
            out = []
            for a_id in actions:
                idx = req.actions.index(a_id) if a_id in req.actions else 0
                base_s = action_scores[idx]
                coord_idx = idx % len(x)
                mod = 0.25 * (float(x[coord_idx]) - 0.5)
                out.append(base_s + mod)
            return np.array(out, dtype=np.float64)

        def revalidation_fn(action: DianoiaWalkAction, snapshot_id: str) -> Tuple[bool, Optional[str]]:
            if action.action_type == "EXPAND" and not action.target:
                return False, "Target empty"
            return True, None

        adapter = DianoiaActionAdapter(scorer_fn=scorer_fn, revalidation_fn=revalidation_fn)

        req = adapter.build_routing_request(
            request_id=f"req_{obs_hash}_{self._step_counter}",
            episode_id=obs.get("episode_id", f"ep_{obs_hash[:8]}"),
            snapshot_id=obs.get("snapshot_id", "snap_active"),
            lthcm_revision=obs.get("lthcm_revision", "rev_1"),
            policy_features=policy_features,
            valid_actions=d_actions,
            authoritative_mask=[True] * len(d_actions),
            policy_profile=PolicyProfile(
                "dianoia_walk",
                PolicyProfileType.EXPLORATORY,
                entropy_target=self.target_entropy_ratio,
                allow_noise=True,
                allow_sampling=True,
                allow_adaptation=self.adapt_temperature,
            ),
        )

        # 4. Propose action via Kleroterion core
        chosen_action, decision = adapter.route_step(req, self.config, self.controller_state, self.rng_state)

        # 5. Commit step and advance controller & RNG states
        adapter.commit_step(decision, self.executed_request_ids)
        self.controller_state = decision.controller_state_next
        self.rng_state = decision.rng_state_next
        self.last_decision = decision

        # 6. Identify runner-up action for epistemic shadow branching if disagreement is high
        if len(decision.action_ids) > 1 and len(decision.marginal_probabilities) > 1:
            sorted_indices = np.argsort(decision.marginal_probabilities)[::-1]
            top_action = decision.selected_action_id
            runner_up_idx = sorted_indices[1] if decision.action_ids[sorted_indices[0]] == top_action else sorted_indices[0]
            runner_up_id = decision.action_ids[runner_up_idx]
            if runner_up_id.startswith("EXPAND:"):
                self.last_runner_up_action = runner_up_id[len("EXPAND:"):]
            else:
                self.last_runner_up_action = runner_up_id
        else:
            self.last_runner_up_action = None

        if chosen_action.action_type == "EXPAND":
            return f"EXPAND {chosen_action.target}"
        elif chosen_action.action_type == "BACKTRACK":
            return "BACKTRACK"
        else:
            return "ANSWER"


APPROVED_CONTROLLER_POLICIES: Dict[str, Any] = {
    "baseline_llm": None,
    "greedy_similarity": GreedySimilarityPolicy,
    "early_exit_answer": EarlyExitAnswerPolicy,
    "sheaf_contradiction_backtrack": SheafContradictionBacktrackPolicy,
    "stochastic_softmax": HighEntropyStochasticPolicy,
    "kleroterion_adaptive": KleroterionAdaptivePolicy if HAS_KLEROTERION else None,
    "kleroterion": KleroterionAdaptivePolicy if HAS_KLEROTERION else None,
}

