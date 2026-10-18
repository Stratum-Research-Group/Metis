"""Anadrome Policy Evaluation & Comparison Harness.

Implements offline replay comparison over recorded decision continuation trees
with mutual support coverage filtering (>= 80% gate) and Dokime paired
10,000-resample bootstrap confidence intervals.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dianoia.anadrome.policies import AnadromePolicy
from dianoia.anadrome.replay import AnadromeReplayEngine, EpisodeReplayResult

logger = logging.getLogger(__name__)

# Try importing dokime statistical functions
try:
    from dokime.stats import paired_bootstrap_test, round_stat, single_run_bootstrap
except ImportError:
    # Graceful pure-python / fallback implementation if dokime not on path
    import random

    def round_stat(val: float | None, default_precision: int = 4, max_precision: int = 8) -> float | None:
        if val is None:
            return None
        return round(float(val), default_precision)

    def single_run_bootstrap(
        scores: list[float],
        num_samples: int = 10000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> dict[str, Any]:
        n = len(scores)
        if n == 0:
            return {"mean": None, "ci_lower": None, "ci_upper": None, "count": 0}
        mean = sum(scores) / n
        if n == 1 or all(s == scores[0] for s in scores):
            return {"mean": mean, "ci_lower": mean, "ci_upper": mean, "count": n}

        rng = random.Random(seed)
        sample_means = []
        for _ in range(num_samples):
            resample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
            sample_means.append(sum(resample) / n)
        sample_means.sort()
        alpha = 1.0 - confidence_level
        lower_idx = int((alpha / 2.0) * num_samples)
        upper_idx = int((1.0 - alpha / 2.0) * num_samples)
        return {
            "mean": round_stat(mean),
            "ci_lower": round_stat(sample_means[lower_idx]),
            "ci_upper": round_stat(sample_means[upper_idx]),
            "count": n,
        }

    def paired_bootstrap_test(
        scores_baseline: list[float],
        scores_candidate: list[float],
        num_samples: int = 10000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> dict[str, Any]:
        n = len(scores_baseline)
        if n == 0:
            return {
                "mean_baseline": 0.0,
                "mean_candidate": 0.0,
                "delta": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "ci_excludes_zero": False,
                "verdict": "NO_CHANGE",
            }
        diffs = [c - b for b, c in zip(scores_baseline, scores_candidate)]
        observed_delta = sum(diffs) / n
        mean_base = sum(scores_baseline) / n
        mean_cand = sum(scores_candidate) / n

        rng = random.Random(seed)
        bootstrap_deltas = []
        for _ in range(num_samples):
            resampled_diffs = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
            bootstrap_deltas.append(sum(resampled_diffs) / n)
        bootstrap_deltas.sort()

        alpha = 1.0 - confidence_level
        lower_idx = int((alpha / 2.0) * num_samples)
        upper_idx = int((1.0 - alpha / 2.0) * num_samples)
        ci_lower = bootstrap_deltas[lower_idx]
        ci_upper = bootstrap_deltas[upper_idx]

        ci_excludes_zero = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
        verdict = "IMPROVEMENT" if ci_lower > 0 else ("REGRESSION" if ci_upper < 0 else "NO_CHANGE")

        return {
            "mean_baseline": round_stat(mean_base),
            "mean_candidate": round_stat(mean_cand),
            "delta": round_stat(observed_delta),
            "ci_lower": round_stat(ci_lower),
            "ci_upper": round_stat(ci_upper),
            "ci_excludes_zero": ci_excludes_zero,
            "verdict": verdict,
            "num_queries": n,
        }


@dataclass
class PolicyEpisodeScore:
    """Detailed outcome and score metrics for a single episode."""
    tree_id: str
    walk_id: str
    question: str
    is_fully_supported: bool
    terminal_status: str
    steps_used: int
    final_answer: Optional[str]
    is_correct: Optional[bool] = None
    is_hard_error: bool = False
    is_abstention: bool = False


@dataclass
class PolicyEvaluationSummary:
    """Aggregated offline evaluation summary for an individual policy."""
    policy_name: str
    total_episodes: int
    supported_count: int
    coverage_rate: float
    passes_coverage_gate: bool
    mean_correctness: Optional[float]
    ci_correctness: Optional[Tuple[float, float]]
    mean_steps: float
    hard_error_rate: float
    abstention_rate: float
    episodes: List[PolicyEpisodeScore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "total_episodes": self.total_episodes,
            "supported_count": self.supported_count,
            "coverage_rate": round(self.coverage_rate, 4),
            "passes_coverage_gate": self.passes_coverage_gate,
            "mean_correctness": round_stat(self.mean_correctness),
            "ci_correctness": (
                round_stat(self.ci_correctness[0]),
                round_stat(self.ci_correctness[1]),
            )
            if self.ci_correctness
            else None,
            "mean_steps": round(self.mean_steps, 2),
            "hard_error_rate": round(self.hard_error_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
        }


@dataclass
class PairedComparisonResult:
    """Detailed paired comparison between a candidate and baseline policy."""
    candidate_name: str
    baseline_name: str
    total_episodes: int
    mutually_supported_count: int
    mutual_coverage_rate: float
    candidate_passes_gate: bool
    primary_metric_name: str
    baseline_mean: float
    candidate_mean: float
    delta_point_estimate: float
    delta_95_ci: Tuple[float, float]
    ci_excludes_zero: bool
    delta_steps: float
    baseline_hard_error: float
    candidate_hard_error: float
    hard_error_regression: bool
    promotion_verdict: str  # "PASS" | "NO_EVIDENCE" | "REGRESSION" | "FAILED_COVERAGE_GATE"
    verdict_explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "baseline_name": self.baseline_name,
            "total_episodes": self.total_episodes,
            "mutually_supported_count": self.mutually_supported_count,
            "mutual_coverage_rate": round(self.mutual_coverage_rate, 4),
            "candidate_passes_gate": self.candidate_passes_gate,
            "primary_metric_name": self.primary_metric_name,
            "baseline_mean": round_stat(self.baseline_mean),
            "candidate_mean": round_stat(self.candidate_mean),
            "delta_point_estimate": round_stat(self.delta_point_estimate),
            "delta_95_ci": (
                round_stat(self.delta_95_ci[0]),
                round_stat(self.delta_95_ci[1]),
            ),
            "ci_excludes_zero": self.ci_excludes_zero,
            "delta_steps": round_stat(self.delta_steps),
            "baseline_hard_error": round_stat(self.baseline_hard_error),
            "candidate_hard_error": round_stat(self.candidate_hard_error),
            "hard_error_regression": self.hard_error_regression,
            "promotion_verdict": self.promotion_verdict,
            "verdict_explanation": self.verdict_explanation,
        }


class AnadromeEvaluator:
    """
    Evaluator for Anadrome walk policy zoos.
    Executes replay over decision continuation trees, applies the >= 80% coverage gate,
    and runs paired 10,000-resample Dokime bootstrap hypothesis testing.
    """

    def __init__(
        self,
        trees: List[Dict[str, Any]],
        coverage_gate: float = 0.80,
        ground_truth_map: Optional[Dict[str, str]] = None,
        custom_scorer: Optional[Callable[[str, str, str], bool]] = None,
        bootstrap_samples: int = 10000,
        seed: int = 42,
    ):
        """
        Args:
            trees: Recorded ReplayTreeRecord dicts.
            coverage_gate: Minimum fraction of fully supported episodes required for rankability.
            ground_truth_map: Optional map of question or tree_id -> ground truth answer.
            custom_scorer: Optional function (question, candidate_answer, gold_answer) -> bool.
            bootstrap_samples: Number of resamples for Dokime bootstrap CIs (default 10,000).
            seed: Deterministic random seed for bootstrap.
        """
        self.trees = trees
        self.coverage_gate = coverage_gate
        self.ground_truth_map = ground_truth_map or {}
        self.custom_scorer = custom_scorer
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed
        self.replay_engine = AnadromeReplayEngine(trees)

    def _score_answer(self, tree: Dict[str, Any], answer: Optional[str]) -> Tuple[Optional[bool], bool, bool]:
        """
        Returns (is_correct, is_hard_error, is_abstention).
        """
        if answer is None or answer.strip() == "":
            return None, False, True

        question = tree.get("question", "")
        tree_id = tree.get("tree_id", "")

        # Look for ground truth
        gold = (
            self.ground_truth_map.get(tree_id)
            or self.ground_truth_map.get(question)
            or tree.get("ground_truth")
        )

        if gold is None:
            # If no explicit ground truth, check if answer was verified in the original tree
            # or if answer probe marked it as verified
            verified = False
            for tr in tree.get("transitions", []):
                probe = tr.get("answer_probe")
                if probe and probe.get("draft_answer") == answer:
                    verified = probe.get("is_verified", False)
                    break
            if verified:
                return True, False, False
            # If original walk had final_answer equal to this answer, inherit terminal verification
            if tree.get("final_answer") == answer and tree.get("terminal_status") == "VERIFIED":
                return True, False, False
            # Default unverified
            return False, True, False

        # Evaluate against gold
        if self.custom_scorer:
            correct = self.custom_scorer(question, answer, gold)
        else:
            # Exact normalized match
            norm_ans = answer.strip().lower()
            norm_gold = gold.strip().lower()
            correct = (norm_ans == norm_gold)


        is_hard_error = not correct
        is_abstention = False
        return correct, is_hard_error, is_abstention

    def evaluate_policy(self, policy: AnadromePolicy | Callable[[Dict[str, Any]], str], policy_name: Optional[str] = None) -> PolicyEvaluationSummary:
        """Run offline replay for a single policy and aggregate performance."""
        name = getattr(policy, "name", policy_name or "unnamed_policy")
        policy_fn = policy if callable(policy) else policy.select_action

        replay_results = [self.replay_engine.replay_tree(tree, policy_fn) for tree in self.trees]
        total_episodes = len(replay_results)
        supported_episodes = [r for r in replay_results if r.is_fully_supported]
        coverage_rate = len(supported_episodes) / total_episodes if total_episodes > 0 else 0.0
        passes_gate = coverage_rate >= self.coverage_gate

        episode_scores: List[PolicyEpisodeScore] = []
        correctness_scores: List[float] = []
        hard_errors = 0
        abstentions = 0
        total_steps = 0

        for tree, res in zip(self.trees, replay_results):
            steps_used = len(res.steps)
            total_steps += steps_used

            if not res.is_fully_supported or res.terminal_status == "UNSUPPORTED":
                episode_scores.append(
                    PolicyEpisodeScore(
                        tree_id=res.tree_id,
                        walk_id=res.walk_id,
                        question=res.question,
                        is_fully_supported=False,
                        terminal_status=res.terminal_status,
                        steps_used=steps_used,
                        final_answer=None,
                        is_correct=None,
                        is_hard_error=False,
                        is_abstention=True,
                    )
                )
                abstentions += 1
                continue

            # Fully supported episode
            correct, hard_err, abstention = self._score_answer(tree, res.final_answer)
            if abstention:
                abstentions += 1
            if hard_err:
                hard_errors += 1
            if correct is not None:
                correctness_scores.append(1.0 if correct else 0.0)

            episode_scores.append(
                PolicyEpisodeScore(
                    tree_id=res.tree_id,
                    walk_id=res.walk_id,
                    question=res.question,
                    is_fully_supported=True,
                    terminal_status=res.terminal_status,
                    steps_used=steps_used,
                    final_answer=res.final_answer,
                    is_correct=correct,
                    is_hard_error=hard_err,
                    is_abstention=abstention,
                )
            )

        mean_steps = total_steps / total_episodes if total_episodes > 0 else 0.0
        hard_error_rate = hard_errors / total_episodes if total_episodes > 0 else 0.0
        abstention_rate = abstentions / total_episodes if total_episodes > 0 else 0.0

        if correctness_scores:
            boot = single_run_bootstrap(
                correctness_scores,
                num_samples=self.bootstrap_samples,
                seed=self.seed,
            )
            mean_corr = boot.get("mean")
            ci_corr = (boot.get("ci_lower"), boot.get("ci_upper"))
        else:
            mean_corr = None
            ci_corr = None

        return PolicyEvaluationSummary(
            policy_name=name,
            total_episodes=total_episodes,
            supported_count=len(supported_episodes),
            coverage_rate=coverage_rate,
            passes_coverage_gate=passes_gate,
            mean_correctness=mean_corr,
            ci_correctness=ci_corr,
            mean_steps=mean_steps,
            hard_error_rate=hard_error_rate,
            abstention_rate=abstention_rate,
            episodes=episode_scores,
        )

    def compare_policies(
        self,
        candidate_policy: AnadromePolicy | Callable[[Dict[str, Any]], str],
        baseline_policy: AnadromePolicy | Callable[[Dict[str, Any]], str],
        candidate_name: Optional[str] = None,
        baseline_name: Optional[str] = None,
    ) -> PairedComparisonResult:
        """
        Compare Candidate vs Baseline on mutually supported episodes per Spec Section 5 & 6.
        Applies Dokime 10,000-resample paired bootstrap hypothesis testing.
        """
        c_name = getattr(candidate_policy, "name", candidate_name or "candidate")
        b_name = getattr(baseline_policy, "name", baseline_name or "baseline")

        cand_eval = self.evaluate_policy(candidate_policy, policy_name=c_name)
        base_eval = self.evaluate_policy(baseline_policy, policy_name=b_name)

        total_episodes = len(self.trees)

        # Identify mutually supported episodes
        cand_by_tree = {ep.tree_id: ep for ep in cand_eval.episodes}
        base_by_tree = {ep.tree_id: ep for ep in base_eval.episodes}

        common_tree_ids = [
            t.get("tree_id", "")
            for t in self.trees
            if t.get("tree_id", "") in cand_by_tree
            and t.get("tree_id", "") in base_by_tree
            and cand_by_tree[t.get("tree_id", "")].is_fully_supported
            and base_by_tree[t.get("tree_id", "")].is_fully_supported
        ]

        mutual_count = len(common_tree_ids)
        mutual_coverage = mutual_count / total_episodes if total_episodes > 0 else 0.0

        # Coverage Gate check (Spec 4.3): candidate must pass coverage gate before ranking
        if not cand_eval.passes_coverage_gate:
            return PairedComparisonResult(
                candidate_name=c_name,
                baseline_name=b_name,
                total_episodes=total_episodes,
                mutually_supported_count=mutual_count,
                mutual_coverage_rate=mutual_coverage,
                candidate_passes_gate=False,
                primary_metric_name="terminal_answer_correctness",
                baseline_mean=base_eval.mean_correctness or 0.0,
                candidate_mean=cand_eval.mean_correctness or 0.0,
                delta_point_estimate=0.0,
                delta_95_ci=(0.0, 0.0),
                ci_excludes_zero=False,
                delta_steps=0.0,
                baseline_hard_error=base_eval.hard_error_rate,
                candidate_hard_error=cand_eval.hard_error_rate,
                hard_error_regression=False,
                promotion_verdict="FAILED_COVERAGE_GATE",
                verdict_explanation=(
                    f"Candidate coverage ({cand_eval.coverage_rate:.1%}) failed the >= "
                    f"{self.coverage_gate:.0%} coverage gate."
                ),
            )

        if mutual_count == 0:
            return PairedComparisonResult(
                candidate_name=c_name,
                baseline_name=b_name,
                total_episodes=total_episodes,
                mutually_supported_count=0,
                mutual_coverage_rate=0.0,
                candidate_passes_gate=True,
                primary_metric_name="terminal_answer_correctness",
                baseline_mean=0.0,
                candidate_mean=0.0,
                delta_point_estimate=0.0,
                delta_95_ci=(0.0, 0.0),
                ci_excludes_zero=False,
                delta_steps=0.0,
                baseline_hard_error=base_eval.hard_error_rate,
                candidate_hard_error=cand_eval.hard_error_rate,
                hard_error_regression=False,
                promotion_verdict="NO_EVIDENCE",
                verdict_explanation="Zero mutually supported episodes between candidate and baseline.",
            )


        # Extract paired correctness scores
        base_scores = []
        cand_scores = []
        step_diffs = []
        base_hard_count = 0
        cand_hard_count = 0

        for t_id in common_tree_ids:
            c_ep = cand_by_tree[t_id]
            b_ep = base_by_tree[t_id]

            c_score = 1.0 if c_ep.is_correct else 0.0
            b_score = 1.0 if b_ep.is_correct else 0.0
            cand_scores.append(c_score)
            base_scores.append(b_score)

            step_diffs.append(c_ep.steps_used - b_ep.steps_used)
            if b_ep.is_hard_error:
                base_hard_count += 1
            if c_ep.is_hard_error:
                cand_hard_count += 1

        # Run Dokime paired bootstrap
        boot_res = paired_bootstrap_test(
            scores_baseline=base_scores,
            scores_candidate=cand_scores,
            num_samples=self.bootstrap_samples,
            confidence_level=0.95,
            seed=self.seed,
        )

        delta_est = boot_res.get("delta", 0.0)
        ci_lower = boot_res.get("ci_lower", 0.0)
        ci_upper = boot_res.get("ci_upper", 0.0)
        ci_excludes_zero = boot_res.get("ci_excludes_zero", False)

        base_hard_rate = base_hard_count / mutual_count
        cand_hard_rate = cand_hard_count / mutual_count
        hard_error_regression = cand_hard_rate > base_hard_rate

        delta_steps = sum(step_diffs) / mutual_count

        # Spec Section 6 Promotion Criteria:
        # A candidate PASSES promotion if:
        # 1. candidate coverage >= 80% (passes_coverage_gate)
        # 2. beats baseline on primary metric with paired 95% CI excluding zero (ci_lower > 0)
        # 3. shows no hard-error regression
        if not cand_eval.passes_coverage_gate:
            promotion_verdict = "FAILED_COVERAGE_GATE"
            verdict_explanation = (
                f"Candidate coverage ({cand_eval.coverage_rate:.1%}) failed the >= "
                f"{self.coverage_gate:.0%} coverage gate."
            )
        elif ci_lower > 0 and not hard_error_regression:
            promotion_verdict = "PASS"
            verdict_explanation = (
                f"Candidate significantly outperformed baseline (delta: +{delta_est:.4f}, "
                f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]) with no hard-error regression."
            )
        elif ci_upper < 0 or hard_error_regression:
            promotion_verdict = "REGRESSION"
            reasons = []
            if ci_upper < 0:
                reasons.append(f"significant correctness regression (delta: {delta_est:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
            if hard_error_regression:
                reasons.append(f"hard-error rate regression ({cand_hard_rate:.1%} vs baseline {base_hard_rate:.1%})")
            verdict_explanation = f"Candidate rejected due to {', and '.join(reasons)}."
        else:
            promotion_verdict = "NO_EVIDENCE"
            verdict_explanation = (
                "No evidence of a better policy among the tested candidates over the supported data "
                f"(delta: {delta_est:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] includes zero)."
            )

        return PairedComparisonResult(
            candidate_name=c_name,
            baseline_name=b_name,
            total_episodes=total_episodes,
            mutually_supported_count=mutual_count,
            mutual_coverage_rate=mutual_coverage,
            candidate_passes_gate=cand_eval.passes_coverage_gate,
            primary_metric_name="terminal_answer_correctness",
            baseline_mean=boot_res.get("mean_baseline", 0.0),
            candidate_mean=boot_res.get("mean_candidate", 0.0),
            delta_point_estimate=delta_est,
            delta_95_ci=(ci_lower, ci_upper),
            ci_excludes_zero=ci_excludes_zero,
            delta_steps=delta_steps,
            baseline_hard_error=base_hard_rate,
            candidate_hard_error=cand_hard_rate,
            hard_error_regression=hard_error_regression,
            promotion_verdict=promotion_verdict,
            verdict_explanation=verdict_explanation,
        )

    def evaluate_zoo(
        self,
        policies: Sequence[AnadromePolicy | Callable[[Dict[str, Any]], str]],
        baseline_policy: AnadromePolicy | Callable[[Dict[str, Any]], str],
        policy_names: Optional[Sequence[str]] = None,
        baseline_name: str = "baseline",
    ) -> Dict[str, Any]:
        """
        Evaluate an entire policy zoo against a baseline reference policy.
        Returns full comparative report with rankability, coverage gate status,
        paired difference CIs, and promotion verdicts.
        """
        summaries: List[PolicyEvaluationSummary] = []
        comparisons: List[PairedComparisonResult] = []

        base_summary = self.evaluate_policy(baseline_policy, policy_name=baseline_name)
        summaries.append(base_summary)

        for i, policy in enumerate(policies):
            name = policy_names[i] if policy_names and i < len(policy_names) else getattr(policy, "name", f"candidate_{i}")
            p_summary = self.evaluate_policy(policy, policy_name=name)
            summaries.append(p_summary)

            comp = self.compare_policies(
                candidate_policy=policy,
                baseline_policy=baseline_policy,
                candidate_name=name,
                baseline_name=baseline_name,
            )
            comparisons.append(comp)

        return {
            "total_recorded_trees": len(self.trees),
            "coverage_gate": self.coverage_gate,
            "baseline_summary": base_summary.to_dict(),
            "candidate_summaries": [s.to_dict() for s in summaries[1:]],
            "paired_comparisons": [c.to_dict() for c in comparisons],
        }
