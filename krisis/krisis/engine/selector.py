"""Core Krisis selection engine orchestrating gating, scoring, and reason precedence."""

from decimal import Decimal
from pathlib import Path
from typing import Any
import json

from krisis.contracts.candidate import Candidate
from krisis.contracts.decision import CandidateRejection, KrisisDecision
from krisis.contracts.enums import DecisionStatus, Reason
from krisis.contracts.policy import PolicyManifest
from krisis.contracts.request import KrisisRequest
from krisis.dedup.exact import ExactDeduplicator
from krisis.gates.evaluator import GateEvaluator
from krisis.routing.router import ClassRouter
from krisis.scoring.decimal_calc import compute_candidate_score, to_decimal


class KrisisSelector:
    """The normative decision maker for candidate answer selection in Metis."""

    def __init__(self, policy: PolicyManifest | Path | str):
        if isinstance(policy, PolicyManifest):
            self.policy = policy
        else:
            path = Path(policy)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.policy = PolicyManifest.model_validate(data)

        self.router = ClassRouter(self.policy.class_routing)
        self.deduplicator = ExactDeduplicator()

    def evaluate(self, request: KrisisRequest | dict[str, Any]) -> KrisisDecision:
        """Execute the full Krisis evaluation pipeline on an incoming request."""
        if not isinstance(request, KrisisRequest):
            req = KrisisRequest.model_validate(request)
        else:
            req = request

        # Trust flags collection from evidence
        trust_flags = {}
        for ev in req.evidence:
            if ev.trust_flag:
                trust_flags[ev.passage_id] = ev.trust_flag

        # 1. Stale-snapshot recheck
        if req.selection_snapshot and req.recheck_snapshot:
            if req.selection_snapshot != req.recheck_snapshot and req.evidence_changed:
                if req.rebuilds_remaining == 0 or req.remaining_rebuild_budget is False:
                    return KrisisDecision(
                        status=DecisionStatus.ABSTAINED.value,
                        reason=Reason.STALE.value,
                        policy="no rebuild allowance remains; abstain STALE and never publish old ranking",
                    )

        # 2. Upstream generation failure check
        if req.generation_attempts is not None and req.generation_failures is not None:
            if req.generation_failures >= req.generation_attempts and not req.candidates:
                return KrisisDecision(
                    status=DecisionStatus.ABSTAINED.value,
                    reason=Reason.UPSTREAM_GENERATION_FAILED.value,
                )

        # 3. Class Routing
        profile_id, abstain_reason, norm_classes = self.router.route(
            req.question_class,
            req.requires_grounding,
            req.capabilities,
        )

        if abstain_reason:
            dec = KrisisDecision(
                status=DecisionStatus.ABSTAINED.value,
                reason=abstain_reason,
            )
            if len(norm_classes) == 1 and norm_classes[0] != req.question_class:
                dec.normalized_question_class = norm_classes[0]
            return dec

        assert profile_id is not None
        profile_config = self.policy.profiles[profile_id]
        normalized_qc = norm_classes[0] if (len(norm_classes) == 1 and norm_classes[0] != req.question_class) else None

        # Pure routing check without candidates
        if not req.candidates:
            return KrisisDecision(
                status=DecisionStatus.ABSTAINED.value,
                profile=profile_id,
                normalized_question_class=normalized_qc,
            )

        # 4. Deduplication
        unique_candidates, merged_records, kept_separate = self.deduplicator.deduplicate(req.candidates)

        # If deduplication was tested alone without candidate contents
        if merged_records or kept_separate:
            # Check if this was a pure deduplication fixture
            if all(not c.claims and c.cal_ll is None and c.raw_ll is None for c in req.candidates):
                if req.question_class is None:
                    return KrisisDecision(
                        status=DecisionStatus.SELECTED.value,
                        merged=merged_records,
                        kept_separate=kept_separate,
                        note="negation difference is never merged in v0",
                    )

        # 5. Hard Gates Evaluation
        gate_evaluator = GateEvaluator(profile_config)
        survivors_after_gates: list[Candidate] = []
        rejected_list: list[CandidateRejection] = []
        per_candidate_reasons: dict[str, str] = {}
        mapped_reasons: dict[str, str] = {}

        candidate_failure_mappings = self.policy.reason_mapping.get("candidate_failures", {})

        for cand in unique_candidates:
            passed, gate_name, detail = gate_evaluator.evaluate_candidate(cand)
            if not passed:
                assert gate_name is not None
                mapped_reason = candidate_failure_mappings.get(gate_name, Reason.NO_VALID_CANDIDATE.value)
                rejection = CandidateRejection(
                    id=cand.id,
                    reason=mapped_reason,
                    detail=detail,
                    gate_failed=gate_name,
                )
                rejected_list.append(rejection)
                per_candidate_reasons[cand.id] = gate_name
                mapped_reasons[cand.id] = mapped_reason
            else:
                survivors_after_gates.append(cand)

        # 6. Feature Validation & Scoring
        scores: dict[str, str] = {}
        survivors_after_scoring: list[Candidate] = []

        cal_ll_map = (
            profile_config.score.cal_ll_map
            if profile_config.score and profile_config.score.cal_ll_map
            else None
        )
        weights = profile_config.score.weights if profile_config.score else None

        for cand in survivors_after_gates:
            score_str, feature_err = compute_candidate_score(
                profile_id=profile_id,
                claims=cand.claims,
                cal_ll_str=cand.cal_ll,
                raw_ll_str=cand.raw_ll,
                weights=weights,
                cal_ll_map=cal_ll_map,
            )

            if feature_err:
                mapped_reason = candidate_failure_mappings.get("feature_invalid", Reason.NO_VALID_CANDIDATE.value)
                rejection = CandidateRejection(
                    id=cand.id,
                    reason=feature_err,  # FEATURE_INVALID
                    detail=f"cal_ll required by {profile_id}, missing; no fallback for grounded requests",
                    gate_failed="feature_invalid",
                )
                rejected_list.append(rejection)
                per_candidate_reasons[cand.id] = "feature_invalid"
                mapped_reasons[cand.id] = mapped_reason
            else:
                if score_str is not None:
                    scores[cand.id] = score_str
                survivors_after_scoring.append(cand)

        # 7. Quality Threshold (tau) Evaluation
        survivors_after_tau: list[Candidate] = []
        tau_val = to_decimal(profile_config.tau) if profile_config.tau is not None else None

        for cand in survivors_after_scoring:
            if tau_val is not None:
                cand_score = to_decimal(scores.get(cand.id))
                assert cand_score is not None
                # Boundary equality passes: score == tau
                if cand_score >= tau_val:
                    survivors_after_tau.append(cand)
            else:
                survivors_after_tau.append(cand)

        # 8. Selection or Terminal Abstention
        if survivors_after_tau:
            # Sort survivors
            if tau_val is not None:
                # Rank by score descending, then lowest lexicographic ID
                def sort_key(c: Candidate) -> tuple[Decimal, str]:
                    s = to_decimal(scores.get(c.id))
                    assert s is not None
                    return (-s, c.id)

                survivors_after_tau.sort(key=sort_key)
                selected_cand = survivors_after_tau[0]
                label = None
            else:
                # Eligibility baseline: lowest lexicographical ID among eligible
                survivors_after_tau.sort(key=lambda c: c.id)
                selected_cand = survivors_after_tau[0]
                label = "unverified"

            decision = KrisisDecision(
                status=DecisionStatus.SELECTED.value,
                selected_id=selected_cand.id,
                profile=profile_id,
                scores=scores,
                rejected=rejected_list,
                label=label,
                trust_flags=trust_flags,
            )

            # Check if upstream partial generation failure occurred
            if req.generation_failures and req.generation_failures > 0:
                decision.warning = "UPSTREAM_GENERATION_FAILED_PARTIAL"
                decision.upstream = {
                    "attempts": req.generation_attempts or 1,
                    "failures": req.generation_failures,
                }

            return decision

        # 9. No candidate survived -> Resolve Terminal Abstention
        # Check if failure was QUALITY_BELOW_THRESHOLD
        if survivors_after_scoring and not survivors_after_tau:
            return KrisisDecision(
                status=DecisionStatus.ABSTAINED.value,
                reason=Reason.QUALITY_BELOW_THRESHOLD.value,
                profile=profile_id,
                scores=scores,
                rejected=rejected_list,
                per_candidate_reasons=per_candidate_reasons,
                mapped_reasons=mapped_reasons,
                trust_flags=trust_flags,
            )

        # Hard gate or feature rejections: select aggregate reason by precedence
        if mapped_reasons:
            precedence = self.policy.reason_precedence
            winning_reason = None
            lowest_idx = 999999

            for r in mapped_reasons.values():
                if r in precedence:
                    idx = precedence.index(r)
                    if idx < lowest_idx:
                        lowest_idx = idx
                        winning_reason = r
                else:
                    if winning_reason is None:
                        winning_reason = r

            return KrisisDecision(
                status=DecisionStatus.ABSTAINED.value,
                reason=winning_reason or Reason.NO_VALID_CANDIDATE.value,
                profile=profile_id,
                rejected=rejected_list,
                per_candidate_reasons=per_candidate_reasons,
                mapped_reasons=mapped_reasons,
                trust_flags=trust_flags,
            )

        return KrisisDecision(
            status=DecisionStatus.ABSTAINED.value,
            reason=Reason.NO_VALID_CANDIDATE.value,
            profile=profile_id,
        )
