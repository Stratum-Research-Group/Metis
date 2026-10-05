"""Hard eligibility gate evaluation engine."""

from decimal import Decimal
from typing import Any
from krisis.contracts.candidate import Candidate
from krisis.contracts.policy import ProfileConfig
from krisis.scoring.decimal_calc import to_decimal


class GateEvaluator:
    """Evaluates hard gates for candidate answers."""

    def __init__(self, profile_config: ProfileConfig):
        self.profile = profile_config
        self.per_claim_min = (
            to_decimal(profile_config.per_claim_min)
            if profile_config.per_claim_min is not None
            else None
        )

    def evaluate_candidate(
        self,
        candidate: Candidate,
        valid_passage_ids: set[str] | None = None,
    ) -> tuple[bool, str | None, str | None]:
        """Evaluate hard gates on candidate.
        
        Returns (passed: bool, gate_name: str | None, detail: str | None).
        """
        # 1. Standard boolean gates (access, provenance, contradiction, citation_resolution)
        for gate_name in ["access", "provenance", "contradiction", "citation_resolution"]:
            if gate_name in self.profile.hard_gates:
                # Check gate_results - must be explicitly True
                res = candidate.get_gate_result(gate_name)
                if not res:
                    return False, gate_name, f"Failed {gate_name} gate (unverified or missing)"

        # Check explicit gate_failure field if present
        if candidate.gate_failure:
            return False, candidate.gate_failure, f"Failed gate: {candidate.gate_failure}"

        # 2. Citation resolution against valid passage IDs if grounded
        if "citation_resolution" in self.profile.hard_gates and valid_passage_ids is not None:
            # Validate candidate-level cited passages
            cand_citations = (candidate.cited_passage_ids or []) + (candidate.citations or [])
            for cid in cand_citations:
                if cid and cid not in valid_passage_ids:
                    return False, "citation_resolution", f"Cited passage '{cid}' does not exist in verified evidence"

            # Validate claim-level cited passages
            for cl in candidate.claims:
                pids = getattr(cl, "cited_passage_ids", None) or (cl.get("cited_passage_ids") if isinstance(cl, dict) else [])
                for pid in pids:
                    if pid and pid not in valid_passage_ids:
                        claim_id = getattr(cl, "claim_id", "unknown") or (cl.get("claim_id", "unknown") if isinstance(cl, dict) else "unknown")
                        return False, "citation_resolution", f"Claim '{claim_id}' cites nonexistent passage '{pid}'"

        # 3. Per-claim support verification (grounded profiles)
        if "per_claim_support" in self.profile.hard_gates:
            if not candidate.claims:
                return False, "per_claim_support", "No claims found for grounded candidate"

            for cl in candidate.claims:
                # Verdict check
                verdict = getattr(cl, "verdict", None) or (cl.get("verdict") if isinstance(cl, dict) else None)
                claim_id = getattr(cl, "claim_id", "unknown") or (cl.get("claim_id", "unknown") if isinstance(cl, dict) else "unknown")
                if verdict != "supported":
                    return (
                        False,
                        "per_claim_support",
                        f"unsupported claim {claim_id}, rejected pre-score",
                    )

                # Calibrated confidence minimum check
                if self.per_claim_min is not None:
                    conf_str = getattr(cl, "calibrated_confidence", None) or (cl.get("calibrated_confidence") if isinstance(cl, dict) else None)
                    if conf_str is None:
                        return (
                            False,
                            "per_claim_support",
                            f"claim {claim_id} missing calibrated_confidence",
                        )
                    conf_dec = to_decimal(conf_str)
                    if conf_dec is None or not conf_dec.is_finite() or conf_dec < Decimal("0.0") or conf_dec > Decimal("1.0"):
                        return (
                            False,
                            "feature_valid",
                            f"claim {claim_id} invalid or non-finite confidence: {conf_str}",
                        )
                    if conf_dec < self.per_claim_min:
                        return (
                            False,
                            "per_claim_support",
                            f"claim {claim_id} confidence {conf_str} < per_claim_min {self.per_claim_min}",
                        )

        return True, None, None
