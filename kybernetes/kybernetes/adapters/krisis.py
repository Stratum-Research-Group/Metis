import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from kybernetes.contracts.enums import TurnOutcomeStatus
from kybernetes.contracts.turn import Citation


class KrisisVerificationReport(BaseModel):
    """Result of Krisis quality gate and evidence verification."""
    status: TurnOutcomeStatus
    score: float = Field(..., ge=0.0, le=1.0)
    supported_claims: int = 0
    unsupported_claims: int = 0
    conflict_detected: bool = False
    explanation: str = ""
    is_simulated: bool = Field(default=True, description="Whether verification was performed via local heuristic adapter")
    verifier_mode: str = Field(default="heuristic", description="Verification engine mode e.g. heuristic, formal, simulated")


class KrisisAdapter:
    """
    Adapter for Krisis verification and quality gate.
    Evaluates evidence support and ensures raw generation is never
    confused with factual verification.
    """

    def __init__(self, verifier_client: Optional[Any] = None):
        self.verifier_client = verifier_client
        self.is_simulated = (verifier_client is None)

    def verify_response(
        self,
        query: str,
        response_text: str,
        citations: List[Citation],
        is_tool_output: bool = False,
    ) -> KrisisVerificationReport:
        """
        Verify that synthesized response claims are grounded in evidence.
        """
        if is_tool_output:
            # Deterministic tools (e.g. calculator) have verified mathematical validity
            return KrisisVerificationReport(
                status=TurnOutcomeStatus.VERIFIED,
                score=1.0,
                supported_claims=1,
                unsupported_claims=0,
                conflict_detected=False,
                explanation="Grounded in deterministic tool execution.",
                is_simulated=False,
                verifier_mode="deterministic_tool",
            )

        if not citations:
            # Direct generation without evidence is unverified draft
            return KrisisVerificationReport(
                status=TurnOutcomeStatus.DRAFT,
                score=0.5,
                supported_claims=0,
                unsupported_claims=0,
                conflict_detected=False,
                explanation="Unverified conversational generation without grounded citations.",
                is_simulated=False,
                verifier_mode="direct_draft",
            )

        # Check for malformed unmatched bracket syntax
        if response_text.count("[") != response_text.count("]"):
            return KrisisVerificationReport(
                status=TurnOutcomeStatus.INSUFFICIENT_EVIDENCE,
                score=0.2,
                supported_claims=0,
                unsupported_claims=1,
                conflict_detected=False,
                explanation="Malformed citation markers in response text (unmatched brackets).",
                is_simulated=True,
                verifier_mode="heuristic",
            )

        # Extract bracketed citation markers from response text
        found_markers = re.findall(r"\[([A-Za-z0-9_\-\.\s]+)\]", response_text)
        if not found_markers:
            return KrisisVerificationReport(
                status=TurnOutcomeStatus.INSUFFICIENT_EVIDENCE,
                score=0.3,
                supported_claims=0,
                unsupported_claims=1,
                conflict_detected=False,
                explanation="Response text omits grounding citations despite retrieved evidence.",
                is_simulated=True,
                verifier_mode="heuristic",
            )

        # Build lookup for known valid citation markers
        known_markers: Dict[str, Citation] = {}
        for c in citations:
            raw_id = c.citation_id
            clean_id = c.citation_id.strip("[]")
            known_markers[raw_id] = c
            known_markers[clean_id] = c
            known_markers[f"[{clean_id}]"] = c

        # Reject unknown/unresolved citation markers (e.g. [999])
        unknown_markers = [m for m in found_markers if m not in known_markers and f"[{m}]" not in known_markers]
        if unknown_markers:
            return KrisisVerificationReport(
                status=TurnOutcomeStatus.INSUFFICIENT_EVIDENCE,
                score=0.2,
                supported_claims=0,
                unsupported_claims=len(unknown_markers),
                conflict_detected=False,
                explanation=f"Response contains unresolved citation marker(s): {unknown_markers}",
                is_simulated=True,
                verifier_mode="heuristic",
            )

        # Validate that claims citing known markers are grounded in the referenced passage
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", response_text) if s.strip()]
        stopwords: Set[str] = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "at",
            "to", "for", "with", "of", "by", "that", "this", "it", "as", "from", "be",
            "according", "records", "evidence"
        }
        contradiction_keywords: Set[str] = {"not", "never", "no", "false", "disproven", "untrue", "neither", "nor"}

        supported_count = 0
        unsupported_count = 0
        contradiction_found = False

        for s in sentences:
            s_markers = re.findall(r"\[([A-Za-z0-9_\-\.\s]+)\]", s)
            if not s_markers:
                continue

            s_clean = re.sub(r"\[[^\]]+\]", "", s).lower()
            s_words = set(re.findall(r"\b[a-z]{3,}\b", s_clean)) - stopwords

            # Check for contradiction signals
            if any(neg in s_clean.split() for neg in contradiction_keywords):
                contradiction_found = True

            for m in s_markers:
                cit = known_markers.get(m) or known_markers.get(f"[{m}]")
                if not cit:
                    unsupported_count += 1
                    continue

                cit_words = set(re.findall(r"\b[a-z]{3,}\b", cit.text.lower())) - stopwords
                title_words = set(re.findall(r"\b[a-z]{3,}\b", cit.source_title.lower())) - stopwords

                # If citation has substantial text content, require semantic grounding overlap
                if len(cit_words) > 1 and not (s_words & (cit_words | title_words)):
                    unsupported_count += 1
                else:
                    supported_count += 1

        if contradiction_found:
            return KrisisVerificationReport(
                status=TurnOutcomeStatus.CONFLICTED,
                score=0.1,
                supported_claims=supported_count,
                unsupported_claims=max(1, unsupported_count),
                conflict_detected=True,
                explanation="Contradiction detected in cited claim against evidence.",
                is_simulated=True,
                verifier_mode="heuristic",
            )

        if unsupported_count > 0:
            return KrisisVerificationReport(
                status=TurnOutcomeStatus.INSUFFICIENT_EVIDENCE,
                score=0.3,
                supported_claims=supported_count,
                unsupported_claims=unsupported_count,
                conflict_detected=False,
                explanation="Claim citations are detached from or unsupported by referenced evidence content.",
                is_simulated=True,
                verifier_mode="heuristic",
            )

        has_simulated_citations = any(c.is_simulated for c in citations)
        avg_citation_score = sum(c.score for c in citations) / len(citations)

        if has_simulated_citations:
            return KrisisVerificationReport(
                status=TurnOutcomeStatus.DRAFT,
                score=round(avg_citation_score, 4),
                supported_claims=supported_count,
                unsupported_claims=0,
                conflict_detected=False,
                explanation=f"Claims grounded in simulated/demo evidence ({len(citations)} citations); marked draft (cannot confer factual verification).",
                is_simulated=True,
                verifier_mode="heuristic_simulated",
            )

        return KrisisVerificationReport(
            status=TurnOutcomeStatus.VERIFIED,
            score=round(avg_citation_score, 4),
            supported_claims=supported_count,
            unsupported_claims=0,
            conflict_detected=False,
            explanation=f"Synthesized claims supported by {len(citations)} verified evidence citations.",
            is_simulated=True,
            verifier_mode="heuristic",
        )
