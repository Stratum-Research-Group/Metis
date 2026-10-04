"""Krisis answer selection and factual verification adapter."""

from typing import Dict, List, Optional
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


class KrisisAdapter:
    """
    Adapter for Krisis verification and quality gate.
    Evaluates evidence support and ensures raw generation is never
    confused with factual verification.
    """

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
            )

        # Check citation alignment with response text
        cited_markers = [c.citation_id for c in citations if c.citation_id in response_text]

        if not cited_markers and "[" not in response_text:
            return KrisisVerificationReport(
                status=TurnOutcomeStatus.INSUFFICIENT_EVIDENCE,
                score=0.4,
                supported_claims=0,
                unsupported_claims=1,
                conflict_detected=False,
                explanation="Response text omits grounding citations despite retrieved evidence.",
            )

        avg_citation_score = sum(c.score for c in citations) / len(citations)

        return KrisisVerificationReport(
            status=TurnOutcomeStatus.VERIFIED,
            score=round(avg_citation_score, 4),
            supported_claims=len(citations),
            unsupported_claims=0,
            conflict_detected=False,
            explanation=f"Synthesized claims supported by {len(citations)} verified evidence citations.",
        )
