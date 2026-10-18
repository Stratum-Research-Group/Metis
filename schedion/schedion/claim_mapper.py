"""Sentence-boundary claim mapper for Schedion candidate generation.

Per Metis Roadmap Section 2.1 & Spec Section 4:
Each sentence in the decoded summary is treated as an unverified candidate claim
anchored directly to the 512-token conditioned source span.
"""

from __future__ import annotations

import re
from typing import List, Optional

from krisis.contracts.candidate import Claim


class SentenceBoundaryClaimMapper:
    """
    Deterministic sentence-boundary segmenter that maps decoded summary text
    into structured Krisis Claim objects anchored to the source passage.
    """

    # Sentence boundary regex: punctuation followed by space or end-of-string
    SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, default_verdict: str = "unverified", default_confidence: str = "0.5000"):
        self.default_verdict = default_verdict
        self.default_confidence = default_confidence

    def split_sentences(self, text: str) -> List[str]:
        """Split text into normalized, non-empty sentences."""
        raw_text = text.strip()
        if not raw_text:
            return []

        parts = self.SENTENCE_SPLIT_PATTERN.split(raw_text)
        sentences: List[str] = []
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                sentences.append(cleaned)

        # Fallback if text didn't contain sentence boundary punctuation
        if not sentences and raw_text:
            sentences.append(raw_text)

        return sentences

    def map_claims(
        self,
        text: str,
        candidate_id: str,
        passage_id: Optional[str] = None,
    ) -> List[Claim]:
        """
        Segment summary text and create Claim instances anchored to passage_id.

        Args:
            text: Decoded summary string.
            candidate_id: Identifier of the parent candidate answer.
            passage_id: Optional identifier of the conditioned source span.

        Returns:
            List of Krisis Claim objects.
        """
        sentences = self.split_sentences(text)
        claims: List[Claim] = []
        cited_pids = [passage_id] if passage_id else []

        for idx, sentence in enumerate(sentences):
            claim = Claim(
                claim_id=f"{candidate_id}_c{idx}",
                text=sentence,
                verdict=self.default_verdict,
                calibrated_confidence=self.default_confidence,
                cited_passage_ids=list(cited_pids),
            )
            claims.append(claim)

        return claims
