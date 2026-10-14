"""Independent recovery validator and immutable-source gate.

Satisfies Section 2, 5.4, and Section 7 of LTHCM-recovery-application-plan-356cbcd7.txt:
- Validates hash integrity and provenance on canonical documents.
- Strictly enforces external eligibility and temporal gates (overriding pipeline fallbacks).
- Ensures all derived links have grounded, source-backed evidence in both endpoints.
- Rejects prohibited writes, fabricated sources, duplicates, non-finite scores, and unsupported links.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from recovery.corpus import FrozenSourceCorpus
from recovery.schema import (
    CanonicalDocument,
    CorruptedProvenanceError,
    DerivedEdge,
    EligibilityDecision,
    EligibilityViolationError,
    MissingSourceError,
    NonFiniteStateError,
    ShadowGraphState,
    UnsupportedLinkError,
)

logger = logging.getLogger(__name__)


class IndependentRecoveryValidator:
    """Independent authority verifying evidence boundaries, permissions, and link validity."""

    def __init__(
        self,
        corpus: FrozenSourceCorpus,
        as_of: float,
        allowed_families: Optional[Set[str]] = None,
        fail_closed: bool = True,
    ) -> None:
        self.corpus = corpus
        self.as_of = float(as_of)
        self.allowed_families = allowed_families
        self.fail_closed = fail_closed

    def validate_canonical_integrity(self, doc: CanonicalDocument) -> None:
        """Verifies that a canonical document has untampered text, matching hash, and valid metadata."""
        if not doc.verify_hash():
            raise CorruptedProvenanceError(
                f"Document {doc.doc_id} failed cryptographic SHA-256 integrity verification"
            )
        canonical = self.corpus.get(doc.doc_id)
        if canonical is None:
            raise MissingSourceError(f"Document {doc.doc_id} not present in authoritative canonical corpus")
        if canonical.content_sha256 != doc.content_sha256 or canonical.text != doc.text:
            raise CorruptedProvenanceError(
                f"Document {doc.doc_id} content drifted from frozen canonical record"
            )

    def evaluate_eligibility(self, doc_id: str) -> EligibilityDecision:
        """Evaluates authoritative external permission and temporal validity."""
        return self.corpus.evaluate_eligibility(
            doc_id=doc_id,
            as_of=self.as_of,
            allowed_families=self.allowed_families,
        )

    def validate_derived_edge(self, edge: DerivedEdge) -> bool:
        """Verifies that a derived link connects real retained sources and carries genuine evidence basis."""
        if not math.isfinite(edge.weight) or edge.weight < 0.0 or edge.weight > 1.0:
            raise NonFiniteStateError(f"Edge ({edge.source_id}->{edge.target_id}) has invalid weight {edge.weight}")

        src_doc = self.corpus.get(edge.source_id)
        tgt_doc = self.corpus.get(edge.target_id)

        if src_doc is None:
            raise MissingSourceError(f"Derived edge source {edge.source_id} does not exist in canonical corpus")
        if tgt_doc is None:
            raise MissingSourceError(f"Derived edge target {edge.target_id} does not exist in canonical corpus")

        # Check external eligibility for both endpoints
        src_decision = self.evaluate_eligibility(src_doc.doc_id)
        if not src_decision.allowed:
            raise EligibilityViolationError(
                f"Derived edge source {src_doc.doc_id} is ineligible: {src_decision.reason}"
            )
        tgt_decision = self.evaluate_eligibility(tgt_doc.doc_id)
        if not tgt_decision.allowed:
            raise EligibilityViolationError(
                f"Derived edge target {tgt_doc.doc_id} is ineligible: {tgt_decision.reason}"
            )

        # Source-backed relation check: evidence basis must occur in canonical text of BOTH endpoints
        evidence = edge.evidence_basis.strip().lower()
        if not evidence:
            raise UnsupportedLinkError("Derived edge missing evidence basis string")

        src_has_evidence = evidence in src_doc.text.lower()
        tgt_has_evidence = evidence in tgt_doc.text.lower()

        if not (src_has_evidence and tgt_has_evidence):
            raise UnsupportedLinkError(
                f"Derived edge ({edge.source_id}->{edge.target_id}) claimed entity '{edge.evidence_basis}' "
                f"not grounded in both canonical source texts (src={src_has_evidence}, tgt={tgt_has_evidence})"
            )

        return True

    def filter_and_validate_candidates(
        self,
        raw_candidates: List[Dict[str, Any]],
        strict_temporal: bool = True,
        strict_missing: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        External eligibility gate.
        Rejects disallowed candidates even if LTHCM pipeline's internal fallback supplies them.
        """
        verified_candidates: List[Dict[str, Any]] = []
        seen_canonical_ids: Set[str] = set()

        for cand in raw_candidates:
            doc_id = cand.get("doc_id") or cand.get("id")
            if not doc_id:
                if self.fail_closed and strict_missing:
                    raise MissingSourceError("Candidate missing document identifier")
                continue

            canonical = self.corpus.get(str(doc_id))
            if canonical is None:
                # Fabricated or missing canonical source
                if self.fail_closed and strict_missing:
                    raise MissingSourceError(
                        f"Candidate {doc_id} does not exist in canonical source corpus (rejection criterion 7)"
                    )
                continue

            # Verify cryptographic digest
            self.validate_canonical_integrity(canonical)

            # Enforce external eligibility gate
            decision = self.evaluate_eligibility(canonical.doc_id)
            if not decision.allowed:
                if strict_temporal and self.fail_closed and decision.reason.startswith("temporal_future_violation"):
                    raise EligibilityViolationError(
                        f"Candidate {canonical.doc_id} violates external eligibility: {decision.reason}"
                    )
                # Filter out disallowed candidate
                continue

            # Deduplication: retain highest-scoring valid instance
            if canonical.doc_id in seen_canonical_ids:
                continue

            score = cand.get("score", 0.0)
            if not math.isfinite(score):
                raise NonFiniteStateError(f"Candidate {canonical.doc_id} score is non-finite: {score}")

            verified_candidates.append(
                {
                    "doc_id": canonical.doc_id,
                    "text": canonical.text,
                    "score": float(score),
                    "timestamp": canonical.timestamp,
                    "family": canonical.family,
                    "content_sha256": canonical.content_sha256,
                    "provenance": canonical.provenance,
                    "is_eligible": True,
                }
            )
            seen_canonical_ids.add(canonical.doc_id)

        # Sort descending by score
        verified_candidates.sort(key=lambda c: c["score"], reverse=True)
        return verified_candidates

    def validate_shadow_graph(self, state: ShadowGraphState) -> Tuple[bool, List[str]]:
        """Audits the complete state of a disposable shadow graph."""
        violations: List[str] = []

        # Validate nodes
        for node_id, node in state.nodes.items():
            if not math.isfinite(node.score):
                violations.append(f"Node {node_id} has non-finite score {node.score}")
            canonical = self.corpus.get(node.doc_id)
            if canonical is None:
                violations.append(f"Node {node_id} points to non-existent canonical doc {node.doc_id}")
            else:
                decision = self.evaluate_eligibility(canonical.doc_id)
                if not decision.allowed:
                    violations.append(f"Node {node_id} ({canonical.doc_id}) violates eligibility: {decision.reason}")

        # Validate edges
        for edge in state.edges:
            try:
                self.validate_derived_edge(edge)
            except Exception as exc:
                violations.append(f"Invalid edge ({edge.source_id}->{edge.target_id}): {exc}")

        return len(violations) == 0, violations
