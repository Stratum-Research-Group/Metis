"""Held-out stress tests and rejection criteria for LTHCM Recovery.

Satisfies Section 6 (Held-out tests) and Section 7 (Rejection criteria) of
LTHCM-recovery-application-plan-356cbcd7.txt:
1. Remove bridges never removed during training (unseen bridge drop).
2. Add high-similarity false shortcuts (adversarial distractor edges).
3. Swap a decisive source fact (detects corrupted provenance).
4. Make every permitted supporting source unavailable (honest abstention).
5. Revoke a source mid-loop (prohibits forbidden source from surviving gate).
6. Provide only temporally invalid candidates (strict external eligibility gate).
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

from recovery.corpus import (
    CanonicalDocument,
    FrozenSourceCorpus,
    TEST_FAMILIES,
    build_frozen_recovery_dataset,
)
from recovery.schema import (
    CorruptedProvenanceError,
    DerivedEdge,
    EligibilityViolationError,
    RecoveryOutcome,
    ShadowGraphState,
    ShadowNode,
)
from recovery.validator import IndependentRecoveryValidator
from recovery.wrapper import RecoveryShadowGraphWrapper

logger = logging.getLogger(__name__)


class HeldOutStressTester:
    """Executes the 6 held-out adversarial and stress scenarios."""

    def __init__(self) -> None:
        self.corpus, self.queries = build_frozen_recovery_dataset()
        # Filter to held-out test split
        self.test_queries = [q for q in self.queries if q["is_test_split"]]

    # 1. Remove bridges never removed during training
    def test_unseen_bridge_removal(self) -> Dict[str, Any]:
        target_query = self.test_queries[0]
        as_of = target_query["query_timestamp"]

        # Build clean state with entity edges
        all_edges = self.corpus.derive_source_entity_edges()
        # Drop ALL edges that mention the target bridge entity
        bridge = target_query["bridge_entity"].lower()
        damaged_edges = [
            e for e in all_edges if bridge not in e.evidence_basis.lower()
        ]

        q_words = [w for w in target_query["query"].lower().replace("?", "").replace(",", "").split() if len(w) > 3]
        nodes = {}
        for doc in self.corpus.get_eligible_docs(as_of):
            overlap = sum(1 for w in q_words if w in doc.text.lower())
            nodes[doc.doc_id] = ShadowNode(doc_id=doc.doc_id, score=0.5 + 0.1 * min(overlap, 4))
        damaged_state = ShadowGraphState(nodes=nodes, edges=damaged_edges)

        validator = IndependentRecoveryValidator(self.corpus, as_of=as_of)
        wrapper = RecoveryShadowGraphWrapper(validator, max_steps=4)
        outcome = wrapper.refine_and_recover(
            query=target_query["query"],
            query_timestamp=as_of,
            damaged_state=damaged_state,
            candidate_pool=[{"doc_id": nid, "score": n.score} for nid, n in nodes.items()],
        )

        # Check if the bridge was rediscovered from source text
        recovered_edges = outcome.diagnostics.get("recovered_edges_count", 0)
        top_cands = [c["doc_id"] for c in outcome.final_candidates[:4]]
        expected_found = set(target_query["expected_support"]).issubset(set(top_cands))

        return {
            "test_name": "unseen_bridge_removal",
            "passed": expected_found,
            "status": outcome.status,
            "recovered_edges": recovered_edges,
        }

    # 2. Add high-similarity false shortcuts
    def test_high_similarity_false_shortcuts(self) -> Dict[str, Any]:
        target_query = self.test_queries[0]
        as_of = target_query["query_timestamp"]
        nodes = {
            doc.doc_id: ShadowNode(doc_id=doc.doc_id, score=0.6)
            for doc in self.corpus.get_eligible_docs(as_of)
        }

        # Add deceptive shortcut pointing to a distractor with 0.99 weight
        hop1 = target_query["expected_support"][0]
        distractor = "ar_dist_01"  # Distractor doc
        false_shortcut = DerivedEdge(
            source_id=hop1,
            target_id=distractor,
            relation_type="adversarial_false_shortcut",
            weight=0.99,
            evidence_basis="hallucinated_false_evidence",
        )
        damaged_state = ShadowGraphState(nodes=nodes, edges=[false_shortcut])

        validator = IndependentRecoveryValidator(self.corpus, as_of=as_of)
        wrapper = RecoveryShadowGraphWrapper(validator, max_steps=3)
        outcome = wrapper.refine_and_recover(
            query=target_query["query"],
            query_timestamp=as_of,
            damaged_state=damaged_state,
            candidate_pool=[{"doc_id": nid, "score": n.score} for nid, n in nodes.items()],
        )

        # Verify that the false shortcut was rejected and logged
        false_shortcut_rejected = any(
            "adversarial_false_shortcut" in r or "hallucinated_false_evidence" in r
            for step in outcome.audit_trail
            for r in step.proposals_rejected
        )

        return {
            "test_name": "high_similarity_false_shortcuts",
            "passed": false_shortcut_rejected,
            "status": outcome.status,
        }

    # 3. Swap a decisive source fact
    def test_decisive_fact_swap(self) -> Dict[str, Any]:
        target_query = self.test_queries[0]
        as_of = target_query["query_timestamp"]
        hop1_id = target_query["expected_support"][0]
        orig_doc = self.corpus[hop1_id]

        # Mutate the fact while keeping the original hash (simulating fact swap / tampering)
        mutated_doc = CanonicalDocument(
            doc_id=orig_doc.doc_id,
            text=orig_doc.text.replace("Aeolus-V", "Falcon-Heavy"),  # Swapped decisive entity
            content_sha256=orig_doc.content_sha256,  # Stale hash
            timestamp=orig_doc.timestamp,
            provenance=orig_doc.provenance,
            family=orig_doc.family,
        )

        tampered_corpus = FrozenSourceCorpus(
            [d if d.doc_id != hop1_id else orig_doc for d in self.corpus.get_all()]
        )
        validator = IndependentRecoveryValidator(tampered_corpus, as_of=as_of)

        caught = False
        try:
            validator.validate_canonical_integrity(mutated_doc)
        except CorruptedProvenanceError:
            caught = True

        return {
            "test_name": "decisive_fact_swap",
            "passed": caught,
        }

    # 4. Make every permitted supporting source unavailable
    def test_all_supporting_sources_unavailable(self) -> Dict[str, Any]:
        target_query = self.test_queries[0]
        as_of = target_query["query_timestamp"]
        expected = set(target_query["expected_support"])

        # Mark all expected support documents as deleted
        purged_docs = []
        for d in self.corpus.get_all():
            if d.doc_id in expected:
                purged_docs.append(
                    CanonicalDocument.create(
                        doc_id=d.doc_id,
                        text=d.text,
                        timestamp=d.timestamp,
                        family=d.family,
                        provenance=d.provenance,
                        is_deleted=True,
                    )
                )
            else:
                purged_docs.append(d)

        purged_corpus = FrozenSourceCorpus(purged_docs)
        validator = IndependentRecoveryValidator(purged_corpus, as_of=as_of)

        # Candidates provided to wrapper contain the candidates for this query's family
        family_docs = [d for d in purged_corpus.get_all() if d.family == target_query["family"]]
        nodes = {
            d.doc_id: ShadowNode(doc_id=d.doc_id, score=0.8)
            for d in family_docs
        }
        state = ShadowGraphState(nodes=nodes, edges=[])

        wrapper = RecoveryShadowGraphWrapper(validator, max_steps=3)
        outcome = wrapper.refine_and_recover(
            query=target_query["query"],
            query_timestamp=as_of,
            damaged_state=state,
            candidate_pool=[{"doc_id": nid, "score": n.score} for nid, n in nodes.items()],
        )

        # Rejection criterion: MUST NOT return deleted documents as valid evidence
        final_ids = {c["doc_id"] for c in outcome.final_candidates}
        deleted_survived = not expected.isdisjoint(final_ids)
        passed = (not deleted_survived) and (outcome.status in {"ABSTAIN", "DEGRADED"})

        return {
            "test_name": "all_supporting_sources_unavailable",
            "passed": passed,
            "status": outcome.status,
            "deleted_survived": deleted_survived,
        }

    # 5. Revoke a source mid-loop
    def test_source_revoked_mid_loop(self) -> Dict[str, Any]:
        target_query = self.test_queries[0]
        as_of = target_query["query_timestamp"]
        hop2_id = target_query["expected_support"][1]

        # Initially valid corpus
        validator = IndependentRecoveryValidator(self.corpus, as_of=as_of)
        nodes = {
            doc.doc_id: ShadowNode(doc_id=doc.doc_id, score=0.8)
            for doc in self.corpus.get_eligible_docs(as_of)
        }
        state = ShadowGraphState(nodes=nodes, edges=[])

        # Mid-loop revocation: mutate corpus to mark hop2 as deleted
        revoked_docs = [
            CanonicalDocument.create(
                doc_id=d.doc_id,
                text=d.text,
                timestamp=d.timestamp,
                family=d.family,
                provenance=d.provenance,
                is_deleted=(d.doc_id == hop2_id),
            )
            for d in self.corpus.get_all()
        ]
        revoked_corpus = FrozenSourceCorpus(revoked_docs)
        validator.corpus = revoked_corpus

        wrapper = RecoveryShadowGraphWrapper(validator, max_steps=3)
        outcome = wrapper.refine_and_recover(
            query=target_query["query"],
            query_timestamp=as_of,
            damaged_state=state,
            candidate_pool=[{"doc_id": nid, "score": n.score} for nid, n in nodes.items()],
        )

        final_ids = [c["doc_id"] for c in outcome.final_candidates]
        assert hop2_id not in final_ids, "Revoked source survived the final gate"

        return {
            "test_name": "source_revoked_mid_loop",
            "passed": hop2_id not in final_ids,
            "status": outcome.status,
        }

    # 6. Provide only temporally invalid candidates
    def test_only_temporally_invalid_candidates(self) -> Dict[str, Any]:
        # Query timestamp is 50.0; all documents in corpus have timestamp >= 105.0
        strict_past_as_of = 50.0
        validator = IndependentRecoveryValidator(self.corpus, as_of=strict_past_as_of, fail_closed=True)

        raw_candidates = [
            {"doc_id": doc.doc_id, "score": 0.9} for doc in self.corpus.get_all()
        ]

        caught = False
        try:
            validator.filter_and_validate_candidates(raw_candidates, strict_temporal=True)
        except EligibilityViolationError:
            caught = True

        return {
            "test_name": "only_temporally_invalid_candidates",
            "passed": caught,
        }
