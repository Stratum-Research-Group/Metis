"""Frozen source corpus, queries, and topic-family splits for LTHCM Recovery Benchmark.

Satisfies Section 5.1 & 5.2 of LTHCM-recovery-application-plan-356cbcd7.txt:
- Document/topic families split cleanly with zero test leakage.
- Edges derived strictly from source text entities, not answer labels.
- Read-only canonical source dictionary with strict hash validation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from lthcm.core.wormhole_builder import extract_salient_entities
from recovery.schema import (
    CanonicalDocument,
    DerivedEdge,
    EligibilityDecision,
    ProhibitedWriteError,
)

TRAIN_FAMILIES = {"crypto_quantum", "clinical_biotech"}
TEST_FAMILIES = {"aerospace_rocketry", "semiconductor_optics"}


class FrozenSourceCorpus:
    """Immutable in-memory store for canonical documents."""

    def __init__(self, documents: List[CanonicalDocument]) -> None:
        self._docs: Dict[str, CanonicalDocument] = {}
        for d in documents:
            if not d.verify_hash():
                raise ValueError(f"Corrupted hash for canonical doc {d.doc_id}")
            self._docs[d.doc_id] = d
        self._frozen = True

    def get(self, doc_id: str) -> Optional[CanonicalDocument]:
        return self._docs.get(doc_id)

    def __getitem__(self, doc_id: str) -> CanonicalDocument:
        return self._docs[doc_id]

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def get_all(self) -> List[CanonicalDocument]:
        return list(self._docs.values())

    def get_eligible_docs(
        self,
        as_of: float,
        allowed_families: Optional[Set[str]] = None,
    ) -> List[CanonicalDocument]:
        """Returns documents strictly satisfying external temporal and family eligibility."""
        eligible = []
        for doc in self._docs.values():
            if doc.is_deleted:
                continue
            if doc.timestamp > as_of:
                continue
            if allowed_families is not None and doc.family not in allowed_families:
                continue
            eligible.append(doc)
        return eligible

    def evaluate_eligibility(
        self,
        doc_id: str,
        as_of: float,
        allowed_families: Optional[Set[str]] = None,
    ) -> EligibilityDecision:
        doc = self._docs.get(doc_id)
        if doc is None:
            return EligibilityDecision(allowed=False, reason="doc_not_found", as_of=as_of)
        if doc.is_deleted:
            return EligibilityDecision(allowed=False, reason="document_deleted", as_of=as_of)
        if doc.timestamp > as_of:
            return EligibilityDecision(
                allowed=False,
                reason=f"temporal_future_violation: doc_t={doc.timestamp} > as_of={as_of}",
                as_of=as_of,
            )
        if allowed_families is not None and doc.family not in allowed_families:
            return EligibilityDecision(
                allowed=False,
                reason=f"family_access_denied: family={doc.family}",
                as_of=as_of,
            )
        return EligibilityDecision(allowed=True, reason="permitted", as_of=as_of)

    def write_document(self, doc: CanonicalDocument) -> None:
        """Enforces that canonical corpus is strictly read-only."""
        raise ProhibitedWriteError("Modification of canonical source corpus is strictly prohibited")

    def derive_source_entity_edges(
        self,
        min_entity_len: int = 4,
        base_weight: float = 0.85,
    ) -> List[DerivedEdge]:
        """Derive edges strictly from text entity overlap across sources, zero label leakage."""
        entity_to_docs: Dict[str, Set[str]] = {}
        for doc_id, doc in self._docs.items():
            if doc.is_deleted:
                continue
            ents = {e for e in extract_salient_entities(doc.text) if len(e) >= min_entity_len}
            for e in ents:
                entity_to_docs.setdefault(e, set()).add(doc_id)

        edges: List[DerivedEdge] = []
        seen = set()
        for ent, matched_docs in entity_to_docs.items():
            doc_list = sorted(list(matched_docs))
            if len(doc_list) < 2 or len(doc_list) > 10:
                continue
            for i in range(len(doc_list)):
                for j in range(i + 1, len(doc_list)):
                    d1, d2 = doc_list[i], doc_list[j]
                    key_fwd = (d1, d2, ent)
                    key_rev = (d2, d1, ent)
                    if key_fwd not in seen:
                        seen.add(key_fwd)
                        edges.append(
                            DerivedEdge(
                                source_id=d1,
                                target_id=d2,
                                relation_type=f"entity_{ent[:24]}",
                                weight=base_weight,
                                evidence_basis=ent,
                            )
                        )
                    if key_rev not in seen:
                        seen.add(key_rev)
                        edges.append(
                            DerivedEdge(
                                source_id=d2,
                                target_id=d1,
                                relation_type=f"entity_{ent[:24]}_rev",
                                weight=base_weight,
                                evidence_basis=ent,
                            )
                        )
        return edges


def build_frozen_recovery_dataset() -> Tuple[FrozenSourceCorpus, List[Dict[str, Any]]]:
    """Constructs the canonical recovery corpus and paired multi-hop query suites."""
    raw_cases = [
        # Family 1: crypto_quantum (TRAIN)
        {
            "case_id": "cq_01",
            "family": "crypto_quantum",
            "query": "Which national honor was bestowed in 2024 upon the pioneer of Kyber-Lattice cryptography?",
            "query_timestamp": 300.0,
            "hop1": {
                "id": "cq_hop1_kyber",
                "text": "The Kyber-Lattice post-quantum cryptographic protocol was developed in 2021 by Dr. Nadia Rostova at the Zurich Quantum Institute.",
                "timestamp": 120.0,
            },
            "hop2": {
                "id": "cq_hop2_nadia",
                "text": "Dr. Nadia Rostova was awarded the Turing Gold Medal in 2024 for groundbreaking contributions to post-quantum lattice security.",
                "timestamp": 240.0,
            },
            "bridge_entity": "Nadia Rostova",
            "distractors": [
                {
                    "id": "cq_dist_01",
                    "text": "The Kyber-Lattice cipher testbed at Zurich evaluated NIST round-3 post-quantum candidates in 2022.",
                    "timestamp": 130.0,
                },
                {
                    "id": "cq_dist_02",
                    "text": "The Turing Gold Medal in 2023 was presented to distributed database researcher Leslie Lamport.",
                    "timestamp": 210.0,
                },
            ],
        },
        # Family 2: clinical_biotech (TRAIN)
        {
            "case_id": "cb_02",
            "family": "clinical_biotech",
            "query": "What clinical trial endpoint was achieved by the therapeutic target discovered by Lumicor?",
            "query_timestamp": 300.0,
            "hop1": {
                "id": "cb_hop1_lumicor",
                "text": "Biotech startup Lumicor discovered that nuclear receptor ROR-gamma inverse agonism reverses chronic dermal inflammation in autoimmune models.",
                "timestamp": 110.0,
            },
            "hop2": {
                "id": "cb_hop2_ror",
                "text": "In the Phase II clinical trial of ROR-gamma inhibitor LC-882, patients achieved a 75% reduction in Psoriasis Area and Severity Index (PASI-75).",
                "timestamp": 250.0,
            },
            "bridge_entity": "ROR-gamma",
            "distractors": [
                {
                    "id": "cb_dist_01",
                    "text": "Lumicor closed a Series B financing round to advance preclinical small molecule inhibitors of kinase cascades.",
                    "timestamp": 140.0,
                },
                {
                    "id": "cb_dist_02",
                    "text": "Phase II psoriasis trials frequently evaluate PASI-75 and PASI-90 clearance benchmarks over twelve weeks.",
                    "timestamp": 220.0,
                },
            ],
        },
        # Family 3: aerospace_rocketry (HELD-OUT TEST)
        {
            "case_id": "ar_03",
            "family": "aerospace_rocketry",
            "query": "What payload capacity does the primary launcher of the consortium led by Stellaris possess?",
            "query_timestamp": 300.0,
            "hop1": {
                "id": "ar_hop1_stellaris",
                "text": "Aerospace manufacturer Stellaris operates as lead contractor for the European heavy-lift rocket architecture known as Aeolus-V.",
                "timestamp": 105.0,
            },
            "hop2": {
                "id": "ar_hop2_aeolus",
                "text": "The Aeolus-V two-stage cryogenic rocket delivers a verified payload capacity of 21.5 metric tons to Geostationary Transfer Orbit (GTO).",
                "timestamp": 230.0,
            },
            "bridge_entity": "Aeolus-V",
            "distractors": [
                {
                    "id": "ar_dist_01",
                    "text": "Stellaris manufactures carbon-composite payload fairings for commercial medium-lift space launches.",
                    "timestamp": 115.0,
                },
                {
                    "id": "ar_dist_02",
                    "text": "Heavy-lift launchers designed for Geostationary Transfer Orbit require high specific impulse upper stages.",
                    "timestamp": 205.0,
                },
            ],
        },
        # Family 4: semiconductor_optics (HELD-OUT TEST)
        {
            "case_id": "so_04",
            "family": "semiconductor_optics",
            "query": "Which optics manufacturer crafts the projection mirror assembly used in Fab-21 scanners?",
            "query_timestamp": 300.0,
            "hop1": {
                "id": "so_hop1_fab21",
                "text": "Advanced lithography lines at semiconductor plant Fab-21 deploy High-NA extreme ultraviolet systems designed by ASML.",
                "timestamp": 125.0,
            },
            "hop2": {
                "id": "so_hop2_zeiss",
                "text": "Carl Zeiss manufactures the 0.55 numerical aperture EUV projection mirrors that power ASML High-NA scanners.",
                "timestamp": 260.0,
            },
            "bridge_entity": "ASML",
            "distractors": [
                {
                    "id": "so_dist_01",
                    "text": "Semiconductor fabrication facility Fab-21 reached commercial 3nm wafer yields in the second quarter.",
                    "timestamp": 150.0,
                },
                {
                    "id": "so_dist_02",
                    "text": "Precision optical manufacturer Carl Zeiss provides industrial metrology instruments for coordinate measurement.",
                    "timestamp": 215.0,
                },
            ],
        },
    ]

    canonical_docs: List[CanonicalDocument] = []
    queries: List[Dict[str, Any]] = []

    for case in raw_cases:
        family = case["family"]
        # Add hop 1
        h1 = case["hop1"]
        canonical_docs.append(
            CanonicalDocument.create(
                doc_id=h1["id"],
                text=h1["text"],
                timestamp=h1["timestamp"],
                family=family,
                provenance={"source": "frozen_recovery_corpus", "case_id": case["case_id"]},
            )
        )
        # Add hop 2
        h2 = case["hop2"]
        canonical_docs.append(
            CanonicalDocument.create(
                doc_id=h2["id"],
                text=h2["text"],
                timestamp=h2["timestamp"],
                family=family,
                provenance={"source": "frozen_recovery_corpus", "case_id": case["case_id"]},
            )
        )
        # Add distractors
        for dist in case["distractors"]:
            canonical_docs.append(
                CanonicalDocument.create(
                    doc_id=dist["id"],
                    text=dist["text"],
                    timestamp=dist["timestamp"],
                    family=family,
                    provenance={"source": "frozen_recovery_corpus", "distractor": True},
                )
            )

        queries.append(
            {
                "case_id": case["case_id"],
                "family": family,
                "is_test_split": family in TEST_FAMILIES,
                "query": case["query"],
                "query_timestamp": case["query_timestamp"],
                "expected_support": [h1["id"], h2["id"]],
                "bridge_entity": case["bridge_entity"],
            }
        )

    corpus = FrozenSourceCorpus(canonical_docs)
    return corpus, queries
