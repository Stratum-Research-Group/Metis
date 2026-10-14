"""Contracts, schemas and failure exceptions for LTHCM Damage-Trained Recovery."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class RecoveryError(Exception):
    """Base exception for recovery harness errors."""


class ProhibitedWriteError(RecoveryError):
    """Raised when an attempt is made to mutate canonical sources or write new source text."""


class MissingSourceError(RecoveryError):
    """Raised when a candidate or link references a non-existent or lost canonical document."""


class CorruptedProvenanceError(RecoveryError):
    """Raised when a document hash or provenance signature fails verification."""


class EligibilityViolationError(RecoveryError):
    """Raised when a candidate violates access control or strict temporal validity."""


class UnsupportedLinkError(RecoveryError):
    """Raised when a proposed derived link lacks verifiable source-backed evidence."""


class NonFiniteStateError(RecoveryError):
    """Raised when a score, weight, or vector contains NaN or Inf."""


class RecoveryBudgetExceededError(RecoveryError):
    """Raised when step count, latency, or memory bounds are exceeded."""


@dataclass(frozen=True)
class CanonicalDocument:
    """Immutable canonical source document."""

    doc_id: str
    text: str
    content_sha256: str
    timestamp: float
    provenance: Dict[str, Any]
    family: str
    is_deleted: bool = False
    policy_version: str = "v1"

    @classmethod
    def create(
        cls,
        doc_id: str,
        text: str,
        timestamp: float,
        family: str,
        provenance: Optional[Dict[str, Any]] = None,
        is_deleted: bool = False,
        policy_version: str = "v1",
    ) -> CanonicalDocument:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(
            doc_id=doc_id,
            text=text,
            content_sha256=digest,
            timestamp=float(timestamp),
            provenance=dict(provenance or {"source": "canonical_dataset"}),
            family=family,
            is_deleted=is_deleted,
            policy_version=policy_version,
        )

    def verify_hash(self) -> bool:
        computed = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        return computed == self.content_sha256


@dataclass(frozen=True)
class EligibilityDecision:
    """Authoritative external eligibility decision."""

    allowed: bool
    reason: str
    as_of: float
    policy_version: str = "v1"


@dataclass(frozen=True)
class DerivedEdge:
    """Directed link between documents in the disposable shadow graph."""

    source_id: str
    target_id: str
    relation_type: str
    weight: float
    evidence_basis: str  # e.g. shared entity string occurring in both canonical texts

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight):
            raise NonFiniteStateError(f"Derived edge weight {self.weight} is non-finite")
        if self.weight < 0.0 or self.weight > 1.0:
            raise ValueError(f"Edge weight {self.weight} must be bounded in [0.0, 1.0]")
        if not self.evidence_basis:
            raise UnsupportedLinkError("Derived edge requires a non-empty evidence basis")


@dataclass
class ShadowNode:
    """Disposable state node representing a candidate document during refinement."""

    doc_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_stale: bool = False
    is_duplicate: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise NonFiniteStateError(f"Shadow node score {self.score} is non-finite")


@dataclass
class ShadowGraphState:
    """Disposable state of the query-conditioned graph undergoing recovery."""

    nodes: Dict[str, ShadowNode] = field(default_factory=dict)
    edges: List[DerivedEdge] = field(default_factory=list)
    step: int = 0
    state_hash: str = ""

    def compute_state_hash(self) -> str:
        payload = {
            "step": self.step,
            "nodes": sorted([(k, round(v.score, 6), v.is_stale, v.is_duplicate) for k, v in self.nodes.items()]),
            "edges": sorted([(e.source_id, e.target_id, e.relation_type, round(e.weight, 6)) for e in self.edges]),
        }
        raw = json.dumps(payload, sort_keys=True)
        self.state_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self.state_hash


@dataclass(frozen=True)
class DamageConfig:
    """Parameters for deterministic damage injection."""

    seed: int = 42
    edge_drop_prob: float = 0.4
    score_noise_std: float = 0.15
    distractor_link_count: int = 3
    stale_candidate_count: int = 2
    duplicate_node_count: int = 1
    interrupted_steps: Optional[int] = None
    out_of_budget_modes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecoveryStepRecord:
    """Audit log record for one local recovery update step."""

    step: int
    rule_version: str
    previous_state_hash: str
    proposals_evaluated: int
    proposals_accepted: int
    proposals_rejected: List[str]
    elapsed_ms: float


@dataclass
class RecoveryOutcome:
    """Final outcome of recovery execution."""

    status: str  # "SUCCESS", "DEGRADED", "ABSTAIN", "REJECTED"
    final_candidates: List[Dict[str, Any]]
    used_nli: bool
    nli_available: bool
    steps_completed: int
    stop_reason: str
    audit_trail: List[RecoveryStepRecord] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
