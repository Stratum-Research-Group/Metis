"""
Frozen experiment manifest specification and validation (Section 10.1 & 10.3).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import numpy as np

from kleroterion.errors import InvalidConfigError


@dataclass(frozen=True)
class QueryItem:
    query_id: str
    query_text: str
    query_type: str  # "exact_fact" or "exploratory"
    relevant_doc_ids: Set[str]
    split: str  # "tune" or "held_out"


@dataclass(frozen=True)
class AcceptanceGates:
    """
    Explicit acceptance gates (Section 10.3).
    Missing values mean evaluation acceptance is blocked.
    """
    overlap_min: Optional[float] = None
    recall_loss_max: Optional[float] = None
    gain_min: Optional[float] = None
    exact_fact_loss_max: Optional[float] = None
    contradiction_max: Optional[float] = None
    p95_latency_budget_sec: Optional[float] = None
    empty_intersection_rate_max: Optional[float] = None

    def is_fully_configured(self) -> bool:
        return all(
            val is not None
            for val in (
                self.overlap_min,
                self.recall_loss_max,
                self.gain_min,
                self.exact_fact_loss_max,
                self.contradiction_max,
                self.p95_latency_budget_sec,
                self.empty_intersection_rate_max,
            )
        )


@dataclass(frozen=True)
class FrozenExperimentManifest:
    """
    Manifest freezing all experimental conditions (Section 10.1).
    """
    manifest_id: str
    lthcm_revision: str
    dianoia_revision: str
    corpus_snapshot_id: str
    graph_snapshot_id: str
    index_snapshot_id: str
    scorer_version: str
    embedding_version: str
    queries: List[QueryItem]
    k: int = 10
    walk_depth: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 1337, 2026])
    proposal_count: int = 4
    acceptance_gates: AcceptanceGates = field(default_factory=AcceptanceGates)

    def validate(self) -> None:
        if not self.lthcm_revision:
            raise InvalidConfigError("Manifest requires pinned lthcm_revision.")
        if not self.dianoia_revision:
            raise InvalidConfigError("Manifest requires pinned dianoia_revision.")
        if not self.corpus_snapshot_id or not self.graph_snapshot_id:
            raise InvalidConfigError("Manifest requires frozen corpus and graph snapshot IDs.")
        if not self.queries:
            raise InvalidConfigError("Manifest must contain at least one QueryItem.")
        if self.k < 1:
            raise InvalidConfigError("k must be >= 1.")
