"""Comparative baseline implementations for LTHCM Recovery Benchmark.

Satisfies Section 6 of LTHCM-recovery-application-plan-356cbcd7.txt:
Compares at equal budget:
1. Unchanged pipeline (raw damaged retrieval)
2. Deterministic rebuild plus rerank
3. Spectral Chebyshev refiner (SpectralWormholeRefiner)
4. Ordinary message-passing GNN
5. Proposed damage-trained recurrent wrapper
6. Ablation: recurrence disabled (single step)
7. Ablation: damage-blind wrapper (no pruning/invalidation)
"""

from __future__ import annotations

import copy
import logging
import math
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lthcm.core.wormhole_builder import extract_salient_entities
from recovery.schema import (
    DerivedEdge,
    NonFiniteStateError,
    RecoveryOutcome,
    RecoveryStepRecord,
    ShadowGraphState,
    ShadowNode,
)
from recovery.validator import IndependentRecoveryValidator
from recovery.wrapper import RecoveryShadowGraphWrapper

logger = logging.getLogger(__name__)


class BaseRecoveryModel(ABC):
    """Abstract interface for budget-matched comparative models."""

    name: str

    @abstractmethod
    def run(
        self,
        query: str,
        query_timestamp: float,
        damaged_state: ShadowGraphState,
        validator: IndependentRecoveryValidator,
        latency_budget_ms: float = 500.0,
    ) -> RecoveryOutcome:
        pass


class UnchangedPipelineBaseline(BaseRecoveryModel):
    """Baseline 1: Unchanged current pipeline (serves damaged state directly)."""

    name = "unchanged_pipeline"

    def run(
        self,
        query: str,
        query_timestamp: float,
        damaged_state: ShadowGraphState,
        validator: IndependentRecoveryValidator,
        latency_budget_ms: float = 500.0,
    ) -> RecoveryOutcome:
        t0 = time.perf_counter()
        raw_candidates = [
            {"doc_id": n.doc_id, "score": n.score, "metadata": n.metadata}
            for n in damaged_state.nodes.values()
        ]
        # Filter through external validator
        try:
            verified = validator.filter_and_validate_candidates(raw_candidates, strict_temporal=True)
            status = "SUCCESS" if len(verified) >= 2 else ("DEGRADED" if len(verified) == 1 else "ABSTAIN")
        except Exception as exc:
            return RecoveryOutcome(
                status="REJECTED",
                final_candidates=[],
                used_nli=False,
                nli_available=False,
                steps_completed=0,
                stop_reason=f"validation_failure:{exc}",
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return RecoveryOutcome(
            status=status,
            final_candidates=verified,
            used_nli=False,
            nli_available=False,
            steps_completed=0,
            stop_reason="unchanged_passthrough",
            diagnostics={"elapsed_ms": elapsed},
        )


class DeterministicRebuildRerankBaseline(BaseRecoveryModel):
    """Baseline 2: Deterministic rebuild from surviving sources + lexical rerank."""

    name = "deterministic_rebuild_rerank"

    def run(
        self,
        query: str,
        query_timestamp: float,
        damaged_state: ShadowGraphState,
        validator: IndependentRecoveryValidator,
        latency_budget_ms: float = 500.0,
    ) -> RecoveryOutcome:
        t0 = time.perf_counter()
        corpus = validator.corpus

        # Step 1: Filter out ineligible / stale candidates
        eligible_candidates: Dict[str, ShadowNode] = {}
        for nid, node in damaged_state.nodes.items():
            if node.is_stale:
                continue
            decision = validator.evaluate_eligibility(node.doc_id)
            if decision.allowed:
                eligible_candidates[node.doc_id] = copy.deepcopy(node)

        # Step 2: Deterministically rebuild entity wormhole edges strictly from canonical source text
        active_ids = list(eligible_candidates.keys())
        rebuilt_edges: List[DerivedEdge] = []
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                d1, d2 = active_ids[i], active_ids[j]
                doc1 = corpus.get(d1)
                doc2 = corpus.get(d2)
                if doc1 and doc2:
                    shared = set(extract_salient_entities(doc1.text)) & set(extract_salient_entities(doc2.text))
                    valid = [e for e in shared if len(e) >= 4]
                    if valid:
                        ent = valid[0]
                        rebuilt_edges.append(
                            DerivedEdge(
                                source_id=d1,
                                target_id=d2,
                                relation_type=f"rebuilt_{ent[:20]}",
                                weight=0.85,
                                evidence_basis=ent,
                            )
                        )

        # Step 3: Lexical BM25/keyword rerank + graph degree boost
        q_tokens = [w for w in query.lower().replace("?", "").replace(",", "").split() if len(w) > 3]
        for doc_id, node in eligible_candidates.items():
            doc = corpus.get(doc_id)
            if doc:
                text_lower = doc.text.lower()
                overlap = sum(1 for tok in q_tokens if tok in text_lower)
                # Count verified rebuilt connections
                deg = sum(1 for e in rebuilt_edges if e.source_id == doc_id or e.target_id == doc_id)
                rebuilt_score = 0.5 + 0.1 * min(overlap, 4) + 0.15 * min(deg, 2)
                node.score = min(1.0, rebuilt_score)

        raw = [{"doc_id": n.doc_id, "score": n.score} for n in eligible_candidates.values()]
        verified = validator.filter_and_validate_candidates(raw, strict_temporal=True)
        elapsed = (time.perf_counter() - t0) * 1000.0

        status = "SUCCESS" if len(verified) >= 2 else ("DEGRADED" if len(verified) == 1 else "ABSTAIN")
        return RecoveryOutcome(
            status=status,
            final_candidates=verified,
            used_nli=False,
            nli_available=False,
            steps_completed=1,
            stop_reason="deterministic_rebuild_complete",
            diagnostics={"elapsed_ms": elapsed, "rebuilt_edge_count": len(rebuilt_edges)},
        )


class SpectralChebyshevBaseline(BaseRecoveryModel):
    """Baseline 3: Spectral Chebyshev graph convolution refiner."""

    name = "spectral_refiner"

    def __init__(self, K_order: int = 3, d_model: int = 16) -> None:
        self.K_order = K_order
        self.d_model = d_model

    def run(
        self,
        query: str,
        query_timestamp: float,
        damaged_state: ShadowGraphState,
        validator: IndependentRecoveryValidator,
        latency_budget_ms: float = 500.0,
    ) -> RecoveryOutcome:
        t0 = time.perf_counter()
        nodes = list(damaged_state.nodes.values())
        N = len(nodes)
        if N == 0:
            return RecoveryOutcome(
                status="ABSTAIN",
                final_candidates=[],
                used_nli=False,
                nli_available=False,
                steps_completed=1,
                stop_reason="empty_graph",
            )

        # Build normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
        id_map = {n.doc_id: i for i, n in enumerate(nodes)}
        A = np.zeros((N, N), dtype=np.float32)
        for e in damaged_state.edges:
            if e.source_id in id_map and e.target_id in id_map:
                u, v = id_map[e.source_id], id_map[e.target_id]
                A[u, v] = e.weight
                A[v, u] = e.weight

        deg = np.sum(A, axis=1)
        deg_inv_sqrt = np.zeros_like(deg)
        mask = deg > 1e-6
        deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
        D_inv = np.diag(deg_inv_sqrt)
        L = np.eye(N) - D_inv @ A @ D_inv

        # Scaled Laplacian \tilde{L} = L - I (assuming \lambda_{max} ~ 2.0)
        L_tilde = L - np.eye(N)

        # Chebyshev polynomials: T_0(X) = X, T_1(X) = \tilde{L} X, T_2(X) = 2 \tilde{L} T_1 - T_0
        x = np.array([n.score for n in nodes], dtype=np.float32)
        t0_x = x
        t1_x = L_tilde @ x
        t2_x = 2.0 * (L_tilde @ t1_x) - t0_x

        # Spectral polynomial filter combination
        refined_scores = 0.5 * t0_x + 0.3 * t1_x + 0.2 * t2_x
        refined_scores = np.clip(refined_scores, 0.0, 1.0)

        raw = []
        for i, n in enumerate(nodes):
            raw.append({"doc_id": n.doc_id, "score": float(refined_scores[i])})

        verified = validator.filter_and_validate_candidates(raw, strict_temporal=True)
        elapsed = (time.perf_counter() - t0) * 1000.0
        status = "SUCCESS" if len(verified) >= 2 else ("DEGRADED" if len(verified) == 1 else "ABSTAIN")

        return RecoveryOutcome(
            status=status,
            final_candidates=verified,
            used_nli=False,
            nli_available=False,
            steps_completed=1,
            stop_reason="spectral_chebyshev_complete",
            diagnostics={"elapsed_ms": elapsed, "K_order": self.K_order},
        )


class StandardGNNBaseline(BaseRecoveryModel):
    """Baseline 4: Ordinary non-recurrent message passing GNN."""

    name = "message_passing_gnn"

    def run(
        self,
        query: str,
        query_timestamp: float,
        damaged_state: ShadowGraphState,
        validator: IndependentRecoveryValidator,
        latency_budget_ms: float = 500.0,
    ) -> RecoveryOutcome:
        t0 = time.perf_counter()
        nodes = list(damaged_state.nodes.values())
        N = len(nodes)
        if N == 0:
            return RecoveryOutcome(
                status="ABSTAIN",
                final_candidates=[],
                used_nli=False,
                nli_available=False,
                steps_completed=1,
                stop_reason="empty_graph",
            )

        # Standard 2-layer message passing A_hat X W
        id_map = {n.doc_id: i for i, n in enumerate(nodes)}
        A = np.eye(N, dtype=np.float32)
        for e in damaged_state.edges:
            if e.source_id in id_map and e.target_id in id_map:
                u, v = id_map[e.source_id], id_map[e.target_id]
                A[u, v] = e.weight
                A[v, u] = e.weight

        deg = np.sum(A, axis=1, keepdims=True)
        A_norm = A / np.maximum(deg, 1.0)

        x = np.array([n.score for n in nodes], dtype=np.float32).reshape(-1, 1)
        # Layer 1 & 2 message passing
        h1 = np.maximum(0.0, A_norm @ x)
        h2 = np.maximum(0.0, A_norm @ h1)
        final_scores = np.clip(h2.flatten(), 0.0, 1.0)

        raw = [{"doc_id": n.doc_id, "score": float(final_scores[i])} for i, n in enumerate(nodes)]
        verified = validator.filter_and_validate_candidates(raw, strict_temporal=True)
        elapsed = (time.perf_counter() - t0) * 1000.0
        status = "SUCCESS" if len(verified) >= 2 else ("DEGRADED" if len(verified) == 1 else "ABSTAIN")

        return RecoveryOutcome(
            status=status,
            final_candidates=verified,
            used_nli=False,
            nli_available=False,
            steps_completed=2,
            stop_reason="gnn_message_passing_complete",
            diagnostics={"elapsed_ms": elapsed},
        )


class RecurrentRecoveryWrapperBaseline(BaseRecoveryModel):
    """Baseline 5: Proposed damage-trained recurrent wrapper."""

    name = "recurrent_recovery_wrapper"

    def __init__(self, max_steps: int = 4) -> None:
        self.max_steps = max_steps

    def run(
        self,
        query: str,
        query_timestamp: float,
        damaged_state: ShadowGraphState,
        validator: IndependentRecoveryValidator,
        latency_budget_ms: float = 500.0,
    ) -> RecoveryOutcome:
        wrapper = RecoveryShadowGraphWrapper(
            validator=validator,
            max_steps=self.max_steps,
            latency_budget_ms=latency_budget_ms,
            rule_version="v1-recurrent-local-recovery",
        )
        return wrapper.refine_and_recover(
            query=query,
            query_timestamp=query_timestamp,
            damaged_state=damaged_state,
            candidate_pool=[{"doc_id": n.doc_id, "score": n.score} for n in damaged_state.nodes.values()],
        )


class RecurrentNoRecurrenceAblation(BaseRecoveryModel):
    """Ablation A: Recurrent wrapper with recurrence disabled (steps=1)."""

    name = "wrapper_no_recurrence"

    def run(
        self,
        query: str,
        query_timestamp: float,
        damaged_state: ShadowGraphState,
        validator: IndependentRecoveryValidator,
        latency_budget_ms: float = 500.0,
    ) -> RecoveryOutcome:
        wrapper = RecoveryShadowGraphWrapper(
            validator=validator,
            max_steps=1,
            latency_budget_ms=latency_budget_ms,
            rule_version="ablation-single-step-no-recurrence",
        )
        return wrapper.refine_and_recover(
            query=query,
            query_timestamp=query_timestamp,
            damaged_state=damaged_state,
            candidate_pool=[{"doc_id": n.doc_id, "score": n.score} for n in damaged_state.nodes.values()],
        )


class RecurrentNoDamageAblation(BaseRecoveryModel):
    """Ablation B: Wrapper without damage-adaptive filtering (retains stale/unfiltered nodes)."""

    name = "wrapper_no_damage"

    def run(
        self,
        query: str,
        query_timestamp: float,
        damaged_state: ShadowGraphState,
        validator: IndependentRecoveryValidator,
        latency_budget_ms: float = 500.0,
    ) -> RecoveryOutcome:
        t0 = time.perf_counter()
        # Direct unconstrained propagation without stale node pruning or proposal checks
        state = copy.deepcopy(damaged_state)
        for _ in range(3):
            for edge in state.edges:
                if edge.source_id in state.nodes and edge.target_id in state.nodes:
                    state.nodes[edge.target_id].score = min(
                        1.0, state.nodes[edge.target_id].score + 0.1 * state.nodes[edge.source_id].score
                    )

        raw = [{"doc_id": n.doc_id, "score": n.score} for n in state.nodes.values()]
        # Passes through validator at end
        try:
            verified = validator.filter_and_validate_candidates(raw, strict_temporal=True)
            status = "SUCCESS" if len(verified) >= 2 else ("DEGRADED" if len(verified) == 1 else "ABSTAIN")
        except Exception as exc:
            return RecoveryOutcome(
                status="REJECTED",
                final_candidates=[],
                used_nli=False,
                nli_available=False,
                steps_completed=3,
                stop_reason=f"ablation_validation_failure:{exc}",
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return RecoveryOutcome(
            status=status,
            final_candidates=verified,
            used_nli=False,
            nli_available=False,
            steps_completed=3,
            stop_reason="no_damage_ablation_complete",
            diagnostics={"elapsed_ms": elapsed},
        )
