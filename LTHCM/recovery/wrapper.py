"""External wrapper around disposable shadow graph for iterative local recovery.

Satisfies Section 3, 4, and 5.4 of LTHCM-recovery-application-plan-356cbcd7.txt:
- Operates strictly on a disposable shadow graph; leaves canonical sources and pinned LTHCM untouched.
- Shared local update rule iterates edge proposals, bounded weights, and ranking scores.
- Conditioned on query, time, source availability, and external eligibility mask.
- Records full audit trail per step (previous state hash, rule version, rejected proposals).
- Stop rule preserves last independently valid state, returning degraded or abstention on failure.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

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

logger = logging.getLogger(__name__)


class RecoveryShadowGraphWrapper:
    """Disposable shadow graph refiner with verifiable local update rules."""

    def __init__(
        self,
        validator: IndependentRecoveryValidator,
        max_steps: int = 5,
        latency_budget_ms: float = 1000.0,
        rule_version: str = "v1-local-message-passing",
    ) -> None:
        self.validator = validator
        self.max_steps = max_steps
        self.latency_budget_ms = latency_budget_ms
        self.rule_version = rule_version

    def refine_and_recover(
        self,
        query: str,
        query_timestamp: float,
        damaged_state: ShadowGraphState,
        candidate_pool: List[Dict[str, Any]],
        nli_available: bool = False,
    ) -> RecoveryOutcome:
        """Executes bounded iterative recovery over the damaged shadow graph."""
        start_time = time.perf_counter()
        audit_trail: List[RecoveryStepRecord] = []

        # Initialize disposable working state
        current_state = copy.deepcopy(damaged_state)
        current_state.compute_state_hash()

        # Retain last independently valid state
        last_valid_state: Optional[ShadowGraphState] = None
        is_init_valid, init_violations = self.validator.validate_shadow_graph(current_state)
        if is_init_valid:
            last_valid_state = copy.deepcopy(current_state)

        stop_reason = "max_steps_reached"
        steps_completed = 0

        for step in range(1, self.max_steps + 1):
            step_start = time.perf_counter()
            elapsed_ms = (step_start - start_time) * 1000.0

            if elapsed_ms >= self.latency_budget_ms:
                stop_reason = "latency_budget_exceeded"
                break

            prev_hash = current_state.state_hash or current_state.compute_state_hash()
            proposals_evaluated = 0
            proposals_accepted = 0
            proposals_rejected: List[str] = []

            # Step A: Filter out stale/disallowed nodes from current state
            cleaned_nodes: Dict[str, ShadowNode] = {}
            for nid, node in list(current_state.nodes.items()):
                proposals_evaluated += 1
                if node.is_stale:
                    proposals_rejected.append(f"rejected_stale_node:{nid}")
                    continue
                decision = self.validator.evaluate_eligibility(node.doc_id)
                if not decision.allowed:
                    proposals_rejected.append(f"rejected_ineligible_node:{nid}:{decision.reason}")
                    continue
                cleaned_nodes[nid] = node
                proposals_accepted += 1
            current_state.nodes = cleaned_nodes

            # Step B: Edge Proposal & Validation
            # Evaluate existing edges and prune invalid ones
            valid_edges: List[DerivedEdge] = []
            for edge in current_state.edges:
                proposals_evaluated += 1
                try:
                    if self.validator.validate_derived_edge(edge):
                        valid_edges.append(edge)
                        proposals_accepted += 1
                except Exception as exc:
                    proposals_rejected.append(f"rejected_edge:{edge.source_id}->{edge.target_id}:{exc}")
            current_state.edges = valid_edges

            # Step C: Grounded Bridge Proposal (NCA-inspired local message passing)
            # Find candidate pairs in the shadow graph that have shared text entities but missing edge
            active_doc_ids = sorted(list({node.doc_id for node in current_state.nodes.values()}))
            for i in range(len(active_doc_ids)):
                for j in range(i + 1, len(active_doc_ids)):
                    d1, d2 = active_doc_ids[i], active_doc_ids[j]
                    # Check if edge already exists
                    has_edge = any(
                        (e.source_id == d1 and e.target_id == d2) or (e.source_id == d2 and e.target_id == d1)
                        for e in current_state.edges
                    )
                    if not has_edge:
                        proposals_evaluated += 1
                        doc1 = self.validator.corpus.get(d1)
                        doc2 = self.validator.corpus.get(d2)
                        if doc1 and doc2:
                            shared_ents = set(extract_salient_entities(doc1.text)) & set(
                                extract_salient_entities(doc2.text)
                            )
                            valid_ents = [e for e in shared_ents if len(e) >= 4]
                            if valid_ents:
                                chosen_ent = valid_ents[0]
                                proposed_edge_fwd = DerivedEdge(
                                    source_id=d1,
                                    target_id=d2,
                                    relation_type=f"entity_{chosen_ent[:20]}",
                                    weight=0.85,
                                    evidence_basis=chosen_ent,
                                )
                                proposed_edge_rev = DerivedEdge(
                                    source_id=d2,
                                    target_id=d1,
                                    relation_type=f"entity_{chosen_ent[:20]}_rev",
                                    weight=0.85,
                                    evidence_basis=chosen_ent,
                                )
                                try:
                                    self.validator.validate_derived_edge(proposed_edge_fwd)
                                    self.validator.validate_derived_edge(proposed_edge_rev)
                                    current_state.edges.extend([proposed_edge_fwd, proposed_edge_rev])
                                    proposals_accepted += 2
                                except Exception as exc:
                                    proposals_rejected.append(f"bridge_proposal_rejected:{exc}")

            # Step D: Local Score Diffusion / Reranking (Query-Conditioned Homeostatic Update)
            # Propagate support scores across verified edges conditioned on query relevance gate
            q_tokens = [w for w in query.lower().replace("?", "").replace(",", "").split() if len(w) > 3]

            def get_query_relevance(doc_id: str) -> float:
                d = self.validator.corpus.get(doc_id)
                if not d:
                    return 0.0
                overlap = sum(1 for tok in q_tokens if tok in d.text.lower())
                return min(1.0, 0.4 * overlap)

            node_deltas: Dict[str, float] = {nid: 0.0 for nid in current_state.nodes}
            for edge in current_state.edges:
                # Target is boosted by source score * weight * query relevance gate
                for nid, node in current_state.nodes.items():
                    if node.doc_id == edge.source_id:
                        gate = get_query_relevance(edge.source_id)
                        if gate > 0.0:
                            for tgt_nid, tgt_node in current_state.nodes.items():
                                if tgt_node.doc_id == edge.target_id:
                                    node_deltas[tgt_nid] += 0.5 * node.score * edge.weight * gate

            max_delta = 0.0
            for nid, node in current_state.nodes.items():
                old_score = node.score
                # 0.6 retention of current score + reinforced neighbor support
                new_score = max(0.0, min(1.0, 0.6 * old_score + node_deltas[nid]))
                if not math.isfinite(new_score):
                    raise NonFiniteStateError(f"Non-finite updated score on node {nid}")
                node.score = new_score
                delta = abs(new_score - old_score)
                if delta > max_delta:
                    max_delta = delta

            current_state.step = step
            current_state.compute_state_hash()
            steps_completed = step
            step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0

            # Audit record
            audit_trail.append(
                RecoveryStepRecord(
                    step=step,
                    rule_version=self.rule_version,
                    previous_state_hash=prev_hash,
                    proposals_evaluated=proposals_evaluated,
                    proposals_accepted=proposals_accepted,
                    proposals_rejected=proposals_rejected,
                    elapsed_ms=step_elapsed_ms,
                )
            )

            # Check full state validity
            is_valid, violations = self.validator.validate_shadow_graph(current_state)
            if is_valid:
                last_valid_state = copy.deepcopy(current_state)
            else:
                stop_reason = f"invalid_state_detected:{violations[0]}"
                break

            # Convergence condition
            if max_delta < 1e-4:
                stop_reason = "converged"
                break

        # Output extraction: use last_valid_state if available
        final_state = last_valid_state if last_valid_state is not None else current_state
        raw_candidates = [
            {"doc_id": node.doc_id, "score": node.score, "metadata": node.metadata}
            for node in final_state.nodes.values()
        ]

        # Final pass through external eligibility gate
        try:
            verified_cands = self.validator.filter_and_validate_candidates(
                raw_candidates, strict_temporal=True
            )
        except Exception as exc:
            return RecoveryOutcome(
                status="REJECTED",
                final_candidates=[],
                used_nli=False,
                nli_available=nli_available,
                steps_completed=steps_completed,
                stop_reason=f"final_validation_failure:{exc}",
                audit_trail=audit_trail,
                diagnostics={"error": str(exc)},
            )

        if not verified_cands:
            status = "ABSTAIN"
        elif len(verified_cands) >= 2:
            top_ids = {c["doc_id"] for c in verified_cands[:4]}
            has_support = any(
                (e.source_id in top_ids and e.target_id in top_ids) for e in final_state.edges
            )
            status = "SUCCESS" if has_support else "DEGRADED"
        else:
            status = "DEGRADED"

        return RecoveryOutcome(
            status=status,
            final_candidates=verified_cands,
            used_nli=False,
            nli_available=nli_available,
            steps_completed=steps_completed,
            stop_reason=stop_reason,
            audit_trail=audit_trail,
            diagnostics={
                "steps_completed": steps_completed,
                "verified_candidate_count": len(verified_cands),
                "recovered_edges_count": len(final_state.edges),
                "total_elapsed_ms": (time.perf_counter() - start_time) * 1000.0,
            },
        )
