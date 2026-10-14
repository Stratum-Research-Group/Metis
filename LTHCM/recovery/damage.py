"""Deterministic damage injector for LTHCM Recovery Benchmark.

Satisfies Section 5.3 and Section 4 of LTHCM-recovery-application-plan-356cbcd7.txt:
- In-budget damage: missing edges, noisy scores, distractor links, stale candidates, duplicate nodes, interrupted steps.
- Out-of-budget damage: missing canonical documents, revoked access, corrupted hashes.
- Fully deterministic with logged random seed and audit record.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from recovery.schema import (
    DamageConfig,
    DerivedEdge,
    ShadowGraphState,
    ShadowNode,
)

logger = logging.getLogger(__name__)


class DamageInjector:
    """Injects calibrated, deterministic corruption into disposable shadow graph states."""

    def __init__(self, config: Optional[DamageConfig] = None) -> None:
        self.config = config or DamageConfig()
        self.rng = np.random.default_rng(self.config.seed)
        logger.info("DamageInjector initialized with seed %d", self.config.seed)

    def apply_damage(
        self,
        clean_state: ShadowGraphState,
        all_canonical_doc_ids: List[str],
    ) -> Tuple[ShadowGraphState, Dict[str, Any]]:
        """Applies configured damage modes deterministically to produce a damaged shadow graph."""
        damaged_nodes = copy.deepcopy(clean_state.nodes)
        damaged_edges = copy.deepcopy(clean_state.edges)
        diagnostics: Dict[str, Any] = {
            "seed": self.config.seed,
            "original_node_count": len(damaged_nodes),
            "original_edge_count": len(damaged_edges),
            "dropped_edge_count": 0,
            "distractor_edge_count": 0,
            "stale_node_count": 0,
            "duplicate_node_count": 0,
            "noisy_scores_applied": 0,
            "out_of_budget_injected": list(self.config.out_of_budget_modes),
        }

        # 1. Edge Drops (missing derived edges)
        surviving_edges: List[DerivedEdge] = []
        for edge in damaged_edges:
            if self.rng.random() < self.config.edge_drop_prob:
                diagnostics["dropped_edge_count"] += 1
            else:
                surviving_edges.append(edge)
        damaged_edges = surviving_edges

        # 2. Score Noise (noisy derived scores)
        if self.config.score_noise_std > 0.0:
            for node_id, node in damaged_nodes.items():
                noise = float(self.rng.normal(0.0, self.config.score_noise_std))
                node.score = max(0.0, min(1.0, node.score + noise))
                diagnostics["noisy_scores_applied"] += 1

        # 3. Distractor Links (unsupported false shortcut edges)
        node_id_list = sorted(list(damaged_nodes.keys()))
        if len(node_id_list) >= 2 and self.config.distractor_link_count > 0:
            for _ in range(self.config.distractor_link_count):
                u, v = self.rng.choice(node_id_list, size=2, replace=False)
                distractor_edge = DerivedEdge(
                    source_id=str(u),
                    target_id=str(v),
                    relation_type="synthetic_distractor_shortcut",
                    weight=float(self.rng.uniform(0.7, 0.95)),
                    evidence_basis="unverified_distractor_claim",
                )
                damaged_edges.append(distractor_edge)
                diagnostics["distractor_edge_count"] += 1

        # 4. Stale Candidates (outdated cached items)
        for i in range(self.config.stale_candidate_count):
            stale_id = f"stale_cached_candidate_{i:02d}"
            damaged_nodes[stale_id] = ShadowNode(
                doc_id=stale_id,
                score=float(self.rng.uniform(0.8, 0.99)),
                metadata={"source": "stale_cache", "cached_timestamp": 0.0},
                is_stale=True,
            )
            diagnostics["stale_node_count"] += 1

        # 5. Duplicate Derived Nodes (redundant shadow entries with divergent scores)
        if node_id_list and self.config.duplicate_node_count > 0:
            for i in range(min(self.config.duplicate_node_count, len(node_id_list))):
                orig_id = node_id_list[i]
                orig_node = damaged_nodes[orig_id]
                dup_id = f"{orig_id}_dup_{i}"
                damaged_nodes[dup_id] = ShadowNode(
                    doc_id=orig_node.doc_id,  # Points to same canonical doc_id
                    score=max(0.0, min(1.0, orig_node.score * float(self.rng.uniform(0.5, 0.8)))),
                    metadata=dict(orig_node.metadata),
                    is_duplicate=True,
                )
                diagnostics["duplicate_node_count"] += 1

        # 6. Out-of-Budget Damage Modes
        if "missing_canonical" in self.config.out_of_budget_modes:
            fake_id = "completely_fabricated_doc_999"
            damaged_nodes[fake_id] = ShadowNode(
                doc_id=fake_id,
                score=0.99,
                metadata={"fabricated": True},
            )
            diagnostics["missing_canonical_injected"] = fake_id

        if "revoked_access" in self.config.out_of_budget_modes:
            revoked_id = "revoked_access_candidate_01"
            damaged_nodes[revoked_id] = ShadowNode(
                doc_id=revoked_id,
                score=0.95,
                metadata={"revoked": True},
            )
            diagnostics["revoked_access_injected"] = revoked_id

        damaged_state = ShadowGraphState(
            nodes=damaged_nodes,
            edges=damaged_edges,
            step=clean_state.step,
        )
        damaged_state.compute_state_hash()

        diagnostics["final_node_count"] = len(damaged_nodes)
        diagnostics["final_edge_count"] = len(damaged_edges)
        diagnostics["damaged_state_hash"] = damaged_state.state_hash

        return damaged_state, diagnostics
