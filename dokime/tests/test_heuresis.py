"""Unit and integration tests for Heuresis Discovery Engine & Dokime Task."""

import pytest

from dokime.tasks import (
    ConceptAtom,
    DiscoveryCandidate,
    HeuresisDiscoveryEngine,
    HeuresisDiscoveryTask,
    get_task,
)
from stoma.crawlers.openreview import CommunityAvailabilityIndex


def test_heuresis_dual_scorer_frontier_preference():
    avail_index = CommunityAvailabilityIndex()
    # Mainstream accepted co-occurrences (multiple accepted papers)
    for _ in range(3):
        avail_index.record_co_occurrence("graph_transformers", "mixture_of_experts", is_accepted=True)
        avail_index.record_co_occurrence("graph_transformers", "diffusion_guidance", is_accepted=True)
        avail_index.record_co_occurrence("mixture_of_experts", "diffusion_guidance", is_accepted=True)

    # Contested boundary from peer reviews (low availability)
    avail_index.record_co_occurrence("graph_transformers", "riemannian_metric", is_accepted=False)
    avail_index.record_co_occurrence("riemannian_metric", "cellular_sheaves", is_accepted=False)

    engine = HeuresisDiscoveryEngine(availability_index=avail_index, beta=1.5)

    saturated_triple = ("graph_transformers", "mixture_of_experts", "diffusion_guidance")
    frontier_triple = ("graph_transformers", "riemannian_metric", "cellular_sheaves")

    cand_saturated = engine.score_combination(saturated_triple)
    cand_frontier = engine.score_combination(frontier_triple)

    assert cand_saturated.availability_score > 0.6
    assert cand_frontier.availability_score <= 0.25
    # Frontier triple must decisively beat the saturated triple on discovery score
    assert cand_frontier.discovery_score > cand_saturated.discovery_score


def test_sheaf_obstruction_and_oov_synthesis():
    engine = HeuresisDiscoveryEngine()

    cand = engine.score_combination(("graph_transformers", "riemannian_metric", "cellular_sheaves"))
    assert cand.is_oov_synthesized is True
    assert cand.obstruction_energy > 0.0
    assert cand.oov_primitive == "poincare_curvature_sheaf_transformer"

    cand_clean = engine.score_combination(("graph_transformers", "mixture_of_experts", "diffusion_guidance"))
    assert cand_clean.is_oov_synthesized is False
    assert cand_clean.obstruction_energy == 0.0
    assert cand_clean.oov_primitive is None


def test_fringe_outlier_quota_preservation():
    engine = HeuresisDiscoveryEngine(fringe_quota=0.15)
    top_candidates = engine.generate_top_candidates(k=10, triple_size=3)

    assert len(top_candidates) == 10
    fringe_count = sum(1 for c in top_candidates if c.has_fringe_unit)
    # 15% of 10 is ceil(1.5) = 2
    assert fringe_count >= 2


def test_heuresis_dokime_task_e2e():
    task = HeuresisDiscoveryTask()
    assert task.task_id == "heuresis_scientific_discovery"

    queries = task.load()
    assert len(queries) == 3

    for q in queries:
        assert q.expected in ("A", "B", "C")
        # Correct answer scores 1.0
        assert task.score(q, q.expected) == 1.0
        # Incorrect answer scores 0.0
        assert task.score(q, "X") == 0.0


def test_discovery_registry_lookup():
    task = get_task("heuresis_scientific_discovery")
    assert isinstance(task, HeuresisDiscoveryTask)
    queries = task.load()
    assert len(queries) >= 3
