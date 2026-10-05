"""Unit tests for Paideia contracts and schemas."""

import pytest
from pydantic import ValidationError

from paideia.contracts.dianoia_schema import (
    DianoiaStep,
    DianoiaVerification,
    DianoiaWalkRecord,
)
from paideia.contracts.lthcm_schema import (
    LTHCMRoutingRecord,
    RouteOutcome,
    RouteUtility,
)
from paideia.contracts.kybernetes_schema import KybernetesRouteDefinition
from paideia.contracts.hestia_schema import HestiaAdapterRegistryEntry
from paideia.contracts.manifest_schema import RunManifest


def test_dianoia_step_validation():
    step = DianoiaStep(node="node_1", action="EXPAND", reason="Search for author")
    assert step.node == "node_1"
    assert step.action == "EXPAND"

    # Empty node should fail
    with pytest.raises(ValidationError):
        DianoiaStep(node="  ", action="EXPAND")

    # Invalid action should fail
    with pytest.raises(ValidationError):
        DianoiaStep(node="node_1", action="INVALID_ACTION")  # type: ignore


def test_dianoia_walk_gold_trajectory_filtering():
    step1 = DianoiaStep(node="n1", action="EXPAND", reason="hop")
    step2 = DianoiaStep(node="n2", action="ANSWER", reason="found")

    # Clean walk -> defensible
    walk = DianoiaWalkRecord(
        walk_id="w-1",
        question="Who was X?",
        path=[step1, step2],
        verification=DianoiaVerification(verified=True, conflicts_found=False, verifier_mode="sheaf", provenance="sheaf", is_correct=True),
        final_answer="Answer X",
        model="test_model",
    )
    assert walk.is_defensible_gold_trajectory() is True

    # Unresolved conflict -> not defensible
    conflicted_walk = DianoiaWalkRecord(
        walk_id="w-2",
        question="Who was X?",
        path=[step1, step2],
        verification=DianoiaVerification(verified=True, conflicts_found=True, rewalked=False, verifier_mode="sheaf", provenance="sheaf", is_correct=True),
        final_answer="Answer X",
        model="test_model",
    )
    assert conflicted_walk.is_defensible_gold_trajectory() is False

    # Resolved conflict via re-walk -> defensible
    resolved_walk = DianoiaWalkRecord(
        walk_id="w-2",
        question="Who was X?",
        path=[step1, step2],
        verification=DianoiaVerification(verified=True, conflicts_found=False, rewalked=True, verifier_mode="sheaf", provenance="sheaf", is_correct=True),
        final_answer="Answer X",
        model="test_model",
    )
    assert resolved_walk.is_defensible_gold_trajectory() is True


def test_lthcm_routing_counterfactuals_and_ties():
    rec = LTHCMRoutingRecord(
        query_id="q1",
        question="Who lived longer, A or B?",
        gold_type="comparison",
        route_outcomes={
            "flat": RouteOutcome(route_id="flat", quality_score=0.90, latency_ms=10.0, compute_cost=1.0),
            "mined": RouteOutcome(route_id="mined", quality_score=0.60, latency_ms=40.0, compute_cost=3.0),
        },
    )

    util = RouteUtility(lambda_latency=0.001, lambda_cost=0.01, tie_tolerance=0.01)
    best = rec.get_best_route(util, eligible_routes=["flat", "mined"])
    assert best == "flat"

    # Missing counterfactual returns None
    assert rec.get_best_route(util, eligible_routes=["flat", "missing_route"]) is None

    # Exact tie within tolerance returns None (ambiguous)
    tied_rec = LTHCMRoutingRecord(
        query_id="q2",
        question="Test tie question",
        gold_type="comparison",
        route_outcomes={
            "r1": RouteOutcome(route_id="r1", quality_score=0.80, latency_ms=10.0, compute_cost=1.0),
            "r2": RouteOutcome(route_id="r2", quality_score=0.80, latency_ms=10.0, compute_cost=1.0),
        },
    )
    assert tied_rec.get_best_route(util, eligible_routes=["r1", "r2"]) is None


def test_manifest_schema_immutability():
    manifest = RunManifest(
        run_id="run-123",
        task_name="test_task",
        status="RUNNING",
        hardware_line="RTX 5060 Ti",
        dataset_checksum="abc123hash",
    )
    assert manifest.status == "RUNNING"
    dump = manifest.model_dump()
    assert dump["run_id"] == "run-123"
