"""Unit tests for the PlanRouter and route selection rules."""

import pytest
from kybernetes.contracts.enums import RouteTag, StageName
from kybernetes.contracts.turn import TurnRequest
from kybernetes.routing.router import PlanRouter


@pytest.fixture
def router():
    return PlanRouter()


def test_route_direct_greeting(router):
    req = TurnRequest(session_id="s1", query="Hello! How are you doing today?")
    plan = router.build_plan(req)
    assert plan.route_label == "direct"
    assert RouteTag.DIRECT in plan.route_tags
    stage_names = [s.stage for s in plan.stages]
    assert StageName.AUTHORIZE in stage_names
    assert StageName.RETRIEVE not in stage_names
    assert StageName.SYNTHESIZE in stage_names


def test_route_factual_retrieval(router):
    req = TurnRequest(session_id="s1", query="What is the architecture of Metis?")
    plan = router.build_plan(req)
    assert RouteTag.MEMORY in plan.route_tags
    stage_names = [s.stage for s in plan.stages]
    assert StageName.RETRIEVE in stage_names


def test_route_deliberation_multi_hop(router):
    req = TurnRequest(session_id="s1", query="Compare and contrast how Phylax and Krisis interact in Metis.")
    plan = router.build_plan(req)
    assert plan.route_label == "deliberate_graph"
    assert RouteTag.DELIBERATE in plan.route_tags
    assert RouteTag.MEMORY in plan.route_tags
    stage_names = [s.stage for s in plan.stages]
    assert StageName.RETRIEVE in stage_names
    assert StageName.DELIBERATE in stage_names
    deliberate_stage = next(s for s in plan.stages if s.stage == StageName.DELIBERATE)
    assert deliberate_stage.params.get("graph_consistency_floor") == 0.95
    assert deliberate_stage.params.get("fallback_to_pure_text_on_inconsistency") is True
    assert deliberate_stage.params.get("two_pass_constructive_planning") is True


def test_route_tool_calculator(router):
    req = TurnRequest(session_id="s1", query="Calculate 25 * 4 + sqrt(81)")
    plan = router.build_plan(req)
    assert RouteTag.TOOL in plan.route_tags
    stage_names = [s.stage for s in plan.stages]
    assert StageName.TOOL in stage_names


def test_route_tool_file_search(router):
    req = TurnRequest(session_id="s1", query="Search file for keyword config in workspace")
    plan = router.build_plan(req)
    assert RouteTag.TOOL in plan.route_tags
    stage_names = [s.stage for s in plan.stages]
    assert StageName.TOOL in stage_names
