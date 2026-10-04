"""Unit tests for Kybernetes data contracts and enumerations."""

import pytest
from kybernetes.contracts.enums import (
    BudgetUnit,
    CheckpointAction,
    EventType,
    RouteTag,
    StageName,
    ToolSideEffectClass,
    TurnOutcomeStatus,
)
from kybernetes.contracts.turn import (
    ApprovalResolution,
    Citation,
    ShowWorkInfo,
    TurnEvent,
    TurnRequest,
)
from kybernetes.contracts.plan import ExecutionPlan, StageOutcome, StagePlan
from kybernetes.contracts.trace import TraceRecord
from kybernetes.contracts.tool_spec import ToolCallRequest, ToolDefinition, ToolResult


def test_turn_event_prosopon_compatibility():
    event = TurnEvent(
        turn_id="turn-123",
        seq_num=0,
        event_type="started",
        payload={"query": "test query"},
        status="draft",
    )
    dumped = event.model_dump()
    assert dumped["turn_id"] == "turn-123"
    assert dumped["seq_num"] == 0
    assert dumped["event_type"] == "started"
    assert dumped["status"] == "draft"
    assert "timestamp_utc" in dumped


def test_citation_model():
    citation = Citation(
        citation_id="[1]",
        passage_id="pass_01",
        source_id="doc_01",
        source_title="Test Doc",
        text="Sample text snippet",
        score=0.95,
    )
    assert citation.citation_id == "[1]"
    assert citation.score == 0.95


def test_execution_plan_structure():
    stage = StagePlan(stage=StageName.AUTHORIZE, description="Auth check")
    plan = ExecutionPlan(
        plan_id="plan-1",
        turn_id="turn-1",
        route_label="direct",
        route_tags=[RouteTag.DIRECT],
        stages=[stage],
    )
    assert len(plan.stages) == 1
    assert plan.route_tags[0] == RouteTag.DIRECT


def test_show_work_info():
    info = ShowWorkInfo(
        turn_id="turn-1",
        route="direct",
        route_tags=["DIRECT"],
        components_used=["Hestia"],
        citations=[],
        timings_ms={"auth": 10.5},
        verifier_status="verified",
        zero_edge_fallback_used=False,
    )
    dumped = info.model_dump()
    assert dumped["verifier_status"] == "verified"
    assert dumped["zero_edge_fallback_used"] is False
