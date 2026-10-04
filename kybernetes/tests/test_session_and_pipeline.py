"""Integration tests for the ExecutionPipeline, SessionManager, and cancellation."""

import asyncio
import pytest
from kybernetes.adapters.dianoia import DianoiaAdapter
from kybernetes.adapters.hestia import HestiaAdapter
from kybernetes.adapters.krisis import KrisisAdapter
from kybernetes.adapters.lthcm import LTHCMAdapter
from kybernetes.adapters.phylax import PhylaxAdapter
from kybernetes.budget.engine import BudgetEngine
from kybernetes.budget.storage import BudgetDatabase
from kybernetes.contracts.enums import EventType, TurnOutcomeStatus
from kybernetes.contracts.turn import TurnRequest
from kybernetes.engine.pipeline import ExecutionPipeline
from kybernetes.engine.session import SessionManager
from kybernetes.routing.router import PlanRouter
from kybernetes.tools.registry import ToolRegistry


@pytest.fixture
def pipeline_setup():
    db = BudgetDatabase(":memory:")
    budget = BudgetEngine(db)
    tools = ToolRegistry()
    hestia = HestiaAdapter()
    lthcm = LTHCMAdapter()
    dianoia = DianoiaAdapter()
    phylax = PhylaxAdapter()
    krisis = KrisisAdapter()
    sessions = SessionManager()
    router = PlanRouter()

    pipeline = ExecutionPipeline(
        router=router,
        budget_engine=budget,
        tool_registry=tools,
        hestia_adapter=hestia,
        lthcm_adapter=lthcm,
        dianoia_adapter=dianoia,
        phylax_adapter=phylax,
        krisis_adapter=krisis,
        session_manager=sessions,
    )
    return pipeline, phylax, sessions


@pytest.mark.asyncio
async def test_pipeline_direct_turn(pipeline_setup):
    pipeline, _, _ = pipeline_setup
    req = TurnRequest(session_id="s1", query="Hello assistant!")
    events = [e async for e in pipeline.execute_turn_stream(req)]

    types = [e.event_type for e in events]
    assert EventType.STARTED.value in types
    assert EventType.PLAN_SELECTED.value in types
    assert EventType.DELTA.value in types
    assert EventType.COMPLETED.value in types

    completed_event = events[-1]
    assert completed_event.event_type == EventType.COMPLETED.value
    assert completed_event.payload["route_summary"]["route"] == "direct"


@pytest.mark.asyncio
async def test_pipeline_tool_math(pipeline_setup):
    pipeline, _, _ = pipeline_setup
    req = TurnRequest(session_id="s1", query="Calculate 100 + 45")
    events = [e async for e in pipeline.execute_turn_stream(req)]

    deltas = "".join([e.payload.get("text", "") for e in events if e.event_type == EventType.DELTA.value])
    assert "145.0" in deltas

    completed_event = events[-1]
    assert completed_event.status == TurnOutcomeStatus.VERIFIED.value


@pytest.mark.asyncio
async def test_pipeline_approval_denial(pipeline_setup):
    pipeline, phylax, _ = pipeline_setup
    req = TurnRequest(session_id="s1", query="Write file to /tmp/metis_task.txt")

    events = []
    async for event in pipeline.execute_turn_stream(req):
        events.append(event)
        if event.event_type == EventType.ASK_APPROVAL.value:
            app_id = event.payload["approval"]["approval_id"]
            phylax.resolve_approval(app_id, "deny")

    types = [e.event_type for e in events]
    assert EventType.ASK_APPROVAL.value in types
    assert EventType.APPROVAL_RESOLVED.value in types
    assert EventType.COMPLETED.value in types

    completed_event = events[-1]
    assert completed_event.payload["route_summary"]["action_status"] == "denied"


@pytest.mark.asyncio
async def test_pipeline_cancellation(pipeline_setup):
    pipeline, _, sessions = pipeline_setup
    req = TurnRequest(turn_id="turn-cancel-test", session_id="s1", query="What is Metis?")

    # Request cancellation before start
    sessions.request_cancellation("turn-cancel-test")

    events = [e async for e in pipeline.execute_turn_stream(req)]
    types = [e.event_type for e in events]
    assert EventType.CANCELLED.value in types


def test_session_manager_redaction_and_reconnect(pipeline_setup):
    _, _, sessions = pipeline_setup
    secret_text = "Connect to server using api_key='secret1234567890abcdef' or Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    redacted = sessions.redact_sensitive_text(secret_text)
    assert "secret1234567890abcdef" not in redacted
    assert "[REDACTED_SECRET]" in redacted
