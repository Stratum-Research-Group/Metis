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


@pytest.mark.asyncio
async def test_pipeline_fallback_marked_simulated(pipeline_setup):
    """Verify that when Hestia is unreachable and fallback synthesis runs, is_simulated is True (Finding R6)."""
    pipeline, _, _ = pipeline_setup
    # HestiaAdapter in offline/unreachable state by default in this test fixture
    req = TurnRequest(session_id="s1", query="Calculate 50 + 50")
    events = [e async for e in pipeline.execute_turn_stream(req)]
    completed = events[-1]
    assert completed.event_type == EventType.COMPLETED.value
    show_work = completed.payload.get("route_summary", {})
    assert show_work.get("is_simulated") is True


@pytest.mark.asyncio
async def test_pipeline_midstream_hestia_error_fallback(pipeline_setup, monkeypatch):
    """Finding B1 regression: mid-stream Hestia error must not concatenate unrelated fallback text."""
    pipeline, _, _ = pipeline_setup

    async def mock_check_health():
        return {"available": True, "model": "mock-hestia"}

    async def mock_stream_chat(prompt):
        # Stream 2 deltas, then encounter an error
        yield {"delta": "Hello "}
        yield {"delta": "world "}
        yield {"error": "Upstream token limit exceeded"}

    monkeypatch.setattr(pipeline.hestia, "check_health", mock_check_health)
    monkeypatch.setattr(pipeline.hestia, "stream_chat", mock_stream_chat)

    req = TurnRequest(session_id="s1", query="Calculate 25 + 25")
    events = [e async for e in pipeline.execute_turn_stream(req)]

    deltas = [e.payload["text"] for e in events if e.event_type == EventType.DELTA.value]
    full_text = "".join(deltas)

    # Must contain the partial stream and explicit interruption notice
    assert "Hello world " in full_text
    assert "[Upstream generation interrupted: Upstream token limit exceeded]" in full_text

    # Must NOT concatenate the local fallback sentence ("The calculation result is ...")
    assert "The calculation result is" not in full_text


def test_thinking_budget_configuration(monkeypatch):
    """Verify thinking_budget_steps in KybernetesConfig and load_config env override."""
    from kybernetes.config import KybernetesConfig, load_config

    default_cfg = KybernetesConfig()
    assert default_cfg.thinking_budget_steps == 10

    custom_cfg = KybernetesConfig(thinking_budget_steps=25)
    assert custom_cfg.thinking_budget_steps == 25

    monkeypatch.setenv("KYBERNETES_THINKING_BUDGET_STEPS", "15")
    loaded = load_config()
    assert loaded.thinking_budget_steps == 15


@pytest.mark.asyncio
async def test_pipeline_hard_policy_denial_no_approval_req(pipeline_setup, monkeypatch):
    """F-07: Verify that hard DENY in Phylax policy immediately terminates with DENIED without executing tool."""
    pipeline, phylax, _ = pipeline_setup
    
    # Mock Phylax is_action_permitted to return (False, None) -> hard deny, no interactive approval possible
    def mock_is_action_permitted(capability, target, arguments=None):
        return (False, None)

    monkeypatch.setattr(phylax, "is_action_permitted", mock_is_action_permitted)

    req = TurnRequest(session_id="s_hard_deny", query="Write file to /etc/shadow")
    events = [e async for e in pipeline.execute_turn_stream(req)]

    types = [e.event_type for e in events]
    assert EventType.ASK_APPROVAL.value not in types
    assert EventType.COMPLETED.value in types
    
    completed_event = events[-1]
    assert completed_event.status == TurnOutcomeStatus.DENIED.value
    assert completed_event.payload["route_summary"]["action_status"] == "denied"



