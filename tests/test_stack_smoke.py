"""
Stack-level smoke and integration tests for Metis sovereign local AI stack.
Validates cross-component interactions across:
  - Stoma / LTHCM (temporal bounding & citation contracts)
  - Hestia (model registry & sampling parameters)
  - Dianoia (zero-edge fallback & simulation disclosure)
  - Phylax (point-of-action capability gating & approval tokens)
  - Krisis (scoring & verification status)
  - Kybernetes (orchestration pipeline & SSE event stream)
  - Prosopon (session storage & client compatibility)
"""

import asyncio
from pathlib import Path
import pytest

from kybernetes.contracts.turn import TurnRequest, Citation, TurnEvent
from kybernetes.engine.pipeline import ExecutionPipeline
from kybernetes.budget.storage import BudgetDatabase
from kybernetes.budget.engine import BudgetEngine
from kybernetes.engine.session import SessionManager
from kybernetes.routing.router import PlanRouter
from kybernetes.tools.registry import ToolRegistry
from kybernetes.adapters.hestia import HestiaAdapter
from kybernetes.adapters.lthcm import LTHCMAdapter
from kybernetes.adapters.dianoia import DianoiaAdapter
from kybernetes.adapters.phylax import PhylaxAdapter
from kybernetes.adapters.krisis import KrisisAdapter
from hestia.registry import ModelRegistry, ModelType
from prosopon.storage.session_store import SessionStore
from prosopon.contracts.session import ChatMessage


@pytest.fixture
def mock_hestia_adapter():
    """Hestia adapter that yields deterministic streaming chunks for offline smoke testing."""
    class MockHestia(HestiaAdapter):
        async def stream_chat(self, prompt, system_prompt="...", model=None, max_tokens=1024, temperature=0.2):
            yield {"delta": "According to the architectural records [1], "}
            yield {"delta": "Kybernetes orchestrates Hestia, LTHCM, and Phylax."}
            yield {"usage": {"input_tokens": 42, "output_tokens": 18}}

        async def check_health(self):
            return {
                "available": True,
                "models": ["llama-3.1-8b-instruct", "nomic-embed-text"],
                "default_model": "llama-3.1-8b-instruct",
                "default_model_registered": True,
                "url": "http://127.0.0.1:8080/v1",
            }
    return MockHestia()


@pytest.mark.asyncio
async def test_stack_e2e_factual_orchestration_flow(mock_hestia_adapter):
    """
    Test complete factual turn flow:
    LTHCM evidence -> Dianoia walk -> Mock Hestia synthesis -> Krisis gate -> Prosopon event stream.
    """
    db = BudgetDatabase(":memory:")
    budget = BudgetEngine(db)
    tools = ToolRegistry()
    sessions = SessionManager()
    router = PlanRouter()

    pipeline = ExecutionPipeline(
        router=router,
        budget_engine=budget,
        tool_registry=tools,
        hestia_adapter=mock_hestia_adapter,
        lthcm_adapter=LTHCMAdapter(),
        dianoia_adapter=DianoiaAdapter(),
        phylax_adapter=PhylaxAdapter(),
        krisis_adapter=KrisisAdapter(),
        session_manager=sessions,
    )

    req = TurnRequest(
        session_id="smoke-test-session",
        query="Explain Metis ecosystem architecture and orchestration.",
        turn_id="turn-smoke-001",
    )

    events = []
    async for event in pipeline.execute_turn_stream(req):
        events.append(event)

    event_types = [e.event_type for e in events]
    assert "started" in event_types
    assert "plan_selected" in event_types
    assert "evidence" in event_types
    assert "delta" in event_types
    assert "completed" in event_types

    # Verify monotonic sequence numbering
    seq_nums = [e.seq_num for e in events]
    assert seq_nums == list(range(len(events)))

    # Verify evidence citations contain expected provenance and disclosure
    evidence_events = [e for e in events if e.event_type == "evidence"]
    assert len(evidence_events) == 1
    citations = evidence_events[0].payload.get("citations", [])
    assert len(citations) > 0
    assert all("citation_id" in c for c in citations)
    assert all(c.get("is_simulated") is True for c in citations)

    # Verify final completed event is verified and route summary is populated
    final_event = events[-1]
    assert final_event.event_type == "completed"
    assert final_event.status == "verified"
    assert "route_summary" in final_event.payload
    assert final_event.payload["route_summary"]["route"] == "retrieval_augmented"
    assert "show_work" in final_event.payload

    # Verify streamed response text contains cited claims
    full_text = "".join(e.payload.get("text", "") for e in events if e.event_type == "delta")
    assert "[1]" in full_text
    assert "Kybernetes orchestrates" in full_text


def test_temporal_lightcone_as_of_filtering():
    """Verify temporal bounds contract: passages created after as_of must be excluded."""
    adapter = LTHCMAdapter()

    # Passages in demo corpus created between epoch 1700000000 and 1715000000
    future_citations = adapter.retrieve("kybernetes metis", as_of=1800000000.0)
    assert len(future_citations) > 0

    past_citations = adapter.retrieve("kybernetes metis", as_of=1600000000.0)
    assert len(past_citations) == 0


def test_dianoia_zero_edge_fallback_contract():
    """Verify Dianoia activates flat fallback and flags is_simulated."""
    adapter = DianoiaAdapter()
    citations = [
        Citation(
            citation_id="[1]",
            passage_id="p1",
            source_id="d1",
            source_title="Doc 1",
            text="Single isolated citation snippet",
            score=0.9,
        )
    ]
    res = adapter.deliberate(query="test", seed_citations=citations)
    assert res.zero_edge_fallback_used is True
    assert res.edges_traversed == 0
    assert res.is_simulated is True
    assert "simulated adapter" in res.deliberation_summary


@pytest.mark.asyncio
async def test_phylax_point_of_action_approval_lifecycle():
    """Verify Phylax denies privileged action until authorized by single-use token."""
    phylax = PhylaxAdapter()

    # Read action allowed immediately
    allowed_read, app_read = phylax.is_action_permitted("read_file", "data.txt", {})
    assert allowed_read is True
    assert app_read is None

    # Write action requires approval
    allowed_write, app_req = phylax.is_action_permitted("write_file", "output.txt", {"bytes": 100})
    assert allowed_write is False
    assert app_req is not None
    assert "approval_id" in app_req

    app_id = app_req["approval_id"]
    # Unknown approval resolution fails
    assert phylax.resolve_approval("nonexistent-approval-id", "deny") is False

    # Resolve approval
    assert phylax.resolve_approval(app_id, "allow_once") is True
    decision = await phylax.wait_for_approval(app_id, timeout_seconds=1.0)
    assert decision == "allow_once"


def test_hestia_model_registry_loading():
    """Verify Hestia loads models.yaml and discovers registered chat model."""
    models_file = Path("hestia/models.yaml")
    assert models_file.is_file()
    reg = ModelRegistry()
    reg.load_from_yaml(models_file)
    model_entry = reg.get("llama-3.1-8b-instruct")
    assert model_entry is not None
    assert model_entry.model_type == ModelType.CHAT
    assert model_entry.sampling.temperature == 0.7


def test_prosopon_session_storage_lifecycle(tmp_path):
    """Verify Prosopon session store persists and retrieves turns atomically."""
    store = SessionStore(tmp_path)
    session = store.create_session("Smoke Test Session")
    assert session.session_id is not None
    assert session.name == "Smoke Test Session"

    msg1 = ChatMessage(message_id="msg-1", role="user", content="Hello Metis!")
    msg2 = ChatMessage(message_id="msg-2", role="assistant", content="Greetings. How can I assist you?")
    store.append_message(session.session_id, msg1)
    store.append_message(session.session_id, msg2)

    retrieved = store.get_session(session.session_id)
    assert retrieved is not None
    assert len(retrieved.messages) == 2
    assert retrieved.messages[0].content == "Hello Metis!"
    assert retrieved.messages[1].role == "assistant"
