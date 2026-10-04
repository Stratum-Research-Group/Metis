"""Unit tests for Kybernetes adapters (Hestia, LTHCM, Dianoia, Phylax, Krisis, Dokime)."""

import asyncio
import pytest
from kybernetes.adapters.dianoia import DianoiaAdapter
from kybernetes.adapters.dokime import DokimeAdapter, EvalOutcomeRecord
from kybernetes.adapters.hestia import HestiaAdapter
from kybernetes.adapters.krisis import KrisisAdapter
from kybernetes.adapters.lthcm import LTHCMAdapter
from kybernetes.adapters.phylax import PhylaxAdapter
from kybernetes.contracts.enums import CheckpointAction, TurnOutcomeStatus
from kybernetes.contracts.turn import Citation


def test_lthcm_temporal_as_of_filtering():
    lthcm = LTHCMAdapter()
    # Query with future timestamp -> all matching passages included
    passages_all = lthcm.retrieve(query="kybernetes metis orchestrator", as_of=2000000000.0)
    assert len(passages_all) > 0

    # Query with past timestamp before passages were created -> empty results
    passages_past = lthcm.retrieve(query="kybernetes metis orchestrator", as_of=1000000000.0)
    assert len(passages_past) == 0


def test_dianoia_zero_edge_flat_fallback():
    dianoia = DianoiaAdapter()
    # When 0 or 1 citation exists, 0 edges are possible -> fallback must trigger
    single_citation = [
        Citation(
            citation_id="[1]",
            passage_id="p1",
            source_id="d1",
            source_title="Doc 1",
            text="Snippet",
            score=0.9,
        )
    ]
    res = dianoia.deliberate(query="concept walk", seed_citations=single_citation)
    assert res.zero_edge_fallback_used is True
    assert res.edges_traversed == 0
    assert "Zero thought-graph edges discovered" in res.fallback_reason


def test_dianoia_multi_edge_deliberation():
    dianoia = DianoiaAdapter()
    citations = [
        Citation(citation_id="[1]", passage_id="p1", source_id="d1", source_title="D1", text="T1", score=0.9),
        Citation(citation_id="[2]", passage_id="p2", source_id="d2", source_title="D2", text="T2", score=0.8),
    ]
    res = dianoia.deliberate(query="compare concept nodes", seed_citations=citations)
    assert res.zero_edge_fallback_used is False
    assert res.edges_traversed > 0
    assert res.walk_steps > 0


@pytest.mark.asyncio
async def test_phylax_interactive_approval():
    phylax = PhylaxAdapter()
    permitted, app_req = phylax.is_action_permitted(
        capability="write_file",
        target="/tmp/test.txt",
        arguments={"bytes": 100},
    )
    assert permitted is False
    assert app_req is not None
    app_id = app_req["approval_id"]

    # In background, resolve approval
    async def _resolve_soon():
        await asyncio.sleep(0.02)
        phylax.resolve_approval(app_id, "allow_once")

    asyncio.create_task(_resolve_soon())
    decision = await phylax.wait_for_approval(app_id, timeout_seconds=1.0)
    assert decision == "allow_once"


def test_phylax_budget_guardrail():
    phylax = PhylaxAdapter()
    assert phylax.enforce_budget_guardrail("t1", CheckpointAction.ALLOW, "ok") is True
    assert phylax.enforce_budget_guardrail("t1", CheckpointAction.WARN, "high burn") is True
    assert phylax.enforce_budget_guardrail("t1", CheckpointAction.BLOCK, "limit hit") is False


def test_krisis_verification_statuses():
    krisis = KrisisAdapter()
    c1 = Citation(citation_id="[1]", passage_id="p1", source_id="s1", source_title="Title", text="Text", score=0.9)

    # 1. Grounded with citation marker
    rep1 = krisis.verify_response("query", "Answer with evidence [1].", [c1])
    assert rep1.status == TurnOutcomeStatus.VERIFIED
    assert rep1.score > 0.8

    # 2. Tool output
    rep2 = krisis.verify_response("query", "Result is 42.", [], is_tool_output=True)
    assert rep2.status == TurnOutcomeStatus.VERIFIED
    assert rep2.score == 1.0

    # 3. Direct generation without citations -> draft
    rep3 = krisis.verify_response("query", "Just chatting.", [])
    assert rep3.status == TurnOutcomeStatus.DRAFT

    # 4. Citations provided but omitted in response text -> insufficient evidence
    rep4 = krisis.verify_response("query", "Answer without citations.", [c1])
    assert rep4.status == TurnOutcomeStatus.INSUFFICIENT_EVIDENCE


def test_dokime_cohort_priors():
    dokime = DokimeAdapter()
    assert dokime.load_cohort_priors("unknown") is None

    dokime.record_run(
        EvalOutcomeRecord(
            eval_id="e1",
            task_type="chat",
            model_name="qwen",
            phase="synth",
            measured_input_tokens=200.0,
            measured_output_tokens=100.0,
            measured_time_seconds=2.0,
            factual_correctness_score=1.0,
            timestamp_utc="2026-09-30T12:00:00Z",
        )
    )
    priors = dokime.load_cohort_priors("chat")
    assert priors is not None
    assert priors["sample_count"] == 1
    assert priors["mean_input_tokens"] == 200.0
