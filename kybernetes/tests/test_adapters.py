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


def test_dianoia_live_walk_delegation():
    class MockLiveWalkController:
        def walk(self, question: str, controller_policy=None):
            return {
                "answer": "Synthesized answer from live thought graph.",
                "path": [{"step": 1}, {"step": 2}, {"step": 3}],
                "visited_passages": [
                    {"doc_id": "live-p1", "source_title": "Live Doc 1", "content": "Live snippet 1", "score": 0.95},
                    {"doc_id": "live-p2", "source_title": "Live Doc 2", "content": "Live snippet 2", "score": 0.88},
                ],
                "steps_used": 3,
                "conflicted": False,
            }

    live_ctrl = MockLiveWalkController()
    dianoia = DianoiaAdapter(walk_controller=live_ctrl)
    assert dianoia.is_simulated is False

    seed_citations = [
        Citation(citation_id="[1]", passage_id="p1", source_id="d1", source_title="D1", text="T1", score=0.9)
    ]
    res = dianoia.deliberate(query="live multi-hop query", seed_citations=seed_citations)
    assert res.success is True
    assert res.is_simulated is False
    assert res.walk_steps == 3
    assert res.edges_traversed == 2
    assert res.zero_edge_fallback_used is False
    assert len(res.refined_citations) == 2
    assert res.refined_citations[0].passage_id == "live-p1"
    assert res.refined_citations[0].is_simulated is False
    assert res.deliberation_summary == "Synthesized answer from live thought graph."


def test_dianoia_live_zero_edge_fallback():
    class MockLiveWalkController:
        def walk(self, question: str, controller_policy=None):
            return {"answer": "Should not be called on force zero edge"}

    live_ctrl = MockLiveWalkController()
    dianoia = DianoiaAdapter(walk_controller=live_ctrl)
    assert dianoia.is_simulated is False

    seed_citations = [
        Citation(citation_id="[1]", passage_id="p1", source_id="d1", source_title="D1", text="T1", score=0.9)
    ]
    res = dianoia.deliberate(query="fallback query", seed_citations=seed_citations, force_zero_edges=True)
    assert res.success is True
    assert res.is_simulated is False
    assert res.zero_edge_fallback_used is True
    assert res.edges_traversed == 0
    assert "live engine fallback" in res.deliberation_summary


def test_dianoia_live_walk_error_graceful_degradation():
    class BrokenWalkController:
        def walk(self, question: str, controller_policy=None):
            raise RuntimeError("Engine memory corruption")

    dianoia = DianoiaAdapter(walk_controller=BrokenWalkController())
    assert dianoia.is_simulated is False

    seed_citations = [
        Citation(citation_id="[1]", passage_id="p1", source_id="d1", source_title="D1", text="T1", score=0.9)
    ]
    res = dianoia.deliberate(query="broken graph query", seed_citations=seed_citations)
    assert res.success is True
    assert res.is_simulated is False
    assert res.zero_edge_fallback_used is True
    assert "Engine memory corruption" in res.fallback_reason
    assert res.refined_citations == seed_citations


def test_dianoia_inconsistency_span_filter_fallback():
    dianoia = DianoiaAdapter()
    seed_citations = [
        Citation(
            citation_id="[1]",
            passage_id="p1",
            source_id="d1",
            source_title="Meeting Notes",
            text="Alice attended the project meeting at hour 10 for duration 2 hours.",
            score=0.9,
        )
    ]
    # Graph block has hallucinated entity and wrong timestamp
    inconsistent_graph = {
        "nodes": [
            {"id": "Bob", "timestamp": 14},  # Bob not in text, hour 14 not in text
            {"id": "Alice", "timestamp": 10}, # Grounded
        ],
        "edges": [
            {"src": "Bob", "dst": "Alice"}
        ],
    }

    res = dianoia.deliberate(
        query="what happened at meeting",
        seed_citations=seed_citations,
        consistency_floor=0.95,
        fallback_to_pure_text_on_inconsistency=True,
        extracted_graph_block=inconsistent_graph,
    )
    assert res.zero_edge_fallback_used is True
    assert res.consistency_fallback_used is True
    assert res.consistency_ratio < 0.95
    assert "fell below floor 0.95" in res.fallback_reason
    assert res.refined_citations == seed_citations


def test_dianoia_consistent_graph_block_accepted():
    dianoia = DianoiaAdapter()
    seed_citations = [
        Citation(
            citation_id="[1]",
            passage_id="p1",
            source_id="d1",
            source_title="Meeting Notes",
            text="Alice attended the project meeting at hour 10.",
            score=0.9,
        ),
        Citation(
            citation_id="[2]",
            passage_id="p2",
            source_id="d2",
            source_title="Project Roster",
            text="Alice is the lead engineer on Metis.",
            score=0.85,
        ),
    ]
    consistent_graph = {
        "nodes": [
            {"id": "Alice", "timestamp": 10},
        ],
        "edges": [],
    }

    res = dianoia.deliberate(
        query="what was Alice doing",
        seed_citations=seed_citations,
        consistency_floor=0.95,
        fallback_to_pure_text_on_inconsistency=True,
        extracted_graph_block=consistent_graph,
    )
    assert res.zero_edge_fallback_used is False
    assert res.consistency_fallback_used is False
    assert res.consistency_ratio == 1.0


def test_dianoia_live_walk_conflict_triggers_fallback():
    class ConflictedWalkController:
        def walk(self, question: str, controller_policy=None):
            return {
                "answer": "Conflicted answer.",
                "path": [{"step": 1}, {"step": 2}],
                "visited_passages": [
                    {"doc_id": "p1", "source_title": "D1", "content": "T1", "score": 0.9},
                ],
                "steps_used": 2,
                "conflicted": True,  # Sheaf/NLI detected contradiction in graph walk
            }

    dianoia = DianoiaAdapter(walk_controller=ConflictedWalkController())
    seed_citations = [
        Citation(citation_id="[1]", passage_id="p1", source_id="d1", source_title="D1", text="T1", score=0.9)
    ]

    res = dianoia.deliberate(
        query="conflicted query",
        seed_citations=seed_citations,
        fallback_to_pure_text_on_inconsistency=True,
    )
    assert res.zero_edge_fallback_used is True
    assert res.consistency_fallback_used is True
    assert "internal contradiction/conflict" in res.fallback_reason
    assert res.refined_citations == seed_citations


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

    # 1. Grounded with citation marker -> capped at CITED_UNVERIFIED for heuristic
    rep1 = krisis.verify_response("query", "Answer with evidence [1].", [c1])
    assert rep1.status == TurnOutcomeStatus.CITED_UNVERIFIED
    assert rep1.score > 0.8
    assert "citations present; no entailment check run" in rep1.explanation

    # 1b. Uncited sentence alongside cited sentence fails full support
    moon_cit = Citation(
        citation_id="[1]",
        passage_id="p_moon",
        source_id="s_moon",
        source_title="Astronomy",
        text="The moon orbits the earth every month.",
        score=0.9,
    )
    rep1b = krisis.verify_response(
        "query",
        "The moon does orbit the earth [1]. The sky is purple.",
        [moon_cit],
    )
    assert rep1b.status != TurnOutcomeStatus.VERIFIED
    assert rep1b.status == TurnOutcomeStatus.INSUFFICIENT_EVIDENCE

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


def test_krisis_polarity_clash_detection():
    """F-08: Verify polarity clash detection differentiates contradiction from agreeing negations."""
    krisis = KrisisAdapter()
    
    pos_cit = Citation(
        citation_id="[1]",
        passage_id="p_pos",
        source_id="s1",
        source_title="Metis Architecture",
        text="Metis supports speculative execution on CPU and GPU pipelines.",
        score=0.9,
    )
    neg_cit = Citation(
        citation_id="[2]",
        passage_id="p_neg",
        source_id="s2",
        source_title="Metis Architecture",
        text="Metis does not support unauthenticated WebRTC network access.",
        score=0.9,
    )

    # 1. Agreeing positive: no clash
    rep_pos_agree = krisis.verify_response(
        "query",
        "Metis supports speculative execution on CPU pipelines [1].",
        [pos_cit],
    )
    assert rep_pos_agree.status == TurnOutcomeStatus.CITED_UNVERIFIED
    assert rep_pos_agree.conflict_detected is False

    # 2. Agreeing negative: "not" in both claim and citation -> no clash
    rep_neg_agree = krisis.verify_response(
        "query",
        "Metis does not support unauthenticated WebRTC network access [2].",
        [neg_cit],
    )
    assert rep_neg_agree.status == TurnOutcomeStatus.CITED_UNVERIFIED
    assert rep_neg_agree.conflict_detected is False

    # 3. Polarity clash: claim negates affirmative citation
    rep_clash_neg_claim = krisis.verify_response(
        "query",
        "Metis does not support speculative execution on CPU pipelines [1].",
        [pos_cit],
    )
    assert rep_clash_neg_claim.status == TurnOutcomeStatus.CONFLICTED
    assert rep_clash_neg_claim.conflict_detected is True
    assert "Contradiction detected in cited claim against evidence" in rep_clash_neg_claim.explanation

    # 4. Polarity clash: claim asserts negative citation
    rep_clash_pos_claim = krisis.verify_response(
        "query",
        "Metis supports unauthenticated WebRTC network access [2].",
        [neg_cit],
    )
    assert rep_clash_pos_claim.status == TurnOutcomeStatus.CONFLICTED
    assert rep_clash_pos_claim.conflict_detected is True



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


def test_phylax_adapter_live_policy_evaluation():
    from pathlib import Path
    policy_path = Path("phylax/policies/default_policy.yaml").resolve()
    if not policy_path.exists():
        policy_path = (Path(__file__).parents[3] / "phylax" / "policies" / "default_policy.yaml").resolve()

    if policy_path.exists():
        phylax = PhylaxAdapter(policy_path=str(policy_path))
        assert phylax.is_simulated is False

        # read_file_range is explicitly allowed in default_policy.yaml
        permitted, _ = phylax.is_action_permitted("execute", "read_file_range", {})
        assert permitted is True

        # file_search and run_code_mode are explicitly allowed
        p_search, _ = phylax.is_action_permitted("file_search", ".", {"pattern": "metis"})
        assert p_search is True
        p_code, _ = phylax.is_action_permitted("run_code_mode", "code", {"script": "x = 1"})
        assert p_code is True

        # edit_file_block requires ask approval
        permitted, app_req = phylax.is_action_permitted("execute", "edit_file_block", {})
        assert permitted is False
        assert app_req is not None

        # write_file and sandboxed_bash require ask approval (Finding A4)
        p_write, app_write = phylax.is_action_permitted("write_file", "/tmp/test.txt", {"content": "data"})
        assert p_write is False
        assert app_write is not None
        assert "approval_id" in app_write

        p_bash, app_bash = phylax.is_action_permitted("sandboxed_bash", "sandboxed_bash", {"command": "ls"})
        assert p_bash is False
        assert app_bash is not None
        assert "approval_id" in app_bash

        # Live gateway approval resolution & consumption flow (Finding A4)
        app_id = app_write["approval_id"]
        assert phylax.resolve_approval(app_id, "allow_once") is True
        consumed, msg = phylax.consume_approval(app_id, "write_file", {"content": "data"})
        assert consumed is True

        # Replay prevention against live gateway
        consumed_replay, _ = phylax.consume_approval(app_id, "write_file", {"content": "data"})
        assert consumed_replay is False

    # Fail-closed check: invalid policy path raises FileNotFoundError (Finding A5)
    import pytest
    with pytest.raises(FileNotFoundError):
        PhylaxAdapter(policy_path="/nonexistent/policy.yaml")
