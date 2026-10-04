"""Demonstration runner for the 6 canonical Kybernetes scenarios."""

import asyncio
from tabulate import tabulate

from kybernetes.adapters.dianoia import DianoiaAdapter
from kybernetes.adapters.hestia import HestiaAdapter
from kybernetes.adapters.krisis import KrisisAdapter
from kybernetes.adapters.lthcm import LTHCMAdapter
from kybernetes.adapters.phylax import PhylaxAdapter
from kybernetes.budget.engine import BudgetEngine
from kybernetes.budget.storage import BudgetDatabase
from kybernetes.contracts.turn import TurnRequest
from kybernetes.engine.pipeline import ExecutionPipeline
from kybernetes.engine.session import SessionManager
from kybernetes.routing.router import PlanRouter
from kybernetes.tools.registry import ToolRegistry


async def run_scenario(pipeline: ExecutionPipeline, phylax: PhylaxAdapter, title: str, query: str, auto_deny: bool = False):
    print("=" * 80)
    print(f"SCENARIO: {title}")
    print(f"QUERY: '{query}'")
    print("-" * 80)

    req = TurnRequest(session_id="demo-session", query=query)
    full_text = ""
    route_summary = {}
    citations = []

    async for event in pipeline.execute_turn_stream(req):
        if event.event_type == "plan_selected":
            print(f"[Plan Selected] Route: {event.payload.get('route')} | Stages: {event.payload.get('stages')}")
        elif event.event_type == "evidence":
            citations = event.payload.get("citations", [])
            print(f"[Evidence Grounding] Found {len(citations)} citations:")
            for c in citations:
                print(f"   {c['citation_id']} {c['source_title']} (score: {c['score']})")
        elif event.event_type == "ask_approval":
            app = event.payload.get("approval", {})
            print(f"[Phylax Security Gateway] APPROVAL REQUIRED:")
            print(f"   ID: {app.get('approval_id')} | Action: {app.get('action')} | Target: {app.get('target')}")
            decision = "deny" if auto_deny else "allow_once"
            print(f"   -> Resolving approval with: '{decision}'")
            phylax.resolve_approval(app.get("approval_id"), decision)
        elif event.event_type == "delta":
            full_text += event.payload.get("text", "")
        elif event.event_type == "completed":
            route_summary = event.payload.get("route_summary", {})
            print(f"\n[Generation Result]:\n{full_text.strip()}\n")
            print(f"[Completed Outcome]: Status={event.status}")
            print(f"   Route={route_summary.get('route')}")
            print(f"   Components={route_summary.get('components_used')}")
            print(f"   Latency={route_summary.get('latency_ms')} ms")
            if "zero_edge_fallback_used" in route_summary:
                print(f"   Zero-Edge Fallback Used: {route_summary.get('zero_edge_fallback_used')}")

    print("=" * 80 + "\n")


async def run_all_demos():
    print("\n================================================================================")
    print("             KYBERNETES LOCAL ASSISTANT ORCHESTRATOR DEMO SUITE                 ")
    print("================================================================================\n")

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

    # 1. Direct conversational turn
    await run_scenario(
        pipeline,
        phylax,
        title="1. Direct Conversational Turn (No Memory or Tool Needed)",
        query="Hello! Tell me a friendly greeting.",
    )

    # 2. Factual Retrieval Turn
    await run_scenario(
        pipeline,
        phylax,
        title="2. Factual Retrieval Turn with LTHCM Citations",
        query="What is the architecture of Metis and how does Kybernetes orchestrate it?",
    )

    # 3. Multi-Hop Deliberation Turn
    await run_scenario(
        pipeline,
        phylax,
        title="3. Multi-Hop Deliberation Graph Walk via Dianoia",
        query="Compare and contrast how Phylax and Krisis connect with Kybernetes in the Metis ecosystem.",
    )

    # 4. Safe Tool Execution (Calculator)
    await run_scenario(
        pipeline,
        phylax,
        title="4. Deterministic Tool Execution (Safe AST Calculator)",
        query="Please compute sqrt(256) + 42 * 2.",
    )

    # 5. Denied File Access / Interactive Approval
    await run_scenario(
        pipeline,
        phylax,
        title="5. Guarded File Action with Interactive Phylax Approval (Denied)",
        query="Please export report and write file to /tmp/metis_task.txt",
        auto_deny=True,
    )

    # 6. Zero-Edge Dianoia Fallback
    print("=" * 80)
    print("SCENARIO: 6. Dianoia Zero-Edge Flat Retrieval Fallback")
    print("QUERY: 'Explain how disconnected concept nodes behave in Dianoia'")
    print("-" * 80)
    # Force single isolated citation to trigger 0-edge fallback cleanly
    req = TurnRequest(session_id="demo-session", query="how do disconnected nodes compare in dianoia")
    citations = lthcm.retrieve(req.query, top_k=1)
    fallback_delib = dianoia.deliberate(req.query, citations, force_zero_edges=True)
    print(f"Graph Traversal Result: edges_traversed={fallback_delib.edges_traversed}, steps={fallback_delib.walk_steps}")
    print(f"Zero-Edge Fallback Activated: {fallback_delib.zero_edge_fallback_used}")
    print(f"Fallback Reason: {fallback_delib.fallback_reason}")
    print(f"Evidence Preserved: {[c.citation_id for c in fallback_delib.refined_citations]}")
    print("=" * 80 + "\n")

    print("[Demo Suite Completed Successfully]")


if __name__ == "__main__":
    asyncio.run(run_all_demos())
