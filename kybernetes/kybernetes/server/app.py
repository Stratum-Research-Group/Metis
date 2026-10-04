"""FastAPI HTTP and SSE streaming server for Kybernetes orchestrator."""

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from kybernetes.adapters.dianoia import DianoiaAdapter
from kybernetes.adapters.dokime import DokimeAdapter
from kybernetes.adapters.hestia import HestiaAdapter
from kybernetes.adapters.krisis import KrisisAdapter
from kybernetes.adapters.lthcm import LTHCMAdapter
from kybernetes.adapters.phylax import PhylaxAdapter
from kybernetes.budget.engine import BudgetEngine
from kybernetes.budget.storage import BudgetDatabase
from kybernetes.contracts.turn import ApprovalResolution, TurnRequest
from kybernetes.engine.pipeline import ExecutionPipeline
from kybernetes.engine.session import SessionManager
from kybernetes.routing.router import PlanRouter
from kybernetes.tools.registry import ToolRegistry


def create_app(db_path: str = ":memory:") -> FastAPI:
    """Create and configure FastAPI application for Kybernetes."""

    db = BudgetDatabase(db_path=db_path)
    budget_engine = BudgetEngine(db=db)
    tool_registry = ToolRegistry()
    hestia_adapter = HestiaAdapter()
    lthcm_adapter = LTHCMAdapter()
    dianoia_adapter = DianoiaAdapter()
    phylax_adapter = PhylaxAdapter()
    krisis_adapter = KrisisAdapter()
    session_manager = SessionManager()
    router = PlanRouter()

    pipeline = ExecutionPipeline(
        router=router,
        budget_engine=budget_engine,
        tool_registry=tool_registry,
        hestia_adapter=hestia_adapter,
        lthcm_adapter=lthcm_adapter,
        dianoia_adapter=dianoia_adapter,
        phylax_adapter=phylax_adapter,
        krisis_adapter=krisis_adapter,
        session_manager=session_manager,
    )

    app = FastAPI(
        title="Kybernetes Orchestrator API",
        version="0.1.0",
        description="Versioned assistant orchestrator for the Metis local AI stack.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.pipeline = pipeline
    app.state.phylax = phylax_adapter
    app.state.budget = budget_engine
    app.state.sessions = session_manager

    @app.get("/v1/health")
    async def health_check() -> Dict[str, Any]:
        """Upstream health and component readiness."""
        h_info = await hestia_adapter.check_health()
        return {
            "status": "ok",
            "service": "kybernetes",
            "version": "0.1.0",
            "dependencies": {
                "hestia": h_info,
                "lthcm": {"status": "ready", "epoch_supported": True},
                "dianoia": {"status": "ready", "zero_edge_fallback": True},
                "phylax": {"status": "ready", "capability_gateway": True},
                "krisis": {"status": "ready", "verifier_gates": True},
            },
        }

    @app.post("/v1/chat/turn")
    @app.post("/v1/chat/stream")
    async def chat_turn_stream(req: TurnRequest) -> EventSourceResponse:
        """
        Prosopon SSE stream endpoint emitting ordered TurnEvents.
        """
        async def event_generator() -> AsyncIterator[Dict[str, str]]:
            async for turn_event in pipeline.execute_turn_stream(req):
                yield {
                    "data": turn_event.model_dump_json(),
                    "event": "message",
                }

        return EventSourceResponse(event_generator())

    @app.post("/v1/approvals/{approval_id}/resolve")
    async def resolve_approval(approval_id: str, res: ApprovalResolution) -> Dict[str, Any]:
        """Resolve pending Phylax capability approval."""
        success = phylax_adapter.resolve_approval(approval_id, res.decision)
        if not success:
            raise HTTPException(status_code=404, detail="Approval request not found or expired")
        return {"status": "ok", "approval_id": approval_id, "decision": res.decision}

    @app.post("/v1/chat/turns/{turn_id}/cancel")
    async def cancel_turn(turn_id: str) -> Dict[str, Any]:
        """Cancel an in-flight conversational turn."""
        session_manager.request_cancellation(turn_id)
        return {"status": "ok", "turn_id": turn_id, "cancelled": True}

    @app.get("/v1/traces/{turn_id}")
    async def get_trace(turn_id: str) -> Dict[str, Any]:
        """Retrieve audit trace record for Dokime evaluation."""
        trace = pipeline.get_trace(turn_id)
        if not trace:
            raise HTTPException(status_code=404, detail=f"No trace found for turn {turn_id}")
        return trace.model_dump()

    @app.get("/v1/budget/report/{task_id}")
    async def get_budget_report(task_id: str) -> Dict[str, Any]:
        """Retrieve reconciliation report for task."""
        rep = db.get_reconciliation(task_id)
        if not rep:
            raise HTTPException(status_code=404, detail=f"No reconciliation report found for task {task_id}")
        return rep.model_dump()

    return app
