"""FastAPI application factory for Kybernetes orchestrator and admin control plane."""

import json
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from sse_starlette.sse import EventSourceResponse

from kybernetes.adapters.dianoia import DianoiaAdapter
from kybernetes.adapters.dokime import DokimeAdapter
from kybernetes.adapters.hestia import HestiaAdapter
from kybernetes.adapters.krisis import KrisisAdapter
from kybernetes.adapters.lthcm import LTHCMAdapter
from kybernetes.adapters.phylax import PhylaxAdapter
from kybernetes.admin.audit import AdminAuditLogger
from kybernetes.admin.probes import ProbeRegistry
from kybernetes.admin.routes import create_admin_router
from kybernetes.admin.status import ComponentState
from kybernetes.budget.engine import BudgetEngine
from kybernetes.budget.storage import BudgetDatabase
from kybernetes.config import KybernetesConfig, load_config
from kybernetes.contracts.turn import ApprovalResolution, TurnRequest
from kybernetes.engine.pipeline import ExecutionPipeline
from kybernetes.engine.session import SessionManager
from kybernetes.routing.router import PlanRouter
from kybernetes.tools.registry import ToolRegistry


class HostHeaderValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate Host header against allowed loopback and configured host patterns to prevent DNS rebinding.
    """
    def __init__(self, app: Any, allowed_hosts: Optional[List[str]] = None):
        super().__init__(app)
        self.allowed_hosts = allowed_hosts or ["127.0.0.1", "localhost", "testserver", "test"]

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        host_header = request.headers.get("host", "").split(":")[0].strip().lower()
        # Allow test environments and local host variations
        if host_header and host_header not in self.allowed_hosts and not host_header.startswith("127.0.0."):
            return Response(
                content="Invalid Host header",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        return await call_next(request)


def create_app(
    db_path: Optional[str] = None,
    service_token: Optional[str] = None,
    allowed_origins: Optional[List[str]] = None,
    memory_root: Optional[str] = None,
    lthcm_adapter: Optional[LTHCMAdapter] = None,
    dianoia_adapter: Optional[DianoiaAdapter] = None,
    phylax_adapter: Optional[PhylaxAdapter] = None,
    krisis_adapter: Optional[KrisisAdapter] = None,
    hestia_adapter: Optional[HestiaAdapter] = None,
    config: Optional[KybernetesConfig] = None,
) -> FastAPI:
    """Create and configure FastAPI application for Kybernetes."""

    cfg = config or load_config()
    
    # CLI / argument overrides take precedence over loaded config
    effective_db_path = db_path if db_path is not None else cfg.db_path
    effective_service_token = service_token or cfg.resolve_service_token()
    effective_allowed_origins = allowed_origins or cfg.allowed_origins
    effective_memory_root = memory_root if memory_root is not None else cfg.memory_root

    db = BudgetDatabase(db_path=effective_db_path)
    budget_engine = BudgetEngine(db=db)
    tool_registry = ToolRegistry()
    hestia_adapter = hestia_adapter or HestiaAdapter(base_url=cfg.hestia_base_url)
    lthcm_adapter = lthcm_adapter or LTHCMAdapter(memory_root=effective_memory_root)
    dianoia_adapter = dianoia_adapter or DianoiaAdapter()
    phylax_adapter = phylax_adapter or PhylaxAdapter()
    krisis_adapter = krisis_adapter or KrisisAdapter()
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

    probe_registry = ProbeRegistry(
        hestia_adapter=hestia_adapter,
        lthcm_adapter=lthcm_adapter,
        dianoia_adapter=dianoia_adapter,
        phylax_adapter=phylax_adapter,
        krisis_adapter=krisis_adapter,
    )

    audit_logger = AdminAuditLogger()

    app = FastAPI(
        title="Kybernetes Orchestrator API",
        version="0.1.0",
        description="Versioned assistant orchestrator and control plane for the Metis local AI stack.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=effective_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(HostHeaderValidationMiddleware)

    app.state.pipeline = pipeline
    app.state.phylax = phylax_adapter
    app.state.budget = budget_engine
    app.state.sessions = session_manager
    app.state.probe_registry = probe_registry
    app.state.audit_logger = audit_logger
    app.state.config = cfg

    # Mount admin control plane routes
    if cfg.admin_enabled:
        admin_router = create_admin_router(
            config=cfg,
            probe_registry=probe_registry,
            audit_logger=audit_logger,
            pipeline=pipeline,
            db=db,
            session_manager=session_manager,
        )
        app.include_router(admin_router)

    async def require_service_auth(authorization: Optional[str] = Header(None)) -> bool:
        if not effective_service_token:
            return True
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer ") :].strip()
        if not token or not secrets.compare_digest(token, effective_service_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Valid Kybernetes service token required.",
            )
        return True

    @app.get("/")
    async def root():
        """Service index and endpoint discovery."""
        return {
            "service": "kybernetes",
            "name": "Kybernetes Local Orchestrator API",
            "version": "0.1.0",
            "endpoints": {
                "health": "/v1/health",
                "chat_turn": "/v1/chat/turn",
                "admin_overview": "/v1/admin/overview",
                "docs": "/docs",
            },
        }

    @app.get("/v1/health")
    async def health_check() -> Dict[str, Any]:
        """
        Public unauthenticated health and component readiness check.
        Strictly redacts sensitive local filesystem paths and internal credentials.
        Reports canonical states (live | simulated | mock | offline | unknown).
        """
        # Execute cheap reachability probe for Hestia
        hestia_status = await probe_registry.check_hestia_reachability()
        
        # Read cached statuses for internal adapters (truthful, unprobed = simulated/unknown)
        lthcm_status = probe_registry.get_cached_status("lthcm")
        dianoia_status = probe_registry.get_cached_status("dianoia")
        phylax_status = probe_registry.get_cached_status("phylax")
        krisis_status = probe_registry.get_cached_status("krisis")

        return {
            "status": "ok",
            "service": "kybernetes",
            "version": "0.1.0",
            "dependencies": {
                "hestia": {
                    "state": hestia_status.state.value,
                    "status": hestia_status.status,
                    "mode": hestia_status.mode,
                    "available": (hestia_status.state == ComponentState.LIVE),
                    "evidence": hestia_status.evidence,
                    "checked_at": hestia_status.checked_at,
                    "freshness": hestia_status.freshness.value,
                },
                "lthcm": {
                    "state": lthcm_status.state.value,
                    "status": lthcm_status.status,
                    "mode": lthcm_status.mode,
                    "epoch_supported": True,
                    "generation_id": lthcm_adapter.generation_id,
                    "evidence": lthcm_status.evidence,
                    "checked_at": lthcm_status.checked_at,
                    "freshness": lthcm_status.freshness.value,
                },
                "dianoia": {
                    "state": dianoia_status.state.value,
                    "status": dianoia_status.status,
                    "mode": dianoia_status.mode,
                    "zero_edge_fallback": True,
                    "evidence": dianoia_status.evidence,
                    "checked_at": dianoia_status.checked_at,
                    "freshness": dianoia_status.freshness.value,
                },
                "phylax": {
                    "state": phylax_status.state.value,
                    "status": phylax_status.status,
                    "mode": phylax_status.mode,
                    "capability_gateway": True,
                    "evidence": phylax_status.evidence,
                    "checked_at": phylax_status.checked_at,
                    "freshness": phylax_status.freshness.value,
                },
                "krisis": {
                    "state": krisis_status.state.value,
                    "status": krisis_status.status,
                    "mode": krisis_status.mode,
                    "verifier_gates": True,
                    "evidence": krisis_status.evidence,
                    "checked_at": krisis_status.checked_at,
                    "freshness": krisis_status.freshness.value,
                },
            },
        }

    @app.post("/v1/chat/turn")
    @app.post("/v1/chat/stream")
    async def chat_turn_stream(req: TurnRequest, _: bool = Depends(require_service_auth)) -> EventSourceResponse:
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
    async def resolve_approval(approval_id: str, res: ApprovalResolution, _: bool = Depends(require_service_auth)) -> Dict[str, Any]:
        """Resolve pending Phylax capability approval."""
        success = phylax_adapter.resolve_approval(approval_id, res.decision)
        if not success:
            raise HTTPException(status_code=404, detail="Approval request not found or expired")
        return {"status": "ok", "approval_id": approval_id, "decision": res.decision}

    @app.post("/v1/chat/turns/{turn_id}/cancel")
    async def cancel_turn(turn_id: str, _: bool = Depends(require_service_auth)) -> Dict[str, Any]:
        """Cancel an in-flight conversational turn."""
        session_manager.request_cancellation(turn_id)
        return {"status": "ok", "turn_id": turn_id, "cancelled": True}

    @app.get("/v1/traces/{turn_id}")
    async def get_trace(turn_id: str, _: bool = Depends(require_service_auth)) -> Dict[str, Any]:
        """Retrieve audit trace record for Dokime evaluation."""
        trace = pipeline.get_trace(turn_id)
        if not trace:
            raise HTTPException(status_code=404, detail=f"No trace found for turn {turn_id}")
        return trace.model_dump()

    @app.get("/v1/budget/report/{task_id}")
    async def get_budget_report(task_id: str, _: bool = Depends(require_service_auth)) -> Dict[str, Any]:
        """Retrieve reconciliation report for task."""
        rep = db.get_reconciliation(task_id)
        if not rep:
            raise HTTPException(status_code=404, detail=f"No reconciliation report found for task {task_id}")
        return rep.model_dump()

    return app
