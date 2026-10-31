"""FastAPI application factory for Kybernetes orchestrator and admin control plane."""

import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
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
from kybernetes.lochos import LochosSquadManager
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
    lochos_squad: Optional[LochosSquadManager] = None,
    tool_registry: Optional[ToolRegistry] = None,
    router: Optional[PlanRouter] = None,
    session_manager: Optional[SessionManager] = None,
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
    phylax_adapter = phylax_adapter or PhylaxAdapter()
    tool_registry = tool_registry or ToolRegistry(phylax_adapter=phylax_adapter)
    hestia_adapter = hestia_adapter or HestiaAdapter(base_url=cfg.hestia_base_url)
    lthcm_adapter = lthcm_adapter or LTHCMAdapter(memory_root=effective_memory_root)
    dianoia_adapter = dianoia_adapter or DianoiaAdapter()
    krisis_adapter = krisis_adapter or KrisisAdapter()
    session_manager = session_manager or SessionManager()
    lochos_squad = lochos_squad or LochosSquadManager()
    router = router or PlanRouter()

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
        thinking_budget_steps=cfg.thinking_budget_steps,
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

    async def require_service_auth(request: Request, authorization: Optional[str] = Header(None)) -> bool:
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer ") :].strip()

        if not effective_service_token:
            if os.environ.get("METIS_INSECURE_DEV") == "1":
                return True
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Kybernetes service token required unless METIS_INSECURE_DEV=1 is set.",
            )
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
                "lochos_squad": "/v1/lochos/squad",
                "lochos_queue": "/v1/lochos/queue",
                "lochos_pipeline": "/v1/lochos/pipeline",
                "lochos_attention": "/v1/lochos/attention",
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

    @app.get("/v1/chat/sessions/{session_id}/turns/{turn_id}/events")
    @app.get("/v1/chat/sessions/{session_id}/turns/{turn_id}/reconnect")
    async def reconnect_turn_stream(
        session_id: str,
        turn_id: str,
        after_seq: int = -1,
        last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
        _: bool = Depends(require_service_auth),
    ) -> EventSourceResponse:
        """
        Reconnect to an active or completed turn stream (Herdr / OpenCode persistent daemon).
        Replays missed historical events and continues streaming if the turn is in-flight.
        """
        effective_after = after_seq
        if last_event_id and last_event_id.isdigit():
            effective_after = max(effective_after, int(last_event_id))

        async def reconnect_generator() -> AsyncIterator[Dict[str, str]]:
            async for turn_event in session_manager.subscribe_turn_events(turn_id, after_seq=effective_after):
                yield {
                    "id": str(turn_event.seq_num),
                    "data": turn_event.model_dump_json(),
                    "event": "message",
                }

        return EventSourceResponse(reconnect_generator())

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
        if not session_manager.is_turn_active(turn_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Turn '{turn_id}' not found or already finished. Cannot cancel.",
            )
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

    # --- Lochos Multi-Agent Squad API ---

    @app.get("/v1/lochos/squad")
    async def get_lochos_squad(_: bool = Depends(require_service_auth)) -> Dict[str, Any]:
        """Retrieve active Lochos multi-agent squad status, seats, and active tasks."""
        return lochos_squad.to_status_dict()

    @app.get("/v1/lochos/queue")
    async def get_lochos_queue(
        status: Optional[str] = None,
        seat_id: Optional[str] = None,
        _: bool = Depends(require_service_auth),
    ) -> List[Dict[str, Any]]:
        """Retrieve work items tracked in the Lochos SQLite queue."""
        tasks = lochos_squad.queue.list_tasks(status=status, seat_id=seat_id)
        return [t.to_dict() for t in tasks]

    @app.post("/v1/lochos/dispatch")
    async def dispatch_lochos_work(
        data: Dict[str, Any],
        _: bool = Depends(require_service_auth),
    ) -> Dict[str, Any]:
        """Dispatch a mission slice task to a specific seat in the squad."""
        mission = data.get("mission", "general")
        slice_id = data.get("slice_id", "001")
        title = data.get("title", "Untitled Task")
        description = data.get("description", "")
        target_seat = data.get("target_seat")
        payload = data.get("payload")

        item = lochos_squad.dispatch_work(
            mission=mission,
            slice_id=slice_id,
            title=title,
            description=description,
            target_seat=target_seat,
            payload=payload,
        )
        return item.to_dict()

    @app.post("/v1/lochos/transfer")
    async def transfer_lochos_baton(
        data: Dict[str, Any],
        _: bool = Depends(require_service_auth),
    ) -> Dict[str, Any]:
        """Transfer work baton from one seat to another."""
        task_id = data.get("task_id")
        from_seat = data.get("from_seat")
        to_seat = data.get("to_seat")
        note = data.get("note")
        if not task_id or not from_seat or not to_seat:
            raise HTTPException(status_code=400, detail="task_id, from_seat, and to_seat are required")

        item = lochos_squad.transfer_baton(
            task_id=task_id,
            from_seat=from_seat,
            to_seat=to_seat,
            note=note,
        )
        return item.to_dict()

    @app.get("/v1/lochos/seats/{seat_id}/brief")
    async def get_lochos_seat_brief(seat_id: str, _: bool = Depends(require_service_auth)) -> Dict[str, Any]:
        """Retrieve the continuity/restore brief for a squad seat."""
        seat = lochos_squad.get_seat(seat_id)
        if not seat:
            raise HTTPException(status_code=404, detail=f"Seat '{seat_id}' not found")
        return seat.generate_restore_brief()

    @app.get("/v1/lochos/transfer/brief")
    async def get_lochos_transfer_brief(
        task_id: str,
        from_seat: str,
        to_seat: str,
        note: Optional[str] = None,
        _: bool = Depends(require_service_auth),
    ) -> Dict[str, Any]:
        """Generate a peer handoff brief between squad seats."""
        try:
            return lochos_squad.generate_handoff_brief(
                from_seat=from_seat,
                to_seat=to_seat,
                task_id=task_id,
                note=note,
            )
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))

    @app.get("/v1/lochos/attention")
    async def get_lochos_attention(_: bool = Depends(require_service_auth)) -> List[Dict[str, Any]]:
        """Retrieve cross-seat attention reconciliation requests."""
        return lochos_squad.get_attention_summary()

    @app.post("/v1/lochos/attention/resolve")
    async def resolve_lochos_attention(
        data: Dict[str, Any],
        _: bool = Depends(require_service_auth),
    ) -> Dict[str, Any]:
        """Resolve a blocked seat's attention request."""
        task_id = data.get("task_id")
        seat_id = data.get("seat_id")
        resolution = data.get("resolution", "approved")
        if not task_id or not seat_id:
            raise HTTPException(status_code=400, detail="task_id and seat_id are required")

        item = lochos_squad.queue.resolve_attention(task_id=task_id, seat_id=seat_id, resolution=resolution)
        return item.to_dict()

    @app.post("/v1/lochos/pipeline")
    @app.post("/api/lochos/pipeline")
    async def run_lochos_pipeline(
        data: Dict[str, Any],
        _: bool = Depends(require_service_auth),
    ) -> Dict[str, Any]:
        """Run an automated multi-seat pipeline (dispatch -> code -> handoff -> verify -> sign-off/rollback)."""
        mission = data.get("mission")
        slice_id = data.get("slice_id")
        title = data.get("title")
        if not mission or not slice_id or not title:
            raise HTTPException(status_code=400, detail="mission, slice_id, and title are required")

        description = data.get("description", "")
        payload = data.get("payload")
        coder_seat = data.get("coder_seat", "coder")
        reviewer_seat = data.get("reviewer_seat", "reviewer")
        auto_approve = bool(data.get("auto_approve", False))
        auto_rollback_on_failure = bool(data.get("auto_rollback_on_failure", True))
        escalate_attention_on_failure = bool(data.get("escalate_attention_on_failure", False))
        test_command = data.get("test_command")

        from kybernetes.lochos.workflow import LochosWorkflowOrchestrator
        orchestrator = LochosWorkflowOrchestrator(squad_manager=lochos_squad)
        result = orchestrator.run_pipeline(
            mission=mission,
            slice_id=slice_id,
            title=title,
            description=description,
            payload=payload,
            coder_seat=coder_seat,
            reviewer_seat=reviewer_seat,
            auto_approve=auto_approve,
            auto_rollback_on_failure=auto_rollback_on_failure,
            escalate_attention_on_failure=escalate_attention_on_failure,
            test_command=test_command,
        )
        return result.to_dict()

    @app.websocket("/v1/lochos/ws")
    async def lochos_websocket(websocket: WebSocket) -> None:
        """Live WebSocket streaming bridge for Lochos squad status and task lifecycle events."""
        await websocket.accept()
        try:
            await websocket.send_json({
                "type": "squad_state",
                "data": lochos_squad.to_status_dict(),
            })
        except Exception:
            return

        event_queue: asyncio.Queue = asyncio.Queue()

        def _on_event(evt: Dict[str, Any]) -> None:
            event_queue.put_nowait(evt)

        lochos_squad.add_event_listener(_on_event)

        async def _reader() -> None:
            try:
                while True:
                    msg = await websocket.receive_text()
                    if msg == "ping":
                        await websocket.send_text("pong")
                    elif msg == "refresh":
                        await websocket.send_json({
                            "type": "squad_state",
                            "data": lochos_squad.to_status_dict(),
                        })
            except Exception:
                pass

        async def _writer() -> None:
            try:
                while True:
                    evt = await event_queue.get()
                    await websocket.send_json(evt)
            except Exception:
                pass

        reader_task = asyncio.create_task(_reader())
        writer_task = asyncio.create_task(_writer())

        try:
            await asyncio.wait(
                [reader_task, writer_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except WebSocketDisconnect:
            pass
        finally:
            reader_task.cancel()
            writer_task.cancel()
            lochos_squad.remove_event_listener(_on_event)

    # --- Pinax Tool Catalogue API (Section 7) ---
    pinax = tool_registry.pinax_catalogue

    @app.get("/v1/pinax/tools")
    async def list_pinax_tools(
        query: str = "",
        category: Optional[str] = None,
        risk_tier: Optional[str] = None,
        limit: int = 50,
        _: bool = Depends(require_service_auth),
    ) -> List[Dict[str, Any]]:
        """List or search tools from Pinax catalogue."""
        return pinax.lookup(query=query, category=category, risk_tier=risk_tier, limit=limit)

    @app.get("/v1/pinax/tools/{tool_id:path}/docs")
    async def get_pinax_tool_docs(
        tool_id: str,
        max_bytes: int = 4096,
        _: bool = Depends(require_service_auth),
    ) -> Dict[str, Any]:
        """Fetch documentation slice for a tool (Level 2 size-capped)."""
        return pinax.get_tool_docs(tool_id, max_bytes=max_bytes)

    @app.get("/v1/pinax/tools/{tool_id:path}")
    async def get_pinax_tool_detail(
        tool_id: str,
        _: bool = Depends(require_service_auth),
    ) -> Dict[str, Any]:
        """Get Level 1 detailed specification, schema, and capabilities for a tool."""
        res = pinax.get_tool_detail(tool_id)
        if not res:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
        return res

    @app.post("/v1/pinax/scan")
    async def trigger_pinax_scan(
        payload: Optional[Dict[str, Any]] = None,
        _: bool = Depends(require_service_auth),
    ) -> Dict[str, Any]:
        """Run passive PATH scan to discover installed tools."""
        path_dirs = payload.get("path_dirs") if payload else None
        return pinax.scan_environment(path_dirs=path_dirs)

    @app.post("/v1/pinax/evaluate")
    async def evaluate_pinax_invocation(
        data: Dict[str, Any],
        _: bool = Depends(require_service_auth),
    ) -> Dict[str, Any]:
        """Evaluate invocation risk for a tool call against catalogue overrides."""
        tool_name = data.get("tool_name", "")
        args = data.get("args", [])
        return pinax.evaluate_invocation(tool_name, args)

    return app
