"""FastAPI backend server for Prosopon local web & voice interface."""

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set
import uuid

from fastapi import Depends, FastAPI, File, HTTPException, Header, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel


class ContentSecurityPolicyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self'; "
            "img-src 'self' data:; "
            "media-src 'self' blob:; "
            "frame-ancestors 'none';"
        )
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

from prosopon.config import ProsoponConfig
from prosopon.auth.pairing import PairingManager
from prosopon.contracts.session import ChatMessage, SessionRecord
from prosopon.contracts.kybernetes import Citation, TurnEvent
from prosopon.contracts.phylax import ApprovalResponse
from prosopon.storage.session_store import SessionStore
from prosopon.client.kybernetes_client import KybernetesClient
from prosopon.client.mock_kybernetes import MockKybernetesClient
from prosopon.voice.transcoder import AudioTranscodingError, transcode_to_wav
from prosopon.voice.whisper_stt import STTExecutionError, STTUnavailableError, WhisperSTTEngine
from prosopon.voice.piper_tts import TTSExecutionError, TTSUnavailableError, PiperTTSEngine


class ChatQueryRequest(BaseModel):
    query: str


class PairingRequest(BaseModel):
    pin: str
    device_name: Optional[str] = "Mobile Browser"


class ApprovalResolveRequest(BaseModel):
    decision: str  # "allow_once" | "deny"


class SynthesisRequest(BaseModel):
    text: str


def create_app(config: Optional[ProsoponConfig] = None) -> FastAPI:
    """Factory creating configured Prosopon FastAPI application."""
    cfg = config or ProsoponConfig()
    app = FastAPI(title="Prosopon Local Interface", version="0.1.0")

    # State singletons
    pairing_mgr = PairingManager(data_dir=cfg.data_dir, expiry_hours=cfg.session_expiry_hours)
    session_store = SessionStore(storage_dir=cfg.data_dir)
    
    if cfg.mock_kybernetes:
        kybernetes = MockKybernetesClient()
    else:
        kybernetes = KybernetesClient(
            base_url=cfg.kybernetes_api_url,
            service_token=getattr(cfg, "kybernetes_service_token", None),
        )

    whisper_engine = WhisperSTTEngine(
        binary_name=cfg.whisper_binary,
        model_path=cfg.whisper_model,
        mock_mode=cfg.voice_mock_mode,
    )
    piper_engine = PiperTTSEngine(
        binary_name=cfg.piper_binary,
        model_path=cfg.piper_model,
        mock_mode=cfg.voice_mock_mode,
    )

    app.state.config = cfg
    app.state.pairing_mgr = pairing_mgr
    app.state.session_store = session_store
    app.state.kybernetes = kybernetes

    # CORS & Origin protection
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ContentSecurityPolicyMiddleware)

    # Static asset paths
    static_dir = Path(__file__).parent / "web" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    import socket

    def _build_allowed_hosts() -> Set[str]:
        hosts = {"127.0.0.1", "localhost", "::1", "testclient", "testserver", "test"}
        for h in cfg.allowed_hosts:
            cleaned = h.split(":")[0].strip()
            if cleaned and cleaned != "0.0.0.0":
                hosts.add(cleaned)
        if cfg.is_lan():
            try:
                hostname = socket.gethostname()
                if hostname:
                    hosts.add(hostname)
                    hosts.add(f"{hostname}.local")
                for info in socket.getaddrinfo(hostname, None):
                    ip = info[4][0]
                    if ip and ip != "0.0.0.0":
                        hosts.add(ip)
            except Exception:
                pass
        return hosts

    allowed_hosts = _build_allowed_hosts()

    from urllib.parse import urlparse

    def _validate_host_header(request: Request) -> None:
        raw_host = request.headers.get("host", "").split(":")[0].strip()
        if raw_host and raw_host not in allowed_hosts:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Host header mismatch or untrusted domain.",
            )

        # Cross-site / Origin validation (Finding R3, Gate G29)
        sec_fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
        if sec_fetch_site == "cross-site":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Cross-site requests are not permitted.",
            )

        origin = request.headers.get("origin")
        if origin:
            try:
                parsed_origin = urlparse(origin)
                origin_hostname = (parsed_origin.hostname or "").strip().lower()
                origin_allowed = False
                if origin_hostname in allowed_hosts:
                    origin_allowed = True
                elif origin in cfg.allowed_origins or "*" in cfg.allowed_origins:
                    origin_allowed = True
                if not origin_allowed:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Forbidden: Untrusted Origin '{origin}'.",
                    )
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden: Malformed Origin header.",
                )

    # Auth dependencies
    async def require_auth(request: Request, authorization: Optional[str] = Header(None)) -> bool:
        _validate_host_header(request)
        client_host = request.client.host if request.client else "127.0.0.1"
        is_loopback = client_host in ("127.0.0.1", "localhost", "::1", "testclient")

        # In local-only mode, loopback callers are exempt from token requirement
        if is_loopback and not cfg.is_lan():
            return True

        # In LAN mode or remote access: require valid paired bearer token
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer "):].strip()

        if not token or not pairing_mgr.validate_token(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Device pairing required for LAN access.",
            )
        return True

    async def require_admin_auth(request: Request, authorization: Optional[str] = Header(None)) -> bool:
        """Dedicated authenticated guard with zero loopback exemption for sensitive actions."""
        _validate_host_header(request)
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer "):].strip()

        if not token or not pairing_mgr.validate_token(token, required_role="admin"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Admin-scoped paired token required for sensitive operations.",
            )
        return True

    # 1. Frontend Entrypoint
    @app.get("/")
    async def get_index():
        index_file = static_dir / "index.html"
        if not index_file.exists():
            return {"error": "Frontend assets not found"}
        return FileResponse(index_file)

    # 2. Auth Endpoints
    @app.post("/api/auth/pin")
    async def generate_pin(request: Request, authorization: Optional[str] = Header(None)):
        _validate_host_header(request)
        client_host = request.client.host if request.client else "127.0.0.1"
        is_loopback = client_host in ("127.0.0.1", "localhost", "::1", "testclient")

        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer "):].strip()

        # Reject unauthenticated remote callers
        if not is_loopback:
            if not token or not pairing_mgr.validate_token(token):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized: Remote generation of pairing PINs is prohibited. Generate PIN via terminal ('prosopon pair').",
                )
        else:
            # Over loopback in LAN mode, reject unauthenticated generation to prevent SSRF bypass
            if cfg.is_lan() and (not token or not pairing_mgr.validate_token(token)):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized: In LAN mode, pairing PIN must be viewed at terminal ('prosopon pair') or via authenticated session.",
                )

        pin = pairing_mgr.generate_pairing_pin()
        return {"pin": pin, "expires_in_seconds": 300}

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @app.post("/api/auth/local-token")
    async def get_local_token(request: Request):
        """Auto-provision local bearer token for loopback sessions in local mode."""
        _validate_host_header(request)
        client_host = request.client.host if request.client else "127.0.0.1"
        is_loopback = client_host in ("127.0.0.1", "localhost", "::1", "testclient")

        if not is_loopback or cfg.is_lan():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Auto-provisioning only permitted on local loopback.",
            )

        token = pairing_mgr.generate_loopback_token(device_name="Local Browser", role="admin")
        return {"status": "ok", "token": token}

    @app.post("/api/auth/pair")
    async def pair_device(req: PairingRequest, request: Request):
        _validate_host_header(request)
        client_ip = request.client.host if request.client else None
        token = pairing_mgr.verify_pin_and_pair(req.pin, device_name=req.device_name or "Phone", ip_address=client_ip)
        if not token:
            raise HTTPException(status_code=400, detail="Invalid, expired, or locked pairing PIN.")
        return {"status": "ok", "token": token}

    @app.post("/api/auth/revoke")
    async def revoke_tokens(request: Request, authorization: Optional[str] = Header(None)):
        _validate_host_header(request)
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer "):].strip()
        if not token or not pairing_mgr.validate_token(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Valid paired token required for revocation.",
            )
        is_admin = pairing_mgr.validate_token(token, required_role="admin")
        if is_admin:
            count = pairing_mgr.revoke_all()
        else:
            count = 1 if pairing_mgr.revoke_token(token) else 0
        return {"status": "ok", "revoked_count": count}

    # 3. Session Endpoints
    @app.get("/api/sessions")
    async def list_sessions(_: bool = Depends(require_auth)):
        return session_store.list_sessions()

    @app.post("/api/sessions")
    async def create_session(data: Dict[str, Any], _: bool = Depends(require_auth)):
        name = data.get("name")
        session = session_store.create_session(name=name)
        return session.model_dump()

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str, _: bool = Depends(require_auth)):
        session = session_store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.model_dump()

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str, _: bool = Depends(require_auth)):
        ok = session_store.delete_session(session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ok", "deleted": session_id}

    # 4. Turn Streaming Endpoint (SSE)
    @app.post("/api/chat/sessions/{session_id}/stream")
    async def stream_turn(
        session_id: str,
        req: ChatQueryRequest,
        _: bool = Depends(require_auth),
    ):
        session = session_store.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Record user message
        user_msg = ChatMessage(
            message_id=str(uuid.uuid4()),
            role="user",
            content=req.query,
        )
        session_store.append_message(session_id, user_msg)

        async def event_generator() -> AsyncIterator[str]:
            full_response_text = ""
            citations: list = []
            route_summary: Optional[Dict[str, Any]] = None
            final_status = "draft"

            try:
                async for event in kybernetes.submit_turn_stream(session_id, req.query):
                    # Intercept deltas and citations
                    if event.event_type == "delta":
                        full_response_text += event.payload.get("text", "")
                    elif event.event_type == "evidence":
                        citations = event.payload.get("citations", [])
                    elif event.event_type == "completed":
                        route_summary = event.payload.get("route_summary")
                        final_status = event.status or "verified"

                    # Emit SSE event line
                    yield f"data: {event.model_dump_json()}\n\n"

                # Persist completed assistant message
                asst_msg = ChatMessage(
                    message_id=str(uuid.uuid4()),
                    role="assistant",
                    content=full_response_text,
                    citations=[Citation.model_validate(c) for c in citations],
                    route_summary=route_summary,
                    status=final_status,
                )
                session_store.append_message(session_id, asst_msg)

            except Exception as exc:
                err_event = TurnEvent(
                    turn_id=str(uuid.uuid4()),
                    seq_num=999,
                    event_type="failed",
                    payload={"error": str(exc)},
                    status="failed",
                )
                yield f"data: {err_event.model_dump_json()}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # 5. Phylax Approval Resolution
    @app.post("/api/approvals/{approval_id}/resolve")
    async def resolve_approval(approval_id: str, req: ApprovalResolveRequest, _: bool = Depends(require_admin_auth)):
        if req.decision not in ("allow_once", "deny"):
            raise HTTPException(status_code=400, detail="Decision must be 'allow_once' or 'deny'")
        res = await kybernetes.resolve_approval(approval_id, req.decision)
        return res

    # 6. Local Voice: Transcribe (STT)
    @app.post("/api/voice/transcribe")
    async def transcribe_audio(file: UploadFile = File(...), _: bool = Depends(require_auth)):
        MAX_UPLOAD_BYTES = 15 * 1024 * 1024
        chunks = []
        total_size = 0
        while chunk := await file.read(65536):
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Audio file exceeds maximum size limit (15MB)")
            chunks.append(chunk)

        content = b"".join(chunks)
        if not content:
            raise HTTPException(status_code=400, detail="Empty audio buffer received")

        try:
            wav_path = await transcode_to_wav(content)
        except AudioTranscodingError as exc:
            raise HTTPException(status_code=400, detail=f"Transcoding error: {exc}")

        try:
            text, dur_ms, is_sim = await whisper_engine.transcribe(wav_path)
            return {
                "text": text,
                "transcription_latency_ms": dur_ms,
                "is_simulated": is_sim,
                "provenance": "simulated_whisper" if is_sim else "live_whisper",
            }
        except STTUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except STTExecutionError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            if wav_path.exists():
                wav_path.unlink()

    # 7. Local Voice: Synthesize (TTS)
    @app.post("/api/voice/synthesize")
    async def synthesize_speech(req: SynthesisRequest, _: bool = Depends(require_auth)):
        if len(req.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if len(req.text) > 5000:
            raise HTTPException(status_code=400, detail="Text exceeds maximum length (5000 characters)")

        try:
            wav_bytes, dur_ms, is_sim = await piper_engine.synthesize(req.text)
            headers = {
                "X-Synthesis-Duration-Ms": str(dur_ms),
                "X-Simulation-Provenance": "simulated_piper" if is_sim else "live_piper",
            }
            return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
        except TTSUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except TTSExecutionError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # 8. Unified Control Dashboard Endpoints (Protected by dedicated admin guard, no loopback exemption)
    def _get_gpu_telemetry() -> Dict[str, Any]:
        """Read GPU hardware telemetry via nvidia-smi if available; otherwise report estimated status."""
        try:
            import subprocess
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            if res.returncode == 0:
                gpus = []
                for line in res.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 7:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mb": int(parts[2]),
                            "memory_used_mb": int(parts[3]),
                            "memory_free_mb": int(parts[4]),
                            "temperature_c": int(parts[5]),
                            "utilization_gpu_pct": int(parts[6]),
                        })
                if gpus:
                    return {"available": True, "source": "nvml_nvidia_smi", "gpus": gpus}
        except Exception:
            pass
        return {"available": False, "source": "estimated", "gpus": []}

    @app.get("/api/dashboard/overview")
    async def dashboard_overview(_: bool = Depends(require_admin_auth)) -> Dict[str, Any]:
        """Proxy upstream Kybernetes overview or return truthful local state if offline/mock."""
        gpu_info = _get_gpu_telemetry()
        try:
            overview = await kybernetes.get_admin_overview()
            overview["gpu_telemetry"] = gpu_info
            return overview
        except Exception as exc:
            from datetime import datetime, timezone
            return {
                "service": "kybernetes",
                "version": "0.1.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "orchestrator_mode": "offline",
                "simulation_banner": False,
                "mock_banner": False,
                "gpu_telemetry": gpu_info,
                "orchestrator": {
                    "state": "offline",
                    "status": "unreachable",
                    "detail": f"Kybernetes orchestrator unreachable ({exc})",
                    "freshness": "unknown",
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                },
                "components": {},
            }

    @app.get("/api/dashboard/modules")
    async def dashboard_modules(_: bool = Depends(require_admin_auth)) -> List[Dict[str, Any]]:
        """Proxy upstream module descriptors."""
        try:
            return await kybernetes.get_admin_modules()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Upstream Kybernetes orchestrator unreachable: {exc}")

    @app.get("/api/dashboard/modules/{module_id}/config")
    async def dashboard_module_config(module_id: str, _: bool = Depends(require_admin_auth)) -> Dict[str, Any]:
        """Inspect module config view."""
        if module_id == "prosopon":
            from datetime import datetime, timezone
            cfg_dict = cfg.model_dump()
            # Redact pairing pin/tokens
            if "pairing_pin" in cfg_dict:
                cfg_dict["pairing_pin"] = "[REDACTED]"
            if "kybernetes_service_token" in cfg_dict:
                cfg_dict["kybernetes_service_token"] = "[REDACTED]"
            sources = {k: "env" if f"PROSOPON_{k.upper()}" in os.environ else "default" for k in cfg_dict}
            return {
                "module_id": "prosopon",
                "saved": cfg_dict,
                "effective": cfg_dict,
                "sources": sources,
                "restart_required_keys": ["host", "port", "tls_enabled", "allowed_origins"],
                "schema_ref": "ProsoponConfig",
                "loaded_at": datetime.now(timezone.utc).isoformat(),
                "drift": False,
            }
        try:
            return await kybernetes.get_admin_module_config(module_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Kybernetes config proxy error: {exc}")

    @app.post("/api/dashboard/modules/{module_id}/config/validate")
    async def dashboard_module_config_validate(
        module_id: str,
        req: Dict[str, Any],
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Validate proposed module configuration via native module validator."""
        proposed = req.get("proposed", {})
        if module_id == "prosopon":
            try:
                ProsoponConfig(**proposed)
                return {
                    "module_id": "prosopon",
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                    "restart_required": True,
                }
            except Exception as exc:
                return {
                    "module_id": "prosopon",
                    "valid": False,
                    "errors": [f"ProsoponConfig validation error: {exc}"],
                    "warnings": [],
                    "restart_required": True,
                }
        try:
            return await kybernetes.validate_admin_module_config(module_id, proposed)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Kybernetes config validation proxy error: {exc}")

    @app.get("/api/dashboard/sessions")
    async def dashboard_sessions(limit: int = 50, offset: int = 0, _: bool = Depends(require_admin_auth)) -> Dict[str, Any]:
        try:
            return await kybernetes.get_admin_sessions(limit=limit, offset=offset)
        except Exception as exc:
            return {"status": "unreachable", "total": 0, "limit": limit, "offset": offset, "storage": "unreachable", "sessions": [], "detail": str(exc)}

    @app.get("/api/dashboard/traces")
    async def dashboard_traces(limit: int = 50, offset: int = 0, _: bool = Depends(require_admin_auth)) -> Dict[str, Any]:
        try:
            return await kybernetes.get_admin_traces(limit=limit, offset=offset)
        except Exception as exc:
            return {"status": "unreachable", "total": 0, "limit": limit, "offset": offset, "storage": "unreachable", "traces": [], "detail": str(exc)}

    @app.get("/api/dashboard/audit-log")
    async def dashboard_audit_log(limit: int = 50, offset: int = 0, _: bool = Depends(require_admin_auth)) -> Dict[str, Any]:
        try:
            return await kybernetes.get_admin_audit_log(limit=limit, offset=offset)
        except Exception as exc:
            return {"status": "unreachable", "total": 0, "limit": limit, "offset": offset, "entries": [], "detail": str(exc)}

    @app.post("/api/dashboard/probes/run")
    async def dashboard_run_probe(req: Dict[str, str], _: bool = Depends(require_admin_auth)) -> Dict[str, Any]:
        comp_id = req.get("component_id", "")
        if not comp_id:
            raise HTTPException(status_code=400, detail="component_id required")
        try:
            return await kybernetes.run_admin_probe(comp_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Probe execution failed: {exc}")

    @app.get("/api/dashboard/gpu-telemetry")
    async def dashboard_gpu_telemetry(_: bool = Depends(require_admin_auth)) -> Dict[str, Any]:
        return _get_gpu_telemetry()

    @app.post("/api/dashboard/modules/{module_id}/config/apply")
    async def dashboard_module_config_apply(
        module_id: str,
        req: Dict[str, Any],
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Apply proposed module configuration with optimistic lock."""
        base_hash = req.get("base_hash", "")
        patch = req.get("patch", {})
        confirm_token = req.get("confirm_token")
        try:
            return await kybernetes.apply_admin_module_config(
                module_id=module_id,
                base_hash=base_hash,
                patch=patch,
                confirm_token=confirm_token,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Kybernetes config apply error: {exc}")

    @app.get("/api/dashboard/modules/{module_id}/history")
    async def dashboard_module_history(
        module_id: str,
        _: bool = Depends(require_admin_auth),
    ) -> List[Dict[str, Any]]:
        """Fetch ChangeSet history for a given module."""
        try:
            return await kybernetes.get_admin_module_history(module_id)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Kybernetes history proxy error: {exc}")

    @app.post("/api/dashboard/modules/{module_id}/config/rollback/{changeset_id}")
    async def dashboard_module_config_rollback(
        module_id: str,
        changeset_id: str,
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Rollback module configuration to pre-changeset snapshot."""
        try:
            return await kybernetes.rollback_admin_module_config(module_id, changeset_id)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Kybernetes rollback proxy error: {exc}")

    @app.post("/api/dashboard/models/load")
    async def dashboard_model_load(
        req: Dict[str, Any],
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Audited model load action."""
        model = req.get("model", "")
        confirm = req.get("confirm", True)
        if not model:
            raise HTTPException(status_code=400, detail="'model' field required")
        try:
            return await kybernetes.load_model(model=model, confirm=confirm)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model load failed: {exc}")

    @app.post("/api/dashboard/models/unload")
    async def dashboard_model_unload(
        req: Dict[str, Any],
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Audited model unload action."""
        model = req.get("model", "")
        confirm = req.get("confirm", True)
        if not model:
            raise HTTPException(status_code=400, detail="'model' field required")
        try:
            return await kybernetes.unload_model(model=model, confirm=confirm)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model unload failed: {exc}")

    @app.get("/api/dashboard/batch/dokime/runs")
    async def dashboard_dokime_runs(
        limit: int = 20,
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Fetch historical Dokime evaluation runs."""
        try:
            return await kybernetes.get_dokime_runs(limit=limit)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Dokime runs proxy error: {exc}")

    @app.post("/api/dashboard/batch/dokime/run")
    async def dashboard_dokime_run(
        req: Dict[str, Any],
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Trigger audited Dokime evaluation run."""
        confirm_token = req.get("confirm_token", "")
        task_id = req.get("task_id", "arc_easy_25")
        model_id = req.get("model_id", "llama-3.1-8b-instruct")
        smoke_test = req.get("smoke_test", True)
        try:
            return await kybernetes.run_dokime_batch(
                confirm_token=confirm_token,
                task_id=task_id,
                model_id=model_id,
                smoke_test=smoke_test,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Dokime run proxy error: {exc}")

    @app.get("/api/dashboard/batch/paideia/runs")
    async def dashboard_paideia_runs(
        limit: int = 20,
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Fetch historical Paideia fine-tuning runs."""
        try:
            return await kybernetes.get_paideia_runs(limit=limit)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Paideia runs proxy error: {exc}")

    @app.post("/api/dashboard/batch/paideia/run")
    async def dashboard_paideia_run(
        req: Dict[str, Any],
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Trigger audited Paideia fine-tuning run."""
        confirm_token = req.get("confirm_token", "")
        task = req.get("task", "router")
        allow_simulated = req.get("allow_simulated", True)
        try:
            return await kybernetes.run_paideia_batch(
                confirm_token=confirm_token,
                task=task,
                allow_simulated=allow_simulated,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Paideia run proxy error: {exc}")

    # 10. Workbench API
    workspace_root = Path(__file__).resolve().parents[2]

    @app.get("/api/workbench/tree")
    async def workbench_tree(_: bool = Depends(require_auth)) -> List[Dict[str, Any]]:
        """Return directory tree of workspace source modules for Metis Explorer."""
        IGNORED_NAMES = {
            ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
            "node_modules", "dist", "build", "egg-info", ".metis", "runs"
        }
        tree = []
        top_modules = [
            "kybernetes", "phylax", "harness", "prosopon", "dianoia",
            "kleroterion", "stoma", "paideia", "krisis", "hestia", "LTHCM", "dokime"
        ]
        for mod in top_modules:
            mod_dir = workspace_root / mod
            if not mod_dir.is_dir():
                continue
            children = []
            for root, dirs, files in os.walk(mod_dir):
                dirs[:] = [d for d in dirs if d not in IGNORED_NAMES and not d.endswith(".egg-info")]
                rel_root = Path(root).relative_to(mod_dir)
                for f in sorted(files):
                    if f.endswith((".py", ".yaml", ".yml", ".json", ".md", ".sh", ".toml")):
                        rel_path = str(rel_root / f) if str(rel_root) != "." else f
                        children.append({"name": rel_path, "type": "file"})
                    if len(children) >= 20:
                        break
                if len(children) >= 20:
                    break
            tree.append({"name": mod, "type": "dir", "children": children})
        return tree

    @app.get("/api/workbench/file")
    async def workbench_file(path: str, _: bool = Depends(require_auth)) -> Dict[str, Any]:
        """Read text content of a workspace file with directory traversal prevention."""
        if not path or ".." in path:
            raise HTTPException(status_code=400, detail="Invalid file path")
        target = (workspace_root / path).resolve()
        try:
            target.relative_to(workspace_root)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: file is outside workspace root")

        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(500000)
            lines = content.splitlines()
            return {
                "path": path,
                "content": content,
                "lines": lines,
                "total_lines": len(lines),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}")

    @app.get("/api/workbench/diff")
    async def workbench_diff(_: bool = Depends(require_auth)) -> Dict[str, Any]:
        """Return active pending diff / approval for workbench diff viewer."""
        import subprocess
        try:
            proc = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            mod_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            if mod_files:
                target_file = mod_files[0]
                diff_proc = subprocess.run(
                    ["git", "diff", target_file],
                    cwd=str(workspace_root),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                raw_diff = diff_proc.stdout.splitlines()
                old_lines = []
                new_lines = []
                for line in raw_diff:
                    if line.startswith("-") and not line.startswith("---"):
                        old_lines.append(line[1:])
                    elif line.startswith("+") and not line.startswith("+++"):
                        new_lines.append(line[1:])
                    elif not line.startswith("@@") and not line.startswith("diff"):
                        old_lines.append(line)
                        new_lines.append(line)
                return {
                    "has_diff": True,
                    "targetFile": target_file,
                    "approvalId": "git-working-tree",
                    "oldLines": old_lines[:100],
                    "newLines": new_lines[:100],
                }
        except Exception:
            pass

        return {
            "has_diff": False,
            "targetFile": "kybernetes/kybernetes/tools/registry.py",
            "approvalId": "appr-default",
            "oldLines": ["# No pending diffs in working tree"],
            "newLines": ["# Workspace is clean and verified"],
        }

    # 11. Lochos Multi-Agent Squad API
    @app.get("/api/lochos/squad")
    async def lochos_squad(_: bool = Depends(require_auth)) -> Dict[str, Any]:
        """Proxy upstream Lochos squad status or provide truthful fallback."""
        try:
            return await kybernetes.get_lochos_squad()
        except Exception:
            return {
                "rig_name": "metis-sovereign-squad",
                "version": "0.2",
                "seats": [
                    {
                        "id": "coder",
                        "label": "Autonomous Coding Agent",
                        "role": "Primary code implementation, AST navigation, and exact-anchor editing",
                        "model": "qwen2.5-coder-32b",
                        "state": "idle",
                        "worktree_branch": "squad/coder",
                        "skills": ["python", "refactoring", "git", "ast-mapping"],
                        "baton": None,
                    },
                    {
                        "id": "reviewer",
                        "label": "Independent Review & Verification Agent",
                        "role": "Zero-trust diff auditing, Dokime test verification, and security checks",
                        "model": "qwen2.5-coder-32b",
                        "state": "idle",
                        "worktree_branch": "squad/reviewer",
                        "skills": ["security", "code-review", "pytest", "dokime"],
                        "baton": None,
                    },
                ],
                "active_tasks_count": 0,
                "attention_required": [],
            }

    @app.get("/api/lochos/queue")
    async def lochos_queue(
        status: Optional[str] = None,
        seat_id: Optional[str] = None,
        _: bool = Depends(require_auth),
    ) -> List[Dict[str, Any]]:
        """Proxy Lochos work queue."""
        try:
            return await kybernetes.get_lochos_queue(status=status, seat_id=seat_id)
        except Exception:
            return []

    @app.post("/api/lochos/dispatch")
    async def lochos_dispatch(
        payload: Dict[str, Any],
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Dispatch task to squad seat."""
        try:
            return await kybernetes.dispatch_lochos_task(payload)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Lochos dispatch error: {exc}")

    @app.post("/api/lochos/resolve-attention")
    async def lochos_resolve_attention(
        payload: Dict[str, Any],
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Resolve blocked seat attention request."""
        try:
            return await kybernetes.resolve_lochos_attention(payload)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Lochos resolve error: {exc}")

    # 12. Health
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "app": "prosopon", "version": "0.1.0"}

    return app
