"""FastAPI backend server for Prosopon local web & voice interface."""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
import uuid

from fastapi import Depends, FastAPI, File, HTTPException, Header, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from prosopon.config import ProsoponConfig
from prosopon.auth.pairing import PairingManager
from prosopon.contracts.session import ChatMessage, SessionRecord
from prosopon.contracts.kybernetes import Citation, TurnEvent
from prosopon.contracts.phylax import ApprovalResponse
from prosopon.storage.session_store import SessionStore
from prosopon.client.kybernetes_client import KybernetesClient
from prosopon.client.mock_kybernetes import MockKybernetesClient
from prosopon.voice.transcoder import transcode_to_wav
from prosopon.voice.whisper_stt import WhisperSTTEngine
from prosopon.voice.piper_tts import PiperTTSEngine


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
        kybernetes = KybernetesClient(base_url=cfg.kybernetes_api_url)

    whisper_engine = WhisperSTTEngine(binary_name=cfg.whisper_binary, model_path=cfg.whisper_model)
    piper_engine = PiperTTSEngine(binary_name=cfg.piper_binary, model_path=cfg.piper_model)

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

    # Static asset paths
    static_dir = Path(__file__).parent / "web" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Auth dependency
    async def require_auth(request: Request, authorization: Optional[str] = Header(None)) -> bool:
        # If accessing over loopback (localhost/127.0.0.1), exempt from token pairing
        client_host = request.client.host if request.client else "127.0.0.1"
        is_loopback = client_host in ("127.0.0.1", "localhost", "::1")
        if is_loopback and not cfg.is_lan():
            return True

        # Over LAN: require valid paired bearer token
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer "):].strip()

        if not token or not pairing_mgr.validate_token(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Device pairing required for LAN access.",
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
    async def generate_pin(request: Request):
        # Only allowed from loopback or already authenticated client
        pin = pairing_mgr.generate_pairing_pin()
        return {"pin": pin, "expires_in_seconds": 300}

    @app.post("/api/auth/pair")
    async def pair_device(req: PairingRequest, request: Request):
        client_ip = request.client.host if request.client else None
        token = pairing_mgr.verify_pin_and_pair(req.pin, device_name=req.device_name or "Phone", ip_address=client_ip)
        if not token:
            raise HTTPException(status_code=400, detail="Invalid, expired, or locked pairing PIN.")
        return {"status": "ok", "token": token}

    @app.post("/api/auth/revoke")
    async def revoke_tokens(_: bool = Depends(require_auth)):
        count = pairing_mgr.revoke_all()
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
    async def resolve_approval(approval_id: str, req: ApprovalResolveRequest, _: bool = Depends(require_auth)):
        if req.decision not in ("allow_once", "deny"):
            raise HTTPException(status_code=400, detail="Decision must be 'allow_once' or 'deny'")
        res = await kybernetes.resolve_approval(approval_id, req.decision)
        return res

    # 6. Local Voice: Transcribe (STT)
    @app.post("/api/voice/transcribe")
    async def transcribe_audio(file: UploadFile = File(...), _: bool = Depends(require_auth)):
        content = await file.read()
        try:
            wav_path = await transcode_to_wav(content)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Transcoding error: {exc}")

        try:
            text, dur_ms = await whisper_engine.transcribe(wav_path)
            return {"text": text, "transcription_latency_ms": dur_ms}
        finally:
            if wav_path.exists():
                wav_path.unlink()

    # 7. Local Voice: Synthesize (TTS)
    @app.post("/api/voice/synthesize")
    async def synthesize_speech(req: SynthesisRequest, _: bool = Depends(require_auth)):
        wav_bytes, dur_ms = await piper_engine.synthesize(req.text)
        return Response(content=wav_bytes, media_type="audio/wav", headers={"X-Synthesis-Duration-Ms": str(dur_ms)})

    # 8. Health
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "app": "prosopon", "version": "0.1.0"}

    return app
