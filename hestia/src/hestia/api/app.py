"""Hestia API — OpenAI-compatible endpoints with routing, admission queue, and strict schemas."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import os
import secrets
from typing import Any
import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from hestia.api.schemas import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ManageModelRequest,
    ModelListResponse,
    ModelObject,
)
from hestia.backend import BackendStartError
from hestia.backend.llamacpp import LlamaCppBackend
from hestia.cascade.engine import CascadeConfig, SpeculativeCascadeEngine
from hestia.config import ServiceConfig
from hestia.registry import ModelRegistry, ModelType
from hestia.server import (
    GpuQueryError,
    InsufficientVramError,
    ModelFailedError,
    ModelNotFoundError,
    NoFreePortError,
    QueueTimeoutError,
    ServerManager,
    ServerOverloadedError,
)
from hestia.vram import get_free_vram_mb, get_gpu_name, get_total_vram_mb

logger = logging.getLogger(__name__)

# Module-level references set during lifespan
_manager: ServerManager | None = None
_registry: ModelRegistry | None = None


class HostHeaderValidationMiddleware(BaseHTTPMiddleware):
    """Validate Host header against allowed loopback and configured host patterns."""

    def __init__(self, app: Any, allowed_hosts: list[str] | None = None):
        super().__init__(app)
        self.allowed_hosts = allowed_hosts or ["127.0.0.1", "localhost", "::1", "testserver", "test"]

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        host_header = request.headers.get("host", "").split(":")[0].strip().lower()
        if host_header and host_header not in self.allowed_hosts and not host_header.startswith("127.0.0."):
            return Response(
                content="Invalid Host header",
                status_code=400,
            )
        return await call_next(request)


class ContentTypeValidationMiddleware(BaseHTTPMiddleware):
    """Require Content-Type application/json on all POST endpoints."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.method == "POST":
            ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if ct != "application/json":
                return JSONResponse(
                    status_code=415,
                    content={"detail": "Unsupported Media Type: Content-Type must be application/json"},
                )
        return await call_next(request)


def require_hestia_auth(request: Request) -> None:
    """Require valid Hestia service token unless local loopback / MSS Void or insecure-dev mode."""
    client_host = request.client.host if request.client else ""
    # Local loopback trust: MSS Void on local machine does not need token authorization friction
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return

    auth_header = request.headers.get("authorization", "")
    token = ""
    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer ") :].strip()
    elif "x-service-token" in request.headers:
        token = request.headers["x-service-token"].strip()

    expected_token = os.environ.get("HESTIA_SERVICE_TOKEN")
    if not expected_token:
        if os.environ.get("METIS_INSECURE_DEV") == "1":
            logger.warning("HESTIA_SERVICE_TOKEN unset. Operating in insecure dev mode.")
            return
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Valid Hestia service token required (HESTIA_SERVICE_TOKEN unset or invalid).",
        )

    if not token or not secrets.compare_digest(token, expected_token):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Valid Hestia service token required.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — start/stop the server manager and shared HTTP client."""
    global _manager, _registry

    config_path = Path(app.state.config_path)
    models_path = Path(app.state.models_path)

    config = ServiceConfig.from_yaml(config_path)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    registry = ModelRegistry()
    registry.load_from_yaml(models_path)
    _registry = registry

    backend = LlamaCppBackend(binary_path=config.llamacpp_binary)
    manager = ServerManager(registry=registry, backend=backend, config=config)
    _manager = manager

    # Shared HTTP client for proxying requests cleanly
    app.state.http_client = httpx.AsyncClient(timeout=300.0)

    await manager.start()
    logger.info("Hestia API ready on port %d", config.listen_port)
    try:
        yield
    finally:
        await app.state.http_client.aclose()
        await manager.shutdown()


def create_app(
    config_path: str = "hestia.yaml",
    models_path: str = "models.yaml",
) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Hestia",
        description="Local AI model serving layer",
        version="0.1.0",
        lifespan=lifespan,
    )
    cfg_p = Path(config_path)
    if not cfg_p.is_file() and (Path("hestia") / config_path).is_file():
        config_path = str(Path("hestia") / config_path)
    mod_p = Path(models_path)
    if not mod_p.is_file() and (Path("hestia") / models_path).is_file():
        models_path = str(Path("hestia") / models_path)

    app.state.config_path = config_path
    app.state.models_path = models_path

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:8081",
            "http://localhost:8081",
            "vscode-file://vscode-app",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(HostHeaderValidationMiddleware)
    app.add_middleware(ContentTypeValidationMiddleware)

    app.include_router(_build_router())
    return app


def _build_router():
    """Build the API routes."""
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/")
    async def root() -> JSONResponse:
        """Service index and endpoint discovery for Hestia."""
        return JSONResponse(
            {
                "service": "hestia",
                "name": "Hestia Local Model Server",
                "version": "0.1.0",
                "endpoints": {
                    "health": "/health",
                    "models": "/v1/models",
                    "chat_completions": "/v1/chat/completions",
                    "embeddings": "/v1/embeddings",
                    "docs": "/docs",
                },
            }
        )

    @router.get("/v1/models")
    async def list_v1_models() -> JSONResponse:
        """OpenAI-compatible model listing endpoint for Dianoia, Kybernetes, and external tools."""
        registry = _get_registry()
        models_data = []
        for m in registry.list_all():
            models_data.append(
                ModelObject(
                    id=m.name,
                    root=m.name,
                    type=m.model_type.value,
                    context_size=m.context_size,
                    quantization=m.quantization,
                    artifact_digest=m.artifact_digest,
                    gpu_target=m.gpu_target,
                    state=m.state.value if hasattr(m.state, "value") else str(m.state),
                ).model_dump()
            )
        return JSONResponse(ModelListResponse(data=models_data).model_dump())

    @router.get("/v1/models/{model_name}")
    async def get_v1_model(model_name: str) -> JSONResponse:
        """OpenAI-compatible single model lookup."""
        registry = _get_registry()
        m = registry.get(model_name)
        if not m:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        return JSONResponse(
            ModelObject(
                id=m.name,
                root=m.name,
                type=m.model_type.value,
                context_size=m.context_size,
                quantization=m.quantization,
                artifact_digest=m.artifact_digest,
                gpu_target=m.gpu_target,
                state=m.state.value if hasattr(m.state, "value") else str(m.state),
            ).model_dump()
        )

    @router.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> Response:
        """OpenAI-compatible chat completions endpoint with admission control."""
        require_hestia_auth(request)
        manager = _get_manager()

        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

        model_name = body.get("model")
        if not model_name:
            raise HTTPException(status_code=400, detail="'model' field is required")

        if model_name.startswith("cascade:"):
            return await cascade_completions(request)

        # Validate request schema
        try:
            validated_req = ChatCompletionRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Validate model type
        model = _get_registry().get(model_name)
        if model and model.model_type != ModelType.CHAT:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' is type '{model.model_type.value}', not 'chat'",
            )

        # Apply model-level sampling defaults if not specified by caller
        if model and model.sampling:
            if "temperature" not in body or body["temperature"] is None:
                body["temperature"] = model.sampling.temperature
            if "top_p" not in body or body["top_p"] is None:
                body["top_p"] = model.sampling.top_p
            if "top_k" not in body or body["top_k"] is None:
                if model.sampling.top_k is not None:
                    body["top_k"] = model.sampling.top_k
            if "repeat_penalty" not in body or body["repeat_penalty"] is None:
                if model.sampling.repeat_penalty is not None:
                    body["repeat_penalty"] = model.sampling.repeat_penalty

        # Concurrency admission control
        try:
            await manager.acquire_admission(model_name)
        except ServerOverloadedError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc
        except QueueTimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc

        try:
            try:
                process = await manager.ensure_loaded(model_name, track_inflight=True)
            except ModelNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            except (ModelFailedError, InsufficientVramError, NoFreePortError, GpuQueryError, BackendStartError) as exc:
                raise HTTPException(status_code=503, detail=str(exc))

            stream = validated_req.stream

            if stream:
                try:
                    client, resp = await _open_stream_request(process.base_url, body)
                except Exception as exc:
                    manager.request_finished(model_name)
                    raise HTTPException(status_code=503, detail=f"Backend connection error: {exc}")

                if resp.status_code != 200:
                    err_bytes = await resp.aread()
                    await resp.aclose()
                    await client.aclose()
                    await _safe_request_finished(manager, model_name)
                    await _safe_release_admission(manager, model_name)
                    try:
                        err_json = json.loads(err_bytes.decode(errors="replace"))
                        return JSONResponse(content=err_json, status_code=resp.status_code)
                    except Exception:
                        return Response(
                            content=err_bytes,
                            status_code=resp.status_code,
                            media_type="application/json",
                        )

                async def _stream_generator() -> AsyncGenerator[str, None]:
                    try:
                        async for line in resp.aiter_lines():
                            yield f"{line}\n"
                    except Exception as exc:
                        logger.error("Error during streaming for %s: %s", model_name, exc)
                        error_payload = json.dumps({"error": str(exc)})
                        yield f"data: {error_payload}\n\n"
                    finally:
                        await resp.aclose()
                        await client.aclose()
                        await _safe_request_finished(manager, model_name)
                        await _safe_release_admission(manager, model_name)

                return StreamingResponse(_stream_generator(), media_type="text/event-stream")
            else:
                try:
                    return await _proxy_request(process.base_url, body, path="/v1/chat/completions")
                finally:
                    await _safe_request_finished(manager, model_name)
                    await _safe_release_admission(manager, model_name)
        except Exception:
            # If an error occurred before returning the streaming response generator
            await _safe_release_admission(manager, model_name)
            raise

    @router.post("/v1/cascade/completions", response_model=None)
    async def cascade_completions(request: Request) -> JSONResponse:
        """Speculative drafter-verifier cascade endpoint with early deferral rules."""
        require_hestia_auth(request)
        manager = _get_manager()
        registry = _get_registry()

        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

        draft_name = body.get("draft_model")
        verifier_name = body.get("verifier_model")
        model_spec = body.get("model", "")
        if (not draft_name or not verifier_name) and model_spec.startswith("cascade:"):
            parts = model_spec.split(":")
            if len(parts) >= 3:
                draft_name = parts[1]
                verifier_name = parts[2]

        if not draft_name or not verifier_name:
            raise HTTPException(
                status_code=400,
                detail="Both 'draft_model' and 'verifier_model' must be specified (or model='cascade:<draft>:<verifier>')",
            )

        draft_model = registry.get(draft_name)
        if not draft_model:
            raise HTTPException(status_code=404, detail=f"Draft model '{draft_name}' not found")
        verifier_model = registry.get(verifier_name)
        if not verifier_model:
            raise HTTPException(status_code=404, detail=f"Verifier model '{verifier_name}' not found")

        threshold = float(body.get("deferral_threshold", 0.80))
        criterion = body.get("criterion", "mean_prob")
        cost_ratio = float(body.get("cost_ratio", 0.25))
        draft_max_tokens = body.get("draft_max_tokens")

        config = CascadeConfig(
            draft_model=draft_name,
            verifier_model=verifier_name,
            deferral_threshold=threshold,
            criterion=criterion,
            draft_max_tokens=draft_max_tokens,
            cost_ratio=cost_ratio,
        )
        engine = SpeculativeCascadeEngine(config)

        async def _forward_call(m_name: str, payload: dict[str, Any]) -> dict[str, Any]:
            return await _forward_cascade_call(manager, m_name, payload)

        resp_content, decision = await engine.execute(body, _forward_call)
        return JSONResponse(content=resp_content)

    @router.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        """OpenAI-compatible embeddings endpoint with admission control."""
        require_hestia_auth(request)
        manager = _get_manager()

        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

        model_name = body.get("model")
        if not model_name:
            raise HTTPException(status_code=400, detail="'model' field is required")

        try:
            EmbeddingRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Validate model type
        model = _get_registry().get(model_name)
        if model and model.model_type != ModelType.EMBEDDING:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' is type '{model.model_type.value}', not 'embedding'",
            )

        # Concurrency admission control
        try:
            await manager.acquire_admission(model_name)
        except ServerOverloadedError as exc:
            raise HTTPException(
                status_code=429,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc
        except QueueTimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=str(exc),
                headers={"Retry-After": "5"},
            ) from exc

        try:
            try:
                process = await manager.ensure_loaded(model_name, track_inflight=True)
            except ModelNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            except (ModelFailedError, InsufficientVramError, NoFreePortError, GpuQueryError, BackendStartError) as exc:
                raise HTTPException(status_code=503, detail=str(exc))

            try:
                return await _proxy_request(process.base_url, body, path="/v1/embeddings")
            finally:
                await _safe_request_finished(manager, model_name)
        finally:
            await _safe_release_admission(manager, model_name)

    @router.get("/models")
    async def list_models(request: Request) -> JSONResponse:
        """List all registered models with their internal operational status."""
        require_hestia_auth(request)
        manager = _get_manager()
        return JSONResponse({"models": manager.get_all_status()})

    @router.post("/manage/load")
    async def manage_load(request: Request) -> JSONResponse:
        """Explicitly load a model (used by CLI)."""
        require_hestia_auth(request)
        manager = _get_manager()
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            raise HTTPException(status_code=400, detail="'model' field is required")

        try:
            ManageModelRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            await manager.load_model_explicit(model_name)
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ModelFailedError, InsufficientVramError, NoFreePortError, GpuQueryError, BackendStartError) as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        return JSONResponse({"status": "loaded", "model": model_name})

    @router.post("/manage/unload")
    async def manage_unload(request: Request) -> JSONResponse:
        """Explicitly unload a model (used by CLI)."""
        require_hestia_auth(request)
        manager = _get_manager()
        body = await request.json()
        model_name = body.get("model")
        if not model_name:
            raise HTTPException(status_code=400, detail="'model' field is required")

        try:
            ManageModelRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            await manager.unload_model_explicit(model_name)
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        return JSONResponse({"status": "unloaded", "model": model_name})

    async def _safe_request_finished(mgr: ServerManager, m_name: str) -> None:
        res = mgr.request_finished(m_name)
        if asyncio.iscoroutine(res):
            await res

    async def _safe_release_admission(mgr: ServerManager, m_name: str) -> None:
        res = mgr.release_admission(m_name)
        if asyncio.iscoroutine(res):
            await res

    @router.get("/health")
    async def health() -> JSONResponse:
        """Service and per-model health with live GPU metrics."""
        manager = _get_manager()
        models_status = manager.get_all_status()

        any_failed = any(m["state"] == "failed" for m in models_status)
        service_status = "degraded" if any_failed else "healthy"

        return JSONResponse(
            {
                "status": service_status,
                "models": models_status,
                "gpu": {
                    "name": get_gpu_name(),
                    "vram_free_mb": get_free_vram_mb(),
                    "vram_total_mb": get_total_vram_mb(),
                },
            }
        )

    return router


async def _proxy_request(
    base_url: str,
    body: dict,
    path: str = "/v1/chat/completions",
) -> Response:
    """Proxy a non-streaming request to the backend."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(f"{base_url}{path}", json=body)
        try:
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except Exception:
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


async def _open_stream_request(
    base_url: str,
    body: dict,
) -> tuple[httpx.AsyncClient, httpx.Response]:
    """Open an upstream streaming request to the backend."""
    client = httpx.AsyncClient(timeout=300.0)
    try:
        req = client.build_request(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=body,
        )
        resp = await client.send(req, stream=True)
        return client, resp
    except Exception:
        await client.aclose()
        raise


async def _forward_cascade_call(
    manager: ServerManager,
    model_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Forward an internal cascade inference request to the loaded backend instance."""
    await manager.acquire_admission(model_name)
    try:
        proc = await manager.ensure_loaded(model_name, track_inflight=True)
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{proc.base_url}/v1/chat/completions", json=payload)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
    except (ModelNotFoundError, ModelFailedError, InsufficientVramError, NoFreePortError, GpuQueryError, BackendStartError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    finally:
        res1 = manager.request_finished(model_name)
        if asyncio.iscoroutine(res1):
            await res1
        res2 = manager.release_admission(model_name)
        if asyncio.iscoroutine(res2):
            await res2


def _get_manager() -> ServerManager:
    if _manager is None:
        raise HTTPException(status_code=503, detail="Server manager not initialized")
    return _manager


def _get_registry() -> ModelRegistry:
    if _registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")
    return _registry
