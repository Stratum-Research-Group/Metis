"""Admin API endpoints for Kybernetes control plane."""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
import httpx
from pydantic import BaseModel, Field

from kybernetes.admin.audit import AdminAuditLogger
from kybernetes.admin.descriptors import (
    MODULE_REGISTRY,
    ConfigView,
    ModuleDescriptor,
    compute_dict_hash,
    redact_sensitive_dict,
)
from kybernetes.admin.probes import ProbeRegistry
from kybernetes.admin.status import ComponentState, ComponentStatus, Freshness
from kybernetes.admin.validators import ConfigValidationResult, validate_module_configuration
from kybernetes.config import KybernetesConfig


from kybernetes.admin.changesets import (
    ChangeSet,
    ChangeSetStatus,
    ConfigApplyRequest,
    changeset_manager,
)


class ModelActionRequest(BaseModel):
    model: str = Field(..., description="Target model identifier")
    confirm: bool = Field(default=True, description="Confirmation of GPU side effect")


class ProbeRunRequest(BaseModel):
    component_id: str = Field(..., description="Identifier of the module or component to probe")


class ConfigValidateRequest(BaseModel):
    proposed: Dict[str, Any] = Field(..., description="Proposed module configuration dictionary to validate")


class OverviewResponse(BaseModel):
    service: str = "kybernetes"
    version: str = "0.1.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git: Dict[str, Any] = Field(default_factory=dict)
    stack_compatibility: str = "unchecked"
    budget_storage_mode: str = "in-memory"
    orchestrator_mode: str = "live"
    simulation_banner: bool = False
    mock_banner: bool = False
    components: Dict[str, ComponentStatus] = Field(default_factory=dict)


def get_git_info() -> Dict[str, Any]:
    """Inspect local git repository for HEAD commit and dirty state without throwing exceptions."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
            dirty_res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
            is_dirty = bool(dirty_res.stdout.strip())
            return {"commit": commit, "dirty": is_dirty}
    except Exception:
        pass
    return {"commit": "unknown", "dirty": False}


def create_admin_router(
    config: KybernetesConfig,
    probe_registry: ProbeRegistry,
    audit_logger: AdminAuditLogger,
    pipeline: Any,
    db: Any,
    session_manager: Any,
) -> APIRouter:
    router = APIRouter(prefix="/v1/admin", tags=["admin"])
    resolved_admin_token = config.resolve_admin_token()

    async def require_admin_auth(
        request: Request,
        authorization: Optional[str] = Header(None),
    ) -> bool:
        client_ip = request.client.host if request.client else None
        
        # Strict security requirement: If admin token is unconfigured, reject all requests
        if not resolved_admin_token:
            audit_logger.log(
                actor_token=None,
                action="admin_access_attempt",
                target=request.url.path,
                status="denied",
                detail="Kybernetes admin token is not configured on server.",
                client_ip=client_ip,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin API disabled: KYBERNETES_ADMIN_TOKEN must be configured.",
            )

        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer ") :].strip()

        import secrets
        if not token or not secrets.compare_digest(token, resolved_admin_token):
            audit_logger.log(
                actor_token=token,
                action="admin_access_attempt",
                target=request.url.path,
                status="denied",
                detail="Invalid or missing bearer token",
                client_ip=client_ip,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Valid Kybernetes admin bearer token required.",
            )

        return True

    @router.get("/overview", response_model=OverviewResponse)
    async def get_overview(
        request: Request,
        _: bool = Depends(require_admin_auth),
    ) -> OverviewResponse:
        """Overview with cached truthful ComponentStatuses, git identity, and banners."""
        client_ip = request.client.host if request.client else None
        audit_logger.log(
            actor_token=request.headers.get("Authorization"),
            action="read_overview",
            target="/v1/admin/overview",
            status="allowed",
            client_ip=client_ip,
        )

        statuses: Dict[str, ComponentStatus] = {}
        for mod_id in MODULE_REGISTRY.keys():
            statuses[mod_id] = probe_registry.get_cached_status(mod_id)

        # Evaluate simulation and mock presence
        any_simulated = any(s.state == ComponentState.SIMULATED or s.is_simulated for s in statuses.values())
        any_mock = any(s.state == ComponentState.MOCK for s in statuses.values())

        budget_mode = "persistent (sqlite)" if config.db_path and config.db_path != ":memory:" else "in-memory"

        return OverviewResponse(
            service="kybernetes",
            version="0.1.0",
            git=get_git_info(),
            stack_compatibility="unchecked",
            budget_storage_mode=budget_mode,
            orchestrator_mode="live",
            simulation_banner=any_simulated,
            mock_banner=any_mock,
            components=statuses,
        )

    @router.get("/modules", response_model=List[ModuleDescriptor])
    async def list_modules(
        request: Request,
        _: bool = Depends(require_admin_auth),
    ) -> List[ModuleDescriptor]:
        """List registered module descriptors for all 11 Metis modules."""
        client_ip = request.client.host if request.client else None
        audit_logger.log(
            actor_token=request.headers.get("Authorization"),
            action="list_modules",
            target="/v1/admin/modules",
            status="allowed",
            client_ip=client_ip,
        )
        return list(MODULE_REGISTRY.values())

    @router.get("/modules/{module_id}/config", response_model=ConfigView)
    async def get_module_config(
        module_id: str,
        request: Request,
        _: bool = Depends(require_admin_auth),
    ) -> ConfigView:
        """Inspect saved and effective configuration for a given module with secrets masked."""
        if module_id not in MODULE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

        client_ip = request.client.host if request.client else None
        audit_logger.log(
            actor_token=request.headers.get("Authorization"),
            action="read_config",
            target=f"/v1/admin/modules/{module_id}/config",
            status="allowed",
            client_ip=client_ip,
        )

        descriptor = MODULE_REGISTRY[module_id]

        if module_id == "kybernetes":
            effective = redact_sensitive_dict(config.model_dump())
            saved_raw, _, saved_hash = changeset_manager.get_current_saved_content_and_hash("kybernetes")
            saved = redact_sensitive_dict(saved_raw) if saved_raw else effective
            sources = {k: "file" if (saved_raw and k in saved_raw) else ("env" if f"KYBERNETES_{k.upper()}" in os.environ else "default") for k in effective}
            drift = (saved != effective) if saved_raw else False
            return ConfigView(
                module_id="kybernetes",
                saved=saved,
                effective=effective,
                sources=sources,
                restart_required_keys=["hestia_base_url", "memory_root", "db_path", "allowed_origins"],
                schema_ref=descriptor.schema_ref,
                config_hash=saved_hash if saved_raw else compute_dict_hash(effective),
                loaded_at=datetime.now(timezone.utc).isoformat(),
                drift=drift,
            )

        # Inspect disk config if file exists
        saved_raw, _, saved_hash = changeset_manager.get_current_saved_content_and_hash(module_id)
        saved_data: Dict[str, Any] = redact_sensitive_dict(saved_raw) if saved_raw else {}
        sources_data: Dict[str, str] = {k: "file" for k in saved_data}

        return ConfigView(
            module_id=module_id,
            saved=saved_data,
            effective=None,
            sources=sources_data,
            restart_required_keys=list(saved_data.keys()),
            schema_ref=descriptor.schema_ref,
            config_hash=saved_hash if saved_data else None,
            loaded_at=datetime.now(timezone.utc).isoformat(),
            drift=None,
        )

    @router.post("/modules/{module_id}/config/validate", response_model=ConfigValidationResult)
    async def validate_module_config_route(
        module_id: str,
        req: ConfigValidateRequest,
        request: Request,
        _: bool = Depends(require_admin_auth),
    ) -> ConfigValidationResult:
        """Validate proposed configuration without side effects using the owning module's native validator."""
        client_ip = request.client.host if request.client else None
        audit_logger.log(
            actor_token=request.headers.get("Authorization"),
            action="validate_config",
            target=f"/v1/admin/modules/{module_id}/config/validate",
            status="allowed",
            client_ip=client_ip,
        )
        if module_id not in MODULE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

        try:
            return validate_module_configuration(module_id, req.proposed)
        except Exception as exc:
            return ConfigValidationResult(
                module_id=module_id,
                valid=False,
                errors=[f"Validation execution failure: {exc}"],
                warnings=[],
                restart_required=MODULE_REGISTRY[module_id].restart_required,
            )

    @router.post("/modules/{module_id}/config/apply", response_model=ChangeSet)
    async def apply_module_config_route(
        module_id: str,
        req: ConfigApplyRequest,
        request: Request,
        _: bool = Depends(require_admin_auth),
    ) -> ChangeSet:
        """Apply proposed configuration updates atomically with base_hash lock and pre-write snapshot."""
        client_ip = request.client.host if request.client else None
        auth_hdr = request.headers.get("Authorization")
        audit_logger.log(
            actor_token=auth_hdr,
            action="apply_config",
            target=f"/v1/admin/modules/{module_id}/config/apply",
            status="requested",
            client_ip=client_ip,
        )
        if module_id not in MODULE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

        actor_name = client_ip or "admin"
        if auth_hdr:
            token_clean = auth_hdr.strip()
            if token_clean.lower().startswith("bearer "):
                token_clean = token_clean[7:].strip()
            actor_name = f"token:{hashlib.sha256(token_clean.encode('utf-8')).hexdigest()[:12]}"

        try:
            cs = changeset_manager.apply(
                module_id=module_id,
                base_hash=req.base_hash,
                patch=req.patch,
                requested_by=actor_name,
                confirm_token=req.confirm_token,
            )
            audit_logger.log(
                actor_token=auth_hdr,
                action="apply_config",
                target=f"/v1/admin/modules/{module_id}/config/apply",
                status="completed",
                detail=f"ChangeSet {cs.id} applied ({cs.status.value})",
                client_ip=client_ip,
            )
            return cs
        except PermissionError as pe:
            audit_logger.log(
                actor_token=auth_hdr,
                action="apply_config",
                target=f"/v1/admin/modules/{module_id}/config/apply",
                status="denied",
                detail=str(pe),
                client_ip=client_ip,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(pe),
            )
        except ValueError as ve:
            err_msg = str(ve)
            audit_logger.log(
                actor_token=auth_hdr,
                action="apply_config",
                target=f"/v1/admin/modules/{module_id}/config/apply",
                status="rejected",
                detail=err_msg,
                client_ip=client_ip,
            )
            if "Concurrent modification" in err_msg:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=err_msg)
            elif "Validation failed" in err_msg:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err_msg)
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)
        except Exception as exc:
            audit_logger.log(
                actor_token=auth_hdr,
                action="apply_config",
                target=f"/v1/admin/modules/{module_id}/config/apply",
                status="failed",
                detail=str(exc),
                client_ip=client_ip,
            )
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @router.get("/modules/{module_id}/history", response_model=List[ChangeSet])
    async def get_module_history_route(
        module_id: str,
        request: Request,
        _: bool = Depends(require_admin_auth),
    ) -> List[ChangeSet]:
        """Retrieve change history for a given module."""
        client_ip = request.client.host if request.client else None
        audit_logger.log(
            actor_token=request.headers.get("Authorization"),
            action="read_history",
            target=f"/v1/admin/modules/{module_id}/history",
            status="allowed",
            client_ip=client_ip,
        )
        if module_id not in MODULE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

        return changeset_manager.get_history(module_id)

    @router.post("/modules/{module_id}/config/rollback/{changeset_id}", response_model=ChangeSet)
    async def rollback_module_config_route(
        module_id: str,
        changeset_id: str,
        request: Request,
        _: bool = Depends(require_admin_auth),
    ) -> ChangeSet:
        """Atomically rollback configuration to the snapshot preceding a specific ChangeSet."""
        client_ip = request.client.host if request.client else None
        auth_hdr = request.headers.get("Authorization")
        audit_logger.log(
            actor_token=auth_hdr,
            action="rollback_config",
            target=f"/v1/admin/modules/{module_id}/config/rollback/{changeset_id}",
            status="requested",
            client_ip=client_ip,
        )
        if module_id not in MODULE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found")

        actor_name = client_ip or "admin"
        if auth_hdr:
            token_clean = auth_hdr.strip()
            if token_clean.lower().startswith("bearer "):
                token_clean = token_clean[7:].strip()
            actor_name = f"token:{hashlib.sha256(token_clean.encode('utf-8')).hexdigest()[:12]}"

        try:
            rb_cs = changeset_manager.rollback(
                module_id=module_id,
                changeset_id=changeset_id,
                requested_by=actor_name,
            )
            audit_logger.log(
                actor_token=auth_hdr,
                action="rollback_config",
                target=f"/v1/admin/modules/{module_id}/config/rollback/{changeset_id}",
                status="completed",
                detail=f"Rollback ChangeSet {rb_cs.id} applied",
                client_ip=client_ip,
            )
            return rb_cs
        except KeyError as ke:
            raise HTTPException(status_code=404, detail=str(ke))
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as exc:
            audit_logger.log(
                actor_token=auth_hdr,
                action="rollback_config",
                target=f"/v1/admin/modules/{module_id}/config/rollback/{changeset_id}",
                status="failed",
                detail=str(exc),
                client_ip=client_ip,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    @router.post("/models/load")
    async def load_model_route(
        req: ModelActionRequest,
        request: Request,
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Audited explicit model load action routed to Hestia inference server."""
        client_ip = request.client.host if request.client else None
        auth_hdr = request.headers.get("Authorization")
        audit_logger.log(
            actor_token=auth_hdr,
            action="model_load",
            target=f"model:{req.model}",
            status="requested",
            client_ip=client_ip,
        )
        base = config.hestia_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        target_url = f"{base}/manage/load"
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("HESTIA_SERVICE_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(target_url, json={"model": req.model}, headers=headers)
                status_str = "completed" if resp.status_code == 200 else "failed"
                audit_logger.log(
                    actor_token=auth_hdr,
                    action="model_load",
                    target=f"model:{req.model}",
                    status=status_str,
                    detail=f"Hestia HTTP {resp.status_code}",
                    client_ip=client_ip,
                )
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=resp.text)
                return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            audit_logger.log(
                actor_token=auth_hdr,
                action="model_load",
                target=f"model:{req.model}",
                status="failed",
                detail=str(exc),
                client_ip=client_ip,
            )
            raise HTTPException(status_code=503, detail=f"Hestia communication error: {exc}")

    @router.post("/models/unload")
    async def unload_model_route(
        req: ModelActionRequest,
        request: Request,
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Audited explicit model unload action routed to Hestia inference server."""
        client_ip = request.client.host if request.client else None
        auth_hdr = request.headers.get("Authorization")
        audit_logger.log(
            actor_token=auth_hdr,
            action="model_unload",
            target=f"model:{req.model}",
            status="requested",
            client_ip=client_ip,
        )
        base = config.hestia_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        target_url = f"{base}/manage/unload"
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("HESTIA_SERVICE_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(target_url, json={"model": req.model}, headers=headers)
                status_str = "completed" if resp.status_code == 200 else "failed"
                audit_logger.log(
                    actor_token=auth_hdr,
                    action="model_unload",
                    target=f"model:{req.model}",
                    status=status_str,
                    detail=f"Hestia HTTP {resp.status_code}",
                    client_ip=client_ip,
                )
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=resp.text)
                return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            audit_logger.log(
                actor_token=auth_hdr,
                action="model_unload",
                target=f"model:{req.model}",
                status="failed",
                detail=str(exc),
                client_ip=client_ip,
            )
            raise HTTPException(status_code=503, detail=f"Hestia communication error: {exc}")

    @router.get("/sessions")
    async def list_sessions(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """List in-memory conversational sessions."""
        client_ip = request.client.host if request.client else None
        audit_logger.log(
            actor_token=request.headers.get("Authorization"),
            action="list_sessions",
            target="/v1/admin/sessions",
            status="allowed",
            client_ip=client_ip,
        )

        all_sessions = []
        if hasattr(session_manager, "_sessions"):
            for sess_id, sess in session_manager._sessions.items():
                all_sessions.append({
                    "session_id": sess_id,
                    "created_at": getattr(sess, "created_at", None),
                    "turn_count": len(getattr(sess, "turns", [])),
                    "active_turn_id": getattr(sess, "active_turn_id", None),
                })

        paginated = all_sessions[offset : offset + limit]
        return {
            "total": len(all_sessions),
            "limit": limit,
            "offset": offset,
            "storage": "in-memory (max 500 LRU, 24h TTL)",
            "sessions": paginated,
        }

    @router.get("/traces")
    async def list_traces(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """List audit trace records for Dokime and evaluation inspection."""
        client_ip = request.client.host if request.client else None
        audit_logger.log(
            actor_token=request.headers.get("Authorization"),
            action="list_traces",
            target="/v1/admin/traces",
            status="allowed",
            client_ip=client_ip,
        )

        all_traces = []
        if hasattr(pipeline, "_traces"):
            for t_id, trace in pipeline._traces.items():
                all_traces.append({
                    "turn_id": t_id,
                    "session_id": getattr(trace, "session_id", None),
                    "timestamp": getattr(trace, "timestamp", None),
                    "total_duration_ms": getattr(trace, "total_duration_ms", None),
                    "tokens_used": getattr(trace, "tokens_used", None),
                    "served_model": getattr(trace, "metadata", {}).get("served_model"),
                    "dependency_versions": getattr(trace, "dependency_versions", {}),
                })

        paginated = list(reversed(all_traces))[offset : offset + limit]
        return {
            "total": len(all_traces),
            "limit": limit,
            "offset": offset,
            "storage": "in-memory (max 1000 buffer)",
            "traces": paginated,
        }

    @router.get("/audit-log")
    async def get_audit_log(
        request: Request,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        _: bool = Depends(require_admin_auth),
    ) -> Dict[str, Any]:
        """Retrieve recent entries from the append-only admin audit log."""
        entries = audit_logger.get_entries(limit=limit, offset=offset)
        return {
            "total": audit_logger.count(),
            "limit": limit,
            "offset": offset,
            "entries": [e.model_dump() for e in entries],
        }

    @router.post("/probes/run", response_model=ComponentStatus)
    async def trigger_probe(
        req: ProbeRunRequest,
        request: Request,
        _: bool = Depends(require_admin_auth),
    ) -> ComponentStatus:
        """Trigger an on-demand non-mutating canary probe for a specific component."""
        client_ip = request.client.host if request.client else None
        audit_logger.log(
            actor_token=request.headers.get("Authorization"),
            action="trigger_probe",
            target=f"/v1/admin/probes/run:{req.component_id}",
            status="allowed",
            client_ip=client_ip,
        )
        try:
            return await probe_registry.run_functional_probe(req.component_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return router
