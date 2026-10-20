# Hestia: Dashboard Implementation Specification & Resource Monitoring

This specification outlines the telemetry, configuration management, and model lifecycle operations exposed by **Hestia** for the Metis Control Dashboard (Section 2: *Models & Resources*).

---

## 1. Architectural Role in Dashboard

Hestia provides real-time model serving and VRAM admission telemetry to the dashboard:
- **Model Registry & Status**: Displays loaded models, operational states, backend processes, and ports.
- **Hardware & VRAM Telemetry**: Displays allocated vs available GPU VRAM (per card: Tesla V100 `sm_70` and RTX 5060 Ti `sm_120`).
- **Lifecycle Control**: Exposes explicit load/unload buttons (never triggered automatically by polling).
- **Service Configuration**: Inspects `hestia.yaml` and `models.yaml`.

```
[Dashboard Models Section]
         │
         ├── GET /v1/admin/modules/hestia/config
         ├── GET /v1/models (Model listing & digests)
         ├── GET /models (Detailed operational status)
         ▼
[Hestia API Server] ─── Manages llama-server subprocesses on ports 9000-9019
```

---

## 2. Module Descriptor for Hestia

```json
{
  "id": "hestia",
  "display_name": "Hestia (Model Server)",
  "section": "models_resources",
  "config_format": "yaml",
  "config_path": "hestia/hestia.yaml",
  "schema_ref": "ServiceConfig",
  "mutable": true,
  "restart_required_fields": ["listen_host", "listen_port", "backend_port_range", "vram_budget_mb"],
  "hot_reloadable_fields": ["idle_timeout_seconds", "max_queue_depth", "max_concurrent_requests"]
}
```

---

## 3. ComponentStatus Probe Specification

The probe connects to Hestia's `/health` endpoint and inspects model readiness:

```python
async def probe_hestia() -> ComponentStatus:
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(f"{hestia_url}/health")
            if resp.status_code != 200:
                return ComponentStatus(
                    state="offline",
                    detail=f"Health returned HTTP {resp.status_code}",
                    freshness="fresh",
                    checked_at=datetime.utcnow().isoformat(),
                    evidence=f"GET {hestia_url}/health"
                )
            
            # Fetch loaded models
            models_resp = await client.get(f"{hestia_url}/models")
            models_data = models_resp.json() if models_resp.status_code == 200 else {}
            loaded_models = [m["model_id"] for m in models_data.get("models", []) if m.get("status") == "ready"]
            
            return ComponentStatus(
                state="live",
                detail=f"{len(loaded_models)} model(s) loaded: {', '.join(loaded_models) or 'none'}",
                freshness="fresh",
                version="0.1.0",
                checked_at=datetime.utcnow().isoformat(),
                evidence="HTTP 200 /health and /models"
            )
    except Exception as exc:
        return ComponentStatus(
            state="offline",
            detail=f"Connection failed: {str(exc)}",
            freshness="fresh",
            checked_at=datetime.utcnow().isoformat(),
            evidence=f"Failed to connect to {hestia_url}"
        )
```

---

## 4. Models & Resources Section Schema

The dashboard displays:
1. **GPU Allocation Gauge**:
   - VRAM Budget (MB)
   - Allocated VRAM (MB)
   - Free VRAM Headroom (MB)
2. **Model Catalog**:
   - Model Alias & Architecture
   - File Path & Weights SHA-256 Digest
   - Status (`ready`, `loading`, `unloaded`, `failed`)
   - Backend Process PID and Allocated Port (within `9000-9019`)
   - Queue Depth & In-Flight Concurrency
   - Actions: `Load` / `Unload` buttons with confirmation
3. **Configuration Inspector**:
   - Saved `hestia.yaml` vs Effective `ServiceConfig`
   - Uncommitted edits badge (`Restart Required`)
