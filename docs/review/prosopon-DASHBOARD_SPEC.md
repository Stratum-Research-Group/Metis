# Prosopon: Dashboard Implementation Specification & BFF Integration

This specification defines the role of **Prosopon** as the frontend host and Backend-for-Frontend (BFF) proxy for the unified Metis Control Dashboard.

---

## 1. Architectural Role in Dashboard

Prosopon serves two primary functions in the dashboard architecture:
1. **Frontend Host (SPA)**: Serves the vanilla JavaScript/HTML/CSS dashboard interface alongside the main chat/voice UI.
2. **Backend-for-Frontend (BFF)**: Exposes `/api/dashboard/*` routes that proxy authenticated requests to Kybernetes' `/v1/admin/*` endpoints, adding origin validation, device pairing verification, and secure context handling.

```
[Browser Client]
       │
       ├── HTTPS (Bearer Token Auth, No Loopback Exemption)
       ▼
[Prosopon Server] ─── /api/dashboard/* 
       │
       ├── Dedicated Admin Auth Check (require_admin_auth)
       ├── Mutual Service Credential
       ▼
[Kybernetes Server] ─── /v1/admin/*
```

---

## 2. Security & Authentication Model

### 1. Dedicated Admin Guard (`require_admin_auth`)
- **Zero Loopback Exemption**: Unlike casual chat, dashboard access requires an active paired session token regardless of whether the request originates from `127.0.0.1` or LAN.
- **CSRF & Origin Protection**: All requests validate the `Host` header against configured bind addresses and enforce `allowed_origins`.
- **Credential Storage**: Bearer tokens are kept in browser memory or session storage, transmitted via the `Authorization: Bearer <token>` header.

### 2. Forwarding to Kybernetes
- Prosopon holds a shared internal service secret (`KYBERNETES_SERVICE_TOKEN`) used to sign or authenticate requests made to Kybernetes `/v1/admin/*`.
- External clients cannot call Kybernetes directly, preventing bypass of Prosopon's security boundary.

---

## 3. Module Descriptor for Prosopon

Prosopon registers the following descriptor with the Kybernetes control plane:

```json
{
  "id": "prosopon",
  "display_name": "Prosopon (UI & Voice)",
  "section": "presentation",
  "config_format": "pydantic",
  "config_path": "prosopon/prosopon/config.py",
  "schema_ref": "ProsoponConfig",
  "mutable": true,
  "restart_required_fields": ["host", "port", "tls", "allowed_origins", "kybernetes_api_url"],
  "hot_reloadable_fields": ["whisper_model_path", "piper_model_path"]
}
```

---

## 4. ComponentStatus Probe Specification

The Prosopon probe evaluates operational health truthfully:

```python
async def probe_prosopon() -> ComponentStatus:
    # 1. Check local audio dependencies
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    whisper_ok = Path(config.whisper_model_path).exists() if config.whisper_model_path else False
    
    # 2. Check upstream orchestrator connection
    upstream_state = "mock" if config.mock_kybernetes else "live"
    upstream_reachable = False
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{config.kybernetes_api_url}/v1/health")
            upstream_reachable = (resp.status_code == 200)
    except Exception:
        upstream_reachable = False
        
    state = "live" if (upstream_reachable and not config.mock_kybernetes) else ("mock" if config.mock_kybernetes else "degraded")
    
    return ComponentStatus(
        state=state,
        detail=f"BFF active; Upstream: {'reachable' if upstream_reachable else 'unreachable'} ({upstream_state}); Voice: ffmpeg={ffmpeg_ok}, whisper={whisper_ok}",
        freshness="fresh",
        version="0.1.0",
        checked_at=datetime.utcnow().isoformat(),
        evidence=f"GET {config.kybernetes_api_url}/v1/health -> {200 if upstream_reachable else 'failed'}"
    )
```

---

## 5. BFF Proxy Endpoints (`/api/dashboard/*`)

Prosopon proxies the following routes to Kybernetes `/v1/admin/*`:

| Endpoint | Method | Upstream Target | Description |
| :--- | :--- | :--- | :--- |
| `/api/dashboard/overview` | `GET` | `/v1/admin/overview` | Aggregated component health grid, active alerts, and resource telemetry. |
| `/api/dashboard/modules` | `GET` | `/v1/admin/modules` | List of registered ModuleDescriptors. |
| `/api/dashboard/modules/{id}/config` | `GET` | `/v1/admin/modules/{id}/config` | Saved vs effective config view with drift detection. |
| `/api/dashboard/modules/{id}/config/validate` | `POST` | `/v1/admin/modules/{id}/config/validate` | Dry-run schema validation without writing to disk. |
| `/api/dashboard/modules/{id}/config/apply` | `POST` | `/v1/admin/modules/{id}/config/apply` | Atomic patch write requiring base hash and confirmation token for sensitive fields. |
| `/api/dashboard/modules/{id}/history` | `GET` | `/v1/admin/modules/{id}/history` | Audit log of previous configuration change sets. |
| `/api/dashboard/modules/{id}/config/rollback/{cs_id}` | `POST` | `/v1/admin/modules/{id}/config/rollback/{cs_id}` | Restores configuration snapshot from change set history. |

---

## 6. Frontend UI View (`web/static/js/dashboard.js`)

The dashboard tab provides:
1. **System Banner**: Highlights if the system is running in `--mock` mode or if any subsystem is `simulated`.
2. **Component Status Grid**: Real-time status cards with color-coded chips (`live`=green, `simulated`=amber, `mock`=blue, `offline`=red, `unknown`=gray) and evidence tooltips.
3. **Module Config Inspector**: Side-by-side comparison of **Saved** vs **Effective** settings, highlighting uncommitted drift.
4. **Guarded Edit Modal**: Diff view showing proposed changes, identifying required service restarts, and prompting for an owner confirmation token when changing network/security settings.
