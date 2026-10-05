# Kybernetes: Dashboard Control Plane Specification (`/v1/admin/*`)

This document defines the central control plane architecture, endpoint specifications, and module coordination protocol exposed by **Kybernetes** for the Metis Control Dashboard.

---

## 1. Architectural Role in Dashboard

Kybernetes is the **authoritative coordinator** for stack administration:
1. **Module Registry**: Maintains registrations of all 11 stack modules via standardized `ModuleDescriptor` records.
2. **Truthful Probing**: Periodically evaluates or caches `ComponentStatus` health probes across modules without executing heavyweight inference.
3. **Configuration Authority**: Mediates schema inspection, validation, atomic change sets, and rollback operations across module-owned configuration files.

```
[Prosopon BFF]
      │
      ├── Authenticated Internal Transport (Bearer Service Token)
      ▼
[Kybernetes Control Plane] ─── /v1/admin/*
      │
      ├── ModuleRegistry
      │     ├── Hestia Adapter / Models API
      │     ├── LTHCM / Stoma Memory Loader
      │     ├── Dianoia Deliberation Controller
      │     ├── Phylax Policy Gateway
      │     └── Krisis Policy Engine
      ▼
[Module Configuration Files & Runtime Reports]
```

---

## 2. ComponentStatus Specification

Kybernetes surfaces subsystem status using a strictly truthful 5-state lifecycle:

```python
class ComponentState(str, Enum):
    LIVE = "live"             # Functional end-to-end check succeeded
    SIMULATED = "simulated"   # Mock adapter or synthetic corpus in use
    MOCK = "mock"             # Explicit mock mode enabled
    OFFLINE = "offline"       # Service unreachable or process down
    UNKNOWN = "unknown"       # Unprobed or expired probe cache

class ComponentStatus(BaseModel):
    state: ComponentState
    detail: str
    freshness: Literal["fresh", "stale", "unknown"]
    version: Optional[str] = None
    checked_at: str
    latency_ms: Optional[int] = None
    evidence: str
```

### Probing Rules
- **No Automatic Model Inference**: Health polling must *never* trigger LLM token generation or model loads in Hestia.
- **Evidence-Backed**: A status of `live` requires concrete evidence (e.g. `HTTP 200 /v1/models`, real index retrieval with positive passage count). Import success or object instantiation is not proof of live enforcement.
- **Cached with TTL**: Probes are cached for 15 seconds; older entries transition to `freshness="stale"`.

---

## 3. ModuleDescriptor Contract

Each registered module declares:

```python
class ModuleDescriptor(BaseModel):
    id: str                                  # e.g. "hestia", "phylax"
    display_name: str                        # Human-readable title
    section: str                             # Dashboard section ID
    config_path: Optional[str]               # Path to primary config file
    config_format: Literal["yaml", "toml", "json", "pydantic", "none"]
    schema_ref: str                          # Name of validation schema
    restart_required: List[str]              # Keys that require process restart
    mutable: bool                            # Whether settings can be edited
```

---

## 4. Control Plane Endpoints (`/v1/admin/*`)

All admin endpoints require bearer token authentication:

### 1. `GET /v1/admin/overview`
Returns the global status array, active alerts (e.g. port clashes, uncommitted configuration drift), and budget DB persistence state (`persistent` vs `memory-only`).

### 2. `GET /v1/admin/modules`
Returns the list of all registered `ModuleDescriptor` objects.

### 3. `GET /v1/admin/modules/{id}/config`
Returns the `ConfigView` for a specific module:
- `saved`: Raw dictionary read from the on-disk configuration file.
- `effective`: Values acknowledged by the active runtime instance.
- `source`: Key-by-key provenance (`default`, `file`, `env`, `cli`, `override`).
- `config_hash`: SHA-256 digest of the saved configuration.
- `drift_detected`: Boolean indicating if `saved != effective`.

### 4. `POST /v1/admin/modules/{id}/config/validate`
Performs a dry-run validation using the module's native validator without disk mutation. Returns error lists or deprecation warnings.

### 5. `POST /v1/admin/modules/{id}/config/apply`
Applies a configuration patch using atomic temporary-file writes and renames.
- Requires `base_hash` matching current `config_hash` (optimistic locking).
- Requires `confirmation_token` for sensitive keys (e.g. changing listen hosts, widening Phylax policies).
- Emits a `ChangeSet` record with status `written` and `restart_pending=True`.

### 6. `POST /v1/admin/modules/{id}/config/rollback/{changeset_id}`
Restores configuration from a previous historical snapshot and updates ChangeSet history.
