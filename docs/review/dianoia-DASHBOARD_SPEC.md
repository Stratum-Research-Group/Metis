# Dianoia: Dashboard Implementation Specification & Deliberation Telemetry

This specification outlines the telemetry, walk configuration, and trajectory metrics exposed by **Dianoia** for the Metis Control Dashboard (Section 4: *Reasoning & Deliberation*).

---

## 1. Architectural Role in Dashboard

Dianoia provides deliberative thought graph traversal and multi-hop reasoning metrics:
- **Walk Configuration**: Primary deliberation model (`qwen2.5-7b-instruct`), fallback model (`llama-3.1-8b-instruct`), max step budget, and top-$k$ graph expansion branching.
- **Thought Trajectory Telemetry**: Average steps per walk, expand/backtrack ratio, fallback activation frequency, and sheaf contradiction resolution rate.
- **Walk Log Inspector**: Recent trajectory records (`~/.dianoia/walks.jsonl`), schema version `1.0.0`, and Paideia gold defensibility status.

```
[Dashboard Reasoning Section]
         │
         ├── GET /v1/admin/modules/dianoia/config
         ├── GET /v1/admin/modules/dianoia/walks/recent
         ▼
[Dianoia Walk Controller & Log Archive]
```

---

## 2. Module Descriptor for Dianoia

```json
{
  "id": "dianoia",
  "display_name": "Dianoia (Deliberative Reasoning)",
  "section": "reasoning",
  "config_format": "pydantic",
  "config_path": "dianoia/dianoia/config.py",
  "schema_ref": "DianoiaConfig",
  "mutable": true,
  "restart_required_fields": ["primary_model", "fallback_model", "llm_url"],
  "hot_reloadable_fields": ["max_steps", "top_k", "log_path"]
}
```

---

## 3. ComponentStatus Probe Specification

The probe verifies LLM endpoint connectivity, primary/fallback model availability, and LTHCM memory graph coupling:

```python
async def probe_dianoia() -> ComponentStatus:
    try:
        # Check LLM server availability via /v1/models
        llm_url = "http://localhost:8080/v1"
        models_available = []
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{llm_url}/models")
            if resp.status_code == 200:
                data = resp.json()
                models_available = [m.get("id") for m in data.get("data", [])]
                
        primary_ok = "qwen2.5-7b-instruct" in models_available
        fallback_ok = "llama-3.1-8b-instruct" in models_available
        
        state = "live" if primary_ok else ("simulated" if not models_available else "degraded")
        detail = f"Primary ({'ready' if primary_ok else 'missing'}), Fallback ({'ready' if fallback_ok else 'disabled'})"
        
        return ComponentStatus(
            state=state,
            detail=detail,
            freshness="fresh",
            version="1.0.0",
            checked_at=datetime.utcnow().isoformat(),
            evidence=f"GET {llm_url}/models -> {len(models_available)} models listed"
        )
    except Exception as exc:
        return ComponentStatus(
            state="unknown",
            detail=f"Probe failed: {str(exc)}",
            freshness="fresh",
            checked_at=datetime.utcnow().isoformat(),
            evidence=str(exc)
        )
```

---

## 4. Reasoning Section Schema (Dianoia Telemetry)

The dashboard displays:
1. **Deliberation Configuration**:
   - Primary LLM and Fallback LLM endpoint settings.
   - Max Step Budget (default 12) & Top-$K$ Expansion (default 8).
   - Contradiction Re-walk Allowance (max 1 re-walk).
2. **Trajectory Metrics (from JSONL logs)**:
   - Total Walks Executed.
   - Mean Step Count & Trajectory Length.
   - Conflict Detection Rate & Successful Re-walk Resolution Rate.
   - Fallback Trigger Frequency (Parse Failures / OOMs).
3. **Recent Walk Viewer**:
   - Interactive visual graph view of the latest thought walk.
   - Expand vs Backtrack paths, node reasoning justifications, and Sheaf verification score.
