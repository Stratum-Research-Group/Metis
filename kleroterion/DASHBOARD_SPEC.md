# Kleroterion: Dashboard Implementation Specification & Stochastic Policy Telemetry

This specification outlines the telemetry, policy profiles, and entropy diagnostics exposed by **Kleroterion** for the Metis Control Dashboard (Section 4: *Reasoning & Deliberation*).

---

## 1. Architectural Role in Dashboard

Kleroterion provides stochastic action proposal and entropy calibration metrics:
- **Active Profile**: Displays the active deliberation profile (`exact_fact`, `exploratory`, `adaptive`).
- **Entropy & Temperature Diagnostics**: Tracks smoothed marginal entropy, current temperature ($\tau$), and mode collapse warnings.
- **Proposal Gating**: Reports authoritative action mask compliance and proposal count.

```
[Dashboard Reasoning Section]
         │
         ├── GET /v1/admin/modules/kleroterion/config
         ├── GET /v1/admin/modules/kleroterion/diagnostics
         ▼
[Kleroterion Stochastic Proposal Engine]
```

---

## 2. Module Descriptor for Kleroterion

```json
{
  "id": "kleroterion",
  "display_name": "Kleroterion (Stochastic Selector)",
  "section": "reasoning",
  "config_format": "pydantic",
  "config_path": "kleroterion/src/kleroterion/interfaces.py",
  "schema_ref": "KleroterionConfig",
  "mutable": true,
  "restart_required_fields": ["proposal_mode", "selection_mode"],
  "hot_reloadable_fields": ["temperature_initial", "noise_rms_budget", "adapt_temperature"]
}
```

---

## 3. ComponentStatus Probe Specification

The probe inspects adapter readiness and profile configuration:

```python
async def probe_kleroterion() -> ComponentStatus:
    try:
        # Check if Dianoia adapter is loaded
        adapter_active = hasattr(kybernetes_app.state, "kleroterion_adapter")
        state = "live" if adapter_active else "simulated"
        detail = "Adapter active (stochastic proposals enabled)" if adapter_active else "Opt-in submodule (dormant in default walk)"
        
        return ComponentStatus(
            state=state,
            detail=detail,
            freshness="fresh",
            version="0.1.0",
            checked_at=datetime.utcnow().isoformat(),
            evidence=f"kleroterion_adapter presence={adapter_active}"
        )
    except Exception as exc:
        return ComponentStatus(
            state="unknown",
            detail=f"Probe error: {str(exc)}",
            freshness="fresh",
            checked_at=datetime.utcnow().isoformat(),
            evidence=str(exc)
        )
```

---

## 4. Reasoning Section Schema (Kleroterion Telemetry)

The dashboard displays:
1. **Stochastic Configuration**:
   - Active Policy Profile: `EXACT_FACT` (locked $T=0, \sigma=0$) vs `EXPLORATORY`.
   - Temperature Scaling: Initial $\tau$, Min $\tau$, Max $\tau$.
   - Dimension Scaling: $1/\sqrt{d}$ normalization status.
2. **Entropy Diagnostics**:
   - Smoothed Marginal Entropy ($H(\bar{\pi})$).
   - Target Entropy vs Empirical Entropy delta.
   - Temperature Adaptation Trajectory.
