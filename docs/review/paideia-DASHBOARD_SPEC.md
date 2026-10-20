# Paideia: Dashboard Implementation Specification & Training Pipeline Monitoring

This specification outlines the telemetry, dataset audit metrics, and adapter registry status exposed by **Paideia** for the Metis Control Dashboard (Section 7: *Training*).

---

## 1. Architectural Role in Dashboard

Paideia provides evidence-to-policy dataset auditing and adapter registry metrics:
- **Training Mode & GPU Admission**: Clearly displays training runner status (`SIMULATED` vs `LIVE`), VRAM headroom ($\ge 12$ GB required), and exclusive lock status (`/tmp/metis_gpu.lock`).
- **Dataset Audit Statistics**: Surfaces counts of parsed vs excluded Dianoia walk trajectories, counterfactual routing pairs, and leak-free split checksums.
- **Baseline Ladder Telemetry**: Compares baseline models (Majority, Regex, Lexical, Small Embedding) across accuracy, p95 latency, and memory footprint.
- **Adapter Registry**: Displays candidate and promoted adapters, base model compatibility, and weights SHA-256 digests.

```
[Dashboard Training Section]
         │
         ├── GET /v1/admin/modules/paideia/config
         ├── GET /v1/admin/modules/paideia/audit/summary
         ├── GET /v1/admin/modules/paideia/registry
         ▼
[Paideia Training Runner & Adapter Registry]
```

---

## 2. Module Descriptor for Paideia

```json
{
  "id": "paideia",
  "display_name": "Paideia (Training & Evidence Pipeline)",
  "section": "training",
  "config_format": "pydantic",
  "config_path": "paideia/paideia/config.py",
  "schema_ref": "PaideiaConfig",
  "mutable": false,
  "restart_required_fields": ["adapter_registry_path", "output_dir"],
  "hot_reloadable_fields": []
}
```

---

## 3. ComponentStatus Probe Specification

The probe evaluates training harness dependencies and adapter registry validity:

```python
async def probe_paideia() -> ComponentStatus:
    try:
        registry_path = Path("runs/adapter_registry.json")
        has_registry = registry_path.exists()
        
        # Check training runner status
        # Note: Current pipeline is marker-only (SIMULATED)
        state = "simulated"
        detail = "Evidence pipeline active; Training runner is marker-only (SIMULATED)"
        
        return ComponentStatus(
            state=state,
            detail=detail,
            freshness="fresh",
            version="0.1.0",
            checked_at=datetime.utcnow().isoformat(),
            evidence=f"Registry present={has_registry}; status=SIMULATED"
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

## 4. Training Section Schema (Paideia Telemetry)

The dashboard displays:
1. **Pipeline & Hardware Status**:
   - Status: `SIMULATED (Marker-Only)`.
   - VRAM Admission Status & Lock State (`/tmp/metis_gpu.lock`).
2. **Dataset Audit Summary**:
   - Ingested Dianoia Walks & Defensible Trajectory Count.
   - Excluded Trajectories (Parse Failures, Missing Verification, Unresolved Conflicts).
   - Ingested LTHCM Router Outcomes & Counterfactual Route Pairs.
3. **Baseline Ladder Leaderboard**:
   - Comparison Table: Model (Majority, Regex, Lexical, Embedding, QLoRA Candidate), Metric, p95 Latency, Peak Memory.
   - Statistical Promotion Gates: 95% Bootstrap CI and $p$-value.
4. **Adapter Registry Catalog**:
   - Adapter ID, Target Task, Base Model, SHA-256 Digest, Status (`Candidate`, `Promoted`, `Simulated`).
