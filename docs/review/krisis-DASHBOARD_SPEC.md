# Krisis: Dashboard Implementation Specification & Answer Selection Telemetry

This specification outlines the telemetry, policy profiles, and quality verification statistics exposed by **Krisis** for the Metis Control Dashboard (Section 4: *Reasoning & Verification*).

---

## 1. Architectural Role in Dashboard

Krisis provides answer eligibility and factual grounding metrics:
- **Policy Manifest Status**: Displays the active policy manifest (`krisis-policy-0.1.json`), profile thresholds ($\tau$, `per_claim_min`), and reason precedence rules.
- **Verification Outcomes**: Tracks the distribution of recent answer evaluations (survivors selected, quality below $\tau$, missing citations, unsupported claims, typed abstentions).
- **Policy Linting & Verification**: Allows dry-run schema validation against `krisis-policy.schema.json`.

```
[Dashboard Reasoning Section]
         │
         ├── GET /v1/admin/modules/krisis/config
         ├── GET /v1/admin/modules/krisis/policy
         ▼
[Krisis Policy Engine & Evaluator]
```

---

## 2. Module Descriptor for Krisis

```json
{
  "id": "krisis",
  "display_name": "Krisis (Candidate Selector)",
  "section": "reasoning",
  "config_format": "json",
  "config_path": "krisis/krisis-policy-0.1.json",
  "schema_ref": "KrisisPolicy",
  "mutable": true,
  "restart_required_fields": ["manifest_id", "schema_version"],
  "hot_reloadable_fields": ["profiles", "reason_precedence"]
}
```

---

## 3. ComponentStatus Probe Specification

The probe validates policy syntax, profile definitions, and test evaluation determinism:

```python
async def probe_krisis() -> ComponentStatus:
    try:
        policy_path = Path("krisis/krisis-policy-0.1.json")
        schema_path = Path("krisis/krisis-policy.schema.json")
        
        if not policy_path.exists():
            return ComponentStatus(
                state="offline",
                detail="Policy file not found",
                freshness="fresh",
                checked_at=datetime.utcnow().isoformat(),
                evidence=str(policy_path)
            )
            
        policy_data = json.loads(policy_path.read_text(encoding="utf-8"))
        schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=policy_data, schema=schema_data)
        
        manifest_id = policy_data.get("manifest_id", "unknown")
        status_label = policy_data.get("status", "draft")
        profiles_count = len(policy_data.get("profiles", {}))
        
        return ComponentStatus(
            state="live",
            detail=f"Policy {manifest_id} ({status_label}, {profiles_count} profiles)",
            freshness="fresh",
            version=policy_data.get("schema_version", "0.1.0"),
            checked_at=datetime.utcnow().isoformat(),
            evidence=f"JSON Schema valid; manifest_id={manifest_id}"
        )
    except Exception as exc:
        return ComponentStatus(
            state="unknown",
            detail=f"Validation error: {str(exc)}",
            freshness="fresh",
            checked_at=datetime.utcnow().isoformat(),
            evidence=str(exc)
        )
```

---

## 4. Reasoning Section Schema (Krisis Telemetry)

The dashboard displays:
1. **Policy Configuration**:
   - Active Manifest ID and Status (`draft` vs `sealed`).
   - Profile Thresholds:
     - `grounded_v1`: $\tau = 0.700000$, $\text{per\_claim\_min} = 0.700000$.
     - `grounded_support_only`: $\tau = 0.850000$, $\text{per\_claim\_min} = 0.700000$.
     - `eligibility_baseline`: Baseline selection without logprob scoring.
   - Reason Precedence Order.
2. **Evaluation Telemetry**:
   - Total Turns Evaluated.
   - Pre-score Elimination Rate (Access, Provenance, Contradictions, Unresolved Citations).
   - Mean Score of Selected Candidates vs $\tau$ margin.
