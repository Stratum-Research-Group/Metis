# Phylax: Dashboard Implementation Specification & Guardrail Monitoring

This specification outlines the telemetry, policy inspection, approval cards, and audit chain verification exposed by **Phylax** for the Metis Control Dashboard (Section 5: *Guardrails*).

---

## 1. Architectural Role in Dashboard

Phylax provides the security posture and interactive capability telemetry for the dashboard:
- **Active Policy Inspection**: Displays the loaded YAML policy, SHA-256 integrity hash, rule count, and default-deny status.
- **Audit Hash-Chain Verification**: Displays the status of the append-only cryptographic audit chain (`.phylax/audit.jsonl`), including total logged events and tamper verification.
- **Pending Approvals Manager**: Surfaces open capability requests awaiting owner decision with live expiration countdowns.
- **Policy Linter & Editor**: Provides static linting (`phylax validate`) before allowing policy updates.

```
[Dashboard Guardrails Section]
         │
         ├── GET /v1/admin/modules/phylax/config
         ├── GET /v1/admin/modules/phylax/audit/status
         ├── GET /v1/admin/modules/phylax/approvals
         ▼
[Phylax Policy Engine & Audit Chain]
```

---

## 2. Module Descriptor for Phylax

```json
{
  "id": "phylax",
  "display_name": "Phylax (Security Guardrails)",
  "section": "guardrails",
  "config_format": "yaml",
  "config_path": "phylax/policies/default_policy.yaml",
  "schema_ref": "PhylaxPolicy",
  "mutable": true,
  "restart_required_fields": ["audit_log_path", "redaction"],
  "hot_reloadable_fields": ["rules"]
}
```

---

## 3. ComponentStatus Probe Specification

The probe verifies policy parsing, audit chain cryptographic continuity, and gateway readiness:

```python
async def probe_phylax() -> ComponentStatus:
    try:
        policy_path = Path("phylax/policies/default_policy.yaml")
        if not policy_path.exists():
            return ComponentStatus(
                state="offline",
                detail="Policy file missing",
                freshness="fresh",
                checked_at=datetime.utcnow().isoformat(),
                evidence="Policy path not found"
            )
            
        policy_mgr = PolicyManager(str(policy_path))
        rule_count = len(policy_mgr.policy.rules)
        
        # Verify audit chain integrity
        audit_file = Path(".phylax/audit.jsonl")
        audit_ok = True
        event_count = 0
        if audit_file.exists():
            verifier = AuditVerifier(str(audit_file))
            audit_ok = verifier.verify_chain()
            event_count = verifier.count_records()
            
        state = "live" if audit_ok else "degraded"
        detail = f"Policy loaded ({rule_count} rules); Audit chain: {'valid' if audit_ok else 'corrupt'} ({event_count} events)"
        
        return ComponentStatus(
            state=state,
            detail=detail,
            freshness="fresh",
            version="1.0.0",
            checked_at=datetime.utcnow().isoformat(),
            evidence=f"Policy SHA-256: {policy_mgr.policy_hash[:8]}, Audit chain valid={audit_ok}"
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

## 4. Guardrails Section Schema

The dashboard displays:
1. **Security Posture Overview**:
   - Default Deny Enforcement (Active / Inactive)
   - Policy File Path & Cryptographic SHA-256 Checksum
   - Audit Log Hash Chain Status (Verified / Disrupted)
   - Active Redaction Patterns (`[token]`, `[secret]`, `[password]`)
2. **Rule Matrix**:
   - Table of rules: Subject, Verb, Resource Class, Path/Destination, Effect (`ALLOW`, `DENY`, `ASK`), Priority.
   - Shadowed / Unreachable Rule Warnings.
3. **Pending Capability Approvals**:
   - Interactive Approval Cards with countdown timer.
   - Action Target, Caller Subject, Content Digest.
   - Buttons: `Allow Once` / `Deny`.
4. **Policy Editor with Diff & Validation**:
   - Dry-run validation via `phylax validate` before applying.
   - Mandatory Owner Confirmation Token required to apply policy changes.
