# Dokime: Dashboard Implementation Specification & Evaluation Telemetry

This specification outlines the telemetry, benchmark run history, and release gate metrics exposed by **Dokime** for the Metis Control Dashboard (Section 6: *Evaluation*).

---

## 1. Architectural Role in Dashboard

Dokime provides benchmark execution, statistical comparison, and release gate telemetry:
- **Evaluation History**: Lists historical evaluation runs, tested models, task suites (e.g. ARC, GSM8K, MuSiQue), and sample sizes.
- **Statistical Release Gates**: Visualizes 10,000 paired bootstrap confidence intervals, null hypothesis $p$-values, and win/loss matrices against baselines.
- **Run Manifest Verifier**: Validates immutable run manifests, git commit provenance, and model weights checksums.

```
[Dashboard Evaluation Section]
         │
         ├── GET /v1/admin/modules/dokime/config
         ├── GET /v1/admin/modules/dokime/runs
         ├── GET /v1/admin/modules/dokime/runs/{run_id}
         ▼
[Dokime Evaluation Runner & Manifest Store]
```

---

## 2. Module Descriptor for Dokime

```json
{
  "id": "dokime",
  "display_name": "Dokime (Evaluation Harness)",
  "section": "evaluation",
  "config_format": "toml",
  "config_path": "dokime/examples/config.example.toml",
  "schema_ref": "DokimeConfig",
  "mutable": false,
  "restart_required_fields": ["runs_dir", "hestia_url"],
  "hot_reloadable_fields": []
}
```

---

## 3. ComponentStatus Probe Specification

The probe checks benchmark runner availability and runs directory integrity:

```python
async def probe_dokime() -> ComponentStatus:
    try:
        runs_dir = Path("runs/dokime")
        total_runs = len(list(runs_dir.glob("*.json"))) if runs_dir.exists() else 0
        
        # Verify Hestia connectivity for evaluations
        hestia_ok = False
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                r = await client.get("http://localhost:8080/health")
                hestia_ok = (r.status_code == 200)
        except Exception:
            hestia_ok = False
            
        state = "live" if hestia_ok else "degraded"
        detail = f"Evaluation harness ready ({total_runs} historical runs); Hestia: {'reachable' if hestia_ok else 'offline'}"
        
        return ComponentStatus(
            state=state,
            detail=detail,
            freshness="fresh",
            version="0.1.0",
            checked_at=datetime.utcnow().isoformat(),
            evidence=f"runs_dir present={runs_dir.exists()}; Hestia health={hestia_ok}"
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

## 4. Evaluation Section Schema (Dokime Telemetry)

The dashboard displays:
1. **Recent Benchmark Runs**:
   - Run ID, Execution Timestamp, Evaluated Model & Backend.
   - Task Suite & Item Count ($N$).
   - Primary Accuracy / Pass Rate.
2. **Statistical Comparison & Release Gates**:
   - Candidate Model vs Baseline Model.
   - Metric Delta ($\Delta$).
   - 95% Bootstrap Confidence Interval $[CI_{\text{lower}}, CI_{\text{upper}}]$.
   - Release Gate Status (`PASS` if $CI_{\text{lower}} > 0$ and $p < 0.05$, else `REJECT`).
3. **Artifact Provenance**:
   - Model Weights SHA-256 Digest.
   - Code Commit Hash & Git Tree Status.
