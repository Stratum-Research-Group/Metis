# LTHCM: Dashboard Implementation Specification & Memory Substrate Monitoring

This specification outlines the telemetry, non-Euclidean manifold statistics, and retrieval metrics exposed by **LTHCM** for the Metis Control Dashboard (Section 3: *Memory & Ingestion*).

---

## 1. Architectural Role in Dashboard

LTHCM provides topological context memory metrics:
- **Index Dimensions & Geometry**: Embedding dimensions ($d=384$), facet count, Poincaré ball curvature ($-c$), and density matrix rank.
- **Topological Invariant Telemetry**: Cellular Sheaf Dirichlet energy distributions, Lorentzian light-cone rejection counts, and Vietoris-Rips Betti numbers ($\beta_0, \beta_1$).
- **Truthful Memory Status**: Explicitly reports `simulated` when running over the 4-passage in-memory demo corpus, and `live` only when a real Stoma generation is actively mounted and verified.

```
[Dashboard Memory Section]
         │
         ├── GET /v1/admin/modules/lthcm/config
         ├── GET /v1/admin/modules/lthcm/manifold/stats
         ▼
[LTHCM Engine & Topological Manifold Store]
```

---

## 2. Module Descriptor for LTHCM

```json
{
  "id": "lthcm",
  "display_name": "LTHCM (Topological Context Memory)",
  "section": "memory_ingestion",
  "config_format": "pydantic",
  "config_path": "lthcm/config.py",
  "schema_ref": "RetrievalConfig",
  "mutable": false,
  "restart_required_fields": ["d_model", "num_facets", "curvature"],
  "hot_reloadable_fields": []
}
```

---

## 3. ComponentStatus Probe Specification

The probe evaluates whether LTHCM is operating over a real memory root or falling back to the simulated demo corpus:

```python
async def probe_lthcm() -> ComponentStatus:
    try:
        # Inspect active Kybernetes LTHCM adapter
        adapter = kybernetes_app.state.lthcm_adapter
        if adapter is None or adapter.is_simulated:
            return ComponentStatus(
                state="simulated",
                detail="In-memory demo corpus (4 synthetic passages)",
                freshness="fresh",
                version="1.0.0",
                checked_at=datetime.utcnow().isoformat(),
                evidence="is_simulated=True (no external memory root mounted)"
            )
            
        # Verify functional retrieval over real index
        res = adapter.retrieve("system health probe test query", top_k=1)
        return ComponentStatus(
            state="live",
            detail=f"Real memory root mounted ({len(adapter.doc_ids)} documents indexed)",
            freshness="fresh",
            version="1.0.0",
            checked_at=datetime.utcnow().isoformat(),
            evidence=f"Functional query executed; top-1 score={res[0].score if res else 'none'}"
        )
    except Exception as exc:
        return ComponentStatus(
            state="unknown",
            detail=f"Engine probe error: {str(exc)}",
            freshness="fresh",
            checked_at=datetime.utcnow().isoformat(),
            evidence=str(exc)
        )
```

---

## 4. Memory & Ingestion Section Schema (LTHCM Telemetry)

The dashboard displays:
1. **Memory Substrate State**:
   - Mode: `Live (Generational)` vs `Simulated (Demo Corpus)`.
   - Dimension: $D = 384$, Facets = 1.
   - Poincaré Ball Curvature ($c$).
2. **Topological Metrics**:
   - Total Stored POVM Density Matrices.
   - Mean Sheaf Dirichlet Obstruction Energy across recent queries.
   - Lorentzian Light-Cone Anachronisms Blocked.
3. **Graph Topology**:
   - Wormhole Bridge Count & Spectral Chebyshev Refinement Status.
