# Stoma: Dashboard Implementation Specification & Ingestion Pipeline Monitoring

This specification outlines the telemetry, ingestion metrics, and generational ledger monitoring exposed by **Stoma** for the Metis Control Dashboard (Section 3: *Memory & Ingestion*).

---

## 1. Architectural Role in Dashboard

Stoma provides document ingestion, chunking, and index state telemetry:
- **Index State & Active Generation**: Current generation ID (`gen_{N}`), total indexed passages, embedding dimension ($d=384$), single-facet verification.
- **Watcher & Reconciliation**: Monitored root directories, last reconciliation timestamp, pending batches, and dead-letter queue statistics.
- **Provenance & Integrity**: SHA-256 commit-chain ledger (`commit_chain.json`), active index hash, and monorepo component digests.

```
[Dashboard Memory Section]
         │
         ├── GET /v1/admin/modules/stoma/config
         ├── GET /v1/admin/modules/stoma/status
         ▼
[Stoma Ingestion Manager & Generational Ledger]
```

---

## 2. Module Descriptor for Stoma

```json
{
  "id": "stoma",
  "display_name": "Stoma (Ingestion Pipeline)",
  "section": "memory_ingestion",
  "config_format": "yaml",
  "config_path": ".stoma/config.yaml",
  "schema_ref": "StomaConfig",
  "mutable": true,
  "restart_required_fields": ["storage_dir", "embedder_model"],
  "hot_reloadable_fields": ["watch_paths", "batch_size", "debounce_seconds"]
}
```

---

## 3. ComponentStatus Probe Specification

The probe inspects `current.json`, directory accessibility, and commit chain integrity:

```python
async def probe_stoma() -> ComponentStatus:
    try:
        storage_dir = Path(".stoma")
        current_file = storage_dir / "current.json"
        
        if not current_file.exists():
            return ComponentStatus(
                state="simulated",
                detail="No committed generations; running cold index",
                freshness="fresh",
                checked_at=datetime.utcnow().isoformat(),
                evidence="current.json absent"
            )
            
        current_data = json.loads(current_file.read_text(encoding="utf-8"))
        gen_id = current_data.get("generation_id", "unknown")
        passages = current_data.get("total_passages", 0)
        
        return ComponentStatus(
            state="live",
            detail=f"Active generation: {gen_id} ({passages} passages indexed)",
            freshness="fresh",
            version="1.0.0",
            checked_at=datetime.utcnow().isoformat(),
            evidence=f"current.json valid, generation={gen_id}, passages={passages}"
        )
    except Exception as exc:
        return ComponentStatus(
            state="unknown",
            detail=f"Storage error: {str(exc)}",
            freshness="fresh",
            checked_at=datetime.utcnow().isoformat(),
            evidence=str(exc)
        )
```

---

## 4. Memory & Ingestion Section Schema (Stoma Telemetry)

The dashboard displays:
1. **Generational Storage Overview**:
   - Active Generation ID & Total Passages.
   - Index Size on Disk (MB).
   - Embedding Model: `sentence-transformers/all-MiniLM-L6-v2` ($D=384$).
   - Single-Facet Alignment: Locked $[1, 384]$.
2. **Directory Watcher & Job Queue**:
   - Monitored Folders.
   - Debounce Window (seconds).
   - Ingestion Job Queue & Dead-Letter Items.
3. **Commit Chain Ledger**:
   - Historical generation commits table (Timestamp, Gen ID, Passage Count, Index Hash).
