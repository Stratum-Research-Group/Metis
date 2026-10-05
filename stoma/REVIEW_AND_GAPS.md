# Stoma: Round 1 Defect Analysis, Gaps & Remediation Plan

This document details the defect findings, memory interchange challenges, and parser boundaries identified in `metis-repo-wide-round1-fixes-6752222c.txt` and `metis-dashboard-implementation-spec-ae93d36c.txt` for **Stoma**, along with the exact technical remediation plan and regression tests.

---

## 1. Defect Findings & Root Cause Analysis

### Item 11 (P1): Stoma-to-Memory Contract & Interchange Format
- **Location**: `stoma/src/stoma/indexer.py:597-630`, `stoma/src/stoma/provenance.py:35-49`.
- **Observed Behavior**: Stoma creates generational indices under `generations/gen_{N}/` containing `index.pkl` and `manifest.json`. The passage provenance objects track file hashes, line ranges, and page offsets, but do not embed raw passage text directly into the manifest. Downstream consumers (like Kybernetes `LTHCMAdapter`) expect text fields or raw passage dumps, causing keyword-fallback crashes or `AttributeError` when `passages.json` is a list.
- **Root Cause**: Lack of a standardized, data-only passage text export alongside index pickles.
- **Remediation**:
  1. Export a structured `passages_export.json` alongside every generation commit containing:
     - `passage_id`, `text`, `source_rel_path`, `content_hash`, `location`, `timestamp`, `tokens`.
  2. Implement an atomic read-only consumer API (`StomaGenerationReader`) to cleanly read the active generation from `current.json`.
  3. Ensure corrupt or partial generation pointer files fail loudly with typed exceptions rather than falling back to guessing `gen-1`.
- **Regression Testing**:
  - Test `test_generation_export_contains_passage_text`: Ingest document; verify `passages_export.json` exists in `gen_{N}/` with full text.
  - Test `test_atomic_generation_reader`: Reader loads active generation and verifies SHA-256 hashes against `manifest.json`.

---

### Item 13 (P1): Removal of Untrusted Pickle from Shared Index Loads
- **Location**: `stoma/src/stoma/adapter.py:325-350`, `stoma/src/stoma/interchange.py:37-49`.
- **Observed Behavior**: Saved indices are restored using `pickle.load()` on `index.pkl`.
- **Root Cause**: Reliance on Python pickle serialization across process boundaries.
- **Remediation**:
  1. Enforce strict permissions (`0600`) on generation directories and index files.
  2. Verify index file SHA-256 digests against `commit_marker.json` and `commit_chain.json` *before* loading.
  3. Prepare for SafeTensors / binary array format migration for hyperbolic and superposition vector stores.
- **Regression Testing**:
  - Test `test_tampered_index_hash_rejected_before_load`: Modifying a byte in `index.pkl` triggers hash mismatch failure and aborts load.
