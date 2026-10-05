# Stoma (Στόμα)

> **Ingestion pipeline and mouth for local AI topological memory (LTHCM)**

**Stoma** (Greek: *mouth*) is the ingestion pipeline and document feeder for the Metis local AI stack. While **LTHCM** is the topological memory substrate, Stoma is the mouth: it takes local documents, source code, and media transcripts, parses them into structured passages with granular location information, and indexes them into LTHCM with durable provenance and light-cone causal timestamps.

---

## 1. Hard Constraints & Core Principles

- **100% Local at Runtime**: No cloud parsers, no hosted OCR, no external APIs. User documents, transcripts, passages, and embeddings never leave the local machine.
- **Offline Runtime Enforcement**: Offline environment variables (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`) are enforced at startup; model hub network calls are blocked.
- **Honest Provenance & Monorepo Verification**:
  - In submodule configurations: Verifies pinned git commits for `deps/LTHCM` and `deps/dianoia`.
  - In integrated Metis monorepo configurations: Verifies cryptographic SHA-256 directory tree digests (`compute_component_digest()`) recorded in `vendored_provenance.json` (or `release_manifest.json`), strictly scoped to the Stoma root, failing closed if provenance is altered or missing.
- **Narrow LTHCM Adapter**: Encapsulates LTHCM engine interactions, enforcing shape validation and provenance tracking.

---

## 2. Locked Architectural Decisions

### 1. Encoder & Single-Facet Geometry
- **Runtime Embedder**: Stoma imports the pinned `dianoia.embedding.TextEmbedder` at runtime (`sentence-transformers/all-MiniLM-L6-v2`).
- **Dynamic Dimension**: `d_model` is read dynamically from the loaded model (`384`).
- **Single-Facet Shape**: `facet_vectors = embedding.unsqueeze(0)` with locked shape `[1, d_model]`, matching dianoia's engine construction `LTHCMEngine(d_model=embedder.d_model, num_facets=1)`.
- **Loud Mismatch Failure**: LTHCM's superposition store silently pads or truncates mismatched facets/dimensions. The Stoma adapter actively validates `engine.d_model == embedder.d_model` and `num_facets == 1`, throwing `ShapeMismatchError` immediately upon any discrepancy.

### 2. Measured Model Identity & Token-Bounded Chunking
- Stoma inspects the loaded encoder's tokenizer and limits at setup without guessing:
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Weights Checksum (state_dict SHA-256): `2c705a5e3279febcd4b210fc95a6231ced10be71e400e863ab41b623d5799508`
  - Tokenizer: `BertTokenizer` (`transformers.models.bert.tokenization_bert.BertTokenizer`)
  - Raw `max_seq_length`: `256`
  - Special token overhead: `2` (`[CLS]`, `[SEP]`)
  - Effective max tokens: `254`
- All chunks are strictly bounded by token count before embedding ($\le 254$ effective tokens), guaranteeing **zero silent truncation** by sentence-transformers. Unbroken strings of arbitrary length are binary-split without discarding characters.
- Duplicate and empty chunks are filtered.

### 3. Stack-Wide Timestamp Contract
- `index_document` is called with `timestamp = time.time()` (epoch seconds).
- `retrieve_and_verify` is called with `query_timestamp = time.time()` (current epoch seconds).
- **Critical Contract Notice**: LTHCM's default `query_timestamp` is `100.0` (`pipeline.py:168`). When epoch-second document timestamps are queried with `query_timestamp=100.0`, all documents fall outside the past Lorentzian light cone ($dt = 100.0 - \text{epoch} \ll 0$), silently triggering `causal_fallback = True` and bypassing causality. Any consumer of this index (including Dianoia) must supply real epoch timestamps.

### 4. Generational Commit-Chain & Crash Durability
- LTHCM's hyperbolic store appends on every `add_concept` without same-id deduplication (`lthcm/core/hyperbolic.py:187-193`).
- Stoma manages every mutation as an isolated generation transaction (`generations/gen_{N}/` containing `index.pkl` and `manifest.json`), committed via a durable protocol:
  * Stage files into temporary staging directory with fsync.
  * Atomic directory rename to `gen_{N}/`.
  * Durable commit marker (`commit_marker.json`) written inside the generation folder with fsync.
  * Append entry to persistent generational ledger (`commit_chain.json`) with fsync.
  * Atomic swap of `current.json` containing recorded `index_hash`, `manifest_hash`, and candidate counts.
- Only provably committed generations may be recovered on startup; uncommitted orphan directories are strictly rejected.
- Failed extractions, mid-stream batch errors, or serialization failures leave the serving generation completely untouched.
- File moves/renames are transactional: removing the old ID set and creating the new path under a new ID set, preventing stale references or duplicate ghosts.

### 5. Streaming Ingestion & Dual-Store Integrity
- Ingestion operates via generators (`extract_file_stream` + `chunk_raw_passages_stream`) in bounded embedding batches.
- Explicit 25MB safe size cap on non-streaming AST (`CodePlugin`) and JSON parsers (`OCRPlugin`, `WhisperPlugin`).
- Adapters validate both hyperbolic `documents` (with duplicate ID detection) and superposition `doc_index`, ensuring 100% parity with manifest passage IDs.

### 6. Dianoia Interchange Compatibility
- Pinned Dianoia CLI requires `graph_edges > 0` for its deliberative walk. Stoma v1 has zero graph edges (graph mining is out of scope for v1).
- Stoma exercises the actual Dianoia loader under controlled subprocess invocation, captures the real edge-gate rejection, and accurately reports `direct_cli_walk_unsupported_zero_edges` (exit code 1).
- Stoma's index matches Dianoia's serialization format (`pickle`), `d_model` (384), and single-facet geometry.

---

## 3. Supported Source Plugins

| Plugin | Formats | Features |
|---|---|---|
| **Text & Markdown** | `.txt`, `.md`, `.markdown`, `.rst` | Structure-aware heading preservation, paragraph splitting, line number ranges, streaming extraction. |
| **PDF** | `.pdf` | Local parsing via `pypdf` streaming page-by-page. Distinguishes text PDFs from scanned/image-only PDFs (reports image-only as unsupported). |
| **Code Repositories** | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.c`, `.cpp`, `.java`, etc. | AST-based function/class splitting for Python; structural regex boundaries for other languages. Safe 25MB parser cap. |
| **OCR Hook** | `.ocr.json`, `.ocr.txt`, `ocr/` | Consumes pre-extracted OCR artifacts from external pipelines without running OCR itself. Safe 25MB JSON cap. |
| **Whisper Transcripts** | `.vtt`, `.srt`, `.transcript.json`, `.transcript.txt` | Parses timestamped audio transcripts from `whisper.cpp`. Safe 25MB JSON cap. |

---

## 4. CLI Usage

### Setup & Model Prefetch
```bash
# Prefetch and verify SentenceTransformer model weights
stoma setup-model
```

### Ingest Documents
```bash
# Ingest single file (streaming bounded batches)
stoma ingest /path/to/document.md

# Scan and ingest entire directory (bounded batches)
stoma ingest /path/to/folder
```

### Query Topological Memory
```bash
stoma query "quantum superposition memory" --top-k 5
```
Every query result displays:
- Passage ID
- Source relative path
- File content SHA-256 hash
- Granular location (page / start and end lines / timestamps)
- Ingestion timestamp (epoch seconds + UTC formatted)
- Causal light-cone verification status (`causal_fallback: False`)

### Watch & Reconcile
```bash
# Start background directory watcher with debounced writes, retries, and dead-letter tracking
stoma watch /path/to/folder

# Perform full manual reconciliation against disk with detailed failure reporting
stoma reconcile
```

### System Status & Interchange
```bash
# View storage, model info, and passage counts
stoma status

# Verify direct interchange against Dianoia
stoma interchange
```

---

## 5. Running Tests

```bash
uv run pytest stoma/tests -v --cov=stoma
```
All unit, integration, and regression tests validate:
- Source plugins & image-only PDF detection with typed exception hierarchy
- Token bounding and deduplication
- Adapter shape validation and loud failure
- Causal light-cone filtering (past document present, future excluded, no fallback)
- Generational commit durability and atomic current swap
- Honest monorepo provenance verification (`vendored_provenance.json`)
