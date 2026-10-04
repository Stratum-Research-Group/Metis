# Stoma

> **Ingestion pipeline and mouth for local AI topological memory (LTHCM)**

Stoma feeds the memory of a local AI stack. While **LTHCM** (topological memory) is the memory substrate, Stoma is the mouth: it takes local documents and media, parses them into passages with granular location information, and indexes them into LTHCM with durable provenance.

---

## Hard Constraints & Architecture

- **100% Local at Runtime**: No cloud parsers, no hosted OCR, no external APIs. User documents, transcripts, passages, and embeddings never leave the local machine.
- **Pinned Git Submodules**:
  * **LTHCM**: `https://gitlab.com/akarta-315h/LTHCM` @ `336f4af9f624aca5b4978a5e3ce222ce9638da35`
  * **dianoia**: `https://gitlab.com/akarta-315h/dianoia` @ `da7ce83cb05f3b44764fc59ab5c325995c6fd07c`
  * **hestia**: `https://gitlab.com/akarta-315h/hestia` @ `edd1bcf411180dd3c9b1171057d03f400b3bfb96`
  * **dokime**: `https://gitlab.com/akarta-315h/dokime` @ `43c025348098aa9511f56acfb26ed19a62983177`
- **Submodules are strictly read-only**: Never modified or copied into Stoma's source tree.
- **Narrow LTHCM Adapter**: Encapsulates LTHCM engine interactions, enforcing shape validation and provenance tracking.

---

## Locked Architectural Decisions

### 1. Encoder & Single-Facet Geometry
- **Runtime Embedder**: Stoma imports the pinned `dianoia.embedding.TextEmbedder` at runtime (`sentence-transformers/all-MiniLM-L6-v2`).
- **Dynamic Dimension**: `d_model` is read dynamically from the loaded model (`384`).
- **Single-Facet Shape**: `facet_vectors = embedding.unsqueeze(0)` with locked shape `[1, d_model]`, matching dianoia's engine construction `LTHCMEngine(d_model=embedder.d_model, num_facets=1)`.
- **Loud Mismatch Failure**: LTHCM's superposition store silently pads or truncates mismatched facets/dimensions. The Stoma adapter actively validates `engine.d_model == embedder.d_model` and `num_facets == 1`, throwing `ShapeMismatchError` immediately upon any discrepancy.

### 2. Measured Model Identity & Token-Bounded Chunking
- Stoma inspects the loaded encoder's tokenizer and limits at setup without guessing:
  - Model: `sentence-transformers/all-MiniLM-L6-v2`
  - Revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
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

### 5. Offline Runtime Enforcement & Strict Submodule Pinning
- Offline environment variables (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`) are enforced before model initialization; model hub network calls are blocked at runtime.
- Startup invokes `verify_runtime_dependencies()` to verify exact submodule commits for `deps/LTHCM` (`336f4af9f624aca5b4978a5e3ce222ce9638da35`) and `deps/dianoia` (`da7ce83cb05f3b44764fc59ab5c325995c6fd07c`), failing loudly on altered commits or wrong origin paths.

### 6. Streaming Ingestion & Dual-Store Integrity
- Ingestion operates via generators (`extract_file_stream` + `chunk_raw_passages_stream`) in bounded embedding batches.
- Explicit 25MB safe size cap on non-streaming AST (`CodePlugin`) and JSON parsers (`OCRPlugin`, `WhisperPlugin`).
- Adapters validate both hyperbolic `documents` (with duplicate ID detection) and superposition `doc_index`, ensuring 100% parity with manifest passage IDs.

### 7. Dianoia Interchange Compatibility
- Pinned Dianoia CLI (`deps/dianoia/dianoia/cli.py:206-219`) requires `graph_edges > 0` for its deliberative walk. Stoma v1 has zero graph edges (graph mining is out of scope for v1).
- Stoma exercises the actual Dianoia loader under controlled subprocess invocation, captures the real edge-gate rejection, and accurately reports `direct_cli_walk_unsupported_zero_edges` (exit code 1).
- Stoma's index matches Dianoia's serialization format (`pickle`), `d_model` (384), and single-facet geometry. Subprocess failures (e.g. timeout, missing module) are distinguished from edge gate refusal and reported honestly.

---

## Supported Source Plugins

| Plugin | Formats | Features |
|---|---|---|
| **Text & Markdown** | `.txt`, `.md`, `.markdown`, `.rst` | Structure-aware heading preservation, paragraph splitting, line number ranges, streaming extraction. |
| **PDF** | `.pdf` | Local parsing via `pypdf` streaming page-by-page. Distinguishes text PDFs from scanned/image-only PDFs (reports image-only as unsupported). |
| **Code Repositories** | `.py`, `.js`, `.ts`, `.go`, `.rs`, `.c`, `.cpp`, `.java`, etc. | AST-based function/class splitting for Python; structural regex boundaries for other languages. Safe 25MB parser cap. |
| **OCR Hook** | `.ocr.json`, `.ocr.txt`, `ocr/` | Consumes pre-extracted OCR artifacts from external pipelines without running OCR itself. Safe 25MB JSON cap. |
| **Whisper Transcripts** | `.vtt`, `.srt`, `.transcript.json`, `.transcript.txt` | Parses timestamped audio transcripts from `whisper.cpp`. Safe 25MB JSON cap. |

---

## Installation & Setup

```bash
# 1. Clone repository with submodules
git clone --recursive <repo-url> stoma
cd stoma

# 2. Set up virtual environment using uv or python
uv venv --python 3.11 .venv
source .venv/bin/activate

# 3. Install dependencies and local submodules
uv pip install -e deps/LTHCM -e deps/dianoia
uv pip install -e .

# 4. Optional: Prefetch embedding model cache ahead of offline runtime
stoma setup-model
# or: python -m stoma.embedder prefetch
```

---

## CLI Usage

### Setup & Model Prefetch
```bash
# Download and cache SentenceTransformer weights while online
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

## Running Tests

```bash
pytest -v
```
All **90 unit, integration, and regression tests** validate:
- Source plugins & image-only PDF detection with typed exception hierarchy
- Token bounding and deduplication
- Adapter shape validation and loud failure
- Causal light-cone filtering (past document present, future excluded, no fallback)
- Lifecycle (unchanged skipped, edited swapped, deleted purged)
- Persistence and restart durability with mismatched model rejection
- Symlink exclusion outside allowed roots
- Direct Dianoia interchange & edge refusal reporting
- Round 1, Round 2, Round 3, and Round 4 regression suites:
  * Single atomic commit point (`current.json` swap) and monotonic prefix recovery from `commit_chain.json`
  * Watcher retries for moves, deletions, and ingests with dead-letter queue and capped error memory
  * Explicit offline cache miss naming setup step and weights checksum verification
  * Dual submodule verification: top-level gitlink and checked-out HEAD
  * Streaming rebuild without full chunk accumulation or duplicate extract passes
  * Subprocess Dianoia interchange capturing real returncode 1 and edge refusal
  * Dual-store unconditional equality enforcement
  * Mid-extraction modification detection and transactional rollback
  * Strict file descriptor and directory fsync durability semantics
  * Robust watcher event scheduling with bounded exponential backoff
