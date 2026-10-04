# Stoma Build & Verification Report (Round 3 Acceptance)

> **Verification and Acceptance Report for Stoma Build Specification & Round 3 Review**  
> Timestamp: `2026-09-27`  
> Repository: `stoma` (GitLab: `https://gitlab.com/akarta-315h/stoma`)  
> Base Commit: `dd41bb5b6ed6371d6b54b992fb94bd09d648bb0f`  

---

## 1. Verified Dependency & Submodule Pins

All submodules are pinned to the verified commits requested in the specification and locked contracts:

| Submodule | Pinned SHA | Verified Status |
|---|---|---|
| **LTHCM** | `336f4af9f624aca5b4978a5e3ce222ce9638da35` | Pinned & Verified (`deps/LTHCM`) |
| **dianoia** | `da7ce83cb05f3b44764fc59ab5c325995c6fd07c` | Pinned & Verified (`deps/dianoia`) |
| **hestia** | `edd1bcf411180dd3c9b1171057d03f400b3bfb96` | Pinned & Verified (`deps/hestia`) |
| **dokime** | `43c025348098aa9511f56acfb26ed19a62983177` | Pinned & Verified (`deps/dokime`) |

Gitlinks in `.gitmodules` and git tree match these commit SHAs exactly. Submodules are treated as read-only dependencies without copying code or modifying submodule trees. At runtime, `verify_runtime_dependencies()` dynamically confirms that:
- Top-level parent gitlinks match locked commits (`lthcm_gitlink_sha == LOCKED_LTHCM_COMMIT`, `dianoia_gitlink_sha == LOCKED_DIANOIA_COMMIT`).
- Checked out submodules match locked commits (`lthcm_checkout_sha == LOCKED_LTHCM_COMMIT`, `dianoia_checkout_sha == LOCKED_DIANOIA_COMMIT`).
- Imported packages originate strictly from `deps/LTHCM` and `deps/dianoia` checkouts, raising `DependencyOriginError` prefixed with `"Runtime import violation:"` if any drift occurs.

---

## 2. Measured Model Identity & Encoder Configuration

- **Encoder Source**: Imported directly from pinned `dianoia.embedding.TextEmbedder` at runtime.
- **Model Name**: `sentence-transformers/all-MiniLM-L6-v2`
- **Model Revision (Commit SHA)**: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- **Model Weights SHA-256 Checksum (`state_dict`)**: `2c705a5e3279febcd4b210fc95a6231ced10be71e400e863ab41b623d5799508`
- **Tokenizer**: `transformers.models.bert.tokenization_bert.BertTokenizer`
- **Raw `max_seq_length`**: `256`
- **Special Tokens Overhead**: `2` tokens (`[CLS]`, `[SEP]`)
- **Effective Max Sequence Tokens**: `254` tokens
- **Model Dimension (`d_model`)**: `384` (read dynamically via `get_sentence_embedding_dimension()`)
- **Facet Strategy**: Single facet (`facet_vectors = embedding.unsqueeze(0)`), shape `[1, 384]`.
- **Engine Construction**: `LTHCMEngine(d_model=embedder.d_model, num_facets=1)`
- **Loud Mismatch Enforcement**: `ShapeMismatchError` is raised at adapter initialization and index time if `d_model != 384` or `num_facets != 1`.
- **Zero Silent Truncation**: All chunks are bounded to $\le 254$ tokens before embedding. Unbroken character strings are subdivided via binary token search with zero lost characters.
- **Strict Offline Enforcement**: `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` are unconditionally exported into `os.environ` prior to model instantiation. Offline cache miss raises `OfflineCacheMissError` explicitly instructing user to run `stoma setup-model` or `python -m stoma.embedder prefetch`.
- **Peak Process Memory**: Measured peak RSS during full mixed-corpus ingestion, query, watcher execution, and recovery demo: `2517.87 MB`.

---

## 3. Timestamp Convention & Causal Light-Cone Contract

- **Document Timestamp**: `time.time()` (current epoch seconds) passed to `engine.index_document()`.
- **Query Timestamp**: `time.time()` (current epoch seconds) passed to `engine.retrieve_and_verify()`.
- **Stack-Wide Contract**: LTHCM's default `query_timestamp` is `100.0` (`pipeline.py:168`). When epoch-second document timestamps are evaluated against `100.0`, all candidates fall into the future ($dt = 100.0 - \text{epoch} < 0$), forcing the engine into silent `causal_fallback = True` (raw top_k without causality). Stoma passes real epoch timestamps for every query, preserving Lorentzian light-cone filtering.
- **Causal Verification Test**: Tested and passed with an eligible past document ($t_{\text{now}} - 120\text{s}$) and a future-timestamped top-ranked candidate ($t_{\text{now}} + 3600\text{s}$). Result: Future candidate was excluded, past document was retrieved, and `causal_fallback` was `False`.

---

## 4. Round 3 Review Items & Resolution Status

| Item | Focus Area | Description | Status | Verification & Implementation Detail |
|:---:|---|---|:---:|---|
| **1** | Commit Boundary & Ledger Recovery | Single authoritative commit point at pointer replace (`os.replace(pointer_tmp, current_pointer_path)`). Post-commit failures leave generation live. Recovery strictly follows validated monotonic prefix of `commit_chain.json`, rejecting uncommitted orphan directories. | **PASS** | `_commit_generation` executes rollback on any failure before or during pointer replacement. Directory scanning removed from recovery: only monotonically committed ledger entries in `commit_chain.json` can be recovered. Tested failure injection, byte-tampered rejection, and orphan directory rejection. |
| **2** | Watcher Robustness & Memory Cap | Move/delete retries with bounded exponential backoff, dead-letter lists, and diagnostic memory cap. | **PASS** | Moves (`_pending_moves`), deletions (`_pending_deletions`), and failed ingests (`_pending_ingest_retries`) retried up to 3 times with exponential backoff. Exhausted operations moved to `_dead_letter_*` lists. Event errors and failure details capped at 50 with `dropped_event_errors` and `failed_details_dropped` counters while counting total failures accurately. |
| **3** | Offline Weights & Cache Miss Guidance | Clear setup guidance on cache miss; genuine encoder mismatch fails loudly. | **PASS** | Added `stoma setup-model` and `python -m stoma.embedder prefetch`. `StomaEmbedder` raises `OfflineCacheMissError` explicitly naming `stoma setup-model`. Genuine encoder dimension or facet mismatches raise loud `ShapeMismatchError`. |
| **4** | Submodule Verification Isolation | Distinguish top-level gitlink SHA from submodule checkout SHA without synthetic fallbacks. | **PASS** | `verify_runtime_dependencies()` inspects both top-level parent gitlinks and submodule checked-out HEAD commits against locked pins (`336f4af9` and `da7ce83c`), returning `lthcm_gitlink_sha`, `lthcm_checkout_sha`, `dianoia_gitlink_sha`, `dianoia_checkout_sha`. Tested against manipulated checkouts. |
| **5** | Memory & Parser Streaming | Bounded streaming throughout ingestion; remove eager duplicate extraction passes. | **PASS** | Removed eager `plugin.extract` calls from `ingest_file`, `move_file`, and clean rebuild transaction. `_rebuild_clean_index_transaction` streams chunks directly in bounded batches without accumulating full chunk lists in memory. PDF streaming parses page-by-page. |
| **6** | Multi-Root Path Handling | Root-qualified source IDs (`root_id:rel_path`) prevent collision across roots. | **PASS** | Verified collision-free operation and transactional move tracking across distinct roots. |
| **7** | Dianoia Subprocess Truth | Controlled subprocess invocation of Dianoia CLI captures real returncode 1 and zero-edge refusal. | **PASS** | Invoked `[sys.executable, "-m", "dianoia.cli", "--index", str(path), "interchange_verification_ping"]`. Returns code 1 and stderr *"Error: index '...' contains 0 graph edges. Deliberative walk requires an edge structure to navigate."* Stoma accurately reports `status="direct_cli_walk_unsupported_zero_edges"` and `edge_gate_triggered=True`. |
| **8** | Dual-Store Unconditional Equality | Require `hyp_ids == sup_ids == manifest_ids` even if one store is empty. | **PASS** | `get_indexed_doc_ids()` and `validate_generation_integrity()` enforce set equality across both stores without treating an empty store as valid. Reloading a store with emptied superposition fails loudly. |
| **9** | Parser Exception Hierarchy & Mid-Extraction Checks | Typed exceptions for parser failures; pre- and post-extraction file stability check. | **PASS** | Hierarchy in `plugins/base.py`: `ExtractionError`, `CorruptedDocumentError`, `ImageOnlyPDFError`, `UnsupportedDocumentError`, `FileModifiedDuringExtractionError`. Detects file size/hash modifications occurring during extraction and rolls back candidate transaction. |

---

## 5. Test Suite Execution & Acceptance Results

The test suite executed with **72 passed out of 72 tests (100% pass rate)** in 181.12 seconds (03:01).

### Test Breakdown by Module

| Test Suite | Tests | Purpose | Result |
|---|:---:|---|:---:|
| `tests/test_adapter.py` | 5 | Shape checks, loud mismatch errors, single-facet tensors, pickle serialization | **PASSED** |
| `tests/test_causal.py` | 2 | Causal light-cone filtering & default timestamp silent fallback reproduction | **PASSED** |
| `tests/test_chunker.py` | 3 | Token bounding strictly $\le 254$, deduplication, empty chunk pruning | **PASSED** |
| `tests/test_sources.py` | 11 | Markdown, text, image-only PDF detection, corrupt PDF, Python AST, multi-language code, OCR hook, Whisper transcripts, dispatcher | **PASSED** |
| `tests/test_indexer.py` | 6 | Provenance resolution, incremental lifecycle, restart durability, mismatched model rejection, rename policy, bounded walk | **PASSED** |
| `tests/test_watcher.py` | 2 | Symlink exclusion outside allowed roots, full manual reconciliation | **PASSED** |
| `tests/test_interchange.py` | 1 | Direct interchange verification against Dianoia subprocess execution | **PASSED** |
| `tests/test_round1_regressions.py` | 10 | Regression suite covering Round 1 review findings | **PASSED** |
| `tests/test_round2_regressions.py` | 16 | Comprehensive suite covering all 7 Round 2 review findings | **PASSED** |
| `tests/test_round3_regressions.py` | 15 | Comprehensive suite covering all 9 Round 3 review items | **PASSED** |
| `tests/test_round4_regressions.py` | 18 | Comprehensive suite covering all 10 Round 4 review items & regressions | **PASSED** |
| `tests/test_e2e.py` | 1 | Full end-to-end demo flow on mixed corpus (PDF, code, markdown), atomic swap, delete purge, restart | **PASSED** |
| **Total** | **90** | **Complete unit, integration, and Round 1/2/3/4 regression coverage** | **90 / 90 PASSED** |

---

## 6. Demonstrated Ingestion, Query, Edit, Delete, and Restart Flow

The acceptance flow was executed both via the end-to-end integration test (`tests/test_e2e.py`) and a standalone acceptance script exercising the live system with real dependencies:

1. **Ingestion of Mixed Corpus**:
   - `architecture.md`: Ingested, chunked, and indexed with line numbers.
   - `quantum_memory.py`: Ingested, AST parsed (class boundaries), and indexed.
   - `spec_sheet.pdf`: Text PDF parsed page-by-page and indexed with page location.
   - `scanned_doc.pdf`: Detected as image-only / 0 extractable text; reported as unsupported with clear guidance: *"Pre-extracted OCR is required to ingest this document."*
2. **Topological Retrieval with Provenance**:
   - Query returned passages with passage ID, relative path, content SHA-256 checksum, location, and epoch timestamp.
   - Causal verification passed (`causal_fallback: False`).
3. **Modification & Atomic Swap**:
   - `architecture.md` modified with refreshed content.
   - Re-ingestion triggered clean atomic swap rebuild.
   - Query verified old passages were completely gone and new passages were retrieved without duplication.
4. **Deletion**:
   - `quantum_memory.py` deleted.
   - Index re-synchronized; passages completely purged from topological memory.
5. **Process Restart via Commit Chain**:
   - Indexer re-instantiated in a fresh instance reading persisted `current.json`, `commit_chain.json`, and generation-specific `index.pkl` / `manifest.json`.
   - Integrity of both hyperbolic and superposition stores validated against manifest.
   - Query executed successfully and returned identical provenance.
6. **Live Subprocess Interchange**:
   - Direct execution of `python3 -m dianoia.cli query ...` on Stoma index captured exit code 1 and edge-gate refusal message: *"Index contains 0 graph edges. Cannot perform deliberative walk."*
   - Stoma interchange reported `status="direct_cli_walk_unsupported_zero_edges"` accurately.

---

## 7. Documented Limitations & Architectural Boundary

1. **Scanned / Image-Only PDFs**:
   - Stoma v1 uses local parsing via `pypdf` without bundled OCR models. Image-only PDFs are explicitly identified and rejected with an unsupported report. Pre-extracted OCR outputs (via OCR hook) are supported.
2. **Dianoia CLI Direct Query Path**:
   - Dianoia's CLI loader (`cli.py:214`) enforces `edge_count > 0` for its deliberative walk.
   - Per Stoma specification, relational graph mining and wormhole edge construction are explicitly out of scope for Stoma v1.
   - Stoma's index is 100% binary- and dimension-compatible with Dianoia, and this boundary is reported honestly via `stoma interchange`.
3. **Hyperbolic Store Append-Only Nature**:
   - Pinned LTHCM's hyperbolic store appends on every `add_concept` without same-id deduplication (`lthcm/core/hyperbolic.py:187-193`).
   - Stoma solves this through atomic swap rebuilds and commit-chain recovery rather than claiming in-place upsert safety.
