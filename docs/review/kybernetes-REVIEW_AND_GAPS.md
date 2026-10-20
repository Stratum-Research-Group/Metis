# Kybernetes: Round 1 Defect Analysis, Gaps & Remediation Plan

This document details the defect findings, security vulnerabilities, and implementation gaps identified in `metis-repo-wide-round1-fixes-6752222c.txt` and `metis-dashboard-implementation-spec-ae93d36c.txt` for **Kybernetes**, along with technical remediations and regression tests.

---

## 1. Defect Findings & Root Cause Analysis

### Item 3 (P0): Unauthenticated Control Boundary & Permissive CORS
- **Location**: `kybernetes/kybernetes/server/app.py:57-63,87-130`.
- **Observed Behavior**: All conversational endpoints (`POST /v1/chat/turn`, `POST /v1/chat/stream`), approval resolution (`POST /v1/approvals/{id}/resolve`), turn cancellation, traces, and budget reports lack authentication. The CORS middleware specifies `allow_origins=["*"]` with `allow_credentials=True`.
- **Root Cause**: Relying on 127.0.0.1 binding as an authorization proxy, permitting cross-origin access and allowing direct bypass of Prosopon's pairing gate.
- **Remediation**:
  1. Replace `allow_origins=["*"]` with explicit configured origin whitelists.
  2. Protect all endpoints under `/v1/` with a shared service secret (`Authorization: Bearer <KYBERNETES_SERVICE_TOKEN>`) configured at startup.
  3. Validate the `Host` header to prevent DNS rebinding attacks.
- **Regression Testing**:
  - Test `test_unauthenticated_chat_rejected`: Calling `/v1/chat/turn` without bearer token returns 401.
  - Test `test_wildcard_cors_disallowed_with_credentials`: Verify CORS configuration rejects arbitrary origins.

---

### Item 10 (P1): Simulated Adapters vs. Advertised Live Modules (F1 / F2)
- **Location**: `kybernetes/kybernetes/server/app.py:25-48,70-85`, `adapters/lthcm.py`, `adapters/dianoia.py`, `adapters/phylax.py`, `adapters/krisis.py`.
- **Observed Behavior**:
  - `GET /v1/health` hardcodes `"ready"` for `lthcm`, `dianoia`, `phylax`, and `krisis`, while only `Hestia` is probed.
  - `create_app()` initializes `LTHCMAdapter()` with no `memory_root`, forcing `is_simulated=True` over a 4-passage in-memory mock corpus. The CLI `kybernetes serve` lacks a `--memory-root` flag.
  - `DianoiaAdapter.deliberate()` always returns `is_simulated=True` and ignores its injected module.
  - `PhylaxAdapter` is initialized with `shim=None`, and `KrisisAdapter` defaults to a local heuristic verifier.
- **Root Cause**: Prematurely advertising live module integration when the server defaults to unconfigured mock fixtures.
- **Remediation**:
  1. Replace hardcoded health checks with real `ComponentStatus` probes using the 5 canonical states: `live`, `simulated`, `mock`, `offline`, `unknown`.
  2. Add CLI and environment parameters: `--memory-root`, `--allowed-origins`, `--phylax-policy`, `--krisis-policy`.
  3. In `LTHCMAdapter`, when `--memory-root` is provided, load the real Stoma generation index; report `live` only if a test retrieval passes.
  4. Explicitly label simulated and mock states in trace metadata and health outputs.
- **Regression Testing**:
  - Test `test_health_reports_simulated_for_default_adapters`: Default startup reports `simulated` for LTHCM/Dianoia/Krisis, never `live` or `ready`.
  - Test `test_serve_cli_accepts_memory_root`: Passing `--memory-root` configures real index loading.

---

### Item 11 (P1): Stoma-to-Memory Contract Mismatch
- **Location**: `kybernetes/kybernetes/adapters/lthcm.py:29-109,147-189`.
- **Observed Behavior**: `LTHCMAdapter` searches for `manifest.json`, `index.json`, and `passages.json` directly at the memory root and performs naive keyword-substring retrieval. In contrast, Stoma writes a generational ledger (`current.json` -> `generations/gen_{N}/`) containing pickled engines and passage provenance without raw text in the manifest. Furthermore, when `passages.json` is a list, `data.get()` crashes with `AttributeError`.
- **Root Cause**: Architectural divergence between Stoma's generational storage and Kybernetes' legacy adapter assumptions.
- **Remediation**:
  1. Update `LTHCMAdapter` to read Stoma's `current.json`, resolve the active generation directory, and verify index/manifest checksums.
  2. Ingest structured passage exports or integrate with Stoma's public query interface.
  3. Fix the JSON parsing logic: inspect `isinstance(data, list)` before calling `.get()`.
- **Regression Testing**:
  - Test `test_load_real_stoma_generation`: Ingest a unique fact in Stoma and retrieve the exact passage through Kybernetes LTHCMAdapter.
  - Test `test_stoma_list_format_handled_gracefully`: List-form passage JSON parsed without AttributeError.

---

### Item 21 (P1): Unbounded Turn Cancellation, Replay, and Retention
- **Location**: `kybernetes/kybernetes/engine/pipeline.py:75-98,157-172`, `engine/session.py:44-79`.
- **Observed Behavior**: Cancellation is checked only at stage boundaries. If a client disconnects while awaiting Phylax approval or during streaming synthesis, execution continues. In-memory sessions and traces grow without bound (no TTL or LRU cap).
- **Root Cause**: Lack of cooperative cancellation propagation and unbounded in-memory dictionaries.
- **Remediation**:
  1. Pass `asyncio.Event` cancellation tokens into tool handlers and model streaming loops.
  2. Implement an LRU and TTL cap on in-memory trace and session dictionaries (e.g. max 500 sessions, 24-hour expiration).
  3. Ensure aborted turns write a terminal trace marked `status="cancelled"` in a `finally` block.
- **Regression Testing**:
  - Test `test_cancellation_during_approval_wait_halts_execution`: Cancelling a turn while awaiting approval prevents tool execution.
  - Test `test_session_store_retention_cap`: Excess sessions are purged under LRU policy.

---

### Item 22 (P1): Enforce Tool Bounds & File Search Hard Stops
- **Location**: `kybernetes/kybernetes/tools/registry.py:50-88`, `tools/file_search.py:77-120`, `tools/calculator.py:76-134`.
- **Observed Behavior**: File search scans without a file count or byte cap if no pattern matches. Relative path resolution fails when multiple roots are provided. The AST calculator lacks expression length and recursion depth caps.
- **Root Cause**: Declared timeouts were not enforced programmatically with hard limits.
- **Remediation**:
  1. Enforce strict max search depth (e.g. 5 levels), max scanned files (e.g. 1,000 files), and byte limits (e.g. 50 MB total) in `file_search`.
  2. Fix path resolution across multiple roots using `os.path.commonpath`.
  3. Bound AST expression length ($\le 500$ chars) and tree depth ($\le 10$ levels) in `calculator`.
- **Regression Testing**:
  - Test `test_file_search_exceeds_file_cap_halts`: Deep search without matches cleanly terminates at limit with a partial result.
  - Test `test_calculator_deeply_nested_ast_rejected`: Excessively nested expressions raise validation errors.
