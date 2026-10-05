# Prosopon: Round 1 Defect Analysis, Gaps & Remediation Plan

This document details the defect findings, security vulnerabilities, and implementation gaps identified in `metis-repo-wide-round1-fixes-6752222c.txt` and `metis-dashboard-implementation-spec-ae93d36c.txt` pertaining to **Prosopon**, along with the exact technical remediation plan and regression tests.

---

## 1. Defect Findings & Root Cause Analysis

### Item 1 (P0): Broken Server Entrypoint in CLI
- **Location**: `prosopon/prosopon/cli.py:29-83` and `scripts/run_stack.sh:119-136`.
- **Observed Behavior**: Invoking `prosopon serve --no-mock` creates `ProsoponConfig` and exits immediately with code 0 without calling `uvicorn.run()`. The `run` alias command crashes with `NameError: name 'tls' is not defined` because variable setup is missing from its local scope.
- **Root Cause**: The startup and Uvicorn launch routine was placed inside `run`, while `serve` merely constructed configuration. The launcher script calls `prosopon serve`, causing silent startup failure where the UI never actually launches.
- **Remediation**:
  1. Consolidate TLS certificate generation, application factory creation (`create_app`), startup banner printing, and `uvicorn.run(...)` into the primary `serve` command function.
  2. Make `run` a clean, thin alias pointing directly to `serve`.
  3. Ensure terminal pairing PIN generation is performed on the *exact same* `PairingManager` instance bound to the running FastAPI application, rather than instantiating an orphaned second manager.
- **Regression Testing**:
  - Test `test_cli_serve_invokes_uvicorn`: Invoke CLI via `CliRunner` with patched `uvicorn.run`; assert Uvicorn is invoked exactly once with matching host, port, app instance, and TLS paths.
  - Test `test_cli_run_alias_parity`: Assert `prosopon run` shares identical options and executes the same startup path.
  - Process liveness integration test: Launch process in a subprocess, wait for `/api/health`, and ensure process does not exit prematurely.

---

### Item 2 (P0): Unauthenticated LAN Self-Pairing Endpoint
- **Location**: `prosopon/prosopon/server.py:108-121` (`POST /api/auth/pin`).
- **Observed Behavior**: The PIN endpoint comments suggest loopback or owner-only access, but has no authentication dependency. On LAN configurations (`--lan`), an unauthorized remote client can call `POST /api/auth/pin` to fetch an active PIN, call `POST /api/auth/pair` with that PIN, obtain a bearer token, and gain full owner access.
- **Root Cause**: Missing authorization gate on PIN creation and exposure of secret pairing PINs over HTTP.
- **Remediation**:
  1. Restrict PIN generation to explicit terminal commands (`prosopon pair`) or an authenticated local owner flow.
  2. Completely remove or strictly gate `POST /api/auth/pin` so remote network callers cannot request pairing secrets.
  3. Enforce rate limiting, 5-minute TTL, single-use token consumption, and lockout after 5 failed attempts.
  4. Ensure pairing logs redact tokens and PINs.
- **Regression Testing**:
  - Test `test_remote_cannot_generate_pin`: Simulated remote IP calling `POST /api/auth/pin` receives `403 Forbidden` / `401 Unauthorized`.
  - Test `test_pairing_rate_limit_and_lockout`: 5 consecutive wrong PINs locks out pairing for 15 minutes.
  - Test `test_single_use_pin`: Successfully paired PIN cannot be reused.

---

### Item 3 (P0): Control Boundary & Loopback Authentication Exemption
- **Location**: `prosopon/prosopon/server.py:81-98`.
- **Observed Behavior**: Prosopon's `require_auth` dependency exempts all loopback (`127.0.0.1`, `::1`) clients from bearer token verification.
- **Root Cause**: Equating network binding (loopback) with caller identity authentication. Malicious scripts or cross-origin requests from an open browser tab could query internal endpoints without credentials.
- **Remediation**:
  1. Define a strict, dedicated authentication dependency for admin/dashboard endpoints (`require_admin_auth`) with **zero loopback exemptions**.
  2. Enforce explicit `allowed_origins` check (no `*` with credentials) and validate the `Host` header.
  3. Ensure cookies (if used) require CSRF tokens; bearer authentication in headers is standard.
- **Regression Testing**:
  - Test `test_loopback_exempt_for_standard_chat_only`: Normal local chat respects loopback mode if configured, but `/api/dashboard/*` and `/api/auth/pin` require explicit authentication.
  - Test `test_invalid_host_header_rejected`: Requests with spoofed `Host` headers are rejected with `400 Bad Request`.

---

### Item 4 (P1): CLI Pairing & Revocation Operating on Live Server State
- **Location**: `prosopon/prosopon/cli.py:70-73,86-103` and `prosopon/prosopon/auth/pairing.py:26-114`.
- **Observed Behavior**: `PairingManager` stores PINs in memory and reads tokens from disk once at startup. Running `prosopon pair` in a separate CLI process creates a PIN unknown to the server. Running `prosopon revoke` updates disk, but the server continues validating in-memory cached tokens.
- **Root Cause**: Multi-process state disconnect between CLI commands and the live server process.
- **Remediation**:
  1. Implement inter-process state coordination: either file-backed atomic token ledgers with inode/mtime reload checks, or have CLI commands communicate with an authenticated local server control socket/endpoint.
  2. Store hashed tokens (`sha256(token)`) with owner-only (`0600`) filesystem permissions.
- **Regression Testing**:
  - Test `test_cli_revocation_invalidates_server_token`: Revoking via CLI immediately prevents the server process from accepting the revoked token.
  - Test `test_restarted_server_retains_revocation`: Server restart maintains revoked state.

---

### Item 23 (P1): Honest Voice Pipeline & Subprocess Cleanup
- **Location**: `prosopon/prosopon/voice/whisper_stt.py`, `piper_tts.py`, `transcoder.py`, `server.py:223-243`.
- **Observed Behavior**: When `whisper.cpp` is missing, the code returns a fabricated question (*"What is hippocampal replay?"*); when `piper` is missing, it returns empty silence. Subprocess timeouts and non-zero exit codes are ignored. Upload payloads are read fully before checking byte caps.
- **Root Cause**: Silently masking missing binaries and failing to track simulation provenance.
- **Remediation**:
  1. In live mode, report typed `VoiceEngineUnavailableError` (HTTP 503) when binaries or weights are absent.
  2. Only return simulated audio/transcripts when `--mock` is explicitly passed, with `is_simulated: True` provenance in the response.
  3. Validate subprocess exit codes, kill hanging child processes on timeout, and clean up temporary WAV files in `finally` blocks.
  4. Stream-read audio uploads with an immediate hard byte cap (e.g. 10 MB) to prevent memory exhaustion.
- **Regression Testing**:
  - Test `test_missing_whisper_returns_503_in_live_mode`: Assert live server returns 503 rather than fake text.
  - Test `test_hanging_ffmpeg_killed_and_cleaned`: Hanging transcoding subprocess is killed, and temporary files purged.

---

### Item 24 (P1): Wheel Packaging & Safe Metadata Rendering
- **Location**: `prosopon/pyproject.toml:37-38`, `prosopon/prosopon/web/static/js/ui.js:13-66`, `prosopon/storage/session_store.py`.
- **Observed Behavior**: Built wheels omit `/static/` web assets. `ui.js` interpolates citation IDs, route tags, and component names directly into `innerHTML`. `SessionStore` creates filesystem paths from session IDs without UUID validation.
- **Root Cause**: Missing hatchling package data inclusion and unsafe DOM string concatenation.
- **Remediation**:
  1. Add `artifacts = ["prosopon/web/static/**"]` to wheel build config.
  2. Use DOM APIs (`document.createElement`, `textContent`) instead of `innerHTML` for metadata.
  3. Validate all session IDs against strict UUIDv4 regex and resolve canonical paths.
- **Regression Testing**:
  - Test `test_built_wheel_contains_static_assets`: Build wheel and verify static files exist.
  - Test `test_xss_payload_in_citation_rendered_as_text`: Hostile script tags in citations are rendered harmlessly as plain text.
  - Test `test_path_traversal_session_id_rejected`: Session ID containing `../../` is rejected.
