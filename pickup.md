# Metis Development Resume Guide (`pickup.md`)

**Last Updated:** 2026-10-18  
**Current Branch:** `main`  
**Current HEAD Commit:** `90ddb918f0376cb42e92c2df9e5fe982b6c7aeb8` (`feat(pyle): implement Phase 2 WebRTC DataChannel engine, signaling, and client controller`)  
**Working Tree Status:** Clean, all tests passing.

---

## 1. Quick Health Check (Run After Reboot)

To verify all services and environments are operational after system reboot:

```bash
# 1. Run Prosopon test suite (66 tests)
./prosopon/.venv/bin/pytest prosopon/tests

# 2. Run Kybernetes test suite (173 tests)
./kybernetes/.venv/bin/pytest kybernetes/tests

# Total: 239 tests passing
```

---

## 2. Completed Sessions (Phase 0, Phase 1 & Phase 2)

### Phase 0: Perimeter Hardening
1. **Perimeter Hardening**:
   - Loopback exemptions eliminated in `prosopon/prosopon/server.py`.
   - All HTTP endpoints strictly require a valid bearer token or host-owner credential.
   - `/api/auth/local-token` secured: requires `X-Host-Owner-Token` or valid bootstrap ticket.
2. **Filesystem Authority**:
   - `data_dir / ".host_owner_token"` written with `0600` permissions on startup.
   - CLI tools and local test suites authenticate seamlessly via filesystem ownership.
3. **Single-Use Browser Launch**:
   - `prosopon open` command added to `prosopon/prosopon/cli.py`.
   - Mints a 60-second single-use ticket (`/?bootstrap_token=<ticket>`) and launches the browser.
   - Frontend (`api.js`) exchanges ticket for a session token and strips ticket from URL.
4. **WebSocket Gate**:
   - `/api/lochos/ws` validates auth tokens *before* `websocket.accept()`. Unauthorized clients receive `WS_1008_POLICY_VIOLATION`.

### Phase 1: SQLite Device Registry & WebAuthn RP (Completed)
1. **SQLite Access Store (`prosopon/storage/access_store.py`)**:
   - Persistent store `prosopon/data/access.db` managing `devices`, `passkeys`, `device_sessions`, `pairing_links`, `routes`, `access_audit`, and `webrtc_sessions`.
   - SHA-256 cryptographic hash-chaining across all access audit rows with verification.
   - Default `host_owner` device provisioned automatically.
2. **W3C WebAuthn Relying Party (`prosopon/auth/webauthn.py`)**:
   - Implemented `WebAuthnManager` utilizing `webauthn>=2.0.0` for registration & authentication ceremonies.
3. **Device Management REST Endpoints (`prosopon/server.py`)**:
   - `GET /api/devices`, `POST /api/devices/pair`, `POST /api/devices/{id}/revoke`, `POST /api/devices/lockdown`, `GET /api/devices/me`, `GET /api/devices/audit`.
   - WebAuthn registration/authentication challenge endpoints: `/api/auth/webauthn/register/options`, `/api/auth/webauthn/register/verify`, `/api/auth/webauthn/auth/options`, `/api/auth/webauthn/auth/verify`.
   - Extended `require_auth` & `require_admin_auth` to validate `access.db` session tokens with role/scope enforcement.
4. **CLI Device Commands (`prosopon/cli.py`)**:
   - `prosopon device list`, `prosopon device pair`, `prosopon device revoke`, `prosopon device lockdown`.
5. **Workbench Settings UI**:
   - Added `Devices & Security (Pyle)` tab in Prosopon Settings with live device table, pairing link generator, panic lockdown button, and cryptographic hash chain verification badge.
6. **Documentation Update**:
   - Updated `docs/MSS.md` Section 3.5 and Section 11.8.

### Phase 2: WebRTC DataChannel Engine & Client Controller (Completed)
1. **WebRTC Gateway (`prosopon/transport/webrtc.py`)**:
   - Implemented `WebRTCGateway` using `aiortc>=1.9.0`.
   - SCTP DataChannel multiplexing:
     - `control`: JSON-RPC 2.0 (ping/heartbeat, `auth.status`, `phylax.sign_approval`).
     - `stream`: SSE formatted real-time event broadcasting.
     - `terminal`: Bidirectional ANSI terminal stream.
   - Phylax biometric action challenge verification with risk-tier scope checks (`approvals:resolve` for T1, `approvals:resolve_high` for T2/T3).
   - Instant peer connection teardown on device revocation (< 500ms security invariant).
2. **Signaling Endpoints (`prosopon/server.py`)**:
   - `POST /api/webrtc/offer`: SDP negotiation with token/host-owner authentication.
   - `POST /api/webrtc/ice-candidate`: ICE candidate ingestion.
   - `GET /api/webrtc/sessions`: Active WebRTC session listing.
   - Wired teardown into device revocation and emergency lockdown endpoints.
3. **Client Controller (`prosopon/web/static/js/webrtc_client.js`)**:
   - `PyleWebRTCClient` managing RTCPeerConnection, signaling, JSON-RPC, SSE streams, and Phylax signing.
4. **Test Suite (`prosopon/tests/test_pyle_webrtc.py`)**:
   - 8 comprehensive tests covering negotiation, unauthorized rejection, ICE candidates, JSON-RPC ping/auth, Phylax approval scope enforcement, SSE/terminal dispatch, and instant revocation.

### Phase 3: NCM Hardware & Memory Validation (Completed)
1. **Pinned NCM Checkout (`.metis/scratch/ncm`)**:
   - Pinned upstream `github.com/HazyResearch/ncm` @ `c70274273f489abc73832b2b98584a8483b6b203`.
   - Downloaded XSum tokenizer (`tokenizer.pt`, 286 MB), generator (`generator.pt`, 338 MB), and test sets.
2. **Empirical Benchmarks on RTX 5060 Ti 16GB**:
   - **Peak VRAM (Standalone)**: `0.78 GB allocated` (`0.80 GB reserved`). Gate G-m0a ($\le 6.5\text{ GB}$) **PASSED**.
   - **Peak VRAM (Co-loaded with GPT-2 Large Scorer)**: `3.67 GB allocated` (`3.77 GB reserved`). Gate G-m0b **PASSED**.
   - **Throughput & Latency**: `122.3 ms/doc` generation; `6.74 ms` for 48-token multiscale VQ-VAE decode (148 windows/s).
   - **Concurrency Verdict**: Peak footprint $< 1.0\text{ GB}$ allows NCM to reside concurrently in VRAM with Hestia; **no sequential unloading required**.
   - **Report**: Recorded in [`docs/review/ncm-hardware-validation-rtx5060ti-report.md`](file:///home/akarta/Projects/Metis/docs/review/ncm-hardware-validation-rtx5060ti-report.md).

---

## 3. Immediate Prioritized Tasks Left to Do

### Task 5: Anadrome Decision-Tree Logging Harness
- **Reference Spec**: `docs/review/metis-anadrome-walk-replay-experiment-spec-aa69f396.txt` and synthesis roadmap Section 2.4.
- Extend Dianoia walk logger to capture continuation trees, eligible actions, and RFC 8785 canonical observation hashes to `~/.dianoia/replay_trees.jsonl`.
- Restrict intermediate answer probes to path depth $\ge 2$ with in-process SQLite cache.
- Bound shadow expansions to exactly 1 branch per walk with depth $\le 2$.
- Verify bit-exact replay over baseline controller (100% agreement sanity check) before evaluating candidate policy zoo.

---

## 4. Key Documentation Links

- **Main Manual**: [`docs/MSS.md`](file:///home/akarta/Projects/Metis/docs/MSS.md)
- **Pyle WebRTC + Passkey Spec**: [`docs/review/mss-pyle-webrtc-passkey-spec.md`](file:///home/akarta/Projects/Metis/docs/review/mss-pyle-webrtc-passkey-spec.md)
- **Synthesis Roadmap (All 5 Subsystems)**: [`docs/review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md`](file:///home/akarta/Projects/Metis/docs/review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md)
- **Review Specs Index**: [`docs/REVIEW_INDEX.md`](file:///home/akarta/Projects/Metis/docs/REVIEW_INDEX.md)
