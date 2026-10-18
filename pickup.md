# Metis Development Resume Guide (`pickup.md`)

**Last Updated:** 2026-10-18  
**Current Branch:** `main`  
**Current HEAD Commit:** `fed42ca2385982e3701fcede1f23260ff84ea949` (`feat(pyle): implement Phase 1 SQLite Device Registry, WebAuthn RP, and management endpoints`)  
**Working Tree Status:** Clean, all tests passing.

---

## 1. Quick Health Check (Run After Reboot)

To verify all services and environments are operational after system reboot:

```bash
# 1. Run Prosopon test suite (58 tests)
./prosopon/.venv/bin/pytest prosopon/tests

# 2. Run Kybernetes test suite (173 tests)
./kybernetes/.venv/bin/pytest kybernetes/tests

# Total: 231 tests passing
```

---

## 2. Completed Sessions (Phase 0 & Phase 1)

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
   - Persistent store `prosopon/data/access.db` managing `devices`, `passkeys`, `device_sessions`, `pairing_links`, `routes`, and `access_audit`.
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

---

## 3. Immediate Prioritized Tasks Left to Do

### Task 3: Pyle Phase 2 — WebRTC DataChannel Engine (`aiortc`)
- Implement WebRTC signaling (`/api/webrtc/offer`, `/api/webrtc/ice-candidate`).
- Establish the 4 multiplexed channels:
  - `control`: RPC, ping/keepalive, session state.
  - `stream`: Lochos squad & chat token streaming.
  - `terminal`: ANSI terminal streaming for `kybernetes code` runs.
  - `audio`: Full-duplex Opus audio for sovereign voice client (bypasses browser self-signed TLS restrictions on mobile).
- Enforce WebAuthn biometric challenge-signing for Phylax T2/T3 approvals.

### Task 4: NCM Hardware Validation (Schedion & Symbolon)
- **Reference Spec**: `docs/review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md`
- Clone pinned NCM checkout (`c702742`) to scratch space.
- Benchmark peak VRAM on the RTX 5060 Ti 16GB.
- If peak VRAM $> 10\text{ GB}$, enforce sequential unloading when switching between Hestia and NCM.

### Task 5: Anadrome Decision-Tree Logging Harness
- Extend Dianoia walk logger to capture continuation trees, eligible actions, and RFC 8785 canonical observation hashes to `~/.dianoia/replay_trees.jsonl`.
- Verify bit-exact replay over baseline controller before evaluating candidate policy zoo.

---

## 4. Key Documentation Links

- **Main Manual**: [`docs/MSS.md`](file:///home/akarta/Projects/Metis/docs/MSS.md)
- **Pyle WebRTC + Passkey Spec**: [`docs/review/mss-pyle-webrtc-passkey-spec.md`](file:///home/akarta/Projects/Metis/docs/review/mss-pyle-webrtc-passkey-spec.md)
- **Synthesis Roadmap (All 5 Subsystems)**: [`docs/review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md`](file:///home/akarta/Projects/Metis/docs/review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md)
- **Review Specs Index**: [`docs/REVIEW_INDEX.md`](file:///home/akarta/Projects/Metis/docs/REVIEW_INDEX.md)
