# Metis Development Resume Guide (`pickup.md`)

**Last Updated:** 2026-10-18  
**Current Branch:** `main`  
**Current HEAD Commit:** `a7cf34079815049a4641cb482ba3627bfd02f5a6` (`feat(pyle): implement Phase 0 perimeter hardening and host-owner bootstrap`)  
**Working Tree Status:** Clean, all tests passing.

---

## 1. Quick Health Check (Run After Reboot)

To verify all services and environments are operational after system reboot:

```bash
# 1. Run Prosopon test suite (49 tests)
./prosopon/.venv/bin/pytest prosopon/tests

# 2. Run Kybernetes test suite (173 tests)
./kybernetes/.venv/bin/pytest kybernetes/tests

# Total: 222 tests passing
```

---

## 2. Completed in Last Session (Phase 0)

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

---

## 3. Immediate Prioritized Tasks Left to Do

### Task 1: Update `docs/MSS.md` (Quick Win)
- Document the new `prosopon open` command in Section 11.8.
- Document the zero-loopback security model and `.host_owner_token` filesystem authority in Section 3.
- Reference the WebRTC + Passkey remote architecture.

### Task 2: Pyle Phase 1 — SQLite Device Registry & WebAuthn RP
- **Reference Spec**: `docs/review/mss-pyle-webrtc-passkey-spec.md` (`METIS-SPEC-PYLE-WEBRTC-001`)
- **Action Items**:
  1. Create SQLite schema `prosopon/data/access.db` (`devices`, `device_sessions`, `pairing_links`, `routes`, `access_audit` with SHA-256 hash chaining).
  2. Implement local FIDO2 WebAuthn Relying Party (`fido2` / `py_webauthn`) in `prosopon`.
  3. Implement device management REST endpoints (`/api/devices`, `/api/devices/{id}/revoke`, `/api/devices/lockdown`).
  4. Add CLI commands: `prosopon device list`, `prosopon device pair`, `prosopon device revoke`, `prosopon device lockdown`.
  5. Add *Manage Devices* modal to Prosopon settings.

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
