# Metis Development Resume Guide (`pickup.md`)

**Last Updated:** 2026-10-18  
**Current Branch:** `main`  
**Current HEAD Commit:** `63f8d1af3b6b15e35dd48911a15bdcdb59dae60c` (`feat(ncm): implement Schedion and Symbolon offline adapters and execute E1 diagnostic`)  
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

### Phase 4: Anadrome Decision-Tree Logging & Offline Replay Engine (Completed)
1. **RFC 8785 Canonical JSON Serialization & Hashing (`dianoia/dianoia/anadrome/canonical.py`)**:
   - Deterministic canonical formatting, lexicographical key ordering, and float stabilization (`round(score, 4)`).
   - Bit-exact `compute_observation_hash()` SHA-256 digests invariant to dict ordering and floating-point drift.
2. **Replay Data Contracts (`dianoia/dianoia/anadrome/contracts.py`)**:
   - `EnvironmentIdentity`, `EligibleAction`, `DecisionObservation`, `RecordedTransition`, `ReplayTreeRecord`.
3. **In-Process SQLite Answer Probe Cache (`dianoia/dianoia/anadrome/probe_cache.py`)**:
   - `AnswerProbeCache` storing draft answer, verification status, and conflict detail keyed by `(state_id_hash, model_id, prompt_hash)`.
4. **Append-Only Tree Logger (`dianoia/dianoia/anadrome/logger.py`)**:
   - `AnadromeLogger` writing and streaming `~/.dianoia/replay_trees.jsonl`.
5. **WalkController Continuation Tree & Shadow Branch Harness (`dianoia/dianoia/walk.py`)**:
   - Unified `walk_id` tracking across `walks.jsonl` and `replay_trees.jsonl`.
   - Intermediate answer probes restricted to path depth $\ge 2$ with SQLite caching.
   - Bounded shadow branch expansion: $\le 1$ alternate branch per walk with depth $\le 2$, logged with `is_shadow=True` and `parent_walk_id`.
6. **Offline Replay Engine (`dianoia/dianoia/anadrome/replay.py`)**:
   - Strict **Zero Imputation Invariant**: unsupported transitions are never smoothed or fabricated.
   - Baseline Sanity Check: `verify_baseline_sanity()` reproducing recorded traces with 100.0% bit-exact fidelity.
7. **Test Suite (`dianoia/tests/test_anadrome.py`)**:
   - 15/15 tests passing across canonical serialization, contracts, probe caching, WalkController integration with shadow branches, and replay engine invariants. Total Dianoia suite: 76 tests passing.

---

### Phase 5: Schedion & Symbolon Offline Adapters (Completed)
1. **Schedion Adapter (`schedion/schedion/`)**:
   - **Candidate Generator (`generator.py`)**: Wraps NCM generation path (`NextConceptTransformer` + `VQVAETokenizer`) with strict 512-token GPT-2 truncation discipline and complete provenance metadata (`SchedionProvenance`).
   - **Deterministic Sentence-Boundary Claim Mapper (`claim_mapper.py`)**: `SentenceBoundaryClaimMapper` segmenting decoded summary sentences into structured `Claim` objects anchored to the 512-token conditioned source span with default `verdict="unverified"`.
   - **Krisis Ingestion Gate**: Emits fully validated `Candidate` instances processed cleanly by `KrisisSelector.evaluate()` without schema errors.
   - **Unit Tests (`tests/test_schedion.py`)**: 6/6 tests passing.
2. **Symbolon Adapter (`symbolon/symbolon/`)**:
   - **Scoped Code Identity (`codes.py`)**: Enforces expanded 6-tuple identity rule `(checkpoint_id, tokenizer_id, level, window_idx, pos, code_int)`. Strict scope matching functions `compute_scoped_code_overlap` and `compute_scoped_code_jaccard`.
   - **Multi-Window Tokenizer Adapter (`tokenizer_adapter.py`)**: `SymbolonTokenizerAdapter` chunking arbitrary passages into 48-token context windows, extracting multiscale discrete codes across all 9 levels, and decoding per-level reconstructions.
   - **Experiment E1 Diagnostic Report (`docs/review/symbolon-reconstruction-fidelity-e1-report.md`)**: Executed on RTX 5060 Ti over representative entity, topological, numerical, and biochemical probes. Calibrated monotonic reconstruction fidelity from L0 (4 codes, 1.2% word overlap) to L8 (48 codes, 82.8% word overlap, 87.5% key term preservation).
   - **Unit Tests (`tests/test_symbolon.py`)**: 8/8 tests passing.

### Phase 6: Anadrome Policy Zoo & Offline Evaluation Harness (Completed)
1. **Candidate Walk Policy Zoo (`dianoia/dianoia/anadrome/policies.py`)**:
   - `AnadromePolicy` abstract base contract and callable protocol (`select_action(obs: Dict[str, Any]) -> str`).
   - **Policy 1: Greedy Similarity EXPAND (`GreedySimilarityPolicy`)**: Selects highest embedding similarity / edge score among eligible unvisited expand targets. Configurable `min_score_threshold`, `avoid_loops`, and fallback action preference.
   - **Policy 2: High-Entropy Stochastic Selection (`HighEntropyStochasticPolicy`)**: Kleroterion-style temperature-scaled softmax categorical sampling over candidate action scores with deterministic pseudo-random draws keyed on `(seed, observation_hash)`.
   - **Policy 3: Early-Exit Answer Threshold (`EarlyExitAnswerPolicy`)**: Monitors evidence accumulation and triggers early `ANSWER` once $\ge \text{min\_passages}$ visited or top candidate score is below threshold $\tau$.
   - **Policy 4: Sheaf Contradiction Backtrack (`SheafContradictionBacktrackPolicy`)**: Inspects observation for cellular sheaf Laplacian topological obstruction/conflict flags or markers and immediately forces `BACKTRACK` on conflicting branches (stack depth > 1).
   - **Baseline Replay Policy (`BaselineReplayPolicy`)**: Replays recorded transitions bit-for-bit from reference continuation trees or mappings.
2. **Offline Replay Evaluation & Comparison Harness (`dianoia/dianoia/anadrome/evaluator.py`)**:
   - Multi-policy zoo evaluation over recorded decision continuation trees.
   - **Coverage Gate ($\ge 80\%$)**: Strict gating per Spec Section 4.3. Any policy with coverage $< 80\%$ is rejected with `promotion_verdict="FAILED_COVERAGE_GATE"`.
   - **Mutual Support Filtering**: Restricts pairwise policy comparisons strictly to questions/episodes where *both* candidate and baseline achieved fully supported walks under the Zero Imputation Invariant.
   - **Dokime Paired Bootstrap Protocol**: 10,000 paired resamples, seed 42, computing 95% confidence intervals on delta correctness/accuracy, step deltas, and hard error regressions.
   - **Spec Section 6 Promotion Criteria**: Emits `PASS`, `NO_EVIDENCE`, `REGRESSION`, or `FAILED_COVERAGE_GATE` with exact statistical justification.
3. **Unit and Integration Test Suite (`dianoia/tests/test_policy_zoo.py`)**:
   - 13 comprehensive tests covering policy decisions, determinism, early exit, sheaf contradiction backtracking, baseline replay, coverage gate rejection, 10k bootstrap evaluation, promotion verdicts, and strict view isolation.
   - Dianoia test suite: **89 passed** (up from 76).
   - Total Metis test suite: **365 passed across all suites** (89 dianoia, 37 root, 66 prosopon, 173 kybernetes).

---

## 3. Immediate Prioritized Tasks Left to Do

### Task 8: Schedion & Symbolon Controlled Studies (Roadmap Milestone 2.3, Tracks C & D)
- **Reference Spec**: Synthesis Roadmap Milestone 2.3.
- **Schedion Stage 3**: Controlled comparison between Hestia summary candidates vs Schedion candidates over identical 512-token conditioned spans:
  - Record Krisis selector pass/fail rates, claim extraction distributions, factual grounding, and wall-clock times.
- **Symbolon E2**: Multiscale discrete code capture and fusion evaluation over benchmark diagnostic passages.

---

## 4. Key Documentation Links

- **Main Manual**: [`docs/MSS.md`](file:///home/akarta/Projects/Metis/docs/MSS.md)
- **Pyle WebRTC + Passkey Spec**: [`docs/review/mss-pyle-webrtc-passkey-spec.md`](file:///home/akarta/Projects/Metis/docs/review/mss-pyle-webrtc-passkey-spec.md)
- **Synthesis Roadmap (All 5 Subsystems)**: [`docs/review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md`](file:///home/akarta/Projects/Metis/docs/review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md)
- **Anadrome Replay Spec**: [`docs/review/metis-anadrome-walk-replay-experiment-spec-aa69f396.txt`](file:///home/akarta/Projects/Metis/docs/review/metis-anadrome-walk-replay-experiment-spec-aa69f396.txt)
- **Review Specs Index**: [`docs/REVIEW_INDEX.md`](file:///home/akarta/Projects/Metis/docs/REVIEW_INDEX.md)

