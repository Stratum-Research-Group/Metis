# Metis Development Resume Guide (`pickup.md`)

**Last Updated:** 2026-10-18  
**Current Branch:** `main`  
**Current HEAD Commit:** `b537f4be4ab52a88ce1b5006416d45af4cd23f2f` (`feat(anadrome): harden canonical JSON normalization against mock OOM and execute Milestone 3.2 frozen live walk test`)  
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

### Phase 7: Schedion & Symbolon Controlled Studies (Completed)
1. **Schedion Stage 3 Controlled Comparison (`schedion/schedion/study.py`)**:
   - Implemented `SchedionStage3Study` evaluating baseline Hestia extractive/autoregressive summaries vs. Schedion NCM generator candidates across identical 512-token conditioned spans under `KrisisSelector` gating.
   - Executed on RTX 5060 Ti GPU with real NCM XSum checkpoint.
   - **Key Finding**: Schedion NCM generator exhibited severe out-of-domain degradation on technical/biochemical/mathematical spans (mean factual grounding 26.3% vs. 100.0% for baseline; latency 813.0 ms vs. < 1.0 ms). KrisisSelector cleanly rejected all ungrounded candidates as `NO_VALID_CANDIDATE`.
   - **Verdict**: `ARCHIVE_NO_ADOPTION`. Negative research report recorded in [`docs/review/schedion-stage3-controlled-comparison-report.md`](file:///home/akarta/Projects/Metis/docs/review/schedion-stage3-controlled-comparison-report.md). Schedion remains disabled by default.
2. **Symbolon E2 Candidate Capture & Fusion Study (`symbolon/symbolon/study.py`)**:
   - Implemented `SymbolonE2Study` evaluating flat dense baseline, Symbolon discrete code-overlap alone, uncapped additive union, and capped Reciprocal Rank Fusion (matched top-$k$) across gold relevance and contradiction labels.
   - Executed on RTX 5060 Ti GPU with real VQ-VAE discrete tokenizer.
   - **Key Finding**: Symbolon alone achieved only 25.0% recall and 25.0% contradiction retention. Capped RRF top-$k=3$ showed zero improvement over flat baseline ($\Delta = 0.0000$). Uncapped union preserved gold evidence but caused a 1.75x candidate set expansion and verification overhead.
   - **Verdict**: `ARCHIVE_UNCAPPED_NO_LOSS_RECORDED`. Negative research report recorded in [`docs/review/symbolon-candidate-fusion-e2-report.md`](file:///home/akarta/Projects/Metis/docs/review/symbolon-candidate-fusion-e2-report.md). Symbolon `FilteredRetrievalAdapter` is not adopted into LTHCM.
3. **Unit & Integration Test Suites**:
   - Added `test_stage3_controlled_comparison_study` in `tests/test_schedion.py` (7/7 passed).
   - Added `TestSymbolonStudyE2` in `tests/test_symbolon.py` (9/9 passed).
   - Total Metis test suite: **367 passed across all suites** (89 dianoia, 39 root, 66 prosopon, 173 kybernetes).

---

### Phase 8: Pyle Remote Operations & Panic Lockdown (Completed)
1. **Operator Preset & Remote Approvals (`prosopon/server.py`, `prosopon/transport/webrtc.py`)**:
   - Enabled `operator` preset holding `approvals:resolve` to remotely approve T1 and T2 actions over WebRTC DataChannel (`control`) and REST `/api/approvals/{id}/resolve`.
   - Enforced host confirmation step-up for T3 actions (system-mutating, shell execution, code installation): rejected with 403 Forbidden / JSON-RPC error unless caller holds `approvals:resolve_high` (Owner only).
   - Biometric challenge verification with user verification (`uv=1`) enforced for high tiers when passkey assertions are provided.
   - Remote approval actions audit-logged to SHA-256 hash-chained `access_audit` ledger.
2. **Active Task & In-Flight Turn Termination (`prosopon/server.py`)**:
   - Tracked in-flight agent tasks and turns per device (`active_device_tasks` and `active_device_turns`).
   - On `POST /api/devices/{id}/revoke`: immediately terminates WebRTC PeerConnections (< 500 ms gate), cancels in-flight `asyncio.Task` handles, and dispatches `kybernetes.cancel_turn(turn_id)`.
   - Added `stop_runs` flag in `RevokeDeviceRequest` (default `True`; allows explicit `stop_runs=False` to preserve running jobs).
   - Added `GET /api/devices/{id}/runs` endpoint for revoke dialog inspection.
3. **Panic Lockdown (`prosopon/server.py`, `prosopon/cli.py`)**:
   - One-command / one-action Panic Lockdown (`POST /api/devices/lockdown` and `prosopon device lockdown`).
   - Terminates all active remote agent tasks and turns across all remote devices.
   - Closes all active WebRTC DataChannel sessions (< 500 ms).
   - Revokes all non-host devices and active session tokens in `access.db`.
4. **Extended Doctor Diagnostics (`prosopon/ops/doctor.py`)**:
   - **TLS State & Secure Context**: Validates certificate/key paths and warns on plain HTTP when bound to LAN.
   - **File Permissions**: Enforces `0600` octal mode on `access.db` and `.host_owner_token`; fails closed if group/other permissions exist.
   - **Perimeter Audit**: Validates absence of `/api/auth/local-token` and verifies zero loopback exemptions.
   - **Legacy Token Migration**: Validates that all active credentials use SHA-256 hash chaining.
5. **Unit & Integration Test Suite (`prosopon/tests/test_pyle_devices.py`, `prosopon/tests/test_pyle_webrtc.py`)**:
   - Added tests covering Operator approval of T1/T2, T3 step-up rejection, Owner approval of T3, active task cancellation, `stop_runs=False` preservation, panic lockdown termination, and doctor diagnostics.
    - Total Metis test suite: **371 passed across all suites** (70 prosopon, 173 kybernetes, 89 dianoia, 39 root).

---

### Phase 9: Anadrome Frozen Live Walk Test & Policy Decision Gate (Completed)
1. **Live Walk Study Harness (`dianoia/dianoia/anadrome/study.py`)**:
   - Implemented `AnadromeLiveStudy` with 10 pre-declared frozen benchmark questions spanning 4 leakage-isolated families (multi-hop entity tracking, single-hop precision, contradiction resolution, temporal reasoning) and pre-declared random seeds (`[42, 137, 2026]`, 30 runs per arm).
   - Evaluated baseline LLM controller against candidate walk policies (`greedy_similarity`, `early_exit_answer`, `sheaf_contradiction_backtrack`, `stochastic_softmax`).
   - Wired into Dokime paired bootstrap analysis (10,000 resamples, 95% CI, seed 42).
2. **Empirical Findings & Decision Gate**:
   - Executed live evaluation and published report: [`docs/review/anadrome-frozen-live-walk-milestone3-report.md`](file:///home/akarta/Projects/Metis/docs/review/anadrome-frozen-live-walk-milestone3-report.md).
   - **Verdict**: `NO_EVIDENCE`. None of the candidate policies exceeded baseline controller correctness with a paired 95% CI excluding zero.
   - **Action**: Per Spec Section 6 ("tie or no evidence keeps the current controller - a recorded result, not a failed experiment"), `baseline_llm` remains the production default. All candidates remain registered in `APPROVED_CONTROLLER_POLICIES` (`dianoia/dianoia/walk.py`) for specialized use cases.
3. **Antigravity IDE Stability Hardening**:
   - Hardened `_normalize_for_canonical_json` in `dianoia/dianoia/anadrome/canonical.py` with cycle detection and explicit mock-safety to prevent infinite recursion and explosive 48GB OOM kills during test runs.
   - Added type-safe attribute and score extraction in `WalkController._get_neighbours`.
4. **Unit & Integration Test Suite (`dianoia/tests/test_live_walk_study.py`)**:
   - 4 comprehensive tests passing covering dataset integrity, single-query execution, sheaf contradiction conflict rejection, and full bootstrap evaluation.
   - Dianoia test suite: **93 passed** (up from 89).
   - Total Metis test suite: **375 passed across all suites** (70 prosopon, 173 kybernetes, 93 dianoia, 39 root).

---

## 3. Implementation Status Summary

All 10 prioritized synthesis roadmap tasks across all 5 tracks are now **fully completed and verified**:
- **Track A (Pyle)**: Loopback closure, SQLite device registry, WebAuthn RP, WebRTC DataChannel engine, Operator remote approvals, panic lockdown, doctor diagnostics (Phases 0, 1, 2, 8).
- **Track B (Anadrome)**: Canonical JSON observation hashing, decision tree logging harness, in-process probe cache, zero imputation replay engine, policy zoo, Dokime paired bootstrap evaluation, frozen live walk test (Phases 4, 6, 9).
- **Tracks C & D (Schedion & Symbolon)**: 5060 Ti hardware validation, candidate generation & code identity adapters, E1 diagnostic, Stage 3 and E2 controlled empirical comparisons, negative research reports and archival verdicts (Phases 3, 5, 7).

---

## 4. Key Documentation Links

- **Main Manual**: [`docs/MSS.md`](file:///home/akarta/Projects/Metis/docs/MSS.md)
- **Pyle WebRTC + Passkey Spec**: [`docs/review/mss-pyle-webrtc-passkey-spec.md`](file:///home/akarta/Projects/Metis/docs/review/mss-pyle-webrtc-passkey-spec.md)
- **Synthesis Roadmap (All 5 Subsystems)**: [`docs/review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md`](file:///home/akarta/Projects/Metis/docs/review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md)
- **Anadrome Replay Spec**: [`docs/review/metis-anadrome-walk-replay-experiment-spec-aa69f396.txt`](file:///home/akarta/Projects/Metis/docs/review/metis-anadrome-walk-replay-experiment-spec-aa69f396.txt)
- **Anadrome Milestone 3.2 Report**: [`docs/review/anadrome-frozen-live-walk-milestone3-report.md`](file:///home/akarta/Projects/Metis/docs/review/anadrome-frozen-live-walk-milestone3-report.md)
- **Schedion Stage 3 Report**: [`docs/review/schedion-stage3-controlled-comparison-report.md`](file:///home/akarta/Projects/Metis/docs/review/schedion-stage3-controlled-comparison-report.md)
- **Symbolon E2 Report**: [`docs/review/symbolon-candidate-fusion-e2-report.md`](file:///home/akarta/Projects/Metis/docs/review/symbolon-candidate-fusion-e2-report.md)
- **Review Specs Index**: [`docs/REVIEW_INDEX.md`](file:///home/akarta/Projects/Metis/docs/REVIEW_INDEX.md)



