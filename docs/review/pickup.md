# Metis Development Resume Guide (`pickup.md`)

**Last Updated:** 2026-10-31  
**Current Branch:** `main`  
**Current HEAD Commit:** `4636a0a`  
**Working Tree Status:** Clean; all services active under systemd; 1,137 passing tests verified across all 11 monorepo packages + root + experiments + VS Code Void IDE client (100% green, 3 skipped).  
**Active In-Flight Experiment:** `experiments/text_vs_graph/` — Confirmatory Phase 1 Run (300/300 stories, 15,600 LLM calls) and Confirmatory Phase 2 Run (300/300 stories, 5,100 LLM calls under real LLM extractor noise) fully completed and audited in `CONFIRMATORY_PHASE1_REPORT.md` and `CONFIRMATORY_PHASE2_REPORT.md`. Both Option 1 (Zero-Hallucination Graph Filtering, Dynamic Consistency Fallback & Constructive Grounding, Item 91) and Option 2 (Heuresis Discovery Engine, OpenReview Boundary Crawler & Sheaf OOV Synthesizer, Item 92) fully implemented and verified across Stoma, Kybernetes, Dianoia, and Dokime.

---

## 1. Quick Health Check & Experiment Resume (Run After Reboot)

To verify all services and environments are operational after system reboot:

```bash
# 1. Check systemd ambient background services
systemctl --user status metis-hestia metis-kybernetes metis-prosopon metis-stoma-watcher

# 2. Resume Confirmatory Phase 1 Run (Picks up at Story 102/300)
PYTHONPATH=. ./.venv/bin/python experiments/text_vs_graph/run_confirmatory.py

# 3. Quick test suites sanity check (optional)
./.venv/bin/pytest kybernetes/tests dokime/tests -q
```

# 2. Run Kybernetes orchestrator & tools test suite (193 tests)
./.venv/bin/pytest kybernetes/tests -q

# 3. Run Dokime evaluation & release gate test suite (213 tests)
./.venv/bin/pytest dokime/tests -q

# 4. Run LTHCM Poincaré memory engine test suite (133 tests)
./LTHCM/.venv/bin/pytest LTHCM/tests -q

# 5. Run Stoma ingestion & structured OCR test suite (118 tests)
./stoma/.venv/bin/pytest stoma/tests -q

# 6. Run Dianoia MCTS deliberation test suite (93 tests)
./.venv/bin/pytest dianoia/tests -q

# 7. Run Hestia speculative cascades & serving test suite (77 tests, 1 skipped)
./hestia/.venv/bin/pytest hestia/tests -q

# 8. Run Prosopon WebRTC & access store test suite (73 tests)
./prosopon/.venv/bin/pytest prosopon/tests -q

# 9. Run Paideia active training pipeline test suite (50 tests)
./paideia/.venv/bin/pytest paideia/tests -q

# 10. Run Kleroterion stochastic policy test suite (43 tests)
./kleroterion/.venv/bin/pytest kleroterion/tests -q

# 11. Run Krisis claim grounding & conformal gates test suite (40 tests)
./.venv/bin/pytest krisis/tests -q

# 12. Run Phylax capability security gateway test suite (36 tests)
./.venv/bin/pytest phylax/tests -q

# 13. Run Root integration test suite (39 tests)
./.venv/bin/pytest tests -q

# 14. Compile VS Code Void IDE client (0 errors)
npm --prefix vscode run compile

# Monorepo Total: 1,110 tests passing across all 11 subsystems + root (100% green, 1 skipped)
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

### Phase 10: Architectural Review Synthesis & Kernel Hardening (Completed)
1. **Live Dianoia Deliberation in Kybernetes (`kybernetes/adapters/dianoia.py`)**:
   - Eliminated stub deliberation; wired live execution into `WalkController`.
   - Maps visited passages into structured `Citation` metadata.
   - Preserves flat fallback upon zero-edge indices or deliberation errors.
   - Decoupled circular imports by re-homing `APPROVED_CONTROLLER_POLICIES` into `dianoia/anadrome/policies.py`.
2. **Sandboxed Execution Boundary (`kybernetes/execution/backend.py`)**:
   - Implemented `BubblewrapBackend` (`bwrap --ro-bind / / --tmpfs /tmp --dir <cwd> --bind <cwd> <cwd> --unshare-pid --unshare-ipc --unshare-uts`) with process group cleanup and fallback to `LocalProcessBackend`.
   - Routed `SafeSandboxedBash`, `TerminalSeatAdapter.launch_harness`, and `TerminalCodingAgent` verification hooks through backend execution.
   - Isolated `$HOME` to `target_cwd`, eliminating Substrate review host-execution leaks.
3. **MSS Context Retention & Content-Addressed Spillover (`kybernetes/engine/terminal_agent.py`)**:
   - Fixed terminal agent context amnesia by capturing alternating user tool observations in `history`.
   - Populated `turn_history` with `CodingTurnState` records.
   - Retained goal and plan constraints permanently in `current_prompt`.
   - Implemented SHA-256 content-addressed disk spillover to `.metis/artifacts/tool_out_<hash>.txt` for outputs $> 1,000$ bytes.
4. **Lochos Recursive Subagent Delegation & Worktree Rollback (`kybernetes/lochos/squad.py`, `adapters.py`)**:
   - Scoped `GitMilestoneCheckpointer` to the seat's isolated worktree directory (`seat.worktree_path`).
   - Wired automatic milestone rollback upon task failure (`rollback(checkpoints[0])`).
   - Added subagent lifecycle management: `get_subagent_manager`, `spawn_subagent`, `merge_subagent`, `discard_subagent`.
5. **Paideia Data Pipeline Anadrome Replay Ingestion (`paideia/data/walk_loader.py`)**:
   - Implemented `load_anadrome_replay_dataset` to parse Anadrome continuation trees and decision records (`ReplayTreeRecord`) into supervised policy steps (`PolicyStepSample`).
   - Added automatic format detection in `load_walk_dataset` to transparently ingest both Dianoia walk logs and Anadrome replay tree logs.
   - Reordered honest refusal gate in `QLoRATrainingRunner` to verify `allow_simulated` prior to hardware VRAM checks.
6. **Stoma Offline Knowledge Graph Edge Builder (`stoma graph build`)**:
   - Implemented `build_knn_knowledge_graph` in `stoma/graph_builder.py` to calculate cosine similarities across chunk embeddings ($\tau \ge 0.75$) and install wormhole graph edges.
   - Exposed opt-in CLI group `stoma graph build` without disturbing Stoma v1 zero-edge default ingestion invariants.
7. **Phase 11: Ambient Daemons, Local Zero-Friction & Hardened Sandboxing (Completed)**:
   - **Systemd User Units**: Active services `metis-hestia.service`, `metis-kybernetes.service`, `metis-prosopon.service`, and `metis-stoma-watcher.service` running as managed ambient background daemons.
   - **Local Loopback Zero-Friction**: Local CLI tools (`kybernetes code`) automatically resolve `HESTIA_SERVICE_TOKEN` from `deploy/metis.env` (`0600`), eliminating manual exports and PIN prompts for local machine operators.
   - **UI Pairing Clutter Removal**: Removed mobile pairing modals, padlock icons, and obsolete PIN handlers from Prosopon UI.
   - **Horizon 2 Bubblewrap Sandboxing Lockdown**: Enforced default `--unshare-net` network namespace isolation and confined `HOME=<cwd>`, `TMPDIR=/tmp` env vars in `BubblewrapBackend` to block rogue socket connections and prevent exfiltration during coding agent runs. Verified via unit tests in `kybernetes/tests/test_execution_backend.py`.

8. **Phase 12: Google Research Evaluation Integration (Tracks A through G) (Completed)**:
   - **Track A (Verifiable Instruction Following - IFEval)**: Implemented 20 programmatic instruction constraint checkers in `dokime/src/dokime/tasks/ifeval_checkers.py` (JSON format/schema, word count, sentence count, paragraph count, keyword presence, forbidden words with inflection guards, keyword frequency, section headers, start/end constraints, numbered lists, postscripts, letter casing constraints, quotation constraints, highlight section markdown constraints, two-response delimiters, no-comma punctuation constraints, bullet lists, all-lowercase constraints, title wrapping delimiters, and letter frequency constraints) and `IFEvalTask` in `dokime/src/dokime/tasks/ifeval.py` evaluated against 30 dense test prompts in `dokime/src/dokime/tasks/data/ifeval_suite.json`. Verified via 25 unit/integration tests in `dokime/tests/test_ifeval.py`.
   - **Track B (Conformal Uncertainty Prediction Sets - KnowNo)**: Implemented `ConformalCalibrator` and `ConformalPredictionSet` with exact finite-sample coverage guarantees ($1 - \alpha$) in `krisis/krisis/gates/conformal.py`. Fully wired into `KrisisSelector.evaluate()` with 7 unit tests in `krisis/tests/test_conformal.py`.
   - **Track C (Sufficient Input Subsets - SIS Explainer)**: Implemented backward greedy token attribution in `dokime/src/dokime/explainers/sis.py` and `dokime explain` CLI command. Isolates minimal sufficient rationales preserving prediction fidelity ($\ge \tau$) with 9 unit tests in `dokime/tests/test_sis.py`.
   - **Track D (Speculative Cascades Serving)**: Implemented speculative drafter-verifier cascading engine in `hestia/src/hestia/cascade/engine.py` and exposed `POST /v1/cascade/completions` API endpoints and `model="cascade:<draft>:<verifier>"` transparent routing. Verified via 5 tests in `hestia/tests/test_speculative_cascades.py`.
   - **Track E (Active Selective Prediction - ASPEST)**: Implemented `KrisisActiveRecord` and `ASPESTSampler` in `paideia/paideia/data/aspest_sampler.py`, exposing `paideia dataset sample-active` CLI for facility location clustering and uncertainty scoring over abstained query pools. Verified via 9 tests in `paideia/tests/test_aspest_sampler.py` and `krisis/tests/test_active_learning.py`.
   - **Track F (Google ScaNN ANN Index & Dokime ANN Yardstick)**: Implemented `ScannIndex` wrapping `scann.scann_ops_pybind` with anisotropic Tree-AH quantization and brute-force SIMD fallback in `LTHCM/lthcm/core/scann_index.py`. Implemented `ANNBenchmarkSuite` and `dokime ann-benchmark` CLI. Achieved >108k QPS (11.47x speedup over linear baseline) with 100% recall. Verified via 9 unit tests in `LTHCM/tests/test_scann_index.py` and `dokime/tests/test_ann_benchmark.py`.
   - **Track G (VRDU Structured Ingestion & Extraction Task)**: Enhanced `stoma/src/stoma/plugins/ocr.py` to parse complex semi-structured document layouts into clean Markdown tables, DePlot/MatCha chart summaries (`# Chart: Title`, `Series: ...`), and key-value blocks. Implemented `VRDUTask` with a 20-sample dense semi-structured document benchmark suite in `dokime/src/dokime/tasks/data/vrdu_suite.json`. Verified via 9 unit tests in `stoma/tests/test_ocr_structured.py` and `dokime/tests/test_vrdu.py`.

9. **Phase 13: Round 5 Security Audit Remediations & Zero-Loopback Model (Completed)**:
   - **F-02 (Code Mode Generator Frame Traversal Sandbox Escape)**: Blocked generator frame attributes (`gi_frame`, `cr_frame`, `ag_frame`, `f_back`, `f_globals`, `f_locals`, `f_builtins`, `f_code`, `format`, `format_map`, `__dict__`, `__closure__`), prohibited `ast.Yield` / `ast.YieldFrom`, blocked format calls in AST, and reclassified `run_code_mode` as `ToolSideEffectClass.SYSTEM_EXECUTION` in `kybernetes/kybernetes/tools/code_mode.py`.
   - **F-03 (Prosopon WebRTC Unauthenticated Device ID Injection)**: Enforced strict authentication in `prosopon/prosopon/server.py:webrtc_offer`; rejected unauthenticated callers with 401 Unauthorized.
   - **F-04 (Kybernetes Zero-Loopback Model & Route Protection)**: Removed loopback exemptions in `require_service_auth`; protected all Lochos squad and Pinax endpoints with authentication; injected shared `phylax_adapter` instance into `ToolRegistry`.
   - **F-05 (Prosopon Workbench File Sensitive Credential Disclosure)**: Added strict path denylisting in `prosopon/prosopon/server.py:workbench_file` blocking `.host_owner_token`, `metis.env`, `paired_tokens.json`, `.env`, keys, SQLite databases, and internal snapshot/policy directories with 403 Forbidden.
   - **F-07 (Execution Pipeline Phylax Gate Hard-Deny Fallthrough)**: Enforced immediate `TurnOutcomeStatus.DENIED` event termination when Phylax denies an action; replaced non-existent `"guarded_action"` with `"write_file"` in `router.py`.
   - **F-08 (Krisis Claim Negation False Contradiction Polarity Clash)**: Replaced naive contradiction keyword checks with semantic polarity clash analysis between claim and evidence in `kybernetes/kybernetes/adapters/krisis.py`.
   - **Pickle Security Hardening**: Flipped `allow_trusted_pickle=False` default across Stoma graph builder and interchange modules.

10. **Phase 14: Dokime Symbolon Discrete Code Retrieval Benchmark Suite (Completed)**:
    - **Symbolon Task Plugin (`dokime/src/dokime/tasks/symbolon_benchmark.py`)**: Implemented `SymbolonBenchmarkTask` conforming to `Task` protocol (`task_id="symbolon_code_retrieval"`, `load()`, `score()`, `score_detailed()`) and registered in `BUILTIN_TASKS` and `list_task_info()`.
    - **Diagnostic Benchmark Dataset (`dokime/src/dokime/tasks/data/symbolon_suite.json`)**: Curated 8 multi-domain benchmark queries across biochemistry, EU geopolitics, aerospace rocketry, topological sheaf theory, EUV lithography, and SHA-256 hash chaining with explicit gold relevance and contradiction labels.
    - **4-Arm Comparative Benchmark Suite (`SymbolonRetrievalBenchmarkSuite`)**: Evaluates (1) Flat dense baseline (LTHCM Flat), (2) Symbolon discrete overlap alone (L4), (3) Uncapped additive union ($C_{\text{base}} \cup C_{\text{sym}}$), and (4) Capped RRF fusion (matched top-$k$). Computes mean recall@k, contradiction retention, retriever agreement, candidate expansion factor, and paired bootstrap deltas.
    - **CLI Command (`dokime symbolon-benchmark`)**: Added CLI subcommand supporting `--top-k`, `--level`, and `--output <path>` JSON report export. Produces formatted GitHub-style Markdown tables and enforces `ARCHIVE_UNCAPPED_NO_LOSS_RECORDED` decision verdict tracking.
    - **Unit & Integration Test Suite (`dokime/tests/test_symbolon_benchmark.py`)**: 6 tests passing covering task protocol compliance, response parsing, metric computation, 4-arm comparative suite execution, and CLI command execution.
    - **Total Dokime Tests**: **201 passed** (up from 195).
    - **Total Metis Monorepo Tests**: **1,094 passed** across all 11 core subsystems.

11. **Phase 15: Dokime Executable Graph Reasoning & Paideia Boundary Practice Selection (Completed)**:
    - **Dokime Executable Graph QA (`dokime/src/dokime/tasks/executable_graph.py`)**: Implemented verifiable multi-hop reasoning benchmark evaluated against deterministic executions over frozen, SHA-256 hashed synthetic fact fixture (`07295795b5fe11657898373822caf8376e0e283f6f3cbf7d978ae8a3846de12e`). `GraphQueryExecutor` evaluates forward/reverse multi-hop chains and multi-constraint intersections with path-level fact attribution. `ExecutableGraphTask` (`task_id="executable_graph_qa"`) scores answer precision/recall/F1, exact match, and evidence citation support, refusing evidence-grounded credit to unsupported guesses.
    - **Suite Dataset (`dokime/src/dokime/tasks/data/executable_graph_suite.json`)**: Curated 20 wording-audited test and practice benchmark cases across single-hop, chain-2hop, chain-3hop, intersection, multi-answer, and negative/empty queries.
    - **Paideia Boundary Practice Selection (`paideia/paideia/data/boundary_sampler.py`)**: Implemented variance-based practice batch selector over repeated rollout attempts ($N \ge 2$), prioritizing learning frontier tasks ($\sigma^2 > 0$) with stratified quotas for retention checks (all-pass) and hard diagnostics (all-fail). Included uniform random baseline sampler for matched-budget empirical comparison.
    - **Dokime CLI Command (`dokime graph-rollout`)**: Added repeated rollout suite (`ExecutableGraphRolloutSuite`) and CLI command for repeated-attempt rollout evaluation and category stratification across policies (including live served Hestia models with knowledge fixture prompt context).
    - **Unit & Integration Test Suites**: 12 tests passing in `dokime/tests/test_executable_graph.py` (213 Dokime tests passing) and 4 tests passing in `paideia/tests/test_boundary_sampler.py` (50 Paideia tests passing).
12. **Phase 16: Zero-Hallucination Graph Filtering & Heuresis Discovery Engine (Completed)**:
    - **Zero-Hallucination Graph Filtering (Item 91)**: Added `DeterministicSpanFilter` in `stoma/src/stoma/filters.py` verifying entity mention, timestamp, and relation spans against raw source text with zero token overhead. CLI `stoma graph verify-grounding`. Integrated dynamic consistency gating (`graph_consistency_floor: 0.95`, `fallback_to_pure_text_on_inconsistency: True`) in `kybernetes/kybernetes/routing/router.py` and `kybernetes/kybernetes/adapters/dianoia.py` with automatic flat text fallback. Enforced constructive forward chronological planning in `dianoia/dianoia/prompts.py` and `kybernetes/kybernetes/engine/pipeline.py`.
    - **Heuresis Discovery Engine & OpenReview Crawler (Item 92)**: Implemented `OpenReviewBoundaryCrawler` and `CommunityAvailabilityIndex` in `stoma/src/stoma/crawlers/openreview.py` parsing peer review contested boundaries ($\sigma^2 \ge 1.5$, $4.0 \le \bar{s} \le 6.5$). CLI `stoma openreview crawl-fixture`. Implemented `HeuresisDiscoveryEngine` in `dokime/src/dokime/tasks/heuresis.py` with fast dual contrastive scoring, Sheaf-Laplacian topological OOV concept synthesis, and 15% fringe outlier quota. Registered `"heuresis_scientific_discovery"` in Dokime task registry.
    - **17 New Unit & Integration Tests**: Verified across `stoma/tests/test_span_filter.py`, `stoma/tests/test_openreview_crawler.py`, `kybernetes/tests/test_adapters.py`, and `dokime/tests/test_heuresis.py`.

13. **Phase 17: Empirical Evaluation of Text vs Graph Under Real LLM Extractor Noise (Phase 2 Confirmatory Completed)**:
    - **Natural Extractor Profile ($N=300$ stories, 5,100 LLM operations)**: Implemented zero-shot LLM extractor in `experiments/text_vs_graph/extractor.py`. Evaluated on 300 stories (`conf_s001`–`conf_s300`, seeds `4001`–`4300`). Natural extractor fidelity achieved **$99.08\%$ mean** ($279/300$ 100% perfect, only 1 entity hallucination across 2,400 gold events, $0.71\%$ omission rate).
    - **H2' Confirmed (Text + Extracted Graph vs Text Baseline)**: Arm C achieved **$96.67\%$** vs **$90.83\%$** for Arm A ($+5.83\text{ pp}$ gain, $95\%\text{ CI } [+0.0417, +0.0742]$, verdict: **BETTER**, $p < 10^{-5}$).
    - **Real vs Synthetic 30% Corruption**: Arm B (Graph Only) achieved **$93.75\%$** (vs $73.83\%$ under synthetic corruption, $+19.92\text{ pp}$, $p < 10^{-5}$, **BETTER**). Arm C (Text + Graph) achieved **$96.67\%$** (vs $86.58\%$, $+10.08\text{ pp}$, **BETTER**).
    - **Arm F Dominance**: Arm F (Constructive Chronological Control) achieved **$99.58\%$ accuracy** ($1,195/1,200$ questions correct), definitively outperforming Arm D ($97.08\%$, $-2.50\text{ pp}$, $95\%\text{ CI } [-0.0350, -0.0158]$) and eliminating 97.9% of length truncations.
    - **Counterfactual Advantage (Q3)**: Arm F ($98.7\%$) and Arm C ($86.7\%$) dramatically outperformed pure text Arm A ($75.3\%$, $+11.4\text{ pp}$ to $+23.4\text{ pp}$ gain).
    - **Audited Report**: Full empirical report and bootstrap statistics compiled in `experiments/text_vs_graph/CONFIRMATORY_PHASE2_REPORT.md`.
    - **Monorepo Total**: **1,137 passed tests** across all 11 core subsystems + experiments + root (100% green, 3 skipped).

---

## 3. Implementation Status Summary

All prioritized synthesis roadmap tasks, review remediation circuits, Google Research tracks, and security hardening horizons are **fully completed, verified, and synchronized**:
- **Track A (Pyle & Remote Control)**: Loopback closure, SQLite device registry, WebAuthn RP, WebRTC DataChannel engine, Operator remote approvals, panic lockdown, doctor diagnostics (Phases 0, 1, 2, 8).
- **Track B (Anadrome)**: Canonical JSON observation hashing, decision tree logging harness, in-process probe cache, zero imputation replay engine, policy zoo, Dokime paired bootstrap evaluation, frozen live walk test (Phases 4, 6, 9).
- **Tracks C & D (Schedion & Symbolon)**: 5060 Ti hardware validation, candidate generation & code identity adapters, E1 diagnostic, Stage 3 and E2 controlled empirical comparisons, negative research reports and archival verdicts (Phases 3, 5, 7).
- **Track E (Architectural Review Remediation & Kernel Hardening)**: Live Dianoia walk deliberation, Bubblewrap sandboxed containment with default network namespace unsharing, MSS terminal context retention, Lochos worktree checkpointer rollback & subagent delegation, Paideia Anadrome ingestion, Stoma offline knowledge graph edge builder (Phase 10).
- **Track F (Sovereign Service Deployment & Sandboxing)**: Systemd ambient daemon orchestrations, local loopback zero-friction authorization, removal of mobile PIN clutter, and Bubblewrap network/home confinement (Phase 11 / Horizons 1 & 2).
- **Track G (Google Research Evaluation Tracks A–G)**: IFEval instruction following, KnowNo conformal uncertainty, SIS backward token attribution, Speculative Cascades serving, ASPEST active prediction, Google ScaNN anisotropic Tree-AH vector index, and VRDU structured multimodal ingestion (Phase 12).
- **Track H (Round 5 Security Hardening & Zero-Loopback Model)**: F-02 to F-08 security defect remediations, generator frame sandbox traversal prevention, path traversal credential denylist, and fail-closed gate pipelines (Phase 13).

---

## 4. Key Documentation Links

- **Main Manual**: [`docs/MSS.md`](file:///home/akarta/Projects/Metis/docs/MSS.md)
- **Comprehensive Gaps and Defects Register**: [`docs/review/METIS_COMPREHENSIVE_GAPS_AND_DEFECTS_REGISTER.md`](file:///home/akarta/Projects/Metis/docs/review/METIS_COMPREHENSIVE_GAPS_AND_DEFECTS_REGISTER.md)
- **Comprehensive Project Review (2026-10-21)**: [`docs/review/COMPREHENSIVE_PROJECT_REVIEW_2026_10_21.md`](file:///home/akarta/Projects/Metis/docs/review/COMPREHENSIVE_PROJECT_REVIEW_2026_10_21.md)
- **Round 5 Audit Brief**: [`docs/review/metis-repo-wide-round5-531c7bb-findings-v5-ea1789c2.txt`](file:///home/akarta/Projects/Metis/docs/review/metis-repo-wide-round5-531c7bb-findings-v5-ea1789c2.txt)
- **Google Research Takeaways (Metis General)**: [`docs/review/metis-google-research-takeaways-2026-10-28-db336cdc-d4cccfb9-0358fb51.txt`](file:///home/akarta/Projects/Metis/docs/review/metis-google-research-takeaways-2026-10-28-db336cdc-d4cccfb9-0358fb51.txt)
- **Google Research Takeaways (LTHCM Memory)**: [`docs/review/lthcm-google-research-takeaways-2026-10-28-db336cdc-3e696024-ea23eeaf.txt`](file:///home/akarta/Projects/Metis/docs/review/lthcm-google-research-takeaways-2026-10-28-db336cdc-3e696024-ea23eeaf.txt)
- **Google Research Takeaways (Dioptra Evaluation)**: [`docs/review/dioptra-google-research-takeaways-2026-10-28-db336cdc-70fd015d-ab6f215f.txt`](file:///home/akarta/Projects/Metis/docs/review/dioptra-google-research-takeaways-2026-10-28-db336cdc-70fd015d-ab6f215f.txt)
- **Pyle WebRTC + Passkey Spec**: [`docs/review/mss-pyle-webrtc-passkey-spec.md`](file:///home/akarta/Projects/Metis/docs/review/mss-pyle-webrtc-passkey-spec.md)
- **Synthesis Roadmap (All 5 Subsystems)**: [`docs/review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md`](file:///home/akarta/Projects/Metis/docs/review/metis-ncm-anadrome-pyle-synthesis-roadmap-2026-10-18.md)
- **Anadrome Milestone 3.2 Report**: [`docs/review/anadrome-frozen-live-walk-milestone3-report.md`](file:///home/akarta/Projects/Metis/docs/review/anadrome-frozen-live-walk-milestone3-report.md)
- **Schedion Stage 3 Report**: [`docs/review/schedion-stage3-controlled-comparison-report.md`](file:///home/akarta/Projects/Metis/docs/review/schedion-stage3-controlled-comparison-report.md)
- **Symbolon E2 Report**: [`docs/review/symbolon-candidate-fusion-e2-report.md`](file:///home/akarta/Projects/Metis/docs/review/symbolon-candidate-fusion-e2-report.md)
- **SwiLA Stage 1 Empirical Report**: [`docs/review/swila-session-state-stage1-report.md`](file:///home/akarta/Projects/Metis/docs/review/swila-session-state-stage1-report.md)
- **Dianoia Compression Stage 1 Report**: [`docs/review/dianoia-compression-stage1-report.md`](file:///home/akarta/Projects/Metis/docs/review/dianoia-compression-stage1-report.md)
- **Review Specs Index**: [`docs/REVIEW_INDEX.md`](file:///home/akarta/Projects/Metis/docs/REVIEW_INDEX.md)

---

## 5. Deployment Horizons & Future Options

With all core engine components, security perimeters (Pyle/Phylax), memory subsystems (Stoma/LTHCM), deliberative walks (Dianoia/Kleroterion), multi-agent coordination (Kybernetes/Lochos/Prosopon), and sandboxed execution (Bubblewrap) verified with 1,094 passing tests:

### Horizon 1: Full-Stack Sovereign Local Service Deployment (Completed)
* Systemd user units (`metis-hestia.service`, `metis-kybernetes.service`, `metis-prosopon.service`, `metis-stoma-watcher.service`) active and persistent.
* Host authority automated (`deploy/metis.env` and `.host_owner_token` with `0600` permissions); zero friction on localhost.
* Ambient `stoma watch` daemon monitoring `/home/akarta/Documents` into Poincaré LTHCM memory.

### Horizon 2: Hardened Sandboxing & Network Egress Lockdown (Completed)
* `BubblewrapBackend` profile lockdown: default unshared network namespace (`--unshare-net`), read-only root (`--ro-bind / /`), isolated tmpfs (`--tmpfs /tmp`), confined `HOME=<cwd>` and `TMPDIR=/tmp`.
* Verified with unit tests ensuring socket calls are rejected inside the sandbox envelope unless explicitly permitted (`allow_network=True`).

### Horizon 3: Multi-Device Remote Control Plane (Pyle WebRTC & Remote MSS Void)
* **Objective**: Connect external MSS Void copies or remote companion devices securely across network boundaries.
* **Architecture**:
  1. **Remote MSS Void Peer Connection**: External peers authenticate via `KYBERNETES_SERVICE_TOKEN` and `HESTIA_SERVICE_TOKEN` over HTTPS or WireGuard/Tailscale VPN without public port exposure.
  2. **WebAuthn Biometric Passkey Enrollment**: Use Prosopon's Relying Party to enroll hardware authenticators (TouchID, FaceID, YubiKey) for companion devices.
  3. **Interactive Remote Approvals**: WebRTC DataChannel alerts prompting 1-tap Approve/Reject for Phylax Tier 2/Tier 3 mutating operations.






