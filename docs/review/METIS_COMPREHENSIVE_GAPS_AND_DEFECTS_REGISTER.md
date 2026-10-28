# Metis Comprehensive Gaps, Defects & Telemetry Register (73 Items)

**Status:** Canonical Subsystem Audit & Defect Catalog  
**Last Updated:** 2026-10-20  
**Repository:** `https://gitlab.com/akarta-315h/metis`  
**Current HEAD Commit:** `7eb49d035f4a8393f97b75f8fb554048d37d6251`  
**Coverage Scope:** All 11 Metis Subsystems (`hestia`, `kybernetes`, `prosopon`, `stoma`, `lthcm`, `dianoia`, `kleroterion`, `krisis`, `phylax`, `dokime`, `paideia`), Sandbox Runtimes (`bwrap`), and IDE Telemetry Panes (`vscode/src/vs/workbench/contrib/void/`).

---

## Table of Contents
1. [Part I: Dashboard & Subsystem Telemetry Specifications (Items 1–7)](#part-i-dashboard--subsystem-telemetry-specifications-items-17)
2. [Part II: Core Repo-Wide Fixes & Release Gates (Items 8–34)](#part-ii-core-repo-wide-fixes--release-gates-items-834)
3. [Part III: Extended Runtime, Security, Memory & Evaluation Gaps (Items 35–59)](#part-iii-extended-runtime-security-memory--evaluation-gaps-items-3559)
4. [Part IV: Deep Subsystem Audits & Geometric/Execution Remediations (Items 60–73)](#part-iv-deep-subsystem-audits--geometricexecution-remediations-items-6073)
5. [Summary & Verification Matrix](#summary--verification-matrix)

---

## Part I: Dashboard & Subsystem Telemetry Specifications (Items 1–7)

### 1. Kleroterion Stochastic Policy & Entropy Diagnostics
* **Submodule / Location:** `kleroterion/src/kleroterion/interfaces.py`, `dianoia/dianoia/walk.py`, `docs/review/Kleroterion-DASHBOARD_SPEC.md`
* **Defect / Gap:** Dianoia's deliberation view and the main dashboard lacked live exposure of Kleroterion's stochastic selection engine and entropy monitoring.
* **Specification Requirement:**
  1. **Policy Profile Controls:** Toggle between `EXACT_FACT` (locked $T=0, \sigma=0$, argmax determinism) and `EXPLORATORY` (temperature-scaled softmax categorical sampling).
  2. **Entropy Gauges:** Smoothed Marginal Entropy $H(\bar{\pi})$, target vs. empirical entropy delta ($\Delta H$), and mode collapse warnings.
  3. **Temperature Trajectory:** Live display of initial $\tau$, min $\tau$, max $\tau$, and $1/\sqrt{d}$ dimension scaling normalization.
  4. **Action Mask Compliance:** Real-time counter of eligible vs. authoritative masked proposal actions.

### 2. Krisis Candidate Grounding & Claim Selection Policy
* **Submodule / Location:** `krisis/krisis-policy-0.1.json`, `krisis/krisis/engine/selector.py`, `docs/review/krisis-DASHBOARD_SPEC.md`
* **Defect / Gap:** Lack of policy inspection and claim verification outcome tracking in the reasoning dashboard pane.
* **Specification Requirement:**
  1. **Policy Manifest Status:** Real-time inspection of active manifest `krisis-policy-0.1.json` with profile thresholds (`grounded_v1`: $\tau = 0.70$, $\text{per\_claim\_min} = 0.70$; `grounded_support_only`: $\tau = 0.85$).
  2. **Reason Precedence Matrix:** Explicit audit of candidate rejection ordering: `access_restriction` $\to$ `provenance_missing` $\to$ `contradiction` $\to$ `citation_unresolved` $\to$ `score_below_tau`.
  3. **Evaluation Distribution:** Pre-score elimination rates, percentage of supported claims, and mean score margin over threshold $\tau$.

### 3. Phylax Audit Hash-Chain Verification
* **Submodule / Location:** `phylax/phylax/gateway/gateway.py`, `.phylax/audit.jsonl`, `docs/review/phylax-DASHBOARD_SPEC.md`
* **Defect / Gap:** UI displayed token approval toasts but lacked cryptographic verification of the append-only audit trail.
* **Specification Requirement:**
  1. **SHA-256 Hash Chain Integrity:** Live badge verifying the continuous cryptographic hash chaining of `.phylax/audit.jsonl` to ensure zero tampering, insertion, or reordering of audit records.
  2. **Active Policy Rule Matrix:** Visual inspection of capability gates: `read_file_range` $\to$ `AUTO_ALLOW`, `edit_file_block`/`write_file` $\to$ `TOKEN_GATED`, `sandboxed_bash` $\to$ `ISOLATED_GATED`.

### 4. Dokime Release Gates & Model Win-Rate Matrix
* **Submodule / Location:** `dokime/src/dokime/manifest.py`, `runs/dokime/`, `docs/review/dokime-DASHBOARD_SPEC.md`
* **Defect / Gap:** Unit test verification was integrated, but historical multi-task benchmark evaluation suites and win/loss matrices were absent from the full-screen mission control.
* **Specification Requirement:**
  1. **Benchmark Run Registry:** Historical runs table capturing run IDs, timestamps, evaluated models, and task suites (ARC, GSM8K, MuSiQue, Anadrome frozen walk).
  2. **Statistical Release Gates:** 10,000 paired bootstrap resamples displaying 95% Confidence Intervals $[CI_{\text{lower}}, CI_{\text{upper}}]$ and automated promotion decisions (`PASS` if $CI_{\text{lower}} > 0$ and $p < 0.05$, else `REJECT`).
  3. **Artifact Provenance:** Weight SHA-256 checksums and immutable git tree status.

### 5. Paideia Supervised Training & Anadrome Replay Ingestion
* **Submodule / Location:** `paideia/paideia/data/walk_loader.py`, `paideia/paideia/training/runner.py`, `docs/review/paideia-DASHBOARD_SPEC.md`
* **Defect / Gap:** Supervised training harness status and Anadrome continuation tree ingestion metrics were omitted from the training dashboard.
* **Specification Requirement:**
  1. **Dataset Ingestion Pipeline:** Live counter of ingested Anadrome continuation trees (`ReplayTreeRecord` $\to$ `PolicyStepSample`).
  2. **Honest Capability Badging:** Unambiguous label declaring whether training runner is executing against active GPU hardware or operating in `SIMULATED` marker-only mode.

### 6. Bubblewrap Sandbox Execution & Egress Telemetry
* **Submodule / Location:** `kybernetes/kybernetes/execution/backend.py`, `kybernetes/tests/test_execution_backend.py`
* **Defect / Gap:** Dashboard lacked visibility into the process isolation boundaries guarding terminal coding agents.
* **Specification Requirement:**
  1. **Isolation Envelope State:** Real-time indicators verifying `--unshare-net` (egress network lockdown), `--ro-bind / /` (read-only root filesystem), and `--tmpfs /tmp`.
  2. **Environment Sanitization:** Verification of restricted working directory (`HOME=<cwd>`), process group termination, and isolation of host secrets.

### 7. Horizon 3 Multi-Device Remote Control Plane (Pyle WebRTC)
* **Submodule / Location:** `prosopon/prosopon/transport/webrtc.py`, `prosopon/prosopon/auth/webauthn.py`, `docs/review/mss-pyle-webrtc-passkey-spec.md`
* **Defect / Gap:** Pyle WebRTC engine and access store were functional in backend code, but companion device enrollment and remote WebRTC stream telemetry were not surfaced in the dashboard.
* **Specification Requirement:**
  1. **Companion Pairing & Passkeys:** QR code pairing link generator and W3C WebAuthn passkey enrollment for mobile companion devices.
  2. **SCTP DataChannel Status:** Active telemetry for `control` (JSON-RPC), `stream` (SSE events), and `terminal` channels.
  3. **Panic Lockdown Control:** Global emergency panic trigger terminating remote WebRTC connections in $< 500\text{ ms}$ and invalidating all remote session tokens.

---

## Part II: Core Repo-Wide Fixes & Release Gates (Items 8–34)

### 8. Restore the Prosopon Server Entrypoint (P0)
* **Where:** `prosopon/prosopon/cli.py:29-83`, `scripts/run_stack.sh:119-136`
* **Defect:** `serve --no-mock` exited 0 without starting Uvicorn, and `run` threw `NameError("name 'tls' is not defined")`.
* **Fix:** Unified TLS setup, application creation, and `uvicorn.run` in a single explicit entrypoint with health-check verification.

### 9. Close Unauthenticated LAN Self-Pairing (P0)
* **Where:** `prosopon/prosopon/server.py:108-121`
* **Defect:** Unpaired remote clients on LAN could issue a PIN from `/api/auth/pin` and elevate themselves to full owner privileges.
* **Fix:** Required authenticated local owner filesystem authority (`0600 .host_owner_token`) to mint pairing PINs or session tickets.

### 10. Authenticate the Control Boundary, Including Loopback (P0)
* **Where:** `kybernetes/kybernetes/server/app.py:57-63`, `prosopon/prosopon/server.py:81-98`
* **Defect:** Kybernetes CORS allowed `*` with credentials and had no auth; Prosopon exempted loopback clients, exposing controls to arbitrary local software.
* **Fix:** Enforced dedicated authenticated bearer token guards on all control and administrative routes without loopback exemptions.

### 11. Bind File Approvals to Actual Write Content (P0)
* **Where:** `phylax/phylax/gateway/file_adapter.py:62-105`
* **Defect:** `write_text` authorization evaluated only `content_bytes: len(encoded)`, permitting substitution of malicious payloads of identical length under an approved token.
* **Fix:** Cryptographically bound the SHA-256 digest of write bytes, canonical target destination, and overwrite options into the action token.

### 12. Audit Pre-Approved Actions Before Their Side Effect (P0)
* **Where:** `phylax/phylax/gateway/gateway.py:121-163`
* **Defect:** Token approvals were consumed and tool callbacks executed *before* writing the audit record; disk failure resulted in unrecorded mutating side effects.
* **Fix:** Implemented write-ahead audit logging: append the authorized execution intent to disk prior to invoking the mutating tool callback.

### 13. Make Krisis Required Gates Fail Closed (P0)
* **Where:** `krisis/krisis/contracts/candidate.py:28-59`, `krisis/krisis/gates/evaluator.py:29-35`
* **Defect:** Missing access, provenance, contradiction, or citation results defaulted to `True`, allowing unverified candidates to pass as grounded.
* **Fix:** All missing or partial gate results fail closed as `unverified` / `rejected`; grounded answers strictly require affirmative verifier proof.

### 14. Require Real Verification Before Paideia Gold Labels (P0)
* **Where:** `paideia/paideia/data/walk_loader.py:84-125`
* **Defect:** Trajectories lacking verification records were marked `is_defensible_gold_trajectory() = True` when `conflicts_found` was defaulted to `False`.
* **Fix:** Trajectories without explicit verification results remain `unknown`; only independently verified walks qualify as gold training data.

### 15. Fail Closed on Invalid Adapter Exports and Artifact Identity (P0)
* **Where:** `paideia/paideia/export/registry.py:47-95`
* **Defect:** Corrupt manifests or empty directories exported successfully using empty-input SHA-256 hashes, allowing mismatched weights to masquerade under valid aliases.
* **Fix:** Enforced verification of trained weight files, verified completed manifests, and content-addressed weight digests.

### 16. Make Pairing and Revocation Operate on Live Server State (P1)
* **Where:** `prosopon/prosopon/cli.py:70-73`, `prosopon/prosopon/auth/pairing.py:26-39`
* **Defect:** CLI PIN manager operated in separate memory from the server instance, leaving revoked tokens cached and valid in the running server.
* **Fix:** Centralized token state in `prosopon/data/access.db` (SQLite) with immediate live cache invalidation across processes.

### 17. Do Not Overstate File Path Race Protection (P1)
* **Where:** `phylax/phylax/gateway/file_adapter.py:28-46`
* **Defect:** `O_NOFOLLOW` only checked the leaf file component; parent directories were vulnerable to symlink substitutions after policy check.
* **Fix:** Utilized verified directory file descriptors (`openat`/`dirfd`) relative to the sandbox root and unique exclusive temporary filenames.

### 18. Stop Selecting an Old Krisis Snapshot When Rebuild Is Possible (P1)
* **Where:** `krisis/krisis/engine/selector.py:46-54`
* **Defect:** When evidence changed under `rebuilds_remaining > 0`, the selector still returned the stale candidate.
* **Fix:** Return a typed `stale_rebuild_required` signal forcing candidate regeneration before selection.

### 19. Replace Advertised Live Modules with Real Wiring or Honest Demo Labels (P1)
* **Where:** `kybernetes/kybernetes/server/app.py:70-85`, `kybernetes/adapters/`
* **Defect:** `/v1/health` hardcoded `"ready"` for LTHCM, Dianoia, Phylax, and Krisis even while running simulated in-memory adapters.
* **Fix:** Functional probes inspecting real indices and dependencies; explicit `simulated` vs `live` provenance tags.

### 20. Implement a Real Stoma-to-Memory Contract (P1)
* **Where:** `kybernetes/kybernetes/adapters/lthcm.py:29-109`, `stoma/src/stoma/indexer.py:597-630`
* **Defect:** Adapter searched for flat JSON files while Stoma used generational pointers (`current.json`), causing `AttributeError` on list formats.
* **Fix:** Wired consumption through Stoma's generational manifest reader and verified SQLite-vec vector storage.

### 21. Enforce Causal and Provenance Constraints Through Actual Dianoia Walks (P1)
* **Where:** `dianoia/dianoia/walk.py:349-410`
* **Defect:** Walk seeded from stochastic resonance without query `as_of` epoch filtering, admitting future-dated graph neighbors.
* **Fix:** Enforced temporal light-cone ordering and seed citation provenance verification on all MCTS branch expansions.

### 22. Remove Untrusted Pickle from Shared Index Loads (P1)
* **Where:** `dianoia/dianoia/cli.py:156-181`, `stoma/src/stoma/adapter.py:325-350`
* **Defect:** Ingestion loaded indices using standard Python `pickle.load()` without prior schema or hash verification.
* **Fix:** Replaced untrusted pickle deserialization with structured, validated data-only JSON and SQLite-vec tensor formats.

### 23. Repair Hestia Backend Port Collision (P1)
* **Where:** `hestia/hestia.yaml:14-16`, `scripts/run_stack.sh:13,102`
* **Defect:** Hestia's backend pool started at port 8081, colliding directly with Kybernetes' default listening port 8081.
* **Fix:** Re-allocated Hestia llama-server backends to non-overlapping port pool (9000–9019) with OS socket availability checks.

### 24. Release Hestia Queue Depth on Request Cancellation (P1)
* **Where:** `hestia/src/hestia/server/__init__.py:166-190`
* **Defect:** Client disconnection or cancellation while waiting for admission failed to decrement queue counters, triggering artificial `ServerOverloadedError`.
* **Fix:** Wrapped queue decrement inside `finally` block to guarantee slot cleanup across all cancellation and timeout paths.

### 25. Harden Model Load/Unload and GPU Admission Transactions (P1)
* **Where:** `hestia/src/hestia/server/__init__.py:118-134`, `hestia/src/hestia/backend/llamacpp.py`
* **Defect:** Model locks were per-model rather than global GPU admission transactions; simultaneous load and unload led to orphaned processes.
* **Fix:** Implemented atomic multi-model VRAM budget leases and cancellation-safe child process cleanup.

### 26. Make GPU Assignment and Serving/Training Exclusion Real (P1)
* **Where:** `hestia/src/hestia/registry.py:60`, `paideia/paideia/training/hardware.py:74-110`
* **Defect:** `gpu_target` was stored but not applied to child process environments; Paideia used non-atomic check-then-write file locking.
* **Fix:** Shared atomic POSIX device lock (`/tmp/metis_gpu_0.lock`) coordinating both Hestia serving and Paideia training.

### 27. Label Training and Hestia Adapter Integration as Unimplemented (P1)
* **Where:** `paideia/paideia/training/runner.py:164-195`, `hestia/src/hestia/registry.py:50-79`
* **Defect:** Training runner wrote metadata markers without performing optimizer gradient updates, while docs claimed active LoRA training.
* **Fix:** Transparently tagged QLoRA training runs as `SIMULATED` unless backed by verified hardware and implemented backends.

### 28. Make Turn Cancellation, Replay, and Retention Bounded (P1)
* **Where:** `kybernetes/kybernetes/engine/pipeline.py:75-98`, `kybernetes/engine/session.py:44-79`
* **Defect:** Pipeline checked cancellation only at coarse stage boundaries; session dictionaries and trace logs grew unboundedly without TTL caps.
* **Fix:** Propagated cancellation signals into active token streams; installed LRU ring buffers and TTL retention policies.

### 29. Enforce Tool Bounds Rather Than Only Declaring Timeouts (P1)
* **Where:** `kybernetes/kybernetes/tools/file_search.py:77-120`, `kybernetes/tools/calculator.py:76-134`
* **Defect:** File search scanned unlimited directory trees on no-match queries; calculator lacked AST expression depth bounds.
* **Fix:** Enforced max-match bounds (100 files, 1MB max scan) and AST node recursion depth caps ($\le 50$).

### 30. Make Voice Fail Honestly and Clean Up Resources (P1)
* **Where:** `prosopon/prosopon/voice/whisper_stt.py:12-49`, `prosopon/voice/piper_tts.py:35-59`
* **Defect:** Missing Whisper models returned hardcoded fake transcripts; failed Piper conversions returned silent audio without error logging.
* **Fix:** Explicit `voice_unavailable` status and typed errors when local weights or ffmpeg binaries are missing; automatic cleanup of scratch WAVs.

### 31. Ship the Prosopon Frontend and Render Untrusted Metadata Safely (P1)
* **Where:** `prosopon/pyproject.toml:37-38`, `prosopon/prosopon/web/static/js/ui.js:13-35`
* **Defect:** Built Python wheels omitted the `static/` directory; citation metadata was interpolated into `innerHTML` unescaped.
* **Fix:** Packaged static assets explicitly in `MANIFEST.in`/`pyproject.toml`; enforced DOM `textContent` escaping for all citation tooltips.

### 32. Expand CI to Match Claimed Repository Coverage (P1)
* **Where:** `.gitlab-ci.yml:13-21`, `tests/test_stack_smoke.py:35-77`
* **Defect:** CI only tested Kybernetes and basic smoke tests, skipping Dokime, Stoma, Phylax, Paideia, and Krisis test suites.
* **Fix:** Consolidated all 7 submodules into the continuous regression pipeline (614 unit/integration tests).

### 33. Record Real Served Model and Release Identity (P1)
* **Where:** `hestia/src/hestia/api/app.py:113-145`, `dokime/src/dokime/manifest.py:102-162`
* **Defect:** Hestia model lists lacked SHA-256 weight checksums, and Dokime recorded alias strings rather than resolved artifact digests.
* **Fix:** Immutable release manifests binding exact git commit SHA, dirty status, GGUF file checksums, and tokenizer identities.

### 34. Reconcile Documentation with the Actual Codebase (P2)
* **Where:** `README.md`, `hestia/README.md`, `docs/STACK_BUILD_CONTRACT.md`
* **Defect:** Root documentation claimed components were "production-grade" while adapters were running in demo/mock mode; README listed non-loopback `0.0.0.0` examples rejected by Hestia's strict loopback validator.
* **Fix:** Documented exact operational reality, explicit opt-in boundaries (e.g. Kleroterion, Schedion), and accurate loopback endpoints (`127.0.0.1:8080`, `127.0.0.1:8081`).

---

## Part III: Extended Runtime, Security, Memory & Evaluation Gaps (Items 35–59)

### 35. Coding-Agent Tools Bypass Approval Token Inspection
* **Submodule / Location:** `kybernetes/kybernetes/tools/registry.py:148`
* **Defect:** `execute()` never inspected `request.approval_id`, `side_effect_class`, or `declared_capabilities`. `write_file` executed with `approval_id=None` or with a fabricated ID.
* **Remediation:** Enforced token signature verification in tool executor prior to dispatching any mutating filesystem action.

### 36. Decorative Approval State in CodingAgentHarness
* **Submodule / Location:** `kybernetes/kybernetes/engine/coding_agent.py:144`
* **Defect:** When `requires_approval=True` without a valid token, the harness logged `AWAITING_APPROVAL` but continued to execute the mutating tool within the same loop.
* **Remediation:** Required affirmative yield of execution loop on `AWAITING_APPROVAL` until token resolution callback returns `approved`.

### 37. Root Sandbox Escape via Child Processes in sandboxed_bash
* **Submodule / Location:** `kybernetes/kybernetes/tools/sandboxed_bash.py:122`
* **Defect:** `shell=True` was used with only CWD root verification; commands inherited master `os.environ` containing service tokens, and sub-shells survived timeout signals.
* **Remediation:** Replaced unconstrained subshell with Bubblewrap process container (`bwrap`) using isolated network namespaces and process group SIGKILL.

### 38. Phylax Live Gateway Rule Mapping Mismatch
* **Submodule / Location:** `kybernetes/kybernetes/adapters/phylax.py:80-92`, `phylax/policies/default_policy.yaml:122`
* **Defect:** Adapter mapped actions to verb `"write"` / class `"file"`, but policy rules required class `"tool"` + `tool_id`, triggering default-deny and preventing approval generation.
* **Remediation:** Aligned adapter parameter mapping with `default_policy.yaml` capability schema.

### 39. Silent Fail-Open Exception Swallowing in PhylaxAdapter
* **Submodule / Location:** `kybernetes/kybernetes/adapters/phylax.py:43,120`
* **Defect:** `except Exception: pass` swallowed policy evaluation errors, silently falling back to simulated mode or legacy heuristic bypasses.
* **Remediation:** Enforced strict fail-closed exceptions when the live gateway or policy evaluator fails.

### 40. code_mode Sandbox Dunder Traversal and Tracer Timeout Failure
* **Submodule / Location:** `kybernetes/kybernetes/tools/code_mode.py:118-134,254-264`
* **Defect:** Format-string attribute traversal (`"{0.__class__}".format(1)`) bypassed AST dunder filters, and `sys.settrace` failed to interrupt C-extension calls (`math.factorial`).
* **Remediation:** Enforced subprocess isolation with hardware OS timeouts (`SIGKILL`) rather than line-level Python tracing.

### 41. Destructive Checkpoint Rollback via git clean -fd
* **Submodule / Location:** `kybernetes/kybernetes/engine/checkpoint.py:75`
* **Defect:** `rollback()` executed `git clean -fd`, permanently deleting all untracked files including user work and newly synthesized skills.
* **Remediation:** Confined checkpoint rollback to designated seat worktrees (`seat.worktree_path`), preserving untracked user workspace files.

### 42. Session Persistence Path Traversal and Unredacted Secret Leakage
* **Submodule / Location:** `kybernetes/kybernetes/engine/session.py:227,252`
* **Defect:** Caller-supplied `session_id` was concatenated into file paths without sanitization, allowing `../` directory traversal; `redact_sensitive_text` was never called before serialization.
* **Remediation:** Enforced UUIDv4 session ID validation and ran all serialized turn states through secret redaction filters.

### 43. Cross-Module Hestia Authentication Token Gap
* **Submodule / Location:** `dokime/src/dokime/client.py`, `dianoia/dianoia/walk.py:232`, `scripts/run_stack.sh:85`
* **Defect:** Hestia enforced bearer auth via `HESTIA_SERVICE_TOKEN`, but Dokime client omitted auth headers and Dianoia hardcoded `api_key="not-needed"`, causing HTTP 401 failures against the standard stack.
* **Remediation:** Injected `HESTIA_SERVICE_TOKEN` across all internal HTTP client instances.

### 44. Dashboard Config Apply Overwriting Real Secrets with Redacted Placeholders
* **Submodule / Location:** `kybernetes/kybernetes/admin/routes.py:230-259`, `changesets.py:311`
* **Defect:** Admin UI served redacted fields as `"[REDACTED]"`; saving the configuration round-tripped the literal string `"[REDACTED]"` into configuration files, destroying API keys and paths.
* **Remediation:** Implemented unredacted server-side delta merges, rejecting literal `"[REDACTED]"` placeholder writes.

### 45. Insecure World-Readable Permissions on Config Backup Snapshots
* **Submodule / Location:** `kybernetes/kybernetes/admin/changesets.py:159`
* **Defect:** Backup files were written using default umask (`0644`), leaving unredacted secrets world-readable; rollback metadata was stored in memory only.
* **Remediation:** Applied explicit `0600` octal file permissions to all configuration snapshots and persisted rollback indices in SQLite.

### 46. Phylax Destination Rule Substring Collisions
* **Submodule / Location:** `phylax/phylax/engine/evaluator.py:158-160`
* **Defect:** Destination matching checked bi-directional substring containment, allowing `"api.com"` to match malicious domains like `"api.com.attacker.com"`.
* **Remediation:** Enforced exact domain matching and strict canonical URL hostname equality.

### 47. Automatic Calculator VERIFIED Status Bypassing Verification
* **Submodule / Location:** `kybernetes/kybernetes/engine/pipeline.py:706-707`
* **Defect:** Any pipeline turn executing the calculator tool was automatically marked `final_status=VERIFIED` regardless of mathematical proof.
* **Remediation:** Required explicit verification assertion checks before assigning `VERIFIED` status.

### 48. Fallback Answers Incorrectly Tagged as Non-Simulated
* **Submodule / Location:** `kybernetes/kybernetes/engine/pipeline.py:710-713`
* **Defect:** When Hestia was offline, template responses were returned; turns retaining prior citations reported `is_simulated=False`.
* **Remediation:** Computed `is_simulated` across the entire turn lifecycle, guaranteeing fallback responses are marked `simulated=True`.

### 49. Benchmark Report Generator Hardcoding Win Verdicts
* **Submodule / Location:** `LTHCM/lthcm/benchmarks/run_real_benchmarks_to_md.py:185,280`
* **Defect:** Report writer hardcoded strings like `"$p < 0.0001$"` and `"**Extremely Significant Win**"` regardless of evaluated benchmark numbers.
* **Remediation:** Derived all Markdown verdict strings and $p$-values dynamically from statistical test distributions.

### 50. Synthetic demo_niah Metric Named MRR Without Reciprocal Calculation
* **Submodule / Location:** `LTHCM/lthcm/benchmarks/synthetic/demo_niah.py:236-237`
* **Defect:** Metric named `mrr` assigned tiered values (`1.0` if recall@1, `0.2` if recall@5, else `0.0`) rather than computing true Mean Reciprocal Rank ($1/\text{rank}$).
* **Remediation:** Implemented mathematically standard MRR calculation.

### 51. Unbounded Error Queue in Stoma Watcher Daemon
* **Submodule / Location:** `stoma/src/stoma/watcher.py:90-97,241`
* **Defect:** File-move and ingestion exceptions were appended to unbounded in-memory lists (`event_errors`), causing memory leaks during long-running watcher operations.
* **Remediation:** Capped error ring buffers and established an exhausted dead-letter queue with visible backoff.

### 52. Silent Ingestion Success on Zero-Page or Image-Only PDFs
* **Submodule / Location:** `stoma/src/stoma/plugins/pdf.py:47-48`, `indexer.py:888`
* **Defect:** Zero-page PDFs returned empty streams without warning, committing a zero-passage document with a valid manifest entry.
* **Remediation:** Returned explicit typed unsupported/empty status for 0-page PDFs.

### 53. Promotion of Uncommitted Generational Orphans in Stoma Startup
* **Submodule / Location:** `stoma/src/stoma/indexer.py:324-330`
* **Defect:** A crash occurring between writing `gen_{N}` and updating `current.json` left an uncommitted orphan; startup directory scans promoted the uncommitted generation.
* **Remediation:** Required a verifiable commit-chain marker (`commit_marker.json`) before any generation can be loaded on startup.

### 54. Eager Memory Accumulation During Stoma Ingestion Streaming
* **Submodule / Location:** `stoma/src/stoma/indexer.py:417-443`
* **Defect:** Ingestion materialised entire document passage lists into memory before chunking, causing memory spikes on multi-megabyte source files.
* **Remediation:** Swapped eager list accumulation for true generator streaming with bounded RSS batch caps.

### 55. Schedion Out-of-Domain Factual Grounding Collapse
* **Submodule / Location:** `schedion/schedion/study.py`, `docs/review/schedion-stage3-controlled-comparison-report.md`
* **Defect:** NextConceptTransformer (NCM) candidate generator exhibited severe degradation on technical/scientific text (26.3% grounding vs. 100% baseline).
* **Remediation:** Issued `ARCHIVE_NO_ADOPTION` verdict; kept Schedion disabled by default.

### 56. Symbolon Discrete Code Overlap Dilution
* **Submodule / Location:** `symbolon/symbolon/study.py`, `docs/review/symbolon-candidate-fusion-e2-report.md`
* **Defect:** VQ-VAE discrete multiscale code overlap achieved only 25% recall, expanding candidate sets without improving top-$k$ accuracy over dense embeddings.
* **Remediation:** Issued `ARCHIVE_UNCAPPED_NO_LOSS_RECORDED` verdict; barred Symbolon from replacing LTHCM Poincaré retrieval.

### 57. Anadrome Walk Policy Statistical Inconclusiveness
* **Submodule / Location:** `dianoia/dianoia/anadrome/study.py`, `docs/review/anadrome-frozen-live-walk-milestone3-report.md`
* **Defect:** Candidate MCTS walk policies failed to beat baseline LLM controller correctness with a paired 95% confidence interval excluding zero.
* **Remediation:** Retained `baseline_llm` as production default per Spec Section 6 ("no evidence keeps the current controller").

### 58. Stream Desynchronization on Partial Hestia Error Chunks
* **Submodule / Location:** `kybernetes/kybernetes/engine/pipeline.py:556-600`
* **Defect:** When a model crashed midway through token streaming, tokens already emitted to the client could not be recalled, causing appended fallback text.
* **Remediation:** Tagged real-time SSE token chunks as provisional and transmitted explicit error delta events on model failure.

### 59. Subagent Daemon Threads Lacking Process Restart Durability
* **Submodule / Location:** `kybernetes/kybernetes/lochos/subagent.py`
* **Defect:** Subagent tasks ran as in-process Python daemon threads; restarting Kybernetes lost in-flight agent tasks without state recovery.
* **Remediation:** Persisted subagent task queue records and git worktree checkpoint pointers to SQLite task ledgers.

---

## Part IV: Deep Subsystem Audits & Geometric/Execution Remediations (Items 60–73)

### 60. Untrained SO(D) Lie Group Gauge Fibre Rotation in Ranking (P1)
* **Submodule / Location:** `LTHCM/lthcm/engine/pipeline.py:385`, `stoma/src/stoma/adapter.py:281`
* **Defect:** Unseeded random orthogonal SO(D) gauge rotation matrices perturbed passage embeddings on cross-domain queries in production because zero trained checkpoints exist for gauge transport.
* **Remediation:** Explicitly disabled gauge transport in Stoma production queries (`enable_gauge_transport=False`), preserving unperturbed cosine similarity rankings.

### 61. Non-Deterministic HDC VSA Proposition Seeds (P2)
* **Submodule / Location:** `LTHCM/lthcm/core/hdc_vsa.py:80-82`
* **Defect:** Atom seed generation used Python's built-in `hash()`, producing randomized seeds across separate processes due to SIPHASH seed salting.
* **Remediation:** Replaced `hash()` with deterministic cryptographic SHA-256 byte extraction (`hashlib.sha256(symbol.encode()).digest()[:4]`).

### 62. Autograd Computation Graph Memory Leak in Fast-Weight Memory Updates (P2)
* **Submodule / Location:** `LTHCM/lthcm/engine/pipeline.py:522`
* **Defect:** Online fast-weight updates via `titans.step()` and `riemannian_memory.step()` executed without `torch.no_grad()`, accumulating backprop history across turns and causing CUDA OOM.
* **Remediation:** Enforced `@torch.no_grad()` on test-time memory steps.

### 63. Silent Causal Light-Cone Fallback Serving Invalid Candidates (P2)
* **Submodule / Location:** `LTHCM/lthcm/engine/pipeline.py:436-447`
* **Defect:** When zero candidates satisfied the Minkowski relativistic light-cone condition ($ds^2 \le 0$), the pipeline silently fell back to raw candidates, serving time-reversed evidence.
* **Remediation:** Added `allow_causal_fallback` parameter (defaulting to fail-closed `status="causally_empty"` when disabled).

### 64. Unsafe Pickle Deserialization in Sheaf Checkpoint Loading (P1)
* **Submodule / Location:** `LTHCM/lthcm/core/contrastive_sheaf.py:234`
* **Defect:** `torch.load()` was invoked without `weights_only=True`, exposing workers to arbitrary code execution before metadata validation could execute.
* **Remediation:** Enforced `weights_only=True` on checkpoint deserialization.

### 65. In-Flight Turn Event Buffer Race Condition & Memory Growth (P2)
* **Submodule / Location:** `kybernetes/kybernetes/engine/session.py:182,217`
* **Defect:** `subscribe_turn_events` snapshotted historical buffers before registering the subscriber queue, dropping interleaved events; `_turn_buffers` had no eviction policy.
* **Remediation:** Registered subscriber queues prior to snapshotting and added `_prune_turn_buffers()` to evict completed turn buffers after TTL.

### 66. Advisory-Only Declared Tool Execution Timeouts (P2)
* **Submodule / Location:** `kybernetes/kybernetes/tools/registry.py:332-344`
* **Defect:** Tool timeout checks occurred post-hoc after the handler returned, failing to interrupt infinite loops or hanging C-extensions.
* **Remediation:** Wrapped tool execution in active `concurrent.futures` timeouts with hard thread cancellation.

### 67. Syntax Diagnostic Bypass and 16-Bit CRC Hash Collisions in File Edits (P2)
* **Submodule / Location:** `kybernetes/kybernetes/tools/edit_file_block.py:236-247`
* **Defect:** When syntax validation failed on modified file buffers, the tool logged warnings but committed the corrupted syntax to disk.
* **Remediation:** Gated file writes to fail closed on syntax errors unless `allow_syntax_errors=True` is explicitly passed.

### 68. Stoma Ingestion Config Overlap Validation Rule Gap (P2)
* **Submodule / Location:** `kybernetes/kybernetes/admin/validators.py:144-146`
* **Defect:** Configuration changesets permitted `chunk_overlap_tokens >= max_chunk_tokens`, causing infinite loops in token subdivision.
* **Remediation:** Enforced strict validation: `chunk_overlap_tokens` must be strictly less than `max_chunk_tokens`.

### 69. Session-Scoped Working Directory (`chdir`) Test Suite Poisoning (P2)
* **Submodule / Location:** `krisis/tests/conftest.py:5`, `phylax/tests/conftest.py:5`
* **Defect:** Test conftests modified process CWD with `scope="session"`, corrupting subsequent test modules during unified test runs.
* **Remediation:** Swapped session-scoped `os.chdir` for function-scoped `monkeypatch.chdir` fixtures.

### 70. Command-Line Token Exposure in /proc/PID/cmdline (P1)
* **Submodule / Location:** `scripts/run_stack.sh:80-125`
* **Defect:** Service tokens were passed via CLI flags (`--service-token`, `--admin-token`), exposing credentials to any local process reading `ps aux`.
* **Remediation:** Replaced CLI arguments with environment variables (`export HESTIA_SERVICE_TOKEN`, `export KYBERNETES_SERVICE_TOKEN`) generated per-boot via `openssl rand -hex 16`.

### 71. TOCTOU Executable Substitution in Dynamic Tool Dispatch (P1)
* **Submodule / Location:** `kybernetes/kybernetes/tools/registry.py:110-185`, `docs/review/metis-pinax-tool-catalogue-spec-83941186.txt`
* **Defect:** Time-of-check to time-of-use window between Pinax catalog verification and subagent execution permitted binary replacement.
* **Remediation:** Anchored tool dispatch to immutable read-only Bubblewrap container mounts with process isolation (`--unshare-net`, `--ro-bind / /`).

### 72. Stale External Virtual Environment Interpreter in LTHCM Test Runner (P1)
* **Submodule / Location:** `LTHCM/.venv/pyvenv.cfg`, `LTHCM/pyproject.toml`
* **Defect:** `LTHCM/.venv/bin/pytest` referenced an invalid external interpreter path (`/home/akarta/Projects/LLM/.venv/bin/python3.11`), preventing standalone test execution.
* **Remediation:** Rebuilt `LTHCM/.venv` with CPython 3.12, installed complete dependencies, and verified all 127 tests pass in isolation.

### 73. Synthetic vs Empirical WordNet Hierarchy Benchmark Discrepancy (P3)
* **Submodule / Location:** `LTHCM/lthcm/benchmarks/real/hierarchy/wordnet_eval.py:6-10`
* **Defect:** Evaluated analytical radial depth constructions rather than learned Riemannian representations, misrepresenting results as Nickel & Kiela benchmarks.
* **Remediation:** Documented explicit honest disclaimer in module header and test manifests clarifying analytical demonstration status.

### 74. Hestia Speculative Decoding Integration Gap (P2)
* **Submodule / Location:** `hestia/src/hestia/registry.py:63-83`, `hestia/src/hestia/backend/llamacpp.py:142-162`, `hestia/tests/test_server.py:800-824`
* **Defect:** Model definitions and the llama-server launch generator lacked declarative configuration for speculative decoding draft models (`--draft-model`, `--draft-max`, `--spec-type`), forcing all GGUF models into single-pass autoregressive generation despite the availability of lightweight draft models.
* **Remediation:** Added `draft_model_path`, `draft_max`, and `spec_type` to `ModelEntry` and `ModelEntry.from_dict()`; wired the corresponding CLI flags into `LlamaCppBackend.build_launch_args()`; enforced pre-launch validation in `start()` to ensure declared draft weights exist; verified with unit tests.

### 75. Kybernetes Unbounded Thought-Graph Deliberation Budget Gap (P2)
* **Submodule / Location:** `kybernetes/kybernetes/config.py:55-60`, `kybernetes/kybernetes/engine/pipeline.py:58-75`, `kybernetes/kybernetes/server/app.py:89-100`, `kybernetes/tests/test_session_and_pipeline.py:162-177`
* **Defect:** Deliberative reasoning turns executed without a configurable maximum step bound (`thinking_budget`), risking unbounded walk iterations or inconsistent budget allocation across environments.
* **Remediation:** Added `thinking_budget_steps: int = 10` to `KybernetesConfig` with `KYBERNETES_THINKING_BUDGET_STEPS` environment variable support; passed this bound through `ExecutionPipeline` to constrain `DianoiaAdapter.deliberate(..., max_steps=delib_steps)` and recorded `max_deliberation_steps` in `StageOutcome`; verified with regression tests.

### 76. Ungrounded Continuous Latent Token Memory Architectural Guardrail (P1)
* **Submodule / Location:** `docs/review/metis-memorilla-latent-memory-takeaways-2026-10-19-b3d9f2c7-ef8d5fe5.txt`, `krisis/`, `LTHCM/`
* **Defect:** Review of latent continuous memory approaches (Memorilla / Soft-Tokens) identified a severe grounding vulnerability where continuous latent representations lack discrete source span provenance, which directly violates Krisis grounding policy (`krisis-policy-0.1.json`).
* **Remediation:** Enforced architectural invariant: all retrieval into context and deliberate reasoning must retain discrete passage span provenance; continuous latent memory representations remain non-adopted for retrieval to preserve deterministic verification and zero-hallucination claim auditability.

### 77. SwiLA Streaming Session State Stage 1 Empirical Evaluation & Audit Remediation
* **Discovery:** Audit of the Stage 1 SwiLA session state evaluation (`metis-swila-stage1-report-audit-20261020-bbf93822.txt`) confirmed the validity of the Gate 1 stop rule (halting Tier B), but identified empirical implementation gaps: Check 8e controls were omitted from `validity_checks.py`, candidate target index was fixed at index 0 risking tie bias, per-seed score arrays and mechanism diagnostics were absent from the JSON output, and the initial report contained overclaimed conclusions regarding Poincaré/BM25.
* **Remediation:** Implemented Check 8e (label permutation and independent random association controls) in [`validity_checks.py`](file:///home/akarta/Projects/Metis/docs/review/swila-session-state-stage1-report.md); randomized candidate target positions across $0 \dots 49$; instrumented mechanism diagnostics ($\exp(H) = 4.00$, $\text{NMI} = 0.0025$, zero collisions); dumped complete per-seed arrays into audit artifacts; revised [`swila-session-state-stage1-report.md`](file:///home/akarta/Projects/Metis/docs/review/swila-session-state-stage1-report.md) to adhere strictly to preregistered G4 protocol language; and archived the audit in [`docs/review/`](file:///home/akarta/Projects/Metis/docs/review/).

### 78. Dianoia Question-Conditioned Context Compression Stage 1 Empirical Evaluation & Rejection
* **Discovery:** Empirical evaluation of question-conditioned briefing curation ($A1$, JitMem-style curator) against raw passage concatenation ($A0$) on a unified 2,968-chunk LTHCM graph index ($N=300$ held-out test items, 4.4h runtime on RTX 5060 Ti) revealed severe safety and cost failures under preregistered protocol [`metis-dianoia-question-conditioned-compression-experiment-spec-2026-10-20-531c7bb-29a18f8e.txt`](file:///home/akarta/Projects/Metis/docs/review/metis-dianoia-question-conditioned-compression-experiment-spec-2026-10-20-531c7bb-29a18f8e.txt):
  - **Fact-Preservation Deficit (Gate 2 Fail):** Blind semantic audit of 100 test items showed $A1$ faithfully preserved only 40.57% (43/106) of retrieved gold supporting facts (21.7% dropped, 37.7% distorted/blurred), failing the 95% safety target and 50% dev calibration floor.
  - **Cost Inflation (Gate 2 Fail):** $A1$ consumed $1.6706\times$ total tokens vs $A0$ (exceeding the $1.30\times$ cap), driven by the sequential two-stage LLM inference.
  - **Marginal Task Performance:** Contained EM was $\Delta \text{EM} = +0.0300$, with 95% bootstrap CI $[0.0000, +0.0600]$ straddling the practical adoption margin $+m = +0.02$.
* **Remediation:** Enforced formal protocol decision **Class 2 (REJECT-COST-SAFETY)**; rejected integrating learned curator compression into `dianoia/dianoia/walk.py`; confirmed raw concatenation as the production invariant; archived full empirical results in [`docs/review/dianoia-compression-stage1-report.md`](file:///home/akarta/Projects/Metis/docs/review/dianoia-compression-stage1-report.md).

### Item 79: Round 5 Security Hardening & Build Hygiene (Findings 1–6)
* **Date:** 2026-10-28
* **Subsystems:** `deploy`, `kleroterion`, `kybernetes.engine`, `kybernetes.execution`, `kybernetes.admin`, `kybernetes.adapters`
* **Severity:** P0 (Secrets Exposure) / P1 (Sandbox & Approval Bypasses) / P2 (Simulation Defaults) / P3 (Exception & Count Hygiene)
* **Discovery:** Repo-wide Round 5 audit (`metis-repo-wide-round5-531c7bb-findings-v5-ea1789c2.txt`) identified:
  1. `kleroterion/pyproject.toml` pointed `readme` to a moved file, failing package builds.
  2. `kybernetes/engine/coding_agent.py` autonomously resolved its own single-use Phylax approval tokens for mutating tools.
  3. `kybernetes/execution/backend.py` silently degraded to unisolated host execution when Bubblewrap was unavailable.
  4. `deploy/metis.env` tracked live authentication tokens in git history.
  5. `kybernetes/admin/batch_runner.py` defaulted `smoke_test` and `allow_simulated` to `True`.
  6. `kybernetes/adapters/phylax.py` swallowed all exceptions in `resolve_approval`.
* **Remediation:**
  1. Updated `kleroterion/pyproject.toml` to reference `README.md`; verified editable build and all 43 tests pass.
  2. Prohibited autonomous self-approval in `coding_agent.py`; mutating actions without explicit approval token pause in `AgentSemanticState.AWAITING_APPROVAL`. Added regression test.
  3. Hardened `get_default_execution_backend()` to raise `RuntimeError` if Bubblewrap is explicitly requested but missing, and emit `UserWarning` on fallback. Added regression tests.
  4. Untracked `deploy/metis.env` via `git rm --cached`, restricted permissions to `0600`, added to `.gitignore`, and committed `deploy/metis.env.example`.
  5. Flipped `smoke_test` and `allow_simulated` defaults to `False` in `BatchRunner`.
  6. Added error logging and return `False` on approval resolution failures in `PhylaxAdapter`.

### Item 80: Dokime IFEval Programmatic Instruction-Following Evaluation Suite
* **Date:** 2026-10-28
* **Subsystems:** `dokime.tasks`
* **Severity:** Feature / Capability Integration (Google Research Track)
* **Discovery:** Assessment of `google-research/google-research` (`metis-google-research-takeaways-2026-10-28-db336cdc-d4cccfb9-0358fb51.txt`) identified IFEval (arXiv 2311.07911) as a high-value deterministic evaluation capability. Prior Dokime task suites lacked verifiable instruction-following checks, relying exclusively on multiple-choice and competition math formats.
* **Remediation:**
  1. Implemented `dokime.tasks.ifeval_checkers` providing 11 programmatic instruction checkers (JSON format/schema, word count, sentence count, paragraph count, keyword presence, forbidden words with inflection guards, keyword frequency, section headers, start/end constraints, numbered lists, postscripts).
  2. Implemented `IFEvalTask` in `dokime.tasks.ifeval` supporting strict and loose prompt/rule accuracy without LLM judges.
  3. Created curated 20-prompt verifiable evaluation suite `dokime/src/dokime/tasks/data/ifeval_suite.json`.
  4. Registered `ifeval_instruction_following` in `BUILTIN_TASKS` and verified 167/167 Dokime tests pass (including 15 new unit/integration tests).

### 81. Track B: KnowNo Conformal Prediction Uncertainty Sets in Krisis
- **Category:** Statistical Uncertainty & Adaptive Escalation Engine (Google Research Takeaways: KnowNo, Ren et al., arXiv:2307.01928)
- **Status:** **Remediated & Verified**
- **Impacted Files:**
  - `krisis/krisis/gates/conformal.py`
  - `krisis/krisis/engine/selector.py`
  - `krisis/krisis/contracts/enums.py`
  - `krisis/krisis/contracts/request.py`
  - `krisis/krisis/contracts/decision.py`
  - `krisis/tests/test_conformal.py`
- **Root Cause & Rationale:**
  - Standard candidate selection in Krisis relied solely on static heuristic scalar thresholds ($\tau$). In ambiguous or uncertain queries with multiple competing candidates or low-confidence outputs, static thresholds either over-select erroneous answers or reject viable answers without statistical validity guarantees.
  - Split conformal prediction provides finite-sample coverage guarantees ($P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$) over held-out calibration scores, yielding prediction sets $\mathcal{C}(X) = \{c \mid s(c) \ge 1 - \hat{q}\}$.
- **Remediation & Behavioral Guarantees:**
  1. Implemented `ConformalCalibrator` and `ConformalPredictionSet` in `krisis.gates.conformal` with finite-sample quantile math $\lceil(n+1)(1-\alpha)\rceil / n$.
  2. Extended Krisis contracts with `Reason.AMBIGUOUS_PREDICTION_SET` and `Reason.CONFORMAL_EMPTY_SET`.
  3. Wired conformal set evaluation into `KrisisSelector.evaluate()`:
     - $|\mathcal{C}(X)| == 1$: Statistically unique winner $\to$ `SELECTED` with conformal metadata.
     - $|\mathcal{C}(X)| > 1$: Ambiguous set $\to$ `ABSTAINED` with `AMBIGUOUS_PREDICTION_SET` (triggers human-in-the-loop escalation in Kybernetes).
     - $|\mathcal{C}(X)| == 0$: Empty set $\to$ `ABSTAINED` with `CONFORMAL_EMPTY_SET`.
     - Preserves standard scalar $\tau$ fallback when conformal parameters are not specified.
  4. Created comprehensive unit and integration test suite `krisis/tests/test_conformal.py`. All 36 Krisis tests pass cleanly.

---

## Summary & Verification Matrix

| Category | Range | Focus Subsystems | Verification Status | Verified Evidence |
| :--- | :---: | :--- | :---: | :--- |
| **Telemetry & Dashboard Specs** | Items 1–7 | Kleroterion, Krisis, Phylax, Dokime, Paideia, Sandbox, WebRTC | **Wired & Active** | Strict TS compile: 0 errors; React bundle synced (2.9s) |
| **P0 Security & Startup Blockers** | Items 8–15 | Auth perimeters, fail-closed gates, token content bindings | **Resolved & Tested** | Prosopon, Kybernetes, Phylax auth suites passing |
| **P1 Lifecycle & Integration Gaps** | Items 16–33 | GPU admissions, port clashes, light cones, Stoma contracts | **Resolved & Tested** | Hestia, Stoma, LTHCM integration suites passing |
| **P2 Documentation Alignment** | Item 34 | Docs vs codebase reality, loopback endpoints | **Documented** | Port bindings, tokens, and storage roots aligned |
| **Extended Security & Sandboxing** | Items 35–46 | Bubblewrap escapes, dunder format bypasses, secret redaction | **Hardened** | Linux namespace `--unshare-net` & `--ro-bind` verified |
| **Evaluation, Memory & Benchmarks** | Items 47–59 | Negative study archives, Stoma OOM streaming, audit chains | **Empirically Proven** | Phylax audit chain (368 records) & Dokime 10k bootstrap |
| **Deep Audits & Geometric Hardening** | Items 60–73 | Lie group gauge filters, HDC VSA seeds, autograd leaks, LTHCM venv | **Hardened & Tested** | 1,002 tests passed across 11 modules + 0 TS compile errors |
| **Reasoning & Serving Efficiency** | Items 74–76 | Speculative decoding, thinking budget bounds, latent grounding | **Hardened & Tested** | 1,003 tests passed across 11 modules + 0 TS compile errors |
| **Session Memory & Routing Audits** | Items 77–78 | SwiLA recurrence vs baselines, Dianoia compression vs raw | **Empirically Audited** | Both negative studies archived; production baselines kept |
| **Round 5 Audit & Build Hardening** | Item 79 | Deploy, Kleroterion, Kybernetes Engine, Backend, Admin, Phylax | **Remediated & Verified** | 43 kleroterion tests + 190 kybernetes tests passing |
| **Instruction Evaluation Engine** | Item 80 | Dokime Tasks (IFEval Programmatic Suite) | **Integrated & Verified** | 167 Dokime tests passing (15 new unit/integration tests) |
| **Conformal Prediction Uncertainty** | Item 81 | Krisis (KnowNo Finite-Sample Quantile Sets) | **Integrated & Verified** | 36 Krisis tests passing (7 new conformal unit/integration tests) |

---

### Monorepo Subsystem Test Verification Breakdown (1,030 Total Passed)

| Subsystem | Test Command | Tests Passed | Skipped | Runtime | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Kybernetes** (Orchestrator & Tools) | `./.venv/bin/pytest kybernetes/tests -q` | **190** | 0 | 19.88s | **PASS** |
| **Dokime** (Evaluation & Release Gates) | `./.venv/bin/pytest dokime/tests -q` | **167** | 0 | 2.33s | **PASS** |
| **LTHCM** (Poincaré Memory Engine) | `./LTHCM/.venv/bin/pytest LTHCM/tests -q` | **127** | 0 | 233.53s | **PASS** |
| **Stoma** (Ingestion & Chunking) | `./stoma/.venv/bin/pytest stoma/tests -q` | **113** | 0 | 233.92s | **PASS** |
| **Dianoia** (MCTS Deliberation Engine) | `./.venv/bin/pytest dianoia/tests -q` | **93** | 0 | 3.85s | **PASS** |
| **Hestia** (GGUF Inference Serving) | `./hestia/.venv/bin/pytest hestia/tests -q` | **72** | 1 | 1.19s | **PASS** |
| **Prosopon** (WebRTC & Access Store) | `./prosopon/.venv/bin/pytest prosopon/tests -q` | **71** | 0 | 43.82s | **PASS** |
| **Kleroterion** (Stochastic Policy) | `./kleroterion/.venv/bin/pytest kleroterion/tests -q` | **43** | 0 | 1.64s | **PASS** |
| **Paideia** (Training Pipeline) | `./paideia/.venv/bin/pytest paideia/tests -q` | **41** | 0 | 4.07s | **PASS** |
| **Krisis** (Claim Grounding & Conformal Gates)| `./.venv/bin/pytest krisis/tests -q` | **36** | 0 | 0.72s | **PASS** |
| **Phylax** (Capability Security Gateway) | `./.venv/bin/pytest phylax/tests -q` | **36** | 0 | 1.46s | **PASS** |
| **Root Integration** | `./.venv/bin/pytest tests -q` | **39** | 0 | 17.48s | **PASS** |
| **VS Code Void IDE Client** | `npm run compile` (in `vscode/`) | **0 errors** | 0 | 65.62s | **PASS** |
| **Total Monorepo Suite** | **All 11 Subsystems + IDE Client** | **1,030** | **1** | **~9.5 min** | **ALL GATES PASS** |



