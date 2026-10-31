# Metis Comprehensive Gaps, Defects & Telemetry Register (88 Items)

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
* **Fix:** Utilized `O_NOFOLLOW` descriptor opening with `os.fstat(fd)` vs `os.stat(target_str)` device and inode equality verification, regular file mode validation (`stat.S_ISREG`), and atomic exclusive temporary file (`.{name}.tmp-{pid}-{rand}`) write/fsync/rename.

### 18. Stop Selecting an Old Krisis Snapshot When Rebuild Is Possible (P1)
* **Where:** `krisis/krisis/engine/selector.py:46-62`
* **Defect:** When evidence changed under `rebuilds_remaining > 0`, the selector still returned the stale candidate.
* **Fix:** Return typed signals: `status=DecisionStatus.REBUILD_REQUIRED`, `reason=Reason.STALE` when budget remains, and `status=DecisionStatus.ABSTAINED`, `reason=Reason.STALE` when budget is exhausted.

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
* **Where:** `dianoia/dianoia/cli.py:156-181`, `stoma/src/stoma/adapter.py:368-380`, `stoma/src/stoma/interchange.py:24`, `stoma/src/stoma/graph_builder.py:24`
* **Defect:** Ingestion loaded indices using standard Python `pickle.load()` without prior authorization or hash verification.
* **Fix:** Enforced `allow_trusted_pickle=False` default across Dianoia loader, Stoma adapter, Stoma interchange, and offline graph builder, strictly requiring explicit `--allow-trusted-pickle` or `STOMA_ALLOW_TRUSTED_PICKLE=1` / `DIANOIA_ALLOW_TRUSTED_PICKLE=1` authorization.

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
* **Fix:** Enforced max-match bounds (1,000 files, 50MB max scan, depth $\le 6$) and AST node recursion depth caps ($\le 50$).

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
* **Submodule / Location:** `symbolon/symbolon/study.py`, `dokime/src/dokime/tasks/symbolon_benchmark.py`, `docs/review/symbolon-candidate-fusion-e2-report.md`
* **Defect:** VQ-VAE discrete multiscale code overlap achieved only 25% recall, expanding candidate sets without improving top-$k$ accuracy over dense embeddings.
* **Remediation:** Issued `ARCHIVE_UNCAPPED_NO_LOSS_RECORDED` verdict; barred Symbolon from replacing LTHCM Poincaré retrieval. Implemented `symbolon_code_retrieval` Task plugin and `dokime symbolon-benchmark` CLI command into Dokime with 8-query multi-domain suite (`symbolon_suite.json`) and 4-arm comparative report generator to permanently serve as an empirical yardstick.

### 57. Anadrome Walk Policy Statistical Inconclusiveness
* **Submodule / Location:** `dianoia/dianoia/anadrome/study.py`, `docs/review/anadrome-frozen-live-walk-milestone3-report.md`
* **Defect:** Candidate MCTS walk policies failed to beat baseline LLM controller correctness with a paired 95% confidence interval excluding zero.
* **Remediation:** Retained `baseline_llm` as production default per Spec Section 6 ("no evidence keeps the current controller").

### 58. Stream Desynchronization on Partial Hestia Error Chunks
* **Submodule / Location:** `kybernetes/kybernetes/engine/pipeline.py:556-600`
* **Defect:** When a model crashed midway through token streaming, tokens already emitted to the client could not be recalled, causing appended fallback text.
* **Remediation:** Emitted interruption delta events (`payload={"text": notice, "stream_interrupted": True, "error": err_msg}`, `status=TurnOutcomeStatus.DRAFT`) appending an explicit notice and breaking the stream without concatenating unrelated fallback completions.

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
* **Submodule / Location:** `LTHCM/lthcm/engine/pipeline.py:436-447`, `stoma/src/stoma/adapter.py:241-285`
* **Defect:** When zero candidates satisfied the Minkowski relativistic light-cone condition ($ds^2 \le 0$), the pipeline silently fell back to raw candidates, serving time-reversed evidence.
* **Remediation:** Added explicit `allow_causal_fallback` parameter (returning `status="causally_empty"` when disabled), while allowing document adapters to explicitly declare fallback policy for sub-second ingestion intervals.

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
* **Date:** 2026-10-28 (Updated 2026-10-29)
* **Subsystems:** `dokime.tasks`
* **Severity:** Feature / Capability Integration (Google Research Track)
* **Discovery:** Assessment of `google-research/google-research` (`metis-google-research-takeaways-2026-10-28-db336cdc-d4cccfb9-0358fb51.txt`) identified IFEval (arXiv 2311.07911) as a high-value deterministic evaluation capability. Prior Dokime task suites lacked verifiable instruction-following checks, relying exclusively on multiple-choice and competition math formats.
* **Remediation:**
  1. Implemented `dokime.tasks.ifeval_checkers` providing 20 programmatic instruction checkers (JSON format/schema, word count, sentence count, paragraph count, keyword presence, forbidden words with inflection guards, keyword frequency, section headers, start/end constraints, numbered lists, postscripts, letter casing constraints, quotation constraints, highlight section markdown constraints, two-response delimiters, no-comma punctuation constraints, bullet lists, all-lowercase constraints, title wrapping delimiters, and letter frequency constraints).
  2. Implemented `IFEvalTask` in `dokime.tasks.ifeval` supporting strict and loose prompt/rule accuracy without LLM judges.
  3. Created curated 30-prompt verifiable evaluation suite `dokime/src/dokime/tasks/data/ifeval_suite.json`.
  4. Registered `ifeval_instruction_following` in `BUILTIN_TASKS` and verified 195/195 Dokime tests pass (including 25 unit/integration tests).

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

### 82. Track C: Sufficient Input Subsets (SIS) Explainer in Dokime
- **Category:** Black-Box Attribution & Prompt Minimization Engine (Google Research Takeaways: SIS, Carter et al., arXiv:2107.09752)
- **Status:** **Integrated & Verified**
- **Impacted Files:**
  - `dokime/src/dokime/explainers/sis.py`
  - `dokime/src/dokime/explainers/__init__.py`
  - `dokime/src/dokime/__init__.py`
  - `dokime/src/dokime/cli.py`
  - `dokime/tests/test_sis.py`
- **Root Cause & Rationale:**
  - Diagnosing why a model produced a particular response or identifying the minimum context driving a factual claim in Metis required gradient access or intermediate activation hooks, which are unavailable when querying black-box served GGUF models via Hestia.
  - Sufficient Input Subsets computes the minimal subset along a greedy backward-selection path that satisfies a decision threshold $f(S) \ge \tau$, with $O(N)$ batched evaluations per step and deterministic tie-breaking.
- **Remediation & Behavioral Guarantees:**
  1. Implemented `SufficientInputSubsets`, `SISResult`, and `SISCollectionResult` in `dokime.explainers.sis`.
  2. Implemented greedy backward elimination: masks each candidate position and removes the item whose omission retains the highest score $\ge \tau$, breaking ties deterministically to lowest index.
  3. Implemented disjoint rationale collection (`find_sis_collection`): iteratively masks previously discovered rationales to reveal redundant or alternative pathways.
  4. Implemented `explain_text_sufficiency` helper supporting sentence, paragraph, and word level granularity.
  5. Added `dokime explain` subcommand to CLI with `--target-keyword`, `--disjoint`, and `--tau` options.
  6. Added 9 unit and integration tests in `dokime/tests/test_sis.py`. All 176 Dokime tests pass cleanly.

---

### Item 83: Speculative Cascades Serving Engine & Early Deferral Endpoints in Hestia
- **Priority:** P1 (Serving Latency & Compute Efficiency)
- **Status:** **Implemented & Fully Verified**
- **Impacted Files:**
  - `hestia/src/hestia/cascade/engine.py`
  - `hestia/src/hestia/cascade/__init__.py`
  - `hestia/src/hestia/api/app.py`
  - `hestia/tests/test_speculative_cascades.py`
- **Root Cause & Rationale:**
  - Hestia served single models per request without speculative cascading. Routine code completions and factual inquiries were routed unconditionally to high-capacity verifier models (e.g. 8B–70B), inflating latency and VRAM/compute consumption when lightweight drafters (e.g. 1B–3B) could have generated acceptable completions with high confidence.
  - Speculative Cascades (ICLR 2025) queries a lightweight draft model first with logprobs enabled. If token-level confidence (mean probability, minimum probability, or margin proxy) meets or exceeds deferral threshold $\tau$, the draft completion exits early, yielding estimated speedups of $3\times$ to $5\times$. If confidence falls below $\tau$, the request defers to the verifier model with fail-safe telemetry.
- **Remediation & Behavioral Guarantees:**
  1. Implemented `CascadeConfig`, `CascadeDecision`, and `SpeculativeCascadeEngine` in `hestia.cascade.engine`.
  2. Supported linear token probability extraction from both OpenAI and llama.cpp native logprob payloads (`math.exp(logprob)`).
  3. Implemented deferral criteria: `mean_prob`, `min_prob`, and `margin`.
  4. Created dedicated `POST /v1/cascade/completions` endpoint and integrated transparent cascade dispatch in `POST /v1/chat/completions` via `model="cascade:<draft>:<verifier>"`.
  5. Enforced admission control, concurrency safety, and telemetry annotation in `response["usage"]["cascade"]`.
  6. Added 5 comprehensive unit and API integration tests in `hestia/tests/test_speculative_cascades.py`. All 77 Hestia tests pass cleanly (1 skipped for hardware).

---

### Item 84: ASPEST Active Selective Prediction Pipeline in Krisis & Paideia
- **Priority:** P1 (Active Learning & Alignment Data Curation)
- **Status:** **Implemented & Fully Verified**
- **Impacted Files:**
  - `krisis/krisis/contracts/active_learning.py`
  - `krisis/krisis/contracts/__init__.py`
  - `krisis/krisis/contracts/decision.py`
  - `krisis/tests/test_active_learning.py`
  - `paideia/paideia/contracts/krisis_schema.py`
  - `paideia/paideia/contracts/__init__.py`
  - `paideia/paideia/data/aspest_sampler.py`
  - `paideia/paideia/data/__init__.py`
  - `paideia/paideia/cli.py`
  - `paideia/tests/test_aspest_sampler.py`
- **Root Cause & Rationale:**
  - When Krisis abstained due to ambiguous prediction sets ($|\mathcal{C}(X)| > 1$), empty sets ($|\mathcal{C}(X)| = 0$), or gate failures, these failure modes remained trapped in decision logs and were not curated for active learning or fine-tuning.
  - Active Selective Prediction (ASPEST, arXiv:2304.03870) closes this loop by assigning rejection scores $U(x)$ to abstained evaluations and balancing uncertainty with query diversity $D(x, S)$ to select high-yield batches for human annotation and model alignment.
- **Remediation & Behavioral Guarantees:**
  1. Implemented `KrisisActiveRecord` and `derive_active_record` in `krisis.contracts.active_learning`, mapping conformal uncertainty and heuristic claim scores to normalized rejection intensity $U(x) \in [0, 1]$.
  2. Added `to_active_record()` convenience method on `KrisisDecision`.
  3. Implemented paired contracts `KrisisActiveRecord`, `ActiveSamplingConfig`, and `ASPESTAuditReport` in `paideia.contracts.krisis_schema`.
  4. Implemented `ASPESTSampler` in `paideia.data.aspest_sampler` balancing uncertainty and diversity via $\alpha(x) = \beta \cdot U(x) + (1-\beta) \cdot D(x, S)$ with deterministic tie-breaking. Supported both token Jaccard and dense vector cosine distance.
  5. Added `paideia dataset sample-active` subcommand to Paideia CLI.
  6. Added 4 unit tests in `krisis/tests/test_active_learning.py` (all 40 Krisis tests passing) and 5 unit/CLI tests in `paideia/tests/test_aspest_sampler.py` (all 46 Paideia tests passing).

### 85. Google Research Takeaway Track F: ScaNN Approximate Nearest Neighbor Indexing & Dokime ANN Yardstick
- **Affected Module(s):**
  - `LTHCM/lthcm/core/scann_index.py`
  - `LTHCM/lthcm/core/__init__.py`
  - `LTHCM/lthcm/engine/retriever.py`
  - `LTHCM/pyproject.toml`
  - `LTHCM/tests/test_scann_index.py`
  - `dokime/src/dokime/tasks/ann_benchmark.py`
  - `dokime/src/dokime/cli.py`
  - `dokime/tests/test_ann_benchmark.py`
  - `stoma/vendored_provenance.json`
  - `stoma/src/stoma/vendored_provenance.json`
- **Root Cause & Rationale:**
  - In LTHCM's deterministic retrieval pipeline (`LTHCMRetriever.search()`), Stage 1 candidate generation relied on exhaustive $O(N \cdot D)$ linear dot-product scans (`np.dot(self.doc_embeddings, q_emb)`). As document collections scale into hundreds of thousands of passages, unquantized linear scans dominate search latency.
  - Google Research evaluation identified ScaNN (Anisotropic Vector Quantization & SOAR, ICML 2020 / NeurIPS 2023; pip-installable wheel without TensorFlow on modern AVX/FMA hardware) as the highest-utility artifact for LTHCM, with the explicit mandate to benchmark empirical Recall@k, latency percentiles, and QPS against exact ground truth on active embeddings before adoption.
- **Remediation & Behavioral Guarantees:**
  1. Implemented `ScannIndex` and `ScannConfig` in `LTHCM/lthcm/core/scann_index.py`, providing a complete typed adapter over `scann.scann_ops_pybind` with tree-AH anisotropic quantization, brute force SIMD fallback, docid mapping, upsert/delete mutation tracking, and filesystem serialization/loading.
  2. Integrated `ann_backend: str = "exact"` ("exact", "scann", "faiss"), `ann_leaves_to_search`, and `ann_reorder_k` into `RetrievalConfig` and `LTHCMRetriever` (`LTHCM/lthcm/engine/retriever.py`), allowing callers to select sub-linear vector search with stage trace telemetry (`traces["ann_backend"] = "scann"`).
  3. Added `scann>=1.4.0` optional dependencies under `bench`, `ann`, and `all` in `LTHCM/pyproject.toml`. Recomputed LTHCM content digest (`2985bf1b819197ce15a63b9d2ee6fd59292ca8395fea0dc67aaf38f3d302f5af`) and synchronized `stoma/vendored_provenance.json` and `stoma/src/stoma/vendored_provenance.json`.
  4. Implemented `ANNBenchmarkSuite`, `ANNBenchmarkConfig`, and `ANNBenchmarkReport` in `dokime/src/dokime/tasks/ann_benchmark.py`, evaluating Recall@1, Recall@K, latency percentiles (mean, p50, p95, p99 ms), QPS, and build times against exact linear dot-product ground truth.
  5. Added `dokime ann-benchmark` subcommand to Dokime CLI (`dokime/src/dokime/cli.py`) with optional JSON report export (`--output`).
  6. Verified via Dokime benchmark at $N=3,000$, $D=128$: ScaNN Tree-AH achieves **85,962 QPS (6.64x speedup)** over exact linear scan; ScaNN Brute-Force SIMD achieves **38,865 QPS (3.00x speedup)** with **1.0000 exact recall**.
  7. Added 6 unit tests in `LTHCM/tests/test_scann_index.py` (133 LTHCM tests passing) and 3 unit/CLI tests in `dokime/tests/test_ann_benchmark.py` (179 Dokime tests passing).

---

### Item 86: Visually-Rich Document Understanding (VRDU) Extraction & Stoma Structured Ingestion
- **Category:** Document Understanding & Multimodal Ingestion (Google Research VRDU & DePlot / MatCha Integration)
- **Priority:** P2 (Empirical Capability & Ingestion Quality)
- **Status:** Resolved & Verified
- **Affected Subsystems:**
  - `stoma/src/stoma/plugins/ocr.py`
  - `stoma/tests/test_ocr_structured.py`
  - `dokime/src/dokime/tasks/vrdu.py`
  - `dokime/src/dokime/tasks/data/vrdu_suite.json`
  - `dokime/src/dokime/tasks/registry.py`
  - `dokime/src/dokime/tasks/__init__.py`
  - `dokime/tests/test_vrdu.py`
- **Root Cause & Rationale:**
  - Real-world technical documentation ingested by Stoma (semiconductor purchase orders, cleanroom equipment specifications, clinical trial protocols, SEC 10-K balance sheets, patents, wafer probing reports) frequently contains dense visual structures: multi-column tables, chart data series, and key-value forms.
  - Previously, Stoma's OCR plugin (`stoma/src/stoma/plugins/ocr.py`) flattened OCR bounding boxes into unformatted text streams, discarding table column boundaries, chart points, and key-value pairings essential for downstream vector search and LLM extraction.
  - Furthermore, Dokime lacked a benchmark task for measuring structured document extraction precision, recall, and F1 across rich entity schemas, leaving Metis without empirical yardsticks for document understanding capabilities.
- **Remediation & Behavioral Guarantees:**
  1. Enhanced `stoma/src/stoma/plugins/ocr.py` with structured parsing:
     - Formats detected tabular grids into clean, aligned Markdown table passages with headers (`| col1 | col2 |`).
     - Formats chart data series into DePlot / MatCha style summary passages (`# Chart: Title`, `Series: X | Y`).
     - Formats key-value pairs (`Key: Value`) into dedicated structured entity passages for exact relational and vector retrieval.
     - Preserves full backwards compatibility with unstructured raw OCR text streams.
  2. Implemented `VRDUTask` in `dokime/src/dokime/tasks/vrdu.py` conforming to the Dokime `Task` protocol (`task_id = "vrdu_document_extraction"`):
     - Robust multi-format JSON extraction (markdown fenced code blocks, raw JSON, embedded outer braces).
     - Flexible normalization: date formats (ISO `YYYY-MM-DD` vs `Month DD, YYYY`), numeric tolerance (with currency symbol and comma stripping), whitespace-collapsed case-insensitive string matching, entity substring containment, and recursive nested list/dict matching.
     - Computes micro-averaged Precision, Recall, and F1 across target schema fields.
  3. Created a 20-sample dense semi-structured document benchmark suite in `dokime/src/dokime/tasks/data/vrdu_suite.json`:
     - ASML EUV purchase orders, Zeiss commercial invoices, TSMC wafer probing reports, Hellenic Oncology clinical protocols, Metis 10-K balance sheets, quantum semiconductor patents, ISO cleanroom lab specs, commercial lease agreements, FAA airworthiness certificates, and material transfer agreements.
  4. Added 3 unit tests in `stoma/tests/test_ocr_structured.py` and 6 unit tests in `dokime/tests/test_vrdu.py` (185 Dokime tests + 118 Stoma tests passing).

### Item 87: Security & Architectural Audit Remediation (F-02, F-03, F-04, F-05, F-07, F-08)
- **Category:** Monorepo Security & Robustness (Code Mode AST, WebRTC Auth, Zero-Loopback Model, Path Traversal, Phylax Gate Fallthrough, Krisis Polarity Clash)
- **Priority:** P0 (Defense-in-Depth, Authentication & Remote Sandbox Escape Prevention)
- **Status:** Resolved & Verified
- **Affected Subsystems:**
  - `kybernetes/kybernetes/tools/code_mode.py`
  - `kybernetes/tests/test_code_mode.py`
  - `prosopon/prosopon/server.py`
  - `prosopon/prosopon/config.py`
  - `prosopon/tests/test_server_api.py`
  - `kybernetes/kybernetes/server/app.py`
  - `kybernetes/kybernetes/cli.py`
  - `kybernetes/tests/test_lochos_api.py`
  - `kybernetes/tests/test_pinax.py`
  - `kybernetes/tests/test_server_api.py`
  - `kybernetes/kybernetes/engine/pipeline.py`
  - `kybernetes/kybernetes/routing/router.py`
  - `kybernetes/tests/test_session_and_pipeline.py`
  - `kybernetes/kybernetes/adapters/krisis.py`
  - `kybernetes/tests/test_adapters.py`
- **Root Cause & Rationale:**
  - **F-02 (Code Mode Generator Frame Traversal Sandbox Escape):** In Python, generator execution frames can be traversed to extract module globals (`(yield 1).gi_frame.f_globals["os"]`) without accessing traditional dunder attributes. Furthermore, Python format string traversal (`"{0.__class__}".format(...)`) provides another introspection vector. Additionally, `run_code_mode` was marked with `READ_ONLY` side effects instead of `SYSTEM_EXECUTION`, bypassing Phylax approval checks when approvals were enforced.
  - **F-03 (Prosopon WebRTC Offer Unauthenticated Device ID Injection):** In `/api/webrtc/offer`, `target_dev_id` initialized directly from `req.device_id` in the request body. If the caller provided arbitrary `device_id` and omitted auth headers, `target_dev_id` was non-null, bypassing the `if not target_dev_id:` check and allowing unauthenticated callers to negotiate WebRTC data channels.
  - **F-04 (Kybernetes Loopback Trust & Unauthenticated Lochos/Pinax Route Exposure):** `require_service_auth` in Kybernetes trusted `127.0.0.1`, `::1`, and `localhost`, allowing local processes, reverse proxies, and SSRF attacks to bypass auth. Furthermore, Lochos squad routes (`/v1/lochos/squad`, `/v1/lochos/queue`, `/v1/lochos/seats/{seat_id}/brief`, `/v1/lochos/transfer/brief`, `/v1/lochos/attention`) and Pinax routes (`/v1/pinax/tools`, `/v1/pinax/tools/{tool_id:path}/docs`, `/v1/pinax/tools/{tool_id:path}`, `/v1/pinax/evaluate`) lacked the `require_service_auth` dependency. Additionally, `ToolRegistry` constructed without an injected `phylax_adapter` created an isolated adapter instance, desynchronizing approval state.
  - **F-05 (Prosopon Workbench File Sensitive Credential & Database Disclosure):** `/api/workbench/file` guarded against directory traversal outside `workspace_root`, but allowed authenticated callers to read sensitive workspace files: `.host_owner_token`, `metis.env`, `paired_tokens.json`, SQLite databases (`*.db`, `*.sqlite`), and internal snapshot/policy directories (`.admin_snapshots/`, `.phylax/`).
  - **F-07 (Kybernetes Execution Pipeline Phylax Hard Deny Fallthrough):** In `pipeline.py`, when Phylax evaluated an action as `not permitted` and returned `app_req is None` (hard security policy DENY, or non-interactive mode), the pipeline checked `if app_req:` which evaluated to `False`, and subsequently fell through to execute the tool! Additionally, `router.py` generated `tool_name = "guarded_action"`, an unregistered tool name.
  - **F-08 (Krisis Adapter Claim Negation False Contradiction Polarity Clash):** `KrisisAdapter.verify_response()` flagged any sentence containing contradiction keywords (`"not"`, `"no"`, `"never"`) as `conflict_detected = True`, even when both the claim and the cited passage were negative (i.e. in agreement).
- **Remediation & Behavioral Guarantees:**
  1. **F-02 Hardening:**
     - Added generator frame attributes (`gi_frame`, `cr_frame`, `ag_frame`, `f_back`, `f_globals`, `f_locals`, `f_builtins`, `f_code`, `format`, `format_map`, `__dict__`, `__closure__`) to `FORBIDDEN_ATTRIBUTES` in `SafeCodeMode`.
     - Explicitly blocked `ast.Yield` and `ast.YieldFrom` in the AST safety walk.
     - Blocked `.format(...)` and `.format_map(...)` attribute calls in AST.
     - Reclassified `run_code_mode` as `ToolSideEffectClass.SYSTEM_EXECUTION` to mandate Phylax approval tokens when `enforce_approvals=True`.
     - Added regression tests in `kybernetes/tests/test_code_mode.py` (11 passing).
  2. **F-03 Hardening:**
     - In `prosopon/prosopon/server.py:webrtc_offer`, initialized `target_dev_id = None`. Authenticated only via valid `x-host-owner-token` header or validated Bearer session/pairing token. Unauthenticated callers are rejected with 401 Unauthorized.
     - Added regression test `test_webrtc_offer_unauthenticated_device_id_rejection` in `prosopon/tests/test_server_api.py`.
  3. **F-04 Hardening:**
     - Removed local loopback bypass in Kybernetes `require_service_auth` (enforcing zero-loopback security model).
     - Protected all Lochos squad endpoints and Pinax tool lookup/detail/docs/evaluate endpoints with `Depends(require_service_auth)`.
     - Injected shared `phylax_adapter` instance into `ToolRegistry(phylax_adapter=phylax_adapter)` in both `create_app()` and CLI entry points.
     - Updated `kybernetes/tests/test_server_api.py`, `test_lochos_api.py`, and `test_pinax.py` to enforce authentication tokens on loopback requests.
  4. **F-05 Hardening:**
     - In `prosopon/prosopon/server.py:workbench_file`, added strict path denylisting blocking `.host_owner_token`, `metis.env`, `paired_tokens.json`, `.env`, files ending with `.key`, `.db`, `.sqlite`, `.sqlite3`, `.pem`, and paths within `.admin_snapshots/`, `.phylax/`, or `.git/` with 403 Forbidden.
     - Added `workspace_root` configuration to `ProsoponConfig` and added regression test `test_workbench_file_sensitive_path_denial` in `prosopon/tests/test_server_api.py`.
  5. **F-07 Hardening:**
     - In `kybernetes/kybernetes/engine/pipeline.py`, when `not permitted`, if `not app_req` or `not needs_interactive`, immediately emit `TurnOutcomeStatus.DENIED` completed event and return. Never fall through to tool execution.
     - In `kybernetes/kybernetes/routing/router.py`, replaced non-existent `"guarded_action"` with registered `"write_file"` tool.
     - Added regression test `test_pipeline_hard_policy_denial_no_approval_req` in `kybernetes/tests/test_session_and_pipeline.py`.
  6. **F-08 Hardening:**
     - In `kybernetes/kybernetes/adapters/krisis.py`, replaced naive negation keyword check with semantic polarity clash detection. A contradiction is flagged if and only if the claim and the cited evidence share semantic entity anchor tokens (`len(shared_tokens) >= 2`) and have conflicting truth-value polarity (`claim_neg != cit_neg`).
     - Added regression test `test_krisis_polarity_clash_detection` in `kybernetes/tests/test_adapters.py`.

### Item 88: Symbolon Discrete Code Retrieval Benchmark & Comparative Fusion Yardstick
- **Category:** Evaluation & Retrieval Diagnostics (Dokime Tasks, VQ-VAE Discrete Overlap, Candidate Fusion, Contradiction Retention)
- **Priority:** P2 (Diagnostic & Release Gating Integrity)
- **Status:** Resolved & Verified
- **Affected Subsystems:**
  - `dokime/src/dokime/tasks/symbolon_benchmark.py`
  - `dokime/src/dokime/tasks/data/symbolon_suite.json`
  - `dokime/src/dokime/tasks/registry.py`
  - `dokime/src/dokime/tasks/__init__.py`
  - `dokime/src/dokime/cli.py`
  - `dokime/tests/test_symbolon_benchmark.py`
- **Root Cause & Specification Requirement:**
  - The offline Symbolon E2 candidate fusion study established that coarse acoustic-style VQ-VAE discrete codebook overlap alone achieves only 25.0% recall and 25.0% contradiction retention. Uncapped additive union ($C_{\text{base}} \cup C_{\text{sym}}$) preserves 100% of gold evidence at the cost of 1.75x candidate set expansion and verification overhead, while capped Reciprocal Rank Fusion ($k=3$) produces no improvement ($\Delta = 0.0000$) over flat dense embeddings (`ARCHIVE_UNCAPPED_NO_LOSS_RECORDED`).
  - To prevent future regressions, permanently bar naive discrete code filtering from LTHCM production retrieval, and provide an empirical testbed for future geometric codebook improvements, this benchmark needed formal integration into Dokime's task suite and CLI.
- **Remediation & Behavioral Guarantees:**
  - Implemented `SymbolonBenchmarkTask` conforming to Dokime's `Task` protocol (`task_id="symbolon_code_retrieval"`, `load()`, `score()`, `score_detailed()`).
  - Added robust response parser `extract_retrieved_doc_ids()` handling JSON arrays, Markdown code blocks, free text `doc_*` token matching, and newline/comma-delimited outputs.
  - Added multi-domain diagnostic dataset (`symbolon_suite.json`) containing 8 query-corpus pairs spanning biochemistry, EU geopolitics, Apollo 11 aerospace, topological sheaf Laplacians, EUV lithography, and SHA-256 audit chaining with explicit gold relevance and contradiction labels.
  - Implemented `SymbolonRetrievalBenchmarkSuite` executing 4 comparative arms (Flat dense baseline, Symbolon alone, Uncapped union, Capped RRF) with statistical metric computation and GitHub-style Markdown report tables.
  - Added `dokime symbolon-benchmark` CLI subcommand supporting `--top-k`, `--level`, and `--output <path>` JSON report export.
  - Verified with 6 unit and integration tests in `dokime/tests/test_symbolon_benchmark.py` (201 Dokime tests passing).

### Item 89: Dokime Executable Multi-Hop Graph Reasoning Task & Verifiable Grounding
- **Category:** Evaluation & Multi-Hop Reasoning Diagnostics (Dokime Tasks, Deterministic Executor, Hashed Fixture, Grounded Evidence Support)
- **Priority:** P2 (Diagnostic & Release Gating Integrity)
- **Status:** Resolved & Verified
- **Affected Subsystems:**
  - `dokime/src/dokime/tasks/executable_graph.py`
  - `dokime/src/dokime/tasks/data/executable_graph_suite.json`
  - `dokime/src/dokime/tasks/registry.py`
  - `dokime/src/dokime/tasks/__init__.py`
  - `dokime/tests/test_executable_graph.py`
- **Root Cause & Specification Requirement:**
  - Multi-hop reasoning benchmarks often suffer from LLM answer key hallucinations, missed path constraints, incomplete answer sets, and false credits for ungrounded guessing.
  - Per Evolve on Graph (arXiv:2610.35598) Section 78–81 takeaways for Metis, Dokime requires inspectable multi-hop reasoning test cases where expected answers and citation chains derive from deterministic query execution over a frozen, SHA-256 hashed synthetic fact fixture (`07295795b5fe11657898373822caf8376e0e283f6f3cbf7d978ae8a3846de12e`).
- **Remediation & Behavioral Guarantees:**
  - Implemented canonical 40-fact directed graph fixture (`SYNTHETIC_KNOWLEDGE_FIXTURE`) spanning Computer Science, Biochemistry, Aerospace, and Distributed Consensus with tamper-evident hashing (`compute_fixture_hash()`).
  - Built `GraphQueryExecutor` supporting forward chains (1-hop, 2-hop, 3-hop), reverse chains, and multi-constraint intersections with path-level fact tracking to eliminate irrelevant dead-end facts.
  - Designed formal record schema `ExecutableCaseRecord` and generated 20 wording-audited benchmark cases in `executable_graph_suite.json` across 6 distinct query patterns and test/practice splits.
  - Implemented `ExecutableGraphTask` (`task_id="executable_graph_qa"`) providing set-based answer Precision, Recall, F1, exact match, and separate evidence support Precision/Recall/F1 to ensure unsupported correct guesses fail `evidence_grounded_success`.
  - Registered `executable_graph_qa` in Dokime's task registry and added `dokime graph-rollout` CLI subcommand with `ExecutableGraphRolloutSuite` for repeated-attempt rollout exports. Verified via `dokime tasks` and 11 unit tests in `dokime/tests/test_executable_graph.py` (212 Dokime tests passing).

### Item 90: Paideia Boundary-Aware Practice Selection & Rollout Variance Diagnostic
- **Category:** Training Data & Policy Difficulty Diagnostics (Paideia Contracts, Boundary Sampler, Rollout Variance, Stratified Selection)
- **Priority:** P2 (Diagnostic & Release Gating Integrity)
- **Status:** Resolved & Verified
- **Affected Subsystems:**
  - `paideia/paideia/contracts/boundary_schema.py`
  - `paideia/paideia/data/boundary_sampler.py`
  - `paideia/paideia/contracts/__init__.py`
  - `paideia/paideia/data/__init__.py`
  - `paideia/paideia/cli.py`
  - `paideia/tests/test_boundary_sampler.py`
- **Root Cause & Specification Requirement:**
  - Training or fine-tuning models on tasks that are either universally solved ($\bar{s} = 1.0$) or universally failed ($\bar{s} = 0.0$) yields zero answer-reward contrast and wastes compute.
  - Per Evolve on Graph (arXiv:2610.35598) Item 2 takeaways for Metis, Paideia requires boundary-aware task selection that evaluates sample variance $\sigma^2$ across repeated rollout attempts ($N \ge 2$), identifying the active learning frontier while maintaining a principled mixture of retention checks and hard diagnostics.
- **Remediation & Behavioral Guarantees:**
  - Defined versioned Pydantic contract schemas (`CaseAttemptRecord`, `CaseAttemptStats`, `BoundarySelectionConfig`, `BoundarySelectionReport`) categorizing tasks into `BOUNDARY` ($\sigma^2 > \text{min\_var}$), `RETENTION_CHECK` ($\bar{s}=1.0, \sigma^2=0$), `HARD_DIAGNOSTIC` ($\bar{s}=0.0, \sigma^2=0$), and `CONSISTENT_PARTIAL` ($0 < \bar{s} < 1.0, \sigma^2=0$).
  - Implemented `boundary_sampler.py` computing unbiased sample variance $s^2 = \frac{1}{N-1}\sum (s_i - \bar{s})^2$ and selecting practice batches with configurable mixture ratios (default 60% boundary, 20% retention, 20% hard), prioritizing highest-variance cases.
  - Implemented `select_uniform_practice_batch` to provide an exact matched-budget baseline comparison.
  - Added CLI command `paideia dataset sample-boundary` supporting custom budget, stratum ratios, variance threshold, uniform baseline comparison, and JSON audit report export.
  - Verified with 4 unit tests in `paideia/tests/test_boundary_sampler.py` (50 Paideia tests passing).

---

### Item 91: Zero-Hallucination Graph Filtering, Dynamic Consistency Fallback & Constructive Grounding
- **Priority:** P1 (Reasoning Reliability & Hallucination Elimination)
- **Status:** **Remediated & Verified**
- **Affected Subsystems:**
  - `stoma/src/stoma/filters.py`
  - `stoma/src/stoma/__init__.py`
  - `stoma/src/stoma/cli.py`
  - `stoma/tests/test_span_filter.py`
  - `kybernetes/kybernetes/routing/router.py`
  - `kybernetes/kybernetes/adapters/dianoia.py`
  - `kybernetes/kybernetes/engine/pipeline.py`
  - `kybernetes/tests/test_routing.py`
  - `kybernetes/tests/test_adapters.py`
  - `docs/review/ZERO_HALLUCINATION_GRAPH_FILTERING_SPEC.md`
- **Root Cause & Specification Requirement:**
  - Confirmatory Phase 1 evaluation (300 stories, 15,600 LLM calls) demonstrated that ungrounded knowledge graph corruption induces severe performance collapse ($\Delta = -0.1700, p < 10^{-5}$) and joint context distraction on multi-hop dependencies ($\Delta = -0.0425, p < 10^{-5}$). Furthermore, explicit error-hunting reconciliation prompts (Arm D) triggered hyper-vigilant false-contradiction hallucinations, underperforming matched constructive planning control (Arm F) by $\Delta = -0.0350$ ($p < 10^{-5}$).
  - A 4-tier zero-hallucination defense was required: Tier 1 programmatic span and temporal grounding verification before ingestion, Tier 2 dynamic consistency gating ($\mathcal{C} \ge 0.95$) with automatic eviction and flat text fallback, Tier 3 constructive forward grounding prompts (*"restate facts and timeline constructively"*), and Tier 4 offline permutation audits.
- **Remediation & Behavioral Guarantees:**
  1. Implemented `DeterministicSpanFilter` in `stoma.filters` providing zero-token programmatic verification of entity mentions, hour timestamps, and relational co-occurrence against raw source text passages.
  2. Added `stoma graph verify-grounding` CLI command with configurable consistency floor (`--consistency-floor`, default 0.95).
  3. Extended `kybernetes.routing.router` so `StageName.DELIBERATE` carries `graph_consistency_floor: 0.95` and `fallback_to_pure_text_on_inconsistency: True`.
  4. Extended `DianoiaAdapter.deliberate()` with dynamic consistency verification, returning `consistency_fallback_used=True`, `consistency_ratio`, and cleanly evicting inconsistent graph blocks to flat text passages.
  5. Wired consistency parameters and outcomes into `ExecutionPipeline` and updated synthesis prompts with constructive grounding and chronological timeline restatement.
  6. Verified with 5 unit tests in `stoma/tests/test_span_filter.py` and 3 integration tests in `kybernetes/tests/test_adapters.py`. All 123 Stoma tests and 196 Kybernetes tests pass.

---

### Item 92: Heuresis Autonomous Scientific Discovery Engine & OpenReview Boundary Crawler
- **Priority:** P2 (Long-Term Autonomous Exploration & Frontier Discovery)
- **Status:** **Integrated & Verified**
- **Affected Subsystems:**
  - `stoma/src/stoma/crawlers/openreview.py`
  - `stoma/src/stoma/crawlers/__init__.py`
  - `stoma/src/stoma/__init__.py`
  - `stoma/src/stoma/cli.py`
  - `stoma/tests/test_openreview_crawler.py`
  - `dokime/src/dokime/tasks/heuresis.py`
  - `dokime/src/dokime/tasks/registry.py`
  - `dokime/src/dokime/tasks/__init__.py`
  - `dokime/tests/test_heuresis.py`
  - `docs/review/HEURESIS_DISCOVERY_ENGINE_SPEC.md`
  - `docs/review/ZERO_HALLUCINATION_GRAPH_FILTERING_SPEC.md`
- **Root Cause & Specification Requirement:**
  - Standard discovery architectures suffer from author availability bias (over-indexing on familiar internal corporate jargon while neglecting critical external community breakthroughs) and representation collapse when encountering Out-of-Vocabulary (OOV) or truly novel concepts.
  - An autonomous scientific discovery engine requires crawling peer-review rejection boundaries (OpenReview contested submissions) to extract high-variance boundary phenomena, combined with Sheaf-Laplacian topological concept synthesis and dual fast contrastive scoring.
- **Remediation & Behavioral Guarantees:**
  1. Implemented `OpenReviewBoundaryCrawler` and `CommunityAvailabilityIndex` in `stoma.crawlers.openreview` parsing peer-review forum ratings, sample variance $\sigma^2 \ge 1.5$, reviewer boundary objections, and contested submissions ($4.0 \le \bar{s} \le 6.5$).
  2. Added `stoma openreview crawl-fixture` CLI command for boundary extraction and availability index generation.
  3. Implemented `HeuresisDiscoveryEngine` in `dokime.tasks.heuresis` with fast dual-scoring:
     $$\text{Score}(A, B, C) = z(S_{\text{coh}}) - \beta \cdot z(S_{\text{avail}}) + 0.35 \cdot \mathcal{O} + 0.20 \cdot \mathbf{1}_{\text{fringe}}$$
  4. Implemented abductive Out-of-Vocabulary (OOV) concept synthesis detecting topological sheaf obstruction energy $\mathcal{O} > 0$ and coining resolving primitives (e.g. `poincare_curvature_sheaf_transformer`).
  5. Enforced 15% fringe outlier exploration quota preserving HDBSCAN noise units to systematically cross-breed non-standard ideas into mainstream paradigms.
  6. Implemented `HeuresisDiscoveryTask` in Dokime and registered `"heuresis_scientific_discovery"` in `BUILTIN_TASKS`.
  7. Verified with 4 unit tests in `stoma/tests/test_openreview_crawler.py` and 5 unit/integration tests in `dokime/tests/test_heuresis.py`. All 127 Stoma tests and 218 Dokime tests pass.

### Item 93: Phase 2 Empirical Evaluation: Text vs Graph Under Real LLM Extractor Noise
- **Priority:** P2 (Empirical Validation & Cognitive Representation Architecture)
- **Status:** **Confirmatory Phase 2 Completed & Audited ($N=300$ stories, 5,100 LLM operations)**
- **Affected Subsystems:**
  - `experiments/text_vs_graph/extractor.py`
  - `experiments/text_vs_graph/run_phase2.py`
  - `experiments/text_vs_graph/analyze_phase2.py`
  - `experiments/text_vs_graph/test_extractor.py`
  - `experiments/text_vs_graph/phase2_results.jsonl`
  - `experiments/text_vs_graph/phase2_extractions.jsonl`
  - `experiments/text_vs_graph/phase2_summary.json`
  - `experiments/text_vs_graph/phase2_confirmatory_analysis.json`
  - `experiments/text_vs_graph/CONFIRMATORY_PHASE2_REPORT.md`
- **Root Cause & Specification Requirement:**
  - Phase 1 evaluated causal reasoning under synthetic 30% corruption (random edge drops, timestamp flips, and attribute mutators), which degraded Arm B to 73.83% accuracy. Real-world sovereign pipelines do not experience random independent synthetic bitflips, but rather real LLM extractor noise directly from unstructured text passages.
  - Phase 2 confirmatory validation was required across all 300 stories to measure the true distribution of real extractor errors and test whether constructive chronological grounding (Arm F) and structured graph representations preserve cognitive superiority under real LLM extraction noise.
- **Remediation & Behavioral Guarantees:**
  1. Implemented zero-shot LLM extractor in `experiments/text_vs_graph/extractor.py` parsing structured event DAGs directly from prose via `gemma-4-26b-a4b-abliterated` on Hestia.
  2. Implemented `experiments/text_vs_graph/run_phase2.py` with atomic checkpointing, extraction caching, and multi-arm evaluation (B, C, D, F).
  3. Completed full 300-story confirmatory evaluation (`conf_s001`–`conf_s300`, seeds `4001`–`4300`, 4,800 evals + 300 extractions = 5,100 LLM calls):
     - **Natural Extractor Profile**: Mean fidelity of $99.08\%$ ($279/300$ stories 100% perfect, only 1 hallucinated event across 2,400 gold events, $0.71\%$ omission rate).
     - **H2' Confirmed (Text + Extracted Graph vs Text Baseline)**: Arm C achieved **$96.67\%$** vs **$90.83\%$** for Arm A ($+5.83\text{ pp}$ gain, $95\%\text{ CI } [+0.0417, +0.0742]$, verdict: **BETTER**, $p < 10^{-5}$).
     - **Real vs Synthetic 30% Corruption**: Arm B achieved **$93.75\%$** (vs $73.83\%$ under synthetic corruption, $+19.92\text{ pp}$ gain, $95\%\text{ CI } [+0.1725, +0.2258]$, verdict: **BETTER**). Arm C achieved **$96.67\%$** (vs $86.58\%$, $+10.08\text{ pp}$ gain, verdict: **BETTER**).
     - **Arm F Dominance**: Arm F (Chronological Control) achieved **$99.58\%$ accuracy** ($1,195/1,200$ questions correct), definitively outperforming Arm D ($97.08\%$, $-2.50\text{ pp}$, $95\%\text{ CI } [-0.0350, -0.0158]$) and eliminating 97.9% of length truncations.
     - **Counterfactual Divide (Q3)**: Arm F achieved **$98.7\%$** and Arm C achieved **$86.7\%$** on multi-hop counterfactual DAG propagation, substantially outperforming single-pass pure text Arm A ($75.3\%$, $+11.4\text{ pp}$ to $+23.4\text{ pp}$ gain).
  4. Authoring and auditing formal confirmatory report `experiments/text_vs_graph/CONFIRMATORY_PHASE2_REPORT.md`.
  5. 27 unit tests in `experiments/text_vs_graph/test_*.py` passing cleanly.

---

### Item 94: Core Engine Arm F Two-Pass Constructive Chronological Deliberation Engine
- **Priority:** P1 (Reasoning Efficiency & Token Runway Truncation Elimination)
- **Status:** **Implemented & Fully Verified**
- **Affected Subsystems:**
  - `kybernetes/kybernetes/adapters/dianoia.py`
  - `kybernetes/kybernetes/routing/router.py`
  - `kybernetes/kybernetes/engine/pipeline.py`
  - `kybernetes/tests/test_adapters.py`
  - `kybernetes/tests/test_routing.py`
- **Root Cause & Specification Requirement:**
  - In Confirmatory Phase 2 evaluation ($N=300$), single-pass multi-hop reasoning over unstructured narrative prose suffered severe length truncations at the 512-token ceiling (Arm A 75.3% on Q3; 94 global truncations). Furthermore, prompting models to search for discrepancies (Arm D) triggered false-contradiction hallucinations and scratchpad bloat.
  - Arm F's two-pass constructive discipline achieved **$99.58\%$ overall accuracy** and **$98.7\%$ on Q3**, reducing length truncations by 95% (only 2 truncations across 1,200 runs).
  - To productionize this finding into the sovereign runtime, `StageName.DELIBERATE` and `ExecutionPipeline` required native two-pass constructive chronological planning support.
- **Remediation & Behavioral Guarantees:**
  1. Extended `DianoiaDeliberationResult` with `constructive_planning_used: bool = False`.
  2. Extended `DianoiaAdapter.deliberate()` with `two_pass_constructive_planning: bool = False` and `hestia_client: Optional[Any] = None`. When enabled, Pass 1 constructs chronological timeline notes (either via live Hestia inference or structured sequence aggregation) and sets `constructive_planning_used = True`.
  3. Updated `kybernetes.routing.router` so `StageName.DELIBERATE` enables `two_pass_constructive_planning: True` by default on multi-hop routes.
  4. Updated `ExecutionPipeline` to wire `two_pass_constructive_planning` and pass `self.hestia` to `deliberate()`, recording `constructive_planning` in stage outcome metrics.
  5. Added unit tests in `kybernetes/tests/test_adapters.py` (`test_dianoia_two_pass_constructive_planning_simulated` and `test_dianoia_two_pass_constructive_planning_live_mock_hestia`) and verified routing in `kybernetes/tests/test_routing.py`. All 198 Kybernetes tests pass cleanly.

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
| **Instruction Evaluation Engine** | Item 80 | Dokime Tasks (IFEval Programmatic Suite) | **Integrated & Verified** | 195 Dokime tests passing (25 IFEval unit/integration tests) |
| **Conformal Prediction Uncertainty** | Item 81 | Krisis (KnowNo Finite-Sample Quantile Sets) | **Integrated & Verified** | 40 Krisis tests passing (7 conformal unit/integration tests) |
| **Sufficient Input Subsets Explainer**| Item 82 | Dokime Explainers (SIS Backward Selection & CLI) | **Integrated & Verified** | 195 Dokime tests passing (9 SIS unit/integration tests) |
| **Speculative Cascades Serving** | Item 83 | Hestia (Speculative Drafter-Verifier Cascades & Deferral) | **Integrated & Verified** | 77 Hestia tests passing (5 cascade unit/API tests) |
| **Active Selective Prediction (ASPEST)** | Item 84 | Krisis + Paideia (Active Abstained Batch Curation & CLI) | **Integrated & Verified** | 40 Krisis + 46 Paideia tests passing (9 ASPEST tests) |
| **ANN Indexing & Empirical Yardstick** | Item 85 | LTHCM + Dokime (Google ScaNN Anisotropic Tree-AH & Benchmarks) | **Integrated & Verified** | 133 LTHCM + 195 Dokime tests passing (9 ScaNN/ANN tests) |
| **VRDU Document Extraction & Ingestion** | Item 86 | Stoma + Dokime (Structured Table/Chart/KV Ingestion & VRDU Task) | **Integrated & Verified** | 118 Stoma + 195 Dokime tests passing (9 VRDU tests) |
| **Security Audit Remediations (Round 5)**| Item 87 | Kybernetes, Prosopon, Phylax, Krisis (F-02, F-03, F-04, F-05, F-07, F-08)| **Remediated & Verified** | 193 Kybernetes + 73 Prosopon tests passing (0 loopback leaks) |
| **Symbolon Retrieval Benchmark & Yardstick** | Item 88 | Dokime Tasks (Symbolon Task, 4-Arm Suite & CLI) | **Integrated & Verified** | 201 Dokime tests passing (6 Symbolon benchmark tests) |
| **Executable Graph Reasoning & Grounds** | Item 89 | Dokime Tasks (Deterministic Executor, Hashed Fixture, Task Plugin, Rollouts)| **Integrated & Verified** | 213 Dokime tests passing (12 executable graph tests) |
| **Boundary-Aware Practice Selection** | Item 90 | Paideia (Attempt Rollout Variance, Stratified Selection & CLI)| **Integrated & Verified** | 50 Paideia tests passing (4 boundary sampler tests) |
| **Zero-Hallucination Graph Filtering** | Item 91 | Stoma + Kybernetes (DeterministicSpanFilter & Dynamic Consistency Gating)| **Integrated & Verified** | 123 Stoma + 196 Kybernetes tests passing (8 new unit/adapter tests) |
| **Heuresis Scientific Discovery Engine**| Item 92 | Stoma + Dokime (OpenReview Crawler, Dual Scorer, Sheaf OOV Synthesizer)| **Integrated & Verified** | 127 Stoma + 218 Dokime tests passing (9 new crawler/discovery tests) |
| **Phase 2 Real Extractor Empirical Study**| Item 93 | Experiments (Real LLM Extractor Noise, 300 Stories, Bootstrap CIs)| **Audited & Verified** | 27 tests passing + 4,800 evals verified ($99.58\%$ on Arm F, H2' BETTER) |
| **Arm F Constructive Deliberation Engine**| Item 94 | Kybernetes + Dianoia (Two-Pass Constructive Planning & Routing)| **Integrated & Verified** | 198 Kybernetes tests passing (Pass 1 timeline notes & truncation elimination) |

---

### Monorepo Subsystem Test Verification Breakdown (1,140 Total Passed)

| Subsystem | Test Command | Tests Passed | Skipped | Runtime | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dokime** (Evaluation & Release Gates) | `./.venv/bin/pytest dokime/tests -q` | **218** | 0 | 9.44s | **PASS** |
| **Kybernetes** (Orchestrator & Tools) | `./.venv/bin/pytest kybernetes/tests -q` | **198** | 0 | 35.35s | **PASS** |
| **LTHCM** (Poincaré Memory Engine) | `./LTHCM/.venv/bin/pytest LTHCM/tests -q` | **119** | 2 | 143.21s | **PASS** |
| **Stoma** (Ingestion & Chunking) | `CUDA_VISIBLE_DEVICES="" ./stoma/.venv/bin/pytest stoma/tests -q` | **127** | 0 | 227.52s | **PASS** |
| **Dianoia** (MCTS Deliberation Engine) | `./.venv/bin/pytest dianoia/tests -q` | **93** | 0 | 6.91s | **PASS** |
| **Hestia** (GGUF Inference Serving) | `./hestia/.venv/bin/pytest hestia/tests -q` | **77** | 1 | 1.54s | **PASS** |
| **Prosopon** (WebRTC & Access Store) | `./prosopon/.venv/bin/pytest prosopon/tests -q` | **73** | 0 | 44.81s | **PASS** |
| **Paideia** (Training Pipeline) | `./paideia/.venv/bin/pytest paideia/tests -q` | **50** | 0 | 2.68s | **PASS** |
| **Kleroterion** (Stochastic Policy) | `./kleroterion/.venv/bin/pytest kleroterion/tests -q` | **43** | 0 | 2.12s | **PASS** |
| **Krisis** (Claim Grounding & Conformal Gates)| `./.venv/bin/pytest krisis/tests -q` | **40** | 0 | 0.42s | **PASS** |
| **Phylax** (Capability Security Gateway) | `./.venv/bin/pytest phylax/tests -q` | **36** | 0 | 1.56s | **PASS** |
| **Experiments** (Text vs Graph Empirical Suite) | `./.venv/bin/pytest experiments/text_vs_graph/test_*.py -q` | **27** | 0 | 0.08s | **PASS** |
| **Root Integration** | `./.venv/bin/pytest tests -q` | **39** | 0 | 20.35s | **PASS** |
| **VS Code Void IDE Client** | `npm run compile` (in `vscode/`) | **0 errors** | 0 | 65.62s | **PASS** |
| **Total Monorepo Suite** | **All 11 Subsystems + Experiments + IDE Client** | **1,140** | **3** | **~8.8 min** | **ALL GATES PASS** |





