# Unified Dashboard Plan — Gap Review

**Reviewed:** `metis-dashboard-implementation-spec-ae93d36c.txt` (the spec), Item 28 of `metis-repo-wide-round1-fixes-6752222c.txt`, the 11 per-module `DASHBOARD_SPEC.md` files, and the code as it stands after the round-1 repairs (uncommitted working tree, 2026-10-05).
**Method:** source reading plus the module test runs from the repair work. No servers were started and no models were loaded for this review.
**Legend:** ✅ met · ⚠️ partial · ❌ not met · 🆕 gap that the spec does not cover

---

## 0. Verdict

The spec's principles are sound and should be kept. Those principles are:
- Prosopon owns the UI and Kybernetes owns `/v1/admin/*`.
- Each module validates its own config.
- Status uses five honest states.
- Saved, desired and effective settings are kept separate.
- Changes go through validate → apply → history.
- The dashboard is built read-only first.

There are three groups of problems:

1. **The Step 1 prerequisite is not actually met yet.** The round-1 `/v1/health` change removed the hardcoded `"ready"`. However, it decides `live` from a non-null shim, module or memory root, and the spec explicitly forbids that (§4.2). The unauthenticated health endpoint also now leaks a filesystem path.
2. **The spec is missing contracts needed to build the sections.** It has no list endpoints, no way for a module to report its runtime config, no probe scheduler design, and no status model for batch tools that are not running services (Stoma, Dokime, Paideia).
3. **The security and audit design is incomplete.** It doesn't cover the lifecycle of the BFF credential, auditing of admin actions (including model load/unload), or a CSP for the UI.

---

## 1. Spec prerequisites (F1–F7) against the current code

| # | Finding | Status | What's still wrong |
|---|---|---|---|
| F1 | Health hardcodes "ready" | ⚠️ | [app.py `health_check`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py) sets `live` when `dianoia_module`, `shim` or `verifier_client` is non-null, or when `memory_root` is set. The spec (§4.2) says these are **not** proof. It still returns `status: "ready"` for live components. It has no `evidence`, `checked_at` or `freshness` fields. An injected component that doesn't work would show as live. The test only covers the default (simulated) case. |
| F2 | Simulated adapters, no memory-root CLI | ⚠️ | `kybernetes serve --memory-root` exists. Dianoia, Krisis and Phylax are still local simulations in `create_app`. There's no Kybernetes config file, and `HESTIA_BASE_URL` is the only env var. |
| F3 | CORS `*`, no auth | ⚠️ | CORS now uses an explicit origin list ✅. But auth is **fail-open**: `require_service_auth` passes every request when `KYBERNETES_SERVICE_TOKEN` is unset. There's no Host-header validation. `/v1/health` is unauthenticated and now returns `memory_root` (an absolute path), the Hestia URL and the model list, which breaks the redaction rule in §4.5. |
| F4 | Prosopon mock by default | ⚠️ | `mock_kybernetes` still exists. Prosopon's `/api/health` is still static (`{"status":"ok"}`) and doesn't check upstream. |
| F5 | Hestia backend ports clash with 8081 | ✅ | `hestia.yaml` backend range is now 9000–9019. Whether a port is actually free is still not checked. |
| F6 | LICENSE link broken | ✅ / owner | The link is removed from the README. The owner still has to choose a license. |
| F7 | Budget DB persistent vs `:memory:` | ❌ | Unchanged. It is not surfaced anywhere yet. |

---

## 2. Gaps in the plan itself

### G1 🆕 No data sources for several sections
- **Sessions and budget:** Kybernetes only offers `GET /v1/traces/{turn_id}` and `GET /v1/budget/report/{task_id}`, both by ID. There are no list endpoints. Sessions are kept in memory only (LRU of 500, 24 h TTL) and traces are capped at 1000, so both are lost on restart. The dashboard needs paginated `GET /v1/admin/sessions` and `GET /v1/admin/traces?limit=&before=`, and should label them "in-memory, since <start time>".
- **Guardrails:** pending approvals, policy hash and audit-chain verification are not exposed over HTTP. Kybernetes' `PhylaxAdapter` keeps its own in-memory approvals, which are separate from the real Phylax audit chain.
- **Reasoning:** "recent verdicts" would have to come from traces, and those are in memory only.

### G2 🆕 The five states don't fit batch tools
Stoma, Dokime and Paideia have no running service, so they can never truthfully be "live". The proposal:
- Add `kind: service | library | batch_tool` to `ModuleDescriptor`.
- For batch tools, show **last-run evidence** (manifest path hash, timestamp, exit status, simulated flag) with freshness, and leave runtime state as `unknown` / `offline`.
- Read runs only from allowlisted directories that the server configures.

### G3 🆕 No module can report its effective config
The spec says effective config must come from the running process, but no module has a runtime-config report today. The plan needs:
- Kybernetes: self-report (in-process).
- Hestia and Prosopon: an authenticated `GET …/runtime-config` that returns the frozen config the process loaded, its hash and its load time.
- Batch tools: effective config is "per run", taken from the run manifest.
- LTHCM, Dianoia and Kleroterion: they have no config file, only CLI or constructor args, so they are marked "not configurable via dashboard".

### G4 🆕 The probe framework is unspecified
The spec defines the rules but not how probes run. The missing pieces:
- **Probe registry and cache.** It sits in Kybernetes and holds a `ProbeResult` for each component.
- **Two tiers, with suggested defaults.**
  - *Reachability:* runs in the background at most every 15 s, with a 2 s timeout, and goes stale after 45 s.
  - *Functional:* runs only when the user triggers it, at most once per minute, with a 10 s timeout, and goes stale after 15 min.
- **Overview behaviour.** It only reads the cache and never probes inline.
- **Staleness.** It is computed on the server from `checked_at` in UTC, not from the browser's clock.
- **Concurrency.** Only one functional probe may run per component at a time.
- **Logging.** Probe results are recorded so they survive a page reload.

### G5 🆕 Functional probes need fixtures that don't change real state
- **Phylax:** use a dedicated test capability and a separate audit namespace, or tagged entries, so the probe never consumes real approvals.
- **LTHCM:** needs a known canary passage or query in the index. Otherwise the probe can only say "index loaded", not that retrieval works.
- **Krisis:** a known-answer fixture.
- **Dianoia walk:** costs a model completion, so it is manual only and only when the model is already loaded (the spec already says this for Hestia).

### G6 🆕 BFF credential lifecycle
- `/v1/admin/*` must **refuse to start** without a token, rather than letting requests through.
- Use a **separate admin token** from the chat token, for least privilege.
- Define how the token is generated (Prosopon at pairing, or the stack launcher), where it is stored (file with `0600` permissions), and how it is rotated.
- Prosopon already has a strict guard with no loopback exemption ([server.py](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py) ~L124). Reuse it for `/api/dashboard/*` and add an explicit test for it.
- Sensitive confirm actions should require re-authentication or a fresh PIN.

### G7 🆕 Admin actions are not audited, and load/unload is not guarded
- ChangeSet history covers config edits only. Add an **append-only admin audit log** that records authenticated reads of sensitive views, denied requests, probe triggers and model load/unload.
- Hestia load/unload is a GPU side effect. Treat it as a confirmed action and show the VRAM admission result. Consider routing it through Phylax policy so there is a single enforcement point.

### G8 🆕 Redaction must also cover health
- Cut unauthenticated `/v1/health` down to `{service, version, overall, per-component state}`.
- Move detail, paths and URLs to the authenticated `/v1/admin/overview`, using the descriptor's redaction rules.
- Same for Prosopon: keep its own `/api/health` public, and add upstream status only on the authenticated dashboard route.

### G9 🆕 Release identity on the Overview
- Overview should show git commit plus dirty flag, each module's version, and the served model identity.
- `artifact_digest` now exists in Hestia `/v1/models` but is **only filled in if `models.yaml` provides it**. It is not computed, so it must be shown as "unverified" when missing.
- The `stack.lock` compatibility check from [STACK_BUILD_CONTRACT.md](STACK_BUILD_CONTRACT.md) is not implemented. Show "compatibility: unchecked".

### G10 🆕 How updates reach the UI
The spec doesn't say. Recommendation for v1: poll the cached overview every 10–15 s using an ETag/`If-None-Match`, and load each section independently with its own timeout. No SSE in v1.

### G11 🆕 Hestia telemetry: estimated vs measured
VRAM figures come from `vram_estimate_mb` (an estimate). Label them **estimated**. Real measurement would need NVML or `nvidia-smi`, which is read-only but GPU-touching, so it needs an owner decision (see §5). The two-GPU split, and `device_lock.py`, should be shown as configured rather than assumed.

### G12 🆕 UI safety and reach
- Probe errors, model names and paths are untrusted strings. Render them with `textContent` or DOM APIs only, following the round-1 `ui.js` fixes. Add a strict **CSP** header and don't use inline handlers.
- Default the dashboard to **loopback only**. Exposing it over LAN should need an explicit flag plus pairing plus TLS.
- Check that it works on mobile (Prosopon is used from a phone) and with basic accessibility tools.

### G13 🆕 Config ownership gaps
- Kybernetes has no config file yet (spec Step 2). Pick a format before Step 4.
- Phylax and Krisis policies are security-critical. Recommend keeping them **read-only plus diff view** through v2. Editing would need a review flow; one-click apply is not enough.

### G14 🆕 Missing tests
- **Shared contract tests:** one JSON Schema for `ComponentStatus` and `ConfigView`, used by both Kybernetes and the Prosopon proxy.
- **No load on poll:** poll the overview N times and assert Hestia `/manage/load` and `/v1/chat/completions` are never called.
- **Auth matrix:** {loopback, remote} × {no token, chat token, admin token} × {bad Origin, bad Host}.
- **Redaction snapshots:** no absolute paths or secrets in any admin or health response.
- **False-live guard:** an injected but broken adapter must never report `live`.

### G15 🆕 The per-module docs overstate
Some of the 11 `DASHBOARD_SPEC.md` files describe the control plane as if it exists. For example, Kybernetes' says it "Maintains registrations of all 11 stack modules". Mark them **[PROPOSAL]** and bring them in line with this review.

### G16 Open questions (unchanged from spec §12)
- Q1: what "project recommendation" means.
- Q2: whether v1 should have edit controls at all.
- Q3: whether the dashboard lives in Prosopon or a separate app.

---

## 3. Unfinished parts of the round-1 repairs

These affect what the dashboard can truthfully show:

- **Item 10:** this is the F1 non-compliance above. It also isn't done that "real denied actions reach the actual GuardedGateway": the server still uses Kybernetes' in-memory Phylax shim.
- **Item 26:** only partly done. There are no tokenizer or chat-template digests, `artifact_digest` is not computed, there is no release mode that fails when identity is unknown, and there are no held-out end-to-end suites.
- **Stoma provenance:** the `dianoia` `content_sha256` in [vendored_provenance.json](file:///home/akarta/Projects/Metis/stoma/vendored_provenance.json) was **re-baselined** after the intentional `walk.py`/`cli.py` edits. This is a tamper check, so review and commit it deliberately.
- **Not run locally:** the Stoma and LTHCM suites (missing `torch`/`pypdf`). The pickle-gating changes in `stoma/adapter.py` and `interchange.py` are **untested**.
- **CI:** the `.gitlab-ci.yml` changes have not been run in GitLab. Kleroterion needs a `PYTHONPATH` override, and test files with clashing basenames have to be run one directory at a time.
- **Behaviour change:** the Prosopon client now defaults to `http://127.0.0.1:8081/v1` and adds `/v1` automatically.

---

## 4. Revised build order

| Step | Content | New vs spec | Implementation Status |
|---|---|---|---|
| **0** | Fix health compliance. Build the probe registry and cache (G4) and add `state` / `evidence` / `checked_at` / `freshness`. Never derive `live` from a non-null dependency. Strip paths from unauthenticated health (G8). Add the false-live test. | 🆕 (finishes spec Step 1) | ✅ Completed & Verified |
| **1** | Kybernetes config file covering `hestia_base_url`, `memory_root`, `db_path`, `allowed_origins` and `admin_token_file`. | spec Step 2 + G13 | ✅ Completed & Verified |
| **2** | Lockdown: admin token required (refuse to start without it), separate admin and chat tokens, Host validation, admin audit log, auth-matrix tests. | spec Step 3 + G6/G7 | ✅ Completed & Verified |
| **3** | Descriptors with `kind`, `/v1/admin/overview`, the list endpoints, and runtime-config reports for Kybernetes, Hestia and Prosopon. | spec Step 4 + G1/G2/G3 | ✅ Completed & Verified |
| **4** | Prosopon read-only UI: strict guard, CSP, rendering with `textContent`, per-section timeouts, ETag polling, loopback by default, no-load-on-poll test. | spec Step 5 + G10/G12/G14 | ✅ Completed & Verified |
| **5** | The remaining modules, read-only. Batch tools show last-run evidence. Phylax and Krisis policy is read-only with a diff view. | spec Step 6 + G2/G13 | ✅ Completed & Verified |
| **6** | Manual functional probes with fixtures that don't touch real state. | 🆕 G5 | ✅ Completed & Verified |
| **7** | Validate endpoint, using each module's existing validators. | spec Step 7 | ✅ Completed & Verified |
| **8** | Guarded apply, history and rollback, with the ChangeSet status model. Load/unload becomes a confirmed, audited action. | spec Step 8 + G7 | ✅ Completed & Verified |
| **9** | Optional Dokime and Paideia run controls. | spec Step 9 | ✅ Completed & Verified |

---

## 5. Decisions needed from the owner

1. **Q1–Q3** from spec §12: what "recommendation" means, whether v1 has edit controls, and Prosopon vs a separate app.
2. Should the dashboard ever be reachable over LAN, or stay loopback only?
3. Is read-only GPU telemetry through NVML acceptable, or should the dashboard stay with estimates only?
4. Should Phylax and Krisis policies ever be editable from the dashboard?
5. Which functional probes may spend a model completion (Hestia, Dianoia), and how often?
6. Should sessions and traces be made persistent (SQLite) so the Sessions section survives restarts?

---

## 6. Additions to the acceptance checks (on top of spec §9)

- Default stack: Overview shows **no** component as `live` without a recorded functional probe. Every chip shows `checked_at` and freshness.
- Injected but broken adapter: shows `simulated` or `unknown`, never `live`.
- Unauthenticated `/v1/health` contains no absolute paths, URLs or model file names.
- Kybernetes refuses to start with admin routes enabled and no admin token.
- 100 overview polls trigger zero Hestia load or completion calls.
- Admin audit log records a denied request, a probe trigger and a load/unload click.
- Dokime and Paideia sections show the last run with its simulated flag and freshness, never `live`.
