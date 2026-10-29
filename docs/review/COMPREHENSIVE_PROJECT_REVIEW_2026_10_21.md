# Metis Comprehensive Repository Review, Gaps & Defects Audit

- **Repository**: `/home/akarta/Projects/Metis` (`https://gitlab.com/akarta-315h/metis.git`)
- **Branch**: `main`
- **Reviewed HEAD Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Submodule Pin (`vscode`)**: `2fd2fa78eebfabc97b8ece56008bc8947de9a46d` (`heads/main`, clean)
- **Working-Tree Status**: Clean (`git status --short` empty prior to writing this report; local gitignored runtime artifacts present in `.venv/`, `kybernetes.db`, `kybernetes_budget.db`, `geometry-run1/`, `passages.jsonl`)
- **Review Scope**: Clean-sweep static code analysis, mathematical/geometric verification, security boundary audit, cross-module contract check, and live test suite execution across all 11 core stacks (`stoma`, `LTHCM`, `dianoia`, `kleroterion`, `hestia`, `krisis`, `kybernetes`, `phylax`, `prosopon`, `paideia`, `dokime`), auxiliary modules (`schedion`, `symbolon`, `harness`), deployment scripts (`deploy/`, `scripts/`), IDE integration (`vscode/src/vs/workbench/contrib/void/`), and root CI/build configuration.
- **Verdict**: **NOT PRODUCTION-READY — BLOCKED BY 6 P0 AND 23 P1 DEFECTS.**
  Despite extensive prior hardening documented in [`METIS_COMPREHENSIVE_GAPS_AND_DEFECTS_REGISTER.md`](file:///home/akarta/Projects/Metis/docs/review/METIS_COMPREHENSIVE_GAPS_AND_DEFECTS_REGISTER.md), the repository at `23d54fe40a52e7dac3a6028b2b41838dab0cf477` has live security exposures (committed service tokens in git, unauthenticated `host_owner` WebRTC takeover + biometric approval bypass, unapproved `run_code_mode` AST sandbox escape, loopback auth bypass in Kybernetes, and arbitrary workspace secret disclosure via `/api/workbench/file`), a complete startup failure in `stoma` due to a stale `dianoia` SHA-256 digest in `vendored_provenance.json`, broken Phylax approval consumption across split `PhylaxAdapter` instances, and algebraic defects in LTHCM's higher-order sheaf cohomology and Born-rule retrieval scoring.

---

## 1. Repository & Module Coverage Inventory

| Module / Area | Language / Stack | Entry Points & Key Files | Review Status |
| :--- | :--- | :--- | :--- |
| **`stoma/`** | Python 3.12 (`src/stoma/`) | [`cli.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/cli.py), [`indexer.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/indexer.py), [`adapter.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/adapter.py), [`config.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/config.py), [`graph_builder.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/graph_builder.py), [`watcher.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/watcher.py), `plugins/*` | **Reviewed (100%)** |
| **`LTHCM/`** | Python 3.12 / PyTorch | [`pipeline.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/engine/pipeline.py), [`retriever.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/engine/retriever.py), `lthcm/core/*`, `lthcm/graph/*`, `recovery/*`, `lthcm/benchmarks/*` | **Reviewed (100%)** |
| **`dianoia/`** | Python 3.12 | [`walk.py`](file:///home/akarta/Projects/Metis/dianoia/dianoia/walk.py), [`verify.py`](file:///home/akarta/Projects/Metis/dianoia/dianoia/verify.py), [`cli.py`](file:///home/akarta/Projects/Metis/dianoia/dianoia/cli.py), `dianoia/anadrome/*` ([`replay.py`](file:///home/akarta/Projects/Metis/dianoia/dianoia/anadrome/replay.py), [`evaluator.py`](file:///home/akarta/Projects/Metis/dianoia/dianoia/anadrome/evaluator.py), [`study.py`](file:///home/akarta/Projects/Metis/dianoia/dianoia/anadrome/study.py)) | **Reviewed (100%)** |
| **`kleroterion/`** | Python 3.12 / NumPy | [`core.py`](file:///home/akarta/Projects/Metis/kleroterion/src/kleroterion/core.py), [`masked_softmax.py`](file:///home/akarta/Projects/Metis/kleroterion/src/kleroterion/masked_softmax.py), [`entropy_feedback.py`](file:///home/akarta/Projects/Metis/kleroterion/src/kleroterion/entropy_feedback.py), [`adapter.py`](file:///home/akarta/Projects/Metis/kleroterion/adapters/dianoia/adapter.py), [`metrics.py`](file:///home/akarta/Projects/Metis/kleroterion/eval/metrics.py) | **Reviewed (100%)** |
| **`hestia/`** | Python 3.12 / FastAPI | [`app.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/api/app.py), [`server/__init__.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/server/__init__.py), [`swap.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/server/swap.py), [`llamacpp.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/backend/llamacpp.py), [`vram.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/vram.py), [`device_lock.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/device_lock.py) | **Reviewed (100%)** |
| **`krisis/`** | Python 3.12 | [`selector.py`](file:///home/akarta/Projects/Metis/krisis/krisis/engine/selector.py), [`evaluator.py`](file:///home/akarta/Projects/Metis/krisis/krisis/gates/evaluator.py), [`decimal_calc.py`](file:///home/akarta/Projects/Metis/krisis/krisis/scoring/decimal_calc.py), [`exact.py`](file:///home/akarta/Projects/Metis/krisis/krisis/dedup/exact.py), [`cli.py`](file:///home/akarta/Projects/Metis/krisis/krisis/cli.py) | **Reviewed (100%)** |
| **`kybernetes/`** | Python 3.12 / FastAPI | [`server/app.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py), [`engine/pipeline.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/pipeline.py), [`terminal_agent.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/terminal_agent.py), `adapters/*`, `admin/*`, `budget/*`, `lochos/*`, `pinax/*`, `tools/*`, `execution/*` | **Reviewed (100%)** |
| **`phylax/`** | Python 3.12 | [`gateway.py`](file:///home/akarta/Projects/Metis/phylax/phylax/gateway/gateway.py), [`file_adapter.py`](file:///home/akarta/Projects/Metis/phylax/phylax/gateway/file_adapter.py), [`evaluator.py`](file:///home/akarta/Projects/Metis/phylax/phylax/engine/evaluator.py), [`canonical.py`](file:///home/akarta/Projects/Metis/phylax/phylax/engine/canonical.py), [`linter.py`](file:///home/akarta/Projects/Metis/phylax/phylax/linter/linter.py), `audit/*`, `approvals/*` | **Reviewed (100%)** |
| **`prosopon/`** | Python 3.12 / FastAPI / JS | [`server.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py), [`webrtc.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/transport/webrtc.py), [`webauthn.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/auth/webauthn.py), [`pairing.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/auth/pairing.py), [`access_store.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/storage/access_store.py), [`kybernetes_client.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/client/kybernetes_client.py), `web/static/js/*` | **Reviewed (100%)** |
| **`paideia/`** | Python 3.12 | [`cli.py`](file:///home/akarta/Projects/Metis/paideia/paideia/cli.py), [`router_loader.py`](file:///home/akarta/Projects/Metis/paideia/paideia/data/router_loader.py), [`walk_loader.py`](file:///home/akarta/Projects/Metis/paideia/paideia/data/walk_loader.py), [`runner.py`](file:///home/akarta/Projects/Metis/paideia/paideia/training/runner.py), [`hardware.py`](file:///home/akarta/Projects/Metis/paideia/paideia/training/hardware.py), `baselines/*`, `evaluation/*`, `export/*` | **Reviewed (100%)** |
| **`dokime/`** | Python 3.12 | [`cli.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/cli.py), [`runner.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/runner.py), [`compare.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/compare.py), [`adversarial.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/adversarial.py), [`client.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/client.py), [`manifest.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/manifest.py), `tasks/*` | **Reviewed (100%)** |
| **`schedion/`, `symbolon/`, `harness/`** | Python 3.12 | [`schedion/generator.py`](file:///home/akarta/Projects/Metis/schedion/schedion/generator.py), [`symbolon/codes.py`](file:///home/akarta/Projects/Metis/symbolon/symbolon/codes.py), [`harness/skill_learner.py`](file:///home/akarta/Projects/Metis/harness/skill_learner.py) | **Reviewed (100%)** |
| **`deploy/`, `scripts/`, Root CI** | Bash / Systemd / YAML / TOML | [`deploy/metis.env`](file:///home/akarta/Projects/Metis/deploy/metis.env), `deploy/systemd/*.service`, [`scripts/run_stack.sh`](file:///home/akarta/Projects/Metis/scripts/run_stack.sh), [`.gitlab-ci.yml`](file:///home/akarta/Projects/Metis/.gitlab-ci.yml), [`pyproject.toml`](file:///home/akarta/Projects/Metis/pyproject.toml), [`conftest.py`](file:///home/akarta/Projects/Metis/conftest.py) | **Reviewed (100%)** |
| **`vscode/` Submodule (Void Contrib)** | TypeScript / React | [`metisViews.ts`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/metisViews.ts), [`metisDashboardPane.ts`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/metisDashboardPane.ts), [`MetisSovereignSettings.tsx`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/react/src/void-settings-tsx/MetisSovereignSettings.tsx), [`MetisDashboard.tsx`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/react/src/void-settings-tsx/MetisDashboard.tsx) | **Reviewed (`contrib/void/` Metis integration; upstream VS Code core excluded as vendored upstream)** |

---

## 2. Prioritized Findings — P0 Blockers (Immediate Severe Exposure, RCE, or Startup Failure)

### [F-01] Live Service Tokens Committed to Git in `deploy/metis.env` with World-Readable Git Mode (`100644`)
- **Priority**: **P0** | **Confidence**: **Confirmed**
- **Affected Module**: `deploy/`, `.gitignore`
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`deploy/metis.env`](file:///home/akarta/Projects/Metis/deploy/metis.env#L12-L13)
  - [`prosopon/data/paired_tokens.json`](file:///home/akarta/Projects/Metis/prosopon/data/paired_tokens.json#L1)
  - [`.gitignore`](file:///home/akarta/Projects/Metis/.gitignore#L53-L61)
- **Expected Behavior**: Runtime secret files (`deploy/metis.env`, `prosopon/data/paired_tokens.json`) must be excluded from version control via `.gitignore` and provisioned locally with `0600` permissions; only `deploy/metis.env.example` should be tracked.
- **Observed Behavior**:
  - `git ls-tree HEAD deploy/metis.env` shows `100644 blob 3f91d5d1e1800443128dab69763545f50915c2f1 deploy/metis.env`.
  - Lines 12–13 of `deploy/metis.env` contain live 48-character production tokens (`HESTIA_SERVICE_TOKEN=<REDACTED>` and `KYBERNETES_SERVICE_TOKEN=<REDACTED>`).
  - Because Git only tracks executable vs. non-executable file modes (`100644` vs `100755`), any `git clone` or checkout writes `deploy/metis.env` as `0644` (world-readable) with active service tokens.
  - Additionally, `KYBERNETES_ADMIN_TOKEN` is missing from `deploy/metis.env`, and `prosopon/data/paired_tokens.json` is tracked in git (`100644`).
- **Trigger Condition**: Cloning, pulling, or inspecting the git repository.
- **Impact**: Complete compromise of inter-service bearer authentication across Hestia and Kybernetes.
- **Fix Direction**:
  1. Untrack both files: `git rm --cached deploy/metis.env prosopon/data/paired_tokens.json`.
  2. Add `deploy/metis.env` and `prosopon/data/` to [`.gitignore`](file:///home/akarta/Projects/Metis/.gitignore).
  3. Immediately rotate `HESTIA_SERVICE_TOKEN` and `KYBERNETES_SERVICE_TOKEN` (and generate `KYBERNETES_ADMIN_TOKEN`) on the host with `chmod 0600 deploy/metis.env`.
- **Regression Check**: Add a test in `tests/test_stack_smoke.py` asserting `"deploy/metis.env"` is not in `subprocess.check_output(["git", "ls-files"]).decode().splitlines()`.

---

### [F-02] Arbitrary Code Execution & Full Sandbox Escape in `SafeCodeMode` (`run_code_mode`) Without Phylax Approval
- **Priority**: **P0** | **Confidence**: **Confirmed**
- **Affected Module**: `kybernetes` (`tools/code_mode.py`, `tools/registry.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`kybernetes/kybernetes/tools/code_mode.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/code_mode.py#L96-L157)
  - [`kybernetes/kybernetes/tools/code_mode.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/code_mode.py#L260-L305)
  - [`kybernetes/kybernetes/tools/registry.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/registry.py#L114-L121) and [`registry.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/registry.py#L287)
- **Expected Behavior**: Any tool executing arbitrary Python code must require Phylax approval (`ToolSideEffectClass.SYSTEM_EXEC`) and run inside a true OS-level sandbox (`BubblewrapBackend`) with scrubbed environment variables—never relying on an in-process Python AST blocklist with a `fork`ed parent process.
- **Observed Behavior**:
  1. **Unapproved Execution**: `SafeCodeMode.side_effect_class` is marked `ToolSideEffectClass.READ_ONLY` ([`code_mode.py:96`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/code_mode.py#L96)). Because [`ToolRegistry.execute`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/registry.py#L287) only requires approval when `spec.side_effect_class != ToolSideEffectClass.READ_ONLY`, `run_code_mode` executes with zero Phylax approval gating.
  2. **AST Blocklist Escape #1 (Generator Frame Traversal — Zero Dunders)**: [`_validate_ast_safety`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/code_mode.py#L118-L157) only blocks `ast.Attribute` names starting/ending with `__`. Python generator objects expose `gi_frame`, `f_back`, `f_globals`, and `f_builtins` (none of which contain dunders):
     ```python
     g = (x for x in [1])
     os_mod = g.gi_frame.f_back.f_globals["os"]
     ```
     Because `_worker` ([`code_mode.py:260`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/code_mode.py#L260)) is in the same module that imports `os`, `sys`, and `multiprocessing` at lines 17–21, `g.gi_frame.f_back.f_globals["os"]` gives immediate, unrestricted access to the `os` module (`os.system(...)`, `os.environ`).
  3. **AST Blocklist Escape #2 (Concatenated Format Strings)**: `_FORMAT_DUNDER_RE` ([`code_mode.py:152`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/code_mode.py#L152)) only inspects `ast.Constant` nodes. Writing `("{0." + "__globals__}").format(...)` creates an `ast.BinOp` node that completely bypasses `_FORMAT_DUNDER_RE`.
  4. **Forked Process Memory & Secret Inheritance**: [`code_mode.py:303`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/code_mode.py#L303) uses `multiprocessing.get_context("fork")`, cloning the entire parent FastAPI process memory and `os.environ` (including `KYBERNETES_ADMIN_TOKEN`, `KYBERNETES_SERVICE_TOKEN`, `HESTIA_SERVICE_TOKEN`) into the worker.
- **Trigger Condition**: Any prompt injection or agent turn invoking `run_code_mode`.
- **Impact**: Unapproved arbitrary OS command execution, filesystem mutation, and theft of all in-memory/environment service tokens.
- **Fix Direction**:
  1. Change `SafeCodeMode.side_effect_class` to `ToolSideEffectClass.SYSTEM_EXEC` in [`code_mode.py:96`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/code_mode.py#L96) and [`registry.py:118`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/registry.py#L118).
  2. Block `gi_frame`, `gi_code`, `cr_frame`, `cr_code`, `ag_frame`, `ag_code`, `f_back`, `f_globals`, `f_locals`, `f_builtins`, `f_code`, `format`, and `format_map` in `_validate_ast_safety`.
  3. Execute `run_code_mode` inside `BubblewrapBackend` (with `--unshare-net`, `--ro-bind / /`, and a scrubbed environment) rather than `multiprocessing.get_context("fork")`.
- **Regression Check**: Add unit tests in [`kybernetes/tests/test_code_mode.py`](file:///home/akarta/Projects/Metis/kybernetes/tests/test_code_mode.py) verifying that generator frame traversal (`g.gi_frame.f_back`) and binary-op format strings (`("{0." + "__class__}").format(1)`) are rejected by `_validate_ast_safety`, and that `ToolRegistry.execute("run_code_mode", ...)` requires an approval token.

---

### [F-03] Unauthenticated WebRTC `host_owner` Account Takeover & Biometric Phylax Approval Bypass in Prosopon
- **Priority**: **P0** | **Confidence**: **Confirmed**
- **Affected Module**: `prosopon` (`server.py`, `transport/webrtc.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`prosopon/prosopon/server.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L755-L785)
  - [`prosopon/prosopon/server.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L806-L818)
  - [`prosopon/prosopon/transport/webrtc.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/transport/webrtc.py#L323-L393)
- **Expected Behavior**:
  1. `POST /api/webrtc/offer` and `POST /api/webrtc/ice-candidate` must require a valid authenticated session (`Depends(require_auth)`) and bind the WebRTC session to the caller's authenticated `device_id`.
  2. `WebRTCGateway._handle_phylax_approval` must reject approval messages that omit a cryptographic WebAuthn passkey `credential` when biometric verification is required (or accurately report `biometric_verified: False` and enforce `approvals:resolve` / `approvals:resolve_high` scope checks before resolving).
- **Observed Behavior**:
  1. **Unauthenticated `host_owner` Session Creation ([`server.py:760-785`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L760-L785))**:
     ```python
     target_dev_id = req.device_id
     owner_tok = request.headers.get("x-host-owner-token")
     if owner_tok and host_owner_mgr.validate_owner_token(owner_tok):
         target_dev_id = "host_owner"
     else:
         auth_header = request.headers.get("Authorization", "")
         if auth_header.startswith("Bearer "):
             ...
     dev = access_store.get_device(target_dev_id)
     if not dev or dev.get("revoked"):
         raise HTTPException(status_code=403, ...)
     ```
     When an unauthenticated client sends `POST /api/webrtc/offer` with **no** `x-host-owner-token` and **no** `Authorization` header, `target_dev_id` defaults to `req.device_id` (`"host_owner"`, defined at [`server.py:756`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L756)). Because `"host_owner"` is pre-seeded in `access.db` on startup ([`access_store.py:130`](file:///home/akarta/Projects/Metis/prosopon/prosopon/storage/access_store.py#L130)), `access_store.get_device("host_owner")` succeeds and creates a privileged `host_owner` WebRTC session without any authentication!
  2. **Biometric Verification Skipped When `credential` Is Omitted ([`webrtc.py:338-387`](file:///home/akarta/Projects/Metis/prosopon/prosopon/transport/webrtc.py#L338-L387))**:
     In `_handle_phylax_approval`, WebAuthn verification is gated by `if credential and approval_id and action_digest:`. If the client sends `{"type": "phylax_approval_response", "approval_id": "...", "decision": "approve"}` without `"credential"`, the WebAuthn check is completely skipped, `self.resolve_phylax_approval(...)` executes at line 367, and line 386 responds with `"biometric_verified": True`!
  3. **Unauthenticated ICE Candidate Endpoint ([`server.py:806-818`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L806-L818))**: `POST /api/webrtc/ice-candidate` has no authentication dependency at all.
- **Trigger Condition**: Sending `POST /api/webrtc/offer` without authentication headers, followed by a `phylax_approval_response` DataChannel message omitting `credential`.
- **Impact**: Unauthenticated remote takeover of the `host_owner` WebRTC control plane and unauthenticated approval of T1/T2/T3 Phylax actions.
- **Fix Direction**:
  1. Add `auth: dict = Depends(require_auth)` to `webrtc_offer` ([`server.py:760`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L760)) and `webrtc_ice_candidate` ([`server.py:807`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L807)), and force `target_dev_id = auth["device_id"]`.
  2. Require a valid WebAuthn assertion in `WebRTCGateway._handle_phylax_approval` for high-tier approvals (or fail closed when `credential` is absent).
- **Regression Check**: Add a test in [`prosopon/tests/test_pyle_webrtc.py`](file:///home/akarta/Projects/Metis/prosopon/tests/test_pyle_webrtc.py) asserting `POST /api/webrtc/offer` with no headers returns `HTTP 401`, and sending `phylax_approval_response` without `credential` for a T3 approval is rejected.

---

### [F-04] Loopback Exemption in Kybernetes `require_service_auth` + Unauthenticated Lochos & Pinax Endpoints
- **Priority**: **P0** | **Confidence**: **Confirmed**
- **Affected Module**: `kybernetes` (`server/app.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`kybernetes/kybernetes/server/app.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py#L148-L153)
  - [`kybernetes/kybernetes/server/app.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py#L339-L352), [`L397-L427`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py#L397-L427), [`L443-L482`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py#L443-L482), [`L534-L540`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py#L534-L540)
- **Expected Behavior**: Per [`docs/MSS.md` Section 3.5](file:///home/akarta/Projects/Metis/docs/MSS.md#L127) (*"Zero-Loopback Security Model: Loopback (`127.0.0.1` / `::1`) exemptions are completely eliminated across Prosopon and Kybernetes"*), `require_service_auth` must validate `KYBERNETES_SERVICE_TOKEN` on all requests, and all `/v1/lochos/*` and `/v1/pinax/*` endpoints must enforce authentication.
- **Observed Behavior**:
  1. [`require_service_auth`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py#L151-L152) still contains:
     ```python
     client_host = request.client.host if request.client else ""
     if client_host in ("127.0.0.1", "::1", "localhost"):
         return True
     ```
     Because Kybernetes listens on `127.0.0.1:8081`, every local process or reverse-proxied request has `client_host == "127.0.0.1"` and bypasses `KYBERNETES_SERVICE_TOKEN` completely.
  2. Ten endpoints in [`kybernetes/server/app.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py) omit `Depends(require_service_auth)` altogether:
     - `GET /v1/pinax/tools` (line 339), `GET /v1/pinax/tools/{tool_name}` (line 397), `GET /v1/pinax/tools/{tool_name}/docs` (line 406), `POST /v1/pinax/evaluate` (line 422)
     - `GET /v1/lochos/squad` (line 443), `GET /v1/lochos/queue` (line 449), `GET /v1/lochos/seats/{seat_id}/brief` (line 456), `GET /v1/lochos/transfer/brief` (line 465), `GET /v1/lochos/attention` (line 480), `WS /v1/lochos/ws` (line 534)
- **Trigger Condition**: Any unauthenticated request from `127.0.0.1` to Kybernetes, or any remote request to the 10 unprotected Lochos/Pinax endpoints when Kybernetes is bound to a network interface.
- **Impact**: Complete bypass of `KYBERNETES_SERVICE_TOKEN` on loopback and unauthenticated exposure of Lochos/Pinax APIs and WebSockets.
- **Fix Direction**: Remove lines 151–152 in [`kybernetes/server/app.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py#L151-L152) when `expected` is set, add `dependencies=[Depends(require_service_auth)]` to all Pinax and Lochos HTTP routes, and verify bearer tokens in `lochos_websocket` (line 534) before `await websocket.accept()`.
- **Regression Check**: Test in [`kybernetes/tests/test_server_api.py`](file:///home/akarta/Projects/Metis/kybernetes/tests/test_server_api.py) that when `KYBERNETES_SERVICE_TOKEN` is set, an unauthenticated request from `TestClient` (`127.0.0.1`) to `/v1/chat/turn` and `/v1/lochos/squad` returns `HTTP 401`.

---

### [F-05] Arbitrary Workspace Secret File Disclosure via `/api/workbench/file` Enabling Privilege Escalation to `host_owner`
- **Priority**: **P0** | **Confidence**: **Confirmed**
- **Affected Module**: `prosopon` (`server.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**: [`prosopon/prosopon/server.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L1413-L1439)
- **Expected Behavior**: `/api/workbench/file` must restrict access to authorized roles/scopes and block reading credential/secret files (`.host_owner_token`, `deploy/metis.env`, `paired_tokens.json`, `pairing_pin.json`, `bootstrap_tickets.json`, `access.db`, `.phylax/`, `.admin_snapshots/`, `*.key`).
- **Observed Behavior**: `GET /api/workbench/file?path=...` only checks `target.is_relative_to(workspace_root)` and allows any `require_auth` caller (including a low-privilege `"viewer"` or `"companion"` device) to read any file inside the repository root:
  - `GET /api/workbench/file?path=data/.host_owner_token` (or `prosopon/data/.host_owner_token`) returns the raw `host_owner` secret token.
  - `GET /api/workbench/file?path=deploy/metis.env` returns `HESTIA_SERVICE_TOKEN` and `KYBERNETES_SERVICE_TOKEN`.
- **Trigger Condition**: Calling `GET /api/workbench/file?path=data/.host_owner_token` or `deploy/metis.env` with any low-privilege session token.
- **Impact**: Trivial privilege escalation from `"viewer"` / `"companion"` to full `"host_owner"` and disclosure of all service tokens.
- **Fix Direction**: Add a sensitive-path denylist (`*.env*`, `*token*`, `*.key`, `*.pem`, `*.db`, `*.sqlite`, `data/`, `.phylax/`, `.metis/`, `.admin_snapshots/`, `.git/`) in `get_workbench_file` and `get_workbench_tree`, and require an explicit `workbench:read` (or `owner`/`operator`) scope.
- **Regression Check**: Test in [`prosopon/tests/test_server_api.py`](file:///home/akarta/Projects/Metis/prosopon/tests/test_server_api.py) asserting `GET /api/workbench/file?path=deploy/metis.env` and `path=data/.host_owner_token` return `HTTP 403`.

---

### [F-06] Stale `dianoia` SHA-256 Digest in Both `vendored_provenance.json` Manifests Breaks All `stoma` Imports & Tests
- **Priority**: **P0** | **Confidence**: **Confirmed**
- **Affected Module**: `stoma` (`config.py`, `vendored_provenance.json`, `src/stoma/vendored_provenance.json`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`stoma/src/stoma/__init__.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/__init__.py#L61)
  - [`stoma/src/stoma/config.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/config.py#L288-L341)
  - [`stoma/vendored_provenance.json`](file:///home/akarta/Projects/Metis/stoma/vendored_provenance.json#L20-L27)
  - [`stoma/src/stoma/vendored_provenance.json`](file:///home/akarta/Projects/Metis/stoma/src/stoma/vendored_provenance.json#L20-L27)
- **Expected Behavior**: `stoma.config.verify_runtime_dependencies()` (called automatically on `import stoma` at [`stoma/src/stoma/__init__.py:61`](file:///home/akarta/Projects/Metis/stoma/src/stoma/__init__.py#L61)) must succeed against the current monorepo `dianoia` package, and both `vendored_provenance.json` files must be synchronized.
- **Observed Behavior**:
  - Commit `51ce25f` modified `dianoia/dianoia/prompts.py` and `dianoia/dianoia/walk.py`, changing `compute_component_digest(Path("dianoia"), ["dianoia", "pyproject.toml"])` to `948f35a52ca090acb715aeecfe17ff6cac061f541a9b124cc5b73836db44df39`.
  - However, [`stoma/vendored_provenance.json:25`](file:///home/akarta/Projects/Metis/stoma/vendored_provenance.json#L25) still has `84ff5168b7c175f6cd1c0916811ab55de6e40d60395ec1cbf86d6bc550a4a62a`, and [`stoma/src/stoma/vendored_provenance.json:25`](file:///home/akarta/Projects/Metis/stoma/src/stoma/vendored_provenance.json#L25) has an even older hash `9f63fb9c17898d65aab3fe1dfa6908119fea41522ffff263054c7edb9d59e4ba`.
  - Reproduction output from `./stoma/.venv/bin/pytest stoma/tests --collect-only -q`:
    ```text
    E stoma.config.DependencyOriginError: Runtime import violation: Content digest mismatch for vendored dianoia:
      expected '84ff5168b7c175f6cd1c0916811ab55de6e40d60395ec1cbf86d6bc550a4a62a',
      got '948f35a52ca090acb715aeecfe17ff6cac061f541a9b124cc5b73836db44df39'.
    ```
- **Trigger Condition**: Any `import stoma` anywhere in the monorepo (`stoma` CLI, `stoma watch` daemon, `stoma/tests`, or `tests/test_round2_fixes.py`).
- **Impact**: 100% of `stoma` functionality and all 113 `stoma` unit tests + 4 root regression tests fail at import time.
- **Fix Direction**: Update `content_sha256` for `dianoia` to `948f35a52ca090acb715aeecfe17ff6cac061f541a9b124cc5b73836db44df39` (and `monorepo_revision` to current HEAD) in both [`stoma/vendored_provenance.json`](file:///home/akarta/Projects/Metis/stoma/vendored_provenance.json) and [`stoma/src/stoma/vendored_provenance.json`](file:///home/akarta/Projects/Metis/stoma/src/stoma/vendored_provenance.json), and deduplicate the two manifest files so one is the canonical source.
- **Regression Check**: Run `./.venv/bin/pytest tests/test_round2_fixes.py -k defect7` and `./stoma/.venv/bin/pytest stoma/tests -q`.

---

## 3. Prioritized Findings — P1 Major Functional, Security & Mathematical Defects

### [F-07] Hard-Denied Phylax Actions Execute Anyway in `ExecutionPipeline`, Split `PhylaxAdapter` Instances Break Approval Consumption, and `PlanRouter` Emits Unregistered `"guarded_action"` Tool
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `kybernetes` (`engine/pipeline.py`, `server/app.py`, `cli.py`, `tools/registry.py`, `routing/router.py`, `adapters/phylax.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`kybernetes/kybernetes/engine/pipeline.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/pipeline.py#L363-L372) and [`L528`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/pipeline.py#L528)
  - [`kybernetes/kybernetes/server/app.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py#L79-L83)
  - [`kybernetes/kybernetes/cli.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/cli.py#L131-L135)
  - [`kybernetes/kybernetes/tools/registry.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/registry.py#L33) and [`L298-L315`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/registry.py#L298-L315)
  - [`kybernetes/kybernetes/routing/router.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/routing/router.py#L93-L95)
  - [`kybernetes/kybernetes/adapters/phylax.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/adapters/phylax.py#L336-L345)
- **Expected Behavior**:
  1. When `phylax.is_action_permitted()` returns `(False, None)` (a hard policy `DENY`), `ExecutionPipeline` must abort the turn immediately.
  2. `ExecutionPipeline` and `ToolRegistry` must share a single `PhylaxAdapter` instance so approvals resolved on the pipeline are visible to `ToolRegistry.execute`.
  3. `PlanRouter` must route mutation intents to registered tools (`write_file`, `edit_file_block`, `sandboxed_bash`).
- **Observed Behavior**:
  1. **Hard-Deny Fallthrough**: In [`pipeline.py:363-372`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/pipeline.py#L363-L372), when `not permitted` and `app_req is None`, there is no `else:` branch! Execution falls straight through to `self.tool_registry.execute(...)` at line 528.
  2. **Split `PhylaxAdapter` Instances**: Both [`server/app.py:79,83`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py#L79-L83) and [`cli.py:131,135`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/cli.py#L131-L135) construct `phylax_adapter = PhylaxAdapter()` and `tool_registry = ToolRegistry()` without passing `phylax_adapter` into `ToolRegistry(phylax_adapter=phylax_adapter)`. `ToolRegistry.__init__` creates a second `PhylaxAdapter` with an empty `_approvals` dict, so `tool_registry.phylax_adapter.consume_approval(approval_id, ...)` always fails with `"Approval token '...' not found in active store"`.
  3. **Unregistered `"guarded_action"` Tool**: [`PlanRouter.build_plan`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/routing/router.py#L94) emits `tool_name = "guarded_action"`, which does not exist in `ToolRegistry`.
  4. **Unchecked Target/Arguments in Simulated `consume_approval`**: [`adapters/phylax.py:336-345`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/adapters/phylax.py#L336-L345) ignores `tool_id` and `arguments` when in simulated fallback mode.
- **Trigger Condition**: Any turn triggering a Phylax `DENY` or `ASK` decision in `ExecutionPipeline`.
- **Impact**: Hard-denied `READ_ONLY` tools execute anyway, while every approved mutating tool in the server/CLI fails at execution time.
- **Fix Direction**: Add the missing `else:` abort branch in `pipeline.py:369`, pass `phylax_adapter=phylax_adapter` into `ToolRegistry(...)` in `app.py` and `cli.py`, replace `"guarded_action"` in `PlanRouter` with registered tool IDs, and verify `tool_id` and `arguments` in `PhylaxAdapter.consume_approval`.
- **Regression Check**: Integration test in `kybernetes/tests/test_server_api.py` submitting a mutating turn, approving the `approval_request` via `/v1/approvals/{id}/resolve`, and verifying the tool executes with `TurnOutcomeStatus.VERIFIED` / `COMPLETED`.

---

### [F-08] `KrisisAdapter` Ignores `verifier_client` & Flags Any Cited Sentence Containing Common Negation Words (`"not"`, `"no"`, `"never"`) as `CONFLICTED`
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `kybernetes` (`adapters/krisis.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**: [`kybernetes/kybernetes/adapters/krisis.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/adapters/krisis.py#L28-L39) and [`L125-L170`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/adapters/krisis.py#L125-L170)
- **Expected Behavior**: `KrisisAdapter` should evaluate candidates via `krisis.engine.selector.KrisisSelector` (or NLI/verifier contradiction checks) and only return `TurnOutcomeStatus.CONFLICTED` when evidence contradicts a claim.
- **Observed Behavior**:
  - `self.verifier_client` is stored on `self` ([`krisis.py:29`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/adapters/krisis.py#L29)) but never used.
  - Lines 137–142 flag `contradiction_found = True` whenever any cited sentence contains `"not"`, `"never"`, `"no"`, `"false"`, `"incorrect"`, `"untrue"`, `"cannot"`, or `"failed"`:
    ```python
    negation_words = {"not", "never", "no", "false", "incorrect", "untrue", "cannot", "failed"}
    s_clean = re.sub(r"[^\w\s]", "", sentence.lower())
    if any(neg in s_clean.split() for neg in negation_words):
        contradiction_found = True
    ```
  - Consequently, an accurate cited sentence such as `"The service does not expose public ports [doc-1]"` causes `verify_response` to return `TurnOutcomeStatus.CONFLICTED` and overwrites the entire answer with `"Evidence conflict detected..."` ([`pipeline.py:725`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/pipeline.py#L725)).
- **Trigger Condition**: Any RAG response containing a cited sentence with a common English negation word (`"not"`, `"no"`, `"cannot"`, etc.).
- **Impact**: Severe false-positive conflict rejections on ordinary factual answers.
- **Fix Direction**: Delegate `KrisisAdapter.verify_response` to `KrisisSelector` and check whether the claim negates the cited passage rather than checking whether the claim contains the word `"not"` in isolation.
- **Regression Check**: Unit test in `kybernetes/tests/test_adapters.py` verifying that a passage `"Port 8080 is bound to localhost and does not accept external connections"` and answer `"Port 8080 does not accept external connections [doc-1]"` returns `TurnOutcomeStatus.VERIFIED`.

---

### [F-09] `BudgetStorage` `phase_plans` Primary Key Collision Overwrites Phase Rows Across All Turns, Plus Unreachable Offline Budget Ingestion & Active Turn Leak
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `kybernetes` (`budget/storage.py`, `engine/pipeline.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`kybernetes/kybernetes/budget/storage.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/budget/storage.py#L66-L76) and [`L204-L219`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/budget/storage.py#L204-L219)
  - [`kybernetes/kybernetes/engine/pipeline.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/pipeline.py#L231), [`L259-L285`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/pipeline.py#L259-L285), [`L673-L694`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/pipeline.py#L673-L694)
- **Expected Behavior**:
  1. `phase_plans` rows in SQLite must be keyed by `(task_id, phase_id)` so creating a new turn does not overwrite phase rows from previous turns.
  2. Offline synthesis budget ingestion ([`pipeline.py:679-693`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/pipeline.py#L679-L693)) must execute on non-cancelled turns.
  3. `self.sessions.unregister_active_turn(turn_id)` must run in a `finally:` block across all exit paths.
- **Observed Behavior**:
  1. `phase_plans` uses `phase_id TEXT PRIMARY KEY` ([`storage.py:68`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/budget/storage.py#L68)) with `INSERT OR REPLACE` ([`storage.py:204`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/budget/storage.py#L204)), while `ExecutionPipeline` ([`pipeline.py:231`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/pipeline.py#L231)) uses static IDs `"phase-authorize"`, `"phase-retrieve"`, etc. Every new turn steals the phase rows of all previous turns!
  2. In [`pipeline.py:673-693`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/pipeline.py#L673-L693), `if self.budget_engine:` is indented inside `if self.sessions.is_turn_cancelled(turn_id):` immediately after `return`—making it unreachable dead code.
  3. Early returns at lines 269 and 285 and SSE disconnects (`GeneratorExit`) skip `self.sessions.unregister_active_turn(turn_id)` (line 759), leaking active turn entries forever.
- **Trigger Condition**: Running two consecutive turns with `BudgetEngine` enabled, running an offline turn, or disconnecting an SSE client mid-turn.
- **Impact**: Corrupted historical task plans in `kybernetes_budget.db`, unrecorded offline usage, and leaked turn state/hung subscribers.
- **Fix Direction**: Change `phase_plans` primary key to `PRIMARY KEY (task_id, phase_id)`, dedent lines 679–693 in `pipeline.py`, and wrap `execute_turn` in `try ... finally: self.sessions.unregister_active_turn(turn_id)`.
- **Regression Check**: Save two `TaskPlan`s with identical `phase_id`s in `BudgetDatabase` and assert `get_task_plan(task_1).phases` is non-empty.

---

### [F-10] Arbitrary Host File Read via Untrusted `root_path` / `rel_path` in `LTHCMAdapter._load_real_index`
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `kybernetes` (`adapters/lthcm.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**: [`kybernetes/kybernetes/adapters/lthcm.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/adapters/lthcm.py#L147-L159) and [`L172-L184`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/adapters/lthcm.py#L172-L184)
- **Expected Behavior**: Passage hydration from `passages.jsonl` must verify that `src_file.resolve()` stays within the indexed corpus root directory.
- **Observed Behavior**: `root_p = Path(pdata.get("root_path", root_str))` and `src_file = root_p / pdata["rel_path"]` read arbitrary host files without checking `src_file.resolve().is_relative_to(Path(root_str).resolve())`.
- **Trigger Condition**: Loading a `.lthcm/passages.jsonl` entry containing `"root_path": "/"` or `"rel_path": "../../etc/passwd"`.
- **Impact**: Arbitrary host file disclosure into RAG passages and LLM prompts.
- **Fix Direction**: Enforce `src_file = (Path(root_str).resolve() / pdata["rel_path"]).resolve()` and verify `src_file.is_relative_to(Path(root_str).resolve())`.
- **Regression Check**: Unit test in `kybernetes/tests/test_lthcm_adapter.py` asserting that a `passages.jsonl` entry with `"rel_path": "../secret.txt"` is rejected and not read into `self._corpus`.

---

### [F-11] `TerminalCodingAgent` Regex Action Parser Truncates Code Containing Braces `{}` / Backticks & Invalid FSM Transitions Crash `run_loop`
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `kybernetes` (`engine/terminal_agent.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**: [`kybernetes/kybernetes/engine/terminal_agent.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/terminal_agent.py#L164-L178), [`L294`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/terminal_agent.py#L294), [`L336-L348`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/terminal_agent.py#L336-L348), [`L421-L438`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/engine/terminal_agent.py#L421-L438)
- **Expected Behavior**: `_extract_action` must parse JSON tool payloads whose string values contain `{`, `}`, or ```` ``` ````, and `run_loop` must only execute legal `AgentStateMachine` state transitions.
- **Observed Behavior**:
  1. `_extract_action` uses non-greedy `r"```(?:json)?\s*(\{.*?\})\s*```"` and `r"(\{[^{}]*\"tool\"[^{}]*\})"`. Any `write_file` or `edit_file_block` payload containing `{}` or fenced code blocks inside string values fails to extract!
  2. Rejecting an interactive approval at lines 345–348 leaves `state_machine` in `AWAITING_APPROVAL`; if the next turn emits `{"tool": "done"}`, `transition(COMPLETED)` (line 294) raises `InvalidStateTransitionError`.
  3. After running `verifier` (lines 421–438), `state_machine` remains in `VERIFYING`; if the next turn requires approval (line 336), `transition(AWAITING_APPROVAL)` raises `InvalidStateTransitionError`.
- **Trigger Condition**: Using `kybernetes code` to write C/JS/Python dicts/f-strings containing `{}` or rejecting an approval prompt.
- **Impact**: `kybernetes code` fails to parse real-world code edits and crashes with `InvalidStateTransitionError`.
- **Fix Direction**: Parse JSON objects using `json.JSONDecoder().raw_decode()` from each `{` position, and transition `self.state_machine` back to `AgentSemanticState.PLANNING` after approval rejection or verification completion.
- **Regression Check**: Unit test in `kybernetes/tests/test_terminal_agent.py` calling `_extract_action` on a `write_file` JSON payload containing nested `{ "a": 1 }` and ```python blocks.

---

### [F-12] `PinaxCatalogue.evaluate_invocation` Risk-Override Bypass via Leading Flags, `--flag=value` Syntax, Early `break`, and Unsafe `T0` Seeds (`tee`, `awk`, `env`, `black`)
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `kybernetes` (`pinax/catalogue.py`, `pinax/seed.py`), `phylax` (`linter/linter.py`, `gateway/gateway.py`, `gateway/file_adapter.py`, `engine/evaluator.py`, `engine/canonical.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`kybernetes/kybernetes/pinax/catalogue.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/pinax/catalogue.py#L296-L315)
  - [`kybernetes/kybernetes/pinax/seed.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/pinax/seed.py#L487), [`L571`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/pinax/seed.py#L571), [`L728`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/pinax/seed.py#L728), [`L768`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/pinax/seed.py#L768)
  - [`phylax/phylax/gateway/gateway.py`](file:///home/akarta/Projects/Metis/phylax/phylax/gateway/gateway.py#L169-L216)
  - [`phylax/phylax/gateway/file_adapter.py`](file:///home/akarta/Projects/Metis/phylax/phylax/gateway/file_adapter.py#L113-L117) vs [`phylax/phylax/engine/evaluator.py`](file:///home/akarta/Projects/Metis/phylax/phylax/engine/evaluator.py#L180-L213)
  - [`phylax/phylax/engine/canonical.py`](file:///home/akarta/Projects/Metis/phylax/phylax/engine/canonical.py#L66-L70)
  - [`phylax/phylax/linter/linter.py`](file:///home/akarta/Projects/Metis/phylax/phylax/linter/linter.py#L164-L255)
- **Expected Behavior**:
  1. Pinax risk overrides must catch subcommands preceded by flags (`git -C /repo push`), `--flag=value` tokens (`curl --upload-file=/etc/passwd`), evaluate all overrides to take the maximum `RiskTier`, and classify mutating/exec tools (`tee`, `awk`, `env`, `black`) as `>= T1`.
  2. Phylax `GuardedGateway.authorize_and_execute(..., approval_id=...)` must call `verify_caller_authorization(subject)` and canonicalize `target` before matching `record.target`.
  3. Phylax `PolicyEvaluator._check_condition` must enforce `max_bytes` against `content_bytes` passed by `GuardedFileAdapter.write_text` and must enforce `environment_tags`.
  4. `PolicyLinter` must sort rules by `(priority, effect_rank)` and treat `Verb.ALL` / `ResourceClass.ALL` as wildcards.
- **Observed Behavior**:
  1. `PinaxCatalogue.evaluate_invocation` only checks `args[0] == ovr.pattern` for subcommands and `ovr.pattern in args` (exact match) for flags, breaking on the first match. `git -C . push` and `curl --data=secret` remain `T0`/`T2` (`requires_approval: False`).
  2. `GuardedGateway.authorize_and_execute` ([`gateway.py:169-214`](file:///home/akarta/Projects/Metis/phylax/phylax/gateway/gateway.py#L169-L214)) skips `verify_caller_authorization(subject)` when `approval_id` is passed and passes uncanonicalized `target` to `consume_for_execution`, causing all pre-approved relative file paths to fail with `"Target mismatch"`.
  3. `GuardedFileAdapter.write_text` passes `content_bytes` while `PolicyEvaluator._check_condition` checks `if "content" in args:`, ignoring `max_bytes`; `cond.environment_tags` is never checked at all (fail-open).
  4. `matches_path_pattern` ([`canonical.py:66,70`](file:///home/akarta/Projects/Metis/phylax/phylax/engine/canonical.py#L66-L70)) matches basenames at arbitrary depth and calls `root.resolve()` (CWD-relative) instead of `canon_root` (`base_dir`-relative).
  5. `PolicyLinter._check_semantic_rules` ([`linter.py:164-255`](file:///home/akarta/Projects/Metis/phylax/phylax/linter/linter.py#L164-L255)) checks raw YAML order instead of `priority` and never matches `ResourceClass.ALL` in `_rule_subsumes`.
- **Trigger Condition**: Evaluating CLI tool risks via Pinax, executing pre-approved relative file paths in Phylax, enforcing `max_bytes` / `environment_tags`, or linting policies.
- **Impact**: High-risk shell invocations bypass escalation; Phylax relative-path approvals fail while `max_bytes` and `environment_tags` fail open.
- **Fix Direction**: Fix subcommand/flag parsing and tier max-reduction in `PinaxCatalogue.evaluate_invocation`; canonicalize `target` and verify caller identity at the start of `GuardedGateway.authorize_and_execute`; check `content_bytes` and `environment_tags` in `PolicyEvaluator._check_condition`; use `canon_root` in `canonical.py:70`; and sort by priority + support wildcards in `PolicyLinter`.
- **Regression Check**: Add unit tests in `kybernetes/tests/test_pinax.py` (`git -C . push` -> `T3`, `curl --data=x` -> `T3`) and `phylax/tests/test_gateway.py` (pre-approved relative path write succeeds; `max_bytes` limit blocks oversized `GuardedFileAdapter.write_text`).

---

### [F-13] `LochosSquadManager` Crashes on Second Dispatch to Active Seat & Attention Endpoints Desynchronize Seat State
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `kybernetes` (`lochos/squad.py`, `lochos/queue.py`, `server/app.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`kybernetes/kybernetes/lochos/squad.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/lochos/squad.py#L220-L226) and [`L259-L263`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/lochos/squad.py#L259-L263)
  - [`kybernetes/kybernetes/lochos/queue.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/lochos/queue.py#L311-L372)
  - [`kybernetes/kybernetes/server/app.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/server/app.py#L490-L519)
- **Expected Behavior**: Dispatching or transferring a task to a seat that is already busy (`current_state != IDLE`) must enqueue the task without crashing or clobbering `seat.current_task`, and `/v1/lochos/attention/*` must validate `task_id` and update `seat.state_machine`.
- **Observed Behavior**: `dispatch_work` and `transfer_baton` unconditionally call `seat.state_machine.transition(AgentSemanticState.PLANNING)`. If the seat is already in `PLANNING`, `AgentStateMachine.transition` raises `InvalidStateTransitionError` (`PLANNING -> PLANNING`) after already committing `status='in_progress'` to SQLite! Meanwhile, `/v1/lochos/attention/raise` and `/resolve` bypass `LochosSquadManager` and crash with `AttributeError: 'NoneType' object has no attribute 'to_dict'` if `task_id` is unknown.
- **Trigger Condition**: Dispatching two tasks in a row to `seat-coder` or raising attention on a non-existent `task_id`.
- **Impact**: HTTP 500 crashes and corrupted squad seat state.
- **Fix Direction**: Guard auto-claim in `dispatch_work` and `transfer_baton` with `if seat and seat.current_state == AgentSemanticState.IDLE:`, and route attention raise/resolve through `LochosSquadManager` with `HTTPException(404)` when `item is None`.
- **Regression Check**: Test dispatching two tasks sequentially to `"seat-coder"` in `kybernetes/tests/test_lochos_squad.py` and asserting the second remains `queued` while the first stays `in_progress`.

---

### [F-14] Missing Scope Enforcement Across Prosopon Endpoints, Incomplete Panic Lockdown, Broken `/v1/v1/pinax` Proxy URLs, WebSocket `AttributeError`, and DOM XSS in Frontend JS
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `prosopon` (`server.py`, `auth/pairing.py`, `client/kybernetes_client.py`, `web/static/js/*`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`prosopon/prosopon/server.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L243-L273), [`L503-L530`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L503-L530), [`L854-L1065`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L854-L1065), [`L1487-L1690`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L1487-L1690)
  - [`prosopon/prosopon/auth/pairing.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/auth/pairing.py#L157)
  - [`prosopon/prosopon/client/kybernetes_client.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/client/kybernetes_client.py#L22-L25) and [`L415-L453`](file:///home/akarta/Projects/Metis/prosopon/prosopon/client/kybernetes_client.py#L415-L453)
  - [`prosopon/prosopon/web/static/js/squad.js`](file:///home/akarta/Projects/Metis/prosopon/prosopon/web/static/js/squad.js#L207-L364), [`pinax.js`](file:///home/akarta/Projects/Metis/prosopon/prosopon/web/static/js/pinax.js#L235-L414), [`workbench.js`](file:///home/akarta/Projects/Metis/prosopon/prosopon/web/static/js/workbench.js#L227), [`settings.js`](file:///home/akarta/Projects/Metis/prosopon/prosopon/web/static/js/settings.js#L747)
- **Expected Behavior**:
  1. `has_scope` must be enforced on all endpoints according to device role (`viewer`, `companion`, `operator`, `owner`), and PIN pairing must not default to `"admin"` (`*` scope).
  2. `POST /api/devices/lockdown` must revoke tokens in both `AccessStore` and `PairingManager`.
  3. `KybernetesClient` Pinax methods must call `{self.base_url}/pinax/...` (not `/v1/v1/pinax/...`).
  4. `/api/lochos/ws` must not call non-existent `pairing_mgr.validate_ticket`.
  5. Frontend JS must escape all dynamic strings before inserting into `innerHTML`.
- **Observed Behavior**:
  1. `has_scope` is only called in 1 route (`resolve_approval` at `server.py:951`); a `"viewer"` token can delete sessions, run chat streams, scan arbitrary host directories via `POST /api/pinax/scan`, and spawn/merge/discard Lochos subagents.
  2. `panic_lockdown` ([`server.py:503-530`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L503-L530)) never calls `pairing_mgr.revoke_all()`, leaving PIN-paired and ticket-redeemed tokens active after emergency lockdown!
  3. `KybernetesClient.__init__` appends `/v1` to `self.base_url` (line 25), and lines 415–453 append `/v1/pinax/...` again (`http://127.0.0.1:8081/v1/v1/pinax/...`), causing every live Pinax request to fail with HTTP 404.
  4. `lochos_websocket_proxy` ([`server.py:1582`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L1582)) calls `pairing_mgr.validate_ticket(token)`, which does not exist (`AttributeError`).
  5. `squad.js`, `pinax.js`, `workbench.js`, and `settings.js` interpolate unescaped task titles, tool paths, diff filenames, and device labels into `innerHTML` under a CSP with `script-src 'self' 'unsafe-inline'`.
- **Trigger Condition**: Using a `viewer` device token, triggering `/api/devices/lockdown`, opening the Pinax tab in Prosopon, connecting to `/api/lochos/ws` with an invalid token, or rendering untrusted strings in the UI.
- **Impact**: Broken RBAC, incomplete emergency lockdown, 100% broken live Pinax proxy, WebSocket handshake crashes, and stored/DOM XSS in Prosopon.
- **Fix Direction**: Enforce `has_scope` per endpoint, call `pairing_mgr.revoke_all()` in `panic_lockdown`, change `/v1/pinax/` to `/pinax/` in `kybernetes_client.py:415-453`, remove `pairing_mgr.validate_ticket(token)` in `server.py:1582`, and escape all dynamic HTML interpolations across `squad.js`, `pinax.js`, `workbench.js`, and `settings.js`.
- **Regression Check**: Test in `prosopon/tests/test_pyle_devices.py` that a PIN-paired token is rejected with `401` after `POST /api/devices/lockdown`, and that a `viewer` token receives `403` on `DELETE /api/sessions/{id}` and `POST /api/lochos/dispatch`.

---

### [F-15] `KrisisSelector` Crashes on Non-Finite / Malformed `Decimal` Inputs, Allows Unbounded `cal_ll` Score Inflation, and Drops Valid Candidates via Pre-Gate Deduplication
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `krisis` (`scoring/decimal_calc.py`, `gates/evaluator.py`, `engine/selector.py`, `dedup/exact.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`krisis/krisis/scoring/decimal_calc.py`](file:///home/akarta/Projects/Metis/krisis/krisis/scoring/decimal_calc.py#L9-L21) and [`L86-L105`](file:///home/akarta/Projects/Metis/krisis/krisis/scoring/decimal_calc.py#L86-L105)
  - [`krisis/krisis/gates/evaluator.py`](file:///home/akarta/Projects/Metis/krisis/krisis/gates/evaluator.py#L50-L63) and [`L90-L96`](file:///home/akarta/Projects/Metis/krisis/krisis/gates/evaluator.py#L90-L96)
  - [`krisis/krisis/engine/selector.py`](file:///home/akarta/Projects/Metis/krisis/krisis/engine/selector.py#L99-L114) and [`L213-L271`](file:///home/akarta/Projects/Metis/krisis/krisis/engine/selector.py#L213-L271)
  - [`krisis/krisis/dedup/exact.py`](file:///home/akarta/Projects/Metis/krisis/krisis/dedup/exact.py#L36-L48)
- **Expected Behavior**:
  1. Non-numeric or non-finite strings (`"NaN"`, `"Infinity"`, `"abc"`) in `calibrated_confidence`, `cal_ll`, or `raw_ll` must be cleanly rejected as `FEATURE_INVALID`, and direct `cal_ll` values must be bounded to `[0, 1]`.
  2. Deduplication must not discard a valid candidate `c2` merely because an identical-text predecessor `c1` failed a hard gate, and `KrisisDecision` must include `merged` and `kept_separate` in real evaluations.
- **Observed Behavior**:
  1. `to_decimal` ([`decimal_calc.py:14`](file:///home/akarta/Projects/Metis/krisis/krisis/scoring/decimal_calc.py#L14)) does not catch `decimal.InvalidOperation` or check `d.is_finite()`; `"NaN"` passes `Decimal("NaN")` and crashes `d.quantize(...)` in `render_6dp` with an unhandled `InvalidOperation`.
  2. When `cal_ll` is supplied directly on a `Candidate`, its range `[0, 1]` is never checked, allowing `cal_ll="5.000000"` to inflate total score above `1.0` and bypass `tau`.
  3. `selector.py:99` runs `deduplicate_candidates` *before* hard gates (line 114), discarding valid duplicates if the first duplicate fails a hard gate, and omits `merged=merged_records, kept_separate=kept_separate` from all real `KrisisDecision` return statements (lines 213–271).
  4. `GateEvaluator.evaluate_citation_resolution` ([`evaluator.py:52,62`](file:///home/akarta/Projects/Metis/krisis/krisis/gates/evaluator.py#L52-L62)) uses `if cid and cid not in valid_passage_ids:`, allowing `cited_passage_ids=[""]` to bypass citation resolution!
- **Trigger Condition**: Passing `"NaN"` / out-of-range `cal_ll`, duplicate candidates where the first fails a gate, or `cited_passage_ids=[""]` to `KrisisSelector.evaluate()`.
- **Impact**: Unhandled crash in `KrisisSelector`, score threshold bypass, false abstentions on duplicates, and citation gate bypass on empty strings.
- **Fix Direction**: Enforce `d.is_finite()` and `[0, 1]` bounds on `cal_ll` in `evaluate_feature_validity` and `compute_candidate_score`; reject empty citation IDs in `evaluate_citation_resolution`; deduplicate after gate evaluation (or promote valid survivors within equivalence classes); and attach `merged` / `kept_separate` to `KrisisDecision`.
- **Regression Check**: Unit tests in `krisis/tests/test_gates.py` and `test_scoring.py` covering `cal_ll="NaN"`, `cal_ll="2.0"`, `cited_passage_ids=[""]`, and duplicate candidates where candidate 1 has `access=False` and candidate 2 has `access=True`.

---

### [F-16] `HyperbolicHierarchicalStore.add_concept` Lacks Upsert Logic, Corrupting Multi-Store Alignment on Document Re-Index
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `LTHCM` (`core/hyperbolic.py`, `engine/pipeline.py`) & `stoma` (`adapter.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`LTHCM/lthcm/core/hyperbolic.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/hyperbolic.py#L167-L194)
  - [`LTHCM/lthcm/engine/pipeline.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/engine/pipeline.py#L126-L162)
  - [`stoma/src/stoma/adapter.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/adapter.py#L104-L126)
- **Expected Behavior**: Re-indexing an existing `doc_id` in `LTHCMEngine` must update the document in `superposition`, `hyperbolic`, and `wormhole` stores in place without duplicating entries.
- **Observed Behavior**: `SuperpositionMemoryStore.add_document` and `WormholeGraphIndex.add_node` perform upserts on existing `doc_id`s, but `HyperbolicHierarchicalStore.add_concept` ([`hyperbolic.py:167-194`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/hyperbolic.py#L167-L194)) unconditionally appends a duplicate entry to `self.documents` and `self.embeddings`. When Stoma subsequently verifies store alignment in `StomaLTHCMAdapter.get_store_doc_ids()` ([`adapter.py:116-119`](file:///home/akarta/Projects/Metis/stoma/src/stoma/adapter.py#L116-L119)), it raises `IndexIntegrityError("Duplicate document IDs detected in hyperbolic store")`.
- **Trigger Condition**: Re-indexing any existing `doc_id` into `LTHCMEngine`.
- **Impact**: Multi-store tensor desynchronization and `IndexIntegrityError` in Stoma.
- **Fix Direction**: Maintain an `_id_to_idx: Dict[str, int]` lookup in `HyperbolicHierarchicalStore` and update `self.embeddings[idx]` and `self.documents[idx]` in place when `doc_id` already exists (both in `add_concept` and `add_concepts_batch`).
- **Regression Check**: Unit test in `LTHCM/tests/test_lthcm.py` indexing `"doc_1"` twice and asserting `len(engine.hyperbolic.documents) == len(engine.superposition.documents) == 1`.

---

### [F-17] `stoma graph build` Mutates Legacy Top-Level `index.pkl` Outside the Generational Hash-Chain Protocol & Bypasses Trusted-Pickle Gate
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `stoma` (`graph_builder.py`, `cli.py`, `indexer.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`stoma/src/stoma/graph_builder.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/graph_builder.py#L22-L108)
  - [`stoma/src/stoma/cli.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/cli.py#L264-L283)
  - [`stoma/src/stoma/indexer.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/indexer.py#L286-L288) and [`L536-L605`](file:///home/akarta/Projects/Metis/stoma/src/stoma/indexer.py#L536-L605)
- **Expected Behavior**: `stoma graph build` must load the active committed generation, construct k-NN wormhole edges, and commit a new generation (`gen_XXXXXX`) updating `current.json`, `commit_marker.json`, and `commit_chain.json` while honoring trusted-pickle gates.
- **Observed Behavior**:
  1. `stoma graph build` defaults to `.stoma/index.pkl` (a write-only legacy compatibility mirror), while `StomaIndexer` loads exclusively from `.stoma/generations/gen_XXXXXX/index.pkl`. Edges built by `stoma graph build` are completely invisible to `StomaIndexer` and `Dianoia`!
  2. If pointed at `.stoma/generations/gen_XXXXXX/index.pkl`, `build_knn_knowledge_graph` overwrites `index.pkl` in place without updating `index_hash` in `current.json` / `commit_chain.json`, causing `StomaIndexer` to fail closed on next startup with `CorruptedStorageError`.
  3. [`graph_builder.py:35-36`](file:///home/akarta/Projects/Metis/stoma/src/stoma/graph_builder.py#L35-L36) calls raw `pickle.load(f)` without checking `STOMA_ALLOW_TRUSTED_PICKLE`.
- **Trigger Condition**: Running `stoma graph build` on a Stoma repository.
- **Impact**: Built knowledge graph edges are either ignored by `StomaIndexer` or corrupt the generation hash chain.
- **Fix Direction**: Integrate `build_knn_knowledge_graph` with `StomaIndexer` so it mutates `indexer.adapter.engine.wormhole` and commits a new generation via `indexer._commit_generation(...)`.
- **Regression Check**: Integration test in `stoma/tests/test_graph_builder.py` running `ingest_file`, then `stoma graph build`, then re-instantiating `StomaIndexer` and asserting `indexer.adapter.engine.wormhole.graph.number_of_edges() > 0`.

---

### [F-18] Algebraic & Geometric Defects in LTHCM (`CellularSheaf.compute_higher_obstruction_h2` $\equiv 0$, Non-Conservative `HamiltonianMetricFlow`, Non-Identity `LieAlgebraGaugeConnection`, and Unrectified Born-Rule Squaring)
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `LTHCM` (`core/sheaf.py`, `core/diffeomorphic.py`, `core/gauge_bundle.py`, `core/superposition.py`, `core/matryoshka.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`LTHCM/lthcm/core/sheaf.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/sheaf.py#L114-L122)
  - [`LTHCM/lthcm/core/diffeomorphic.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/diffeomorphic.py#L88-L107) and [`L174-L197`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/diffeomorphic.py#L174-L197)
  - [`LTHCM/lthcm/core/gauge_bundle.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/gauge_bundle.py#L48-L50)
  - [`LTHCM/lthcm/core/superposition.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/superposition.py#L258-L261) and [`LTHCM/lthcm/core/matryoshka.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/matryoshka.py#L198)
- **Expected Behavior**:
  1. `CellularSheaf.compute_higher_obstruction_h2` should compute non-trivial 2-cochain holonomy/obstruction on contradictory 3-cycles $(u, v, w)$.
  2. `HamiltonianMetricFlow` should integrate a conservative force $F = -\nabla_x V(x, q)$ consistent with `compute_hamiltonian`.
  3. `LieAlgebraGaugeConnection` should satisfy $\Omega(x, x) = 0$ (identity transport within the same domain) and $\Omega(y, x) = -\Omega(x, y)$.
  4. `SuperpositionMemoryStore.retrieve` and `MatryoshkaCompressor.coarse_to_fine_search` should not rank anti-parallel vectors ($\langle q, v \rangle = -1.0$) as top matches (`1.0`).
- **Observed Behavior**:
  1. **`h2_energy` is identically zero**: In [`sheaf.py:114-122`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/sheaf.py#L114-L122), $e_1 = F(v) - F(u)$, $e_2 = F(w) - F(v)$, $e_3 = F(u) - F(w)$ use the same vertex map $F$, so $e_1 + e_2 + e_3 \equiv 0$, and since `higher_restriction` $G$ has `bias=False`, $G(e_1) + G(e_2) + G(e_3) = G(0) \equiv 0$ for **all** inputs and weights!
  2. **Non-conservative force & mismatched Hamiltonian**: `drift_net` ([`diffeomorphic.py:102`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/diffeomorphic.py#L102)) is an unconstrained vector MLP $\mathbb{R}^{2D} \to \mathbb{R}^D$ (non-zero curl), and `compute_hamiltonian` (line 192) omits `drift_net` entirely.
  3. **Non-identity self-transport**: `compute_connection_matrix(x, x)` ([`gauge_bundle.py:48-50`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/gauge_bundle.py#L48-L50)) feeds `torch.cat([x, x])` into an unconstrained MLP with biases, producing $\Omega(x, x) \ne 0$.
  4. **Sign-blind Born-rule squaring**: `transition_probs = inner_products ** 2` ([`superposition.py:261`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/superposition.py#L261), [`matryoshka.py:198,247`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/matryoshka.py#L198)) squares negative dot products without `F.relu(...)`, giving an opposite vector ($\cos = -1.0$) the maximum score `1.0`.
- **Trigger Condition**: Running 3-way sheaf contradiction filtering, Hamiltonian flow diagnostics, gauge parallel transport, or superposition/Matryoshka retrieval on embeddings with negative cosine similarity.
- **Impact**: 3-way cohomological contradiction detection is completely inert (`0.0`); anti-parallel embeddings outrank aligned embeddings.
- **Fix Direction**:
  1. Condition edge restriction maps on ordered edge pairs $(u, v)$ in `compute_higher_obstruction_h2` so cycle sums do not telescope to zero.
  2. Parameterize `drift_net` as the negative gradient of a scalar potential $V_\theta(x, q)$ and include $V_\theta(x, q)$ in `compute_hamiltonian`.
  3. Symmetrize/antisymmetrize `coupling_net`: `coeffs = net(cat([x, y])) - net(cat([y, x]))`.
  4. Apply `F.relu(inner_products) ** 2` in `SuperpositionMemoryStore.retrieve` and `MatryoshkaCompressor.coarse_to_fine_search`.
- **Regression Check**: Add unit tests in `LTHCM/tests/test_lthcm.py` verifying `h2_energy > 0` on contradictory triples, `parallel_transport(v, x, x) == v`, and `retrieve` ranking $\langle q, v_+ \rangle = +0.8$ above $\langle q, v_- \rangle = -1.0$.

---

### [F-19] `WalkController._get_neighbours` Temporal Cutoff (`as_of`) Bypassed by Fallback Graph Successor Loop
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `dianoia` (`walk.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**: [`dianoia/dianoia/walk.py`](file:///home/akarta/Projects/Metis/dianoia/dianoia/walk.py#L469-L528)
- **Expected Behavior**: When `as_of is not None`, neighbours missing a timestamp must be rejected across all neighbour discovery loops in `_get_neighbours`, and `access_filter` exceptions must fail closed.
- **Observed Behavior**: The first loop over `self.engine.wormhole.get_bridges(node_id)` ([`walk.py:478-480`](file:///home/akarta/Projects/Metis/dianoia/dianoia/walk.py#L478-L480)) rejects nodes where `not has_time`. Immediately afterward, the fallback loop over `self.engine.wormhole.graph.successors(node_id)` ([`walk.py:501-509`](file:///home/akarta/Projects/Metis/dianoia/dianoia/walk.py#L501-L509)) defaults missing timestamps to `created = 0.0` (`0.0 <= as_of`), re-admitting every timestamp-less neighbour that the first loop just rejected!
- **Trigger Condition**: Calling `WalkController.walk(..., as_of=...)` when graph nodes lack `"created_epoch"` / `"timestamp"` metadata.
- **Impact**: Temporal causal light-cone filtering (`as_of`) is bypassed for untimestamped graph edges.
- **Fix Direction**: Apply the same `has_time` check and `try...except` around `access_filter(meta)` in the `graph.successors` loop at [`walk.py:504-509`](file:///home/akarta/Projects/Metis/dianoia/dianoia/walk.py#L504-L509).
- **Regression Check**: Unit test in `dianoia/tests/test_walk.py` verifying that a wormhole edge to a node with empty metadata `{}` is excluded when `as_of=1700000000.0`.

---

### [F-20] `AnadromeEvaluator`, `AnadromeReplayEngine`, and `AnadromeLiveStudy` Broken by Schema Key Mismatches, Linear Transition Iteration, and Mock Conflict Crash
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `dianoia` (`anadrome/evaluator.py`, `anadrome/replay.py`, `anadrome/study.py`, `walk.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`dianoia/dianoia/anadrome/evaluator.py`](file:///home/akarta/Projects/Metis/dianoia/dianoia/anadrome/evaluator.py#L265-L276)
  - [`dianoia/dianoia/anadrome/replay.py`](file:///home/akarta/Projects/Metis/dianoia/dianoia/anadrome/replay.py#L99-L194)
  - [`dianoia/dianoia/anadrome/study.py`](file:///home/akarta/Projects/Metis/dianoia/dianoia/anadrome/study.py#L441-L451) and [`L505-L519`](file:///home/akarta/Projects/Metis/dianoia/dianoia/anadrome/study.py#L505-L519)
  - [`dianoia/dianoia/walk.py`](file:///home/akarta/Projects/Metis/dianoia/dianoia/walk.py#L890-L923) and [`L1026-L1039`](file:///home/akarta/Projects/Metis/dianoia/dianoia/walk.py#L1026-L1039)
- **Expected Behavior**:
  1. `AnadromeEvaluator._score_answer` must read the actual verification boolean key (`"verified"`) written by `WalkController` and `AnswerProbeCache`.
  2. `AnadromeReplayEngine.replay_tree` must include `"realization_ids"` in `policy_view` and follow shadow branch transitions when a policy diverges onto a recorded shadow branch.
  3. `AnadromeLiveStudy` must detect conflict node visits via `p.get("node")`, return valid conflict pair keys (`"content_a"`, `"content_b"`, `"obstruction_energy"`) in its mock sheaf filter, and pass `seed` and `observation_hash` to `HighEntropyStochasticPolicy`.
- **Observed Behavior**:
  1. `evaluator.py:268` checks `probe.get("is_verified", False)` while `walk.py:1028` and `probe_cache.py:68` write `"verified"`.
  2. `replay.py:100-113` strips `"realization_ids"` from `policy_view` (breaking `SheafContradictionBacktrackPolicy` in offline replay) and steps linearly through `tree["transitions"]` even after switching to a shadow branch action.
  3. `study.py:509` checks `p.get("current_node")` and `p.get("target")` on `walk_path` dicts that only have key `"node"` (`hard_error` is always `False`), and `study.py:446` omits `"content_a"` / `"content_b"`, causing `verify.py:90` to raise `KeyError: 'content_a'` whenever a conflict node is visited.
- **Trigger Condition**: Running `AnadromeEvaluator`, `AnadromeReplayEngine`, or `AnadromeLiveStudy`.
- **Impact**: Inaccurate offline/live policy evaluations, dead hard-error detection, and crashes when conflict nodes are visited in `AnadromeLiveStudy`.
- **Fix Direction**: Align the dict keys across `evaluator.py`, `replay.py`, `study.py`, and `walk.py` as detailed above.
- **Regression Check**: Add a test in `dianoia/tests/test_live_walk_study.py` where a policy visits `doc_biochem_conflict` and assert `hard_error is True` without `KeyError`.

---

### [F-21] Hestia Uncaught `GpuDeviceConflictError` (HTTP 500), Leaked OS GPU `flock` on Crash Exhaustion, 60s Global Admission Lock Stall, and Multi-GPU VRAM Index Bug
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `hestia` (`api/app.py`, `server/__init__.py`, `vram.py`, `registry.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`hestia/src/hestia/api/app.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/api/app.py#L318-L319), [`L421-L422`](file:///home/akarta/Projects/Metis/hestia/src/hestia/api/app.py#L421-L422), [`L457-L458`](file:///home/akarta/Projects/Metis/hestia/src/hestia/api/app.py#L457-L458)
  - [`hestia/src/hestia/server/__init__.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/server/__init__.py#L217-L227), [`L303-L315`](file:///home/akarta/Projects/Metis/hestia/src/hestia/server/__init__.py#L303-L315), [`L359-L400`](file:///home/akarta/Projects/Metis/hestia/src/hestia/server/__init__.py#L359-L400), [`L503-L516`](file:///home/akarta/Projects/Metis/hestia/src/hestia/server/__init__.py#L503-L516), [`L570-L582`](file:///home/akarta/Projects/Metis/hestia/src/hestia/server/__init__.py#L570-L582)
  - [`hestia/src/hestia/vram.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/vram.py#L12-L28)
- **Expected Behavior**:
  1. `GpuDeviceConflictError` must return `HTTP 503 Service Unavailable`.
  2. `_handle_crash` must release the OS GPU lease (`_device_leases[device]`) when a crashed model transitions to `ModelState.FAILED`.
  3. `_try_fit` must not hold `_admission_lock` across a 60s sleep loop or ignore `_unload_model` returning `False`.
  4. `_try_fit`, `total_vram_reserved_mb`, and `_get_eviction_candidates` must query/filter by the target GPU (`model.gpu_target`) rather than hardcoding `gpu_index=0`.
- **Observed Behavior**:
  1. `GpuDeviceConflictError` is not caught in `app.py`, causing unhandled `HTTP 500` responses when Paideia holds the GPU lock.
  2. When a model exceeds `_MAX_RESTART_RETRIES` or fails to restart in `_handle_crash` (lines 515 and 581), `_device_model_counts[device]` is never decremented and the OS `flock` is leaked until Hestia exits.
  3. `_unload_model` sleeps up to 60s while `_load_model` holds `self._admission_lock`, and `_try_fit` (line 379) ignores its boolean return value.
  4. `get_free_vram_mb()` and `get_total_vram_mb()` are called with default `gpu_index=0` even when `model.gpu_target` is `"rtx-5060-ti"` (mapped to GPU 1 in `llamacpp.py:85`), and `pynvml.nvmlShutdown()` in `vram.py` is not in a `finally:` block.
- **Trigger Condition**: Loading a model while Paideia holds the GPU lock, model crash-looping >3 times, concurrent request during eviction, or multi-GPU deployment.
- **Impact**: HTTP 500 errors, permanent GPU lock starvation blocking Paideia training, 60s server stalls, and wrong multi-GPU VRAM accounting.
- **Fix Direction**: Catch `GpuDeviceConflictError` as HTTP 503 in `app.py`; release device leases on `FAILED` transitions in `_handle_crash`; pass `abort_if_inflight=True` and check return status in `_try_fit`; and scope VRAM queries/evictions by `gpu_target` with `try...finally: nvmlShutdown()`.
- **Regression Check**: Unit tests in `hestia/tests/test_server.py` verifying HTTP 503 on `GpuDeviceConflictError` and verifying `_device_leases` is empty after a model transitions to `ModelState.FAILED`.

---

### [F-22] Dokime `AdversarialGateAuditor` Schema Mismatch with `dokime run`, Leaked `httpx.Client` Per Query in `HestiaClient`, and Survivorship Bias in `dokime compare`
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `dokime` (`adversarial.py`, `runner.py`, `manifest.py`, `client.py`, `compare.py`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`dokime/src/dokime/adversarial.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/adversarial.py#L120-L238)
  - [`dokime/src/dokime/runner.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/runner.py#L145-L159) and [`manifest.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/manifest.py#L177-L197)
  - [`dokime/src/dokime/client.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/client.py#L62-L65)
  - [`dokime/src/dokime/compare.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/compare.py#L340-L345) and [`L584-L603`](file:///home/akarta/Projects/Metis/dokime/src/dokime/compare.py#L584-L603)
- **Expected Behavior**:
  1. `AdversarialGateAuditor.audit_run` must accept manifests and results produced by `dokime run`, and `compute_global_p99_ns` must reject `NaN`/`Inf`/`bool` latencies cleanly with `AuditFailure`.
  2. `HestiaClient._get_client()` must cache and reuse `self._client`.
  3. `run_comparison` and `run_cross_model_comparison` must not silently exclude queries where the candidate model errored (`score is None`) from paired accuracy bootstrap tests.
- **Observed Behavior**:
  1. `AdversarialGateAuditor.audit_run` requires `manifest["execution_kind"]`, `manifest["task_data_kind"]`, `obs["status"]`, and integer `obs["latency_ns"]`, whereas `create_manifest` and `run_evaluation` emit `latency_ms` and `score` without those fields—so every `dokime run` output fails `AdversarialGateAuditor`! Also `compute_global_p99_ns([float("nan")])` crashes with `ValueError: cannot convert float NaN to integer`.
  2. `HestiaClient._get_client()` ([`client.py:62-65`](file:///home/akarta/Projects/Metis/dokime/src/dokime/client.py#L62-L65)) instantiates a new `httpx.Client` on every call without assigning `self._client` or closing it.
  3. `compare.py:340-345` and `584-603` restrict `paired_bootstrap_test` to `paired_success_keys`, dropping any query where the candidate failed/timed out and creating severe survivorship bias.
- **Trigger Condition**: Auditing a `dokime run` directory with `AdversarialGateAuditor`, running `dokime run` over large benchmarks, or comparing a candidate model that fails on difficult queries.
- **Impact**: Adversarial auditor unusable on real runs, socket exhaustion during benchmarks, and false `[IMPROVEMENT]` verdicts for unstable models.
- **Fix Direction**: Emit `execution_kind`, `task_data_kind`, `status`, and `latency_ns` in `manifest.py`/`runner.py`; validate `math.isfinite(x) and not isinstance(x, bool)` in `adversarial.py`; assign `self._client = httpx.Client(...)` in `HestiaClient._get_client()`; and score candidate execution failures as `0.0` (or block promotion) in `compare.py`.
- **Regression Check**: Add a test in `dokime/tests/test_adversarial_integrity.py` running `run_evaluation` + `create_manifest` and passing the generated files directly to `AdversarialGateAuditor().audit_run(...)`.

---

### [F-23] Paideia `load_router_dataset` Appends Excluded Tie / Incomplete-Counterfactual Records to `clean_records`, and `paideia` CLI Omits `click` from Core Dependencies
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `paideia` (`data/router_loader.py`, `pyproject.toml`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`paideia/paideia/data/router_loader.py`](file:///home/akarta/Projects/Metis/paideia/paideia/data/router_loader.py#L104-L126)
  - [`paideia/pyproject.toml`](file:///home/akarta/Projects/Metis/paideia/pyproject.toml#L15-L26) and [`L51-L52`](file:///home/akarta/Projects/Metis/paideia/pyproject.toml#L51-L52)
- **Expected Behavior**:
  1. When `best_route is None` in `load_router_dataset` ([`router_loader.py:107-123`](file:///home/akarta/Projects/Metis/paideia/paideia/data/router_loader.py#L107-L123)), the record is counted as excluded (`missing_counterfactuals` or `ties_excluded`) and must be skipped (`continue`), not appended to `clean_records`.
  2. `click>=8.1.0` must be in `[project].dependencies` in `paideia/pyproject.toml` since `[project.scripts]` (`paideia = "paideia.cli:cli"`) imports `click` unconditionally.
- **Observed Behavior**:
  1. `if best_route is None:` increments `audit.missing_counterfactuals` / `audit.ties_excluded` and then falls through without `continue` to `clean_records.append(record)` and `audit.valid_records += 1`!
  2. `click` is only listed in `[project.optional-dependencies].cli`.
- **Trigger Condition**: Calling `load_router_dataset(..., eligible_routes=[...])` on records with utility ties or missing counterfactuals, or installing `paideia` without `[cli]`.
- **Impact**: Excluded records contaminate training/baseline datasets and `audit.valid_records` is miscounted.
- **Fix Direction**: Add `continue` at the end of `if best_route is None:` ([`router_loader.py:123`](file:///home/akarta/Projects/Metis/paideia/paideia/data/router_loader.py#L123)) and move `"click>=8.1.0"` into `[project].dependencies` in [`paideia/pyproject.toml`](file:///home/akarta/Projects/Metis/paideia/pyproject.toml#L15-L20).
- **Regression Check**: Unit test in `paideia/tests/test_data_pipeline.py` passing a tied router record with `eligible_routes=["r1", "r2"]` and asserting `len(clean_records) == 0` and `audit.ties_excluded == 1`.

---

### [F-24] VS Code Submodule (`contrib/void`) Unauthenticated / 404 Hestia API Calls, Token Mismatches, and Electron Renderer DOM XSS via `safeSetInnerHTML`
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: `vscode` submodule (`src/vs/workbench/contrib/void/`) & `hestia` (`config.py`, `hestia.yaml`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477` (`vscode` @ `2fd2fa78eebfabc97b8ece56008bc8947de9a46d`)
- **Locations**:
  - [`vscode/src/vs/workbench/contrib/void/browser/metisViews.ts`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/metisViews.ts#L38-L54), [`L409-L469`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/metisViews.ts#L409-L469), [`L1412-L1416`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/metisViews.ts#L1412-L1416), [`L2903`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/metisViews.ts#L2903), [`L3088`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/metisViews.ts#L3088)
  - [`vscode/src/vs/workbench/contrib/void/browser/metisDashboardPane.ts`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/metisDashboardPane.ts#L328)
  - [`vscode/src/vs/workbench/contrib/void/browser/react/src/void-settings-tsx/MetisSovereignSettings.tsx`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/react/src/void-settings-tsx/MetisSovereignSettings.tsx#L85-L95) and [`L196-L404`](file:///home/akarta/Projects/Metis/vscode/src/vs/workbench/contrib/void/browser/react/src/void-settings-tsx/MetisSovereignSettings.tsx#L196-L404)
  - [`hestia/src/hestia/config.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/config.py#L72) and [`hestia/hestia.yaml`](file:///home/akarta/Projects/Metis/hestia/hestia.yaml#L36)
- **Expected Behavior**:
  1. VS Code views must attach `Authorization: Bearer <token>` on all Hestia requests (`/models`, `/v1/models`, `/manage/load`, `/manage/unload`), call the real `/manage/load` endpoint, and share a consistent token with Hestia (`HESTIA_SERVICE_TOKEN`).
  2. All dynamic strings (especially LLM chat outputs `m.text` at `metisViews.ts:2903`) must be HTML-escaped before passing to `safeSetInnerHTML` (whose TrustedTypes policy is an identity pass-through `createHTML: (v) => v`).
- **Observed Behavior**:
  1. `HestiaViewPane` (`metisViews.ts:409,440,465`), `metisDashboardPane.ts:328`, and `MetisSovereignSettings.tsx:85` call Hestia without an `Authorization` header, receiving HTTP 401 and showing Hestia as `OFFLINE` / `0 Models`.
  2. `MetisLaunchpadViewPane._switchModel` (`metisViews.ts:1412`) calls non-existent `POST /v1/models/load` (HTTP 404) without auth and displays `"Successfully activated"` even on 404.
  3. `hestia/src/hestia/config.py:72` ignores `HESTIA_SERVICE_TOKEN` from the environment, and `hestia/hestia.yaml:36` uses `"metis-local-hestia-token"` while VS Code sends `"metis-sovereign"`, causing HTTP 403 on chat completions.
  4. `MetisChatViewPane._renderContent` (`metisViews.ts:2903`) injects raw LLM output `m.text` directly into `safeSetInnerHTML`, enabling DOM XSS inside the privileged Electron workbench renderer.
- **Trigger Condition**: Using the Metis VS Code IDE views against a running Hestia/Kybernetes stack, or receiving HTML tags in an LLM chat response.
- **Impact**: Broken model polling/loading/chat in VS Code and renderer code execution via XSS.
- **Fix Direction**: Honor `HESTIA_SERVICE_TOKEN` in `hestia/config.py:72`, attach bearer headers to all Hestia/Kybernetes fetches in VS Code using stored settings, fix `/v1/models/load` -> `/manage/load`, and HTML-escape all dynamic interpolations passed to `safeSetInnerHTML`.
- **Regression Check**: Verify `npm run compile` in `vscode/` and test `HestiaConfig.from_yaml` with `HESTIA_SERVICE_TOKEN` set in `os.environ`.

---

### [F-25] Unified Pytest Collection Failure (`import file mismatch` on `test_cli.py`) & Omission of `schedion` / `symbolon` from CI and `conftest.py`
- **Priority**: **P1** | **Confidence**: **Confirmed**
- **Affected Module**: Root Test & CI Infrastructure (`pyproject.toml`, `conftest.py`, `.gitlab-ci.yml`, `docs/review/pickup.md`)
- **Reviewed Commit**: `23d54fe40a52e7dac3a6028b2b41838dab0cf477`
- **Locations**:
  - [`pyproject.toml`](file:///home/akarta/Projects/Metis/pyproject.toml#L55-L79)
  - [`conftest.py`](file:///home/akarta/Projects/Metis/conftest.py#L15-L45)
  - [`.gitlab-ci.yml`](file:///home/akarta/Projects/Metis/.gitlab-ci.yml#L7) and [`L22`](file:///home/akarta/Projects/Metis/.gitlab-ci.yml#L22)
  - [`docs/review/pickup.md`](file:///home/akarta/Projects/Metis/docs/review/pickup.md#L43)
- **Expected Behavior**:
  1. Running unified multi-suite pytest commands (such as `./.venv/bin/pytest kleroterion/tests krisis/tests phylax/tests` documented in `pickup.md:43`) must collect and run cleanly without `import file mismatch` errors.
  2. `schedion` and `symbolon` (`tests/test_schedion.py` and `tests/test_symbolon.py`) must be included in `conftest.py` and `.gitlab-ci.yml`.
- **Observed Behavior**:
  1. Because multiple packages have `tests/test_cli.py` and `tests/test_contracts.py` without `addopts = ["--import-mode=importlib"]` in root `pyproject.toml`, running `./.venv/bin/pytest kleroterion/tests krisis/tests phylax/tests` fails with exit code 2:
     ```text
     ERROR collecting phylax/tests/test_cli.py
     import file mismatch: imported module 'test_cli' has this __file__ attribute:
       /home/akarta/Projects/Metis/krisis/tests/test_cli.py
     ```
  2. `conftest.py` and `.gitlab-ci.yml` omit `schedion` and `symbolon`, and `.gitlab-ci.yml:22` only runs `tests/test_stack_smoke.py tests/test_round2_fixes.py`, skipping `tests/test_schedion.py` and `tests/test_symbolon.py`.
- **Trigger Condition**: Running unified pytest across multiple submodules or running GitLab CI.
- **Impact**: Documented multi-suite health check command in `pickup.md:43` fails at collection time; 16 root tests for `schedion` and `symbolon` never run in CI.
- **Fix Direction**: Add `addopts = ["--import-mode=importlib"]` and `"vscode"`, `"runs"`, `"logs"`, `"data"` to `norecursedirs` in [`pyproject.toml`](file:///home/akarta/Projects/Metis/pyproject.toml#L55-L64); add `schedion` and `symbolon` to [`conftest.py`](file:///home/akarta/Projects/Metis/conftest.py#L15-L45) and [`.gitlab-ci.yml`](file:///home/akarta/Projects/Metis/.gitlab-ci.yml#L7); and run `uv run pytest tests -v` in `.gitlab-ci.yml:22`.
- **Regression Check**: Run `./.venv/bin/pytest kleroterion/tests krisis/tests phylax/tests -q` and verify 108 tests pass with exit code 0.

---

## 4. Prioritized Findings — P2 & P3 Defects (Reliability, Concurrency, Drift & Edge Cases)

### [F-26] Systemd Service Units Hardcode Flags Overriding `metis.env`, Point to Non-Existent Per-Package `.venv`s, and `run_stack.sh` Readiness Probe Fails HTTP 401
- **Priority**: **P2** | **Confidence**: **Confirmed**
- **Affected Module**: `deploy/systemd/`, `scripts/run_stack.sh`
- **Locations**:
  - [`deploy/systemd/metis-hestia.service`](file:///home/akarta/Projects/Metis/deploy/systemd/metis-hestia.service#L9-L17), [`metis-kybernetes.service`](file:///home/akarta/Projects/Metis/deploy/systemd/metis-kybernetes.service#L10-L21), [`metis-prosopon.service`](file:///home/akarta/Projects/Metis/deploy/systemd/metis-prosopon.service#L10-L18), [`metis-stoma-watcher.service`](file:///home/akarta/Projects/Metis/deploy/systemd/metis-stoma-watcher.service#L9-L17)
  - [`scripts/run_stack.sh`](file:///home/akarta/Projects/Metis/scripts/run_stack.sh#L38), [`L85-L87`](file:///home/akarta/Projects/Metis/scripts/run_stack.sh#L85-L87), [`L106-L109`](file:///home/akarta/Projects/Metis/scripts/run_stack.sh#L106-L109)
- **Observed Behavior**: Systemd units invoke `/opt/metis/<pkg>/.venv/bin/...` (whereas `setup_workspace.sh` creates a single root `/opt/metis/.venv`), hardcode `--host 127.0.0.1` and `--db-path kybernetes.db` (diverging from `kybernetes_budget.db`), and omit `ProtectSystem=strict` on 3 of 4 units. In `scripts/run_stack.sh:106`, `wait_for_url` polls `/v1/models` without `Authorization: Bearer ${HESTIA_SERVICE_TOKEN}`, always failing with HTTP 401 until timeout.
- **Fix Direction**: Point systemd units to `/opt/metis/.venv/bin/...`, honor `metis.env` variables, unify on `kybernetes_budget.db`, and poll `/health` (or pass the bearer header) in `scripts/run_stack.sh`.

---

### [F-27] Kybernetes Admin Probes, Traces, ChangeSet Secret Persistence, and Tool Registry Keyword Defects
- **Priority**: **P2** | **Confidence**: **Confirmed**
- **Affected Module**: `kybernetes` (`admin/probes.py`, `admin/routes.py`, `admin/changesets.py`, `tools/registry.py`, `tools/edit_file_block.py`, `tools/file_search.py`, `cli.py`)
- **Locations**:
  - [`kybernetes/kybernetes/admin/probes.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/admin/probes.py#L83-L85)
  - [`kybernetes/kybernetes/admin/routes.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/admin/routes.py#L618-L620)
  - [`kybernetes/kybernetes/admin/changesets.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/admin/changesets.py#L387-L388), [`L498`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/admin/changesets.py#L498), [`L520`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/admin/changesets.py#L520)
  - [`kybernetes/kybernetes/tools/registry.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/registry.py#L132), [`L163`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/registry.py#L163), [`L191`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/registry.py#L191), [`L358-L363`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/registry.py#L358-L363)
  - [`kybernetes/kybernetes/tools/file_search.py`](file:///home/akarta/Projects/Metis/kybernetes/kybernetes/tools/file_search.py#L89)
- **Observed Behavior**:
  1. `check_hestia_reachability` (`probes.py:83`) omits `HESTIA_SERVICE_TOKEN`, reporting Hestia `OFFLINE (HTTP 401)`.
  2. `/v1/admin/traces` (`routes.py:618-620`) reads non-existent `trace.timestamp`, `trace.total_duration_ms`, and `trace.tokens_used` (actual attributes on `TraceRecord` are `timestamp_utc`, `total_latency_ms`, `budget_usage`).
  3. `ChangeSetManager.apply` and `rollback` write unredacted secret values in `cs.patch` and `cs.diff` to `.admin_snapshots/cs_*.json`.
  4. `ToolRegistry._register_defaults` (`registry.py:132,163,191`) passes `side_effect=...` instead of `side_effect_class=...` (leaving `catalogue_probe` as `READ_ONLY`), and `edit_file_block` syntax failures return `ToolResult(success=True, output={"replaced": False, "error": ...})`.
  5. `SafeFileSearch` (`file_search.py:89`) resolves relative directories against `Path.cwd()` instead of `self.allowed_roots[0]`.
- **Fix Direction**: Pass `HESTIA_SERVICE_TOKEN` in `probes.py:83`, read the actual `TraceRecord` attributes in `routes.py:618-620`, redact `cs.patch`/`cs.diff` before saving `cs_*.json`, fix `side_effect_class` in `registry.py`, set `success=False` when `output.get("error")` is present, and resolve relative directories against `self.allowed_roots[0]` in `file_search.py`.

---

### [F-28] Multi-Process Audit Hash-Chain Race in Phylax `AuditLogger`, Unredacted Arguments in `.phylax/approvals.json`, and Lost `ASK` Requests in `phylax decide` CLI
- **Priority**: **P2** | **Confidence**: **Confirmed**
- **Affected Module**: `phylax` (`audit/logger.py`, `approvals/manager.py`, `cli.py`)
- **Locations**:
  - [`phylax/phylax/audit/logger.py`](file:///home/akarta/Projects/Metis/phylax/phylax/audit/logger.py#L130-L175)
  - [`phylax/phylax/approvals/manager.py`](file:///home/akarta/Projects/Metis/phylax/phylax/approvals/manager.py#L71-L95)
  - [`phylax/phylax/cli.py`](file:///home/akarta/Projects/Metis/phylax/phylax/cli.py#L121-L130)
- **Observed Behavior**: `AuditLogger.log_event` computes `next_id` and `prev_hash` from in-memory attributes *before* acquiring `fcntl.flock(f, fcntl.LOCK_EX)`, so concurrent processes fork and corrupt the audit hash chain. `ApprovalManager` writes raw unredacted arguments to `.phylax/approvals.json` without `0600` permissions. `phylax decide` (`cli.py:126`) instantiates `GuardedGateway(pm, audit)` without `ApprovalManager(DEFAULT_APPROVAL_PATH)`, discarding `ASK` approval requests.
- **Fix Direction**: Acquire `flock` before reading the last entry of `audit.jsonl`, redact sensitive arguments and set `0600` on `approvals.json`, and pass `ApprovalManager(DEFAULT_APPROVAL_PATH)` in `phylax/cli.py:126`.

---

### [F-29] Prosopon WebAuthn Origin Spoofing, Unexpired Pairing Links in `register/options`, `0644` TLS Private Key, Doctor False Positive, External Google Fonts, and `localStorage`-Only Settings UI
- **Priority**: **P2** | **Confidence**: **Confirmed**
- **Affected Module**: `prosopon` (`auth/webauthn.py`, `auth/tls.py`, `ops/doctor.py`, `server.py`, `web/static/index.html`, `web/static/js/settings.js`)
- **Locations**:
  - [`prosopon/prosopon/auth/webauthn.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/auth/webauthn.py#L138) and [`L212`](file:///home/akarta/Projects/Metis/prosopon/prosopon/auth/webauthn.py#L212)
  - [`prosopon/prosopon/server.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/server.py#L585-L591)
  - [`prosopon/prosopon/auth/tls.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/auth/tls.py#L64-L71)
  - [`prosopon/prosopon/ops/doctor.py`](file:///home/akarta/Projects/Metis/prosopon/prosopon/ops/doctor.py#L131-L135)
  - [`prosopon/prosopon/web/static/index.html`](file:///home/akarta/Projects/Metis/prosopon/prosopon/web/static/index.html#L7-L9)
  - [`prosopon/prosopon/web/static/js/settings.js`](file:///home/akarta/Projects/Metis/prosopon/prosopon/web/static/js/settings.js#L269-L280)
- **Observed Behavior**:
  1. `webauthn.py:138,212` sets `origins = [origin] if origin else list(self.allowed_origins)`, trusting any caller-supplied `Origin` header instead of intersecting with `self.allowed_origins`.
  2. `server.py:585-591` queries `pairing_links` without checking `expires_at_utc`.
  3. `tls.py:64-71` writes `prosopon.key` without `chmod(0o600)`.
  4. `doctor.py:131,135` hardcodes `PASS: Insecure /api/auth/local-token endpoint removed` (while `/api/auth/local-token` still exists at `server.py:340`) and checks `tokens.json` instead of `paired_tokens.json`.
  5. `index.html:7-9` requests external Google Fonts (`https://fonts.googleapis.com`), violating offline sovereignty and CSP.
  6. `settings.js:269-280` only saves settings to browser `localStorage` and never calls any backend API.
- **Fix Direction**: Validate `origin in self.allowed_origins` in `webauthn.py`; check `expires_at_utc` in `server.py:585`; set `0600` on `prosopon.key`; fix `doctor.py`; remove Google Fonts from `index.html`; and wire `settings.js` to backend config endpoints.

---

### [F-30] Stoma Watcher Thread-Safety Race on `StomaIndexer` & Ignored `--domain` Option in `stoma query` CLI
- **Priority**: **P2** | **Confidence**: **Confirmed**
- **Affected Module**: `stoma` (`watcher.py`, `indexer.py`, `cli.py`)
- **Locations**:
  - [`stoma/src/stoma/watcher.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/watcher.py#L330-L395)
  - [`stoma/src/stoma/cli.py`](file:///home/akarta/Projects/Metis/stoma/src/stoma/cli.py#L98-L105)
- **Observed Behavior**:
  1. `StomaWatchEventHandler._flush_pending` releases `self._lock` before calling `self.indexer.ingest_file` / `delete_file` / `move_file`, while `StomaIndexer` has no internal lock protecting `self.current_generation` or `gen_XXXXXX` directory creation against concurrent `watcher.reconcile()` or `indexer.query()` calls.
  2. `stoma query` ([`cli.py:101-105`](file:///home/akarta/Projects/Metis/stoma/src/stoma/cli.py#L101-L105)) accepts `--domain` / `-d` but calls `indexer.query(query_text=query_text, top_k=top_k)` without passing `domain=domain`.
- **Fix Direction**: Add a `threading.RLock` inside `StomaIndexer` guarding all index mutations and queries, and pass `domain=domain` in `stoma/src/stoma/cli.py:105`.

---

### [F-31] LTHCM Batch-Size Crash in `RiemannianLangevinMemory` / `TitansNeuralMemory`, Unsorted Deduplication in `IndependentRecoveryValidator`, Substring False Positives in `LearnedEdgeScorer`, and Unbounded Caches
- **Priority**: **P2** | **Confidence**: **Confirmed**
- **Affected Module**: `LTHCM` (`core/riemannian_langevin.py`, `core/titans_memory.py`, `recovery/validator.py`, `graph/learned_edge_scorer.py`, `graph/build_edges.py`, `core/cross_encoder.py`, `graph/community.py`)
- **Locations**:
  - [`LTHCM/lthcm/core/riemannian_langevin.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/riemannian_langevin.py#L206-L214) and [`titans_memory.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/titans_memory.py#L108-L115)
  - [`LTHCM/recovery/validator.py`](file:///home/akarta/Projects/Metis/LTHCM/recovery/validator.py#L120-L176)
  - [`LTHCM/lthcm/graph/learned_edge_scorer.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/graph/learned_edge_scorer.py#L107-L110) and [`build_edges.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/graph/build_edges.py#L149)
  - [`LTHCM/lthcm/core/cross_encoder.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/cross_encoder.py#L44) and [`community.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/graph/community.py#L242)
- **Observed Behavior**:
  1. `repeat_interleave(B_cur // self.M.shape[0], dim=0)` produces a 0-row tensor when `B_cur < self.M.shape[0]`, crashing `torch.matmul`.
  2. `IndependentRecoveryValidator.filter_and_validate_candidates` deduplicates by `seen_canonical_ids` *before* sorting by score, discarding higher-scoring duplicates that appear later in the list.
  3. `LearnedEdgeScorer.extract_features` uses raw `text_i.find(title_j)` without word boundaries (`\b`), matching `"art"` inside `"part"`; `build_edges.py` overwrites duplicate titles in `title_lookup[norm_t] = did`.
  4. `CrossEncoderReranker._score_cache` and `CommunitySummarizer._cache` grow without bound, and `CommunitySummarizer._cache` keys only by `doc_ids` without content hashes.
- **Fix Direction**: Handle `B_cur < self.M.shape[0]` in `retrieve`, sort candidates by score descending before deduplicating in `validator.py`, use word-boundary regexes and `Dict[str, List[str]]` in `learned_edge_scorer.py` / `build_edges.py`, and bound caches with LRU eviction + content hashes.

---

### [F-32] Kleroterion `compute_recall_at_k` / `compute_ndcg_at_k` Overcount Duplicate IDs (`Recall@k > 1.0`) and `DianoiaWalkAction` Colon Delimiter Mismatch
- **Priority**: **P2** | **Confidence**: **Confirmed**
- **Affected Module**: `kleroterion` (`eval/metrics.py`, `adapters/dianoia/adapter.py`)
- **Locations**:
  - [`kleroterion/eval/metrics.py`](file:///home/akarta/Projects/Metis/kleroterion/eval/metrics.py#L9-L24) and [`L42-L64`](file:///home/akarta/Projects/Metis/kleroterion/eval/metrics.py#L42-L64)
  - [`kleroterion/adapters/dianoia/adapter.py`](file:///home/akarta/Projects/Metis/kleroterion/adapters/dianoia/adapter.py#L31-L50)
- **Observed Behavior**: `compute_recall_at_k` and `compute_ndcg_at_k` do not deduplicate `ranked_doc_ids[:k]`, returning `2.0` (`200%` recall) when a relevant document ID appears twice in `top_k`. `DianoiaWalkAction.to_action_id` emits `"EXPAND:<target>"` whereas `dianoia/walk.py:110` parses `"EXPAND <target>"`.
- **Fix Direction**: Deduplicate `ranked_doc_ids[:k]` preserving first-seen rank in `metrics.py`, and support space-delimited `"EXPAND <target>"` in `DianoiaWalkAction`.

---

### [F-33] Hestia Stream Error Inflight Counter Leak, Unreset `ModelSwapManager` Drain Event, CLI Missing `Optional` Import, and Paideia/Dokime P2 Defects
- **Priority**: **P2** | **Confidence**: **Confirmed**
- **Affected Module**: `hestia`, `dokime`, `paideia`
- **Locations**:
  - [`hestia/src/hestia/api/app.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/api/app.py#L330-L335)
  - [`hestia/src/hestia/server/swap.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/server/swap.py#L236-L253)
  - [`hestia/src/hestia/cli/__init__.py`](file:///home/akarta/Projects/Metis/hestia/src/hestia/cli/__init__.py#L135-L235)
  - [`dokime/src/dokime/tasks/pinax_lookup.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/tasks/pinax_lookup.py#L53-L56) and [`dokime/src/dokime/cli.py`](file:///home/akarta/Projects/Metis/dokime/src/dokime/cli.py#L448)
  - [`paideia/paideia/training/runner.py`](file:///home/akarta/Projects/Metis/paideia/paideia/training/runner.py#L131-L141), [`hardware.py`](file:///home/akarta/Projects/Metis/paideia/paideia/training/hardware.py#L21-L38), [`promotion.py`](file:///home/akarta/Projects/Metis/paideia/paideia/evaluation/promotion.py#L66), [`admission_pool.py`](file:///home/akarta/Projects/Metis/paideia/paideia/baselines/admission_pool.py#L193-L212)
- **Observed Behavior**:
  1. In `hestia/api/app.py:330-335`, if `await resp.aread()` raises on a non-200 streaming error, `_safe_request_finished()` is skipped and `_inflight[model_name]` leaks permanently.
  2. In `hestia/server/swap.py:236-253`, `self._pin_events[generation]` is never cleared on subsequent `acquire_stream_lease` calls after draining once.
  3. `hestia/cli/__init__.py` uses `Optional[str]` without importing `Optional` and omits auth headers on `hestia status`.
  4. `PinaxLookup50Task.format_prompt` (`pinax_lookup.py:53-56`) drops `query.prompt` when given a prefix template without `{question}`/`{prompt}` placeholders; `dokime compare` (`cli.py:448`) exits `0` even on `[REGRESSION]`.
  5. `paideia/training/runner.py:134-137` calls non-existent `DatasetBOM.generate_from_file` inside `except Exception: pass` (`dataset_bom` is always `None`); `hardware.py:21` ignores `METIS_LOCK_DIR`; `promotion.py:66` indexes `p100` instead of `np.percentile(..., 95)` for $n \le 20$; and `AdmissionGatedPool.predict_one` mutates model state (spawns/prunes experts) during evaluation inference.
- **Fix Direction**: Wrap `resp.aread()` cleanup in `try...finally`; call `clear()` on `_pin_events[generation]` in `acquire_stream_lease`; fix `Optional` and auth headers in `hestia/cli`; append `query.prompt` in `PinaxLookup50Task.format_prompt`; call `generate_dataset_bom` in `runner.py`; honor `METIS_LOCK_DIR` in `hardware.py`; use `np.percentile` in `promotion.py`; and keep `AdmissionGatedPool.predict_one` side-effect-free during evaluation.

---

### [F-34] `DynamicSkillLearner` Path Traversal & YAML Frontmatter Injection, Plus `FisherRaoMetric.geodesic_midpoint` Unnormalized Input Degeneracy
- **Priority**: **P3** | **Confidence**: **Confirmed**
- **Affected Module**: `harness` (`skill_learner.py`), `LTHCM` (`core/fisher_rao.py`, `core/categorical.py`)
- **Locations**:
  - [`harness/skill_learner.py`](file:///home/akarta/Projects/Metis/harness/skill_learner.py#L42-L73)
  - [`LTHCM/lthcm/core/fisher_rao.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/fisher_rao.py#L121-L150)
  - [`LTHCM/lthcm/core/categorical.py`](file:///home/akarta/Projects/Metis/LTHCM/lthcm/core/categorical.py#L298-L302)
- **Observed Behavior**:
  1. `DynamicSkillLearner.synthesize_skill` sanitizes `name` into `slug` but passes raw `domain` into `self.skills_dir / domain / slug` without sanitization or `is_relative_to` verification, and interpolates unescaped `description` strings into YAML frontmatter.
  2. `FisherRaoMetric.geodesic_midpoint` does not call `self.to_simplex(p, is_logits=False)` before taking `torch.sqrt(p)`, collapsing geodesic interpolation when given unnormalized positive weights; `AdjointFunctorPair._init_orthogonal` crashes when `d_hdc < d_model`.
- **Fix Direction**: Sanitize `domain` and JSON-encode YAML frontmatter strings in `skill_learner.py`; normalize `p, q` via `self.to_simplex` in `FisherRaoMetric.geodesic_midpoint`; and validate `d_hdc >= d_model` (or transpose QR) in `AdjointFunctorPair`.

---

## 5. Testing & Verification Evidence

All test suites were inventoried and executed against `HEAD` (`23d54fe40a52e7dac3a6028b2b41838dab0cf477`) on Linux (CPython `3.12` via `uv` virtual environments, NVIDIA RTX 5060 Ti + Tesla V100 host):

| Suite / Command Executed | Exit Code | Passed | Failed | Skipped / Deselected | Duration | Notes / Failure Root Cause |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `./kybernetes/.venv/bin/pytest kybernetes/tests -q` | `0` | **187** | 0 | 0 | 17.15s | All 187 unit/integration tests pass. |
| `./dokime/.venv/bin/pytest dokime/tests -q` | `0` | **152** | 0 | 0 | 2.07s | All 152 tests pass. |
| `./LTHCM/.venv/bin/pytest LTHCM/tests -k "not test_universal_32_frontiers_benchmark_suite and not test_standard_community_benchmarks_suite" -q` | `0` | **125** | 0 | 2 deselected | 223.74s | 125/125 unit & recovery tests pass; 2 multi-minute full-corpus benchmark runners deselected after collection check (127 total collected). |
| `./.venv/bin/pytest dianoia/tests -q` | `0` | **93** | 0 | 0 | 3.94s | All 93 tests pass. |
| `./hestia/.venv/bin/pytest hestia/tests -q` | `0` | **72** | 0 | 1 skipped | 1.25s | 72 passed, 1 skipped (live llama-server binary test). |
| `./prosopon/.venv/bin/pytest prosopon/tests -q` | `0` | **71** | 0 | 0 | 44.61s | All 71 WebRTC, auth, device, and voice tests pass. |
| `./.venv/bin/pytest kleroterion/tests -q` | `0` | **43** | 0 | 0 | 1.60s | All 43 tests pass when run in isolation. |
| `./paideia/.venv/bin/pytest paideia/tests -q` | `0` | **41** | 0 | 0 | 1.23s | All 41 tests pass. |
| `./.venv/bin/pytest phylax/tests -q` | `0` | **36** | 0 | 0 | 1.42s | All 36 tests pass when run in isolation. |
| `./.venv/bin/pytest krisis/tests -q` | `0` | **29** | 0 | 0 | 0.23s | All 29 tests pass when run in isolation. |
| `./.venv/bin/pytest tests -q` | `1` | **35** | **4** | 0 | 18.37s | **4 FAILURES** in `tests/test_round2_fixes.py` (`test_defect7_*`) due to Finding **[F-06]** (`DependencyOriginError` on stale `dianoia` SHA-256 in `stoma/vendored_provenance.json`). |
| `./stoma/.venv/bin/pytest stoma/tests --collect-only -q` | `4` | **0** | **113 (collection error)** | 0 | 1.12s | **COLLECTION ERROR** in `stoma/tests/conftest.py` due to Finding **[F-06]** (`DependencyOriginError` on `import stoma`). |
| `./.venv/bin/pytest kleroterion/tests krisis/tests phylax/tests -q` | `2` | **0** | **1 (collection error)** | 0 | 7.34s | **COLLECTION ERROR** (`import file mismatch` on `phylax/tests/test_cli.py` vs `krisis/tests/test_cli.py`) due to Finding **[F-25]**. |
| **Total Reconciled Across Monorepo** | — | **884** | **4 failed + 113 blocked by [F-06]** | **1 skipped, 2 deselected** | ~323s | Fixing **[F-06]** unblocks the 113 `stoma` tests and 4 root provenance tests (`1,001` tests). |

---

## 6. Open Gates, Decisions Needed & Exact Next Actions

### Open Gates Before Production Readiness
1. **Gate 1 — Secret Hygiene & Rotation ([F-01], [F-05], [F-27], [F-28])**: Remove `deploy/metis.env` and `prosopon/data/paired_tokens.json` from git tracking, add them to `.gitignore`, rotate `HESTIA_SERVICE_TOKEN` and `KYBERNETES_SERVICE_TOKEN`, and block secret files in `/api/workbench/file`.
2. **Gate 2 — Stoma Provenance & Generational Graph Integrity ([F-06], [F-16], [F-17])**: Synchronize `stoma/vendored_provenance.json` and `stoma/src/stoma/vendored_provenance.json` with `dianoia`'s current SHA-256 digest (`948f35a52ca090acb715aeecfe17ff6cac061f541a9b124cc5b73836db44df39`), add upsert logic to `HyperbolicHierarchicalStore.add_concept`, and route `stoma graph build` through `StomaIndexer` generational commits.
3. **Gate 3 — Control-Plane Authentication & Tool Sandboxing ([F-02], [F-03], [F-04], [F-07], [F-14])**: Remove the loopback exemption in `kybernetes/server/app.py:require_service_auth`, protect all Lochos/Pinax routes, require authentication on `/api/webrtc/offer` and `/api/webrtc/ice-candidate`, require WebAuthn credentials on WebRTC Phylax approvals, reclassify `SafeCodeMode` as `SYSTEM_EXEC` inside `BubblewrapBackend`, and share a single `PhylaxAdapter` instance between `ExecutionPipeline` and `ToolRegistry`.
4. **Gate 4 — Mathematical & Verification Integrity ([F-08], [F-15], [F-18], [F-19], [F-20], [F-22])**: Fix `KrisisAdapter` negation-word false positives, rectify negative dot products before Born-rule squaring in `LTHCM`, fix edge-dependent restriction maps in `CellularSheaf.compute_higher_obstruction_h2`, enforce `has_time` in `WalkController._get_neighbours`, and align `dokime.runner` outputs with `AdversarialGateAuditor`.

---

## 7. Better Features & Architectural Improvements List

Beyond resolving the defects catalogued above, the following high-leverage architectural enhancements and features will significantly elevate Metis's capability, performance, and sovereignty:

### 7.1 Unify the Policy, Risk & Approval Plane Across `Prosopon ↔ Kybernetes ↔ Phylax ↔ Pinax`
- **Current State**: Approval state is fragmented across four disconnected stores: `Prosopon.WebRTCGateway._pending_approvals`, `Kybernetes.ExecutionPipeline.phylax`, `Kybernetes.ToolRegistry.phylax_adapter`, and `.phylax/approvals.json`. Resolving a biometric approval over WebRTC in Prosopon never notifies Kybernetes, and `ToolRegistry.execute` does not consult `PinaxCatalogue` for command-level risk escalation (`T0`–`T3`).
- **Improvement**:
  1. Establish a single SQLite-backed, SHA-256 hash-chained approval ledger in Phylax shared by `ExecutionPipeline`, `ToolRegistry`, and `CodingAgentHarness`.
  2. Wire `WebRTCGateway.add_approval_listener` in `prosopon/server.py` directly to `kybernetes_client.resolve_approval(...)` only after cryptographic WebAuthn (`uv=1`) verification succeeds.
  3. Invoke `PinaxCatalogue.evaluate_invocation` and `PinaxScanner.verify_executable_integrity` automatically inside `SafeSandboxedBash` and `ToolRegistry.execute` so any command dynamically escalating to `T2`/`T3` triggers a hardware-bound passkey approval card on the operator's companion device.

### 7.2 Replace Heuristic `KrisisAdapter` & `PlanRouter` with Native `KrisisSelector` + Pinax-Guided Structured Tool Routing
- **Current State**: `kybernetes/adapters/krisis.py` ignores the `krisis` package and uses a naive word-overlap + negation-word heuristic, while `kybernetes/routing/router.py` uses substring matching (`"calculate"`, `"write file"`) and emits an unregistered `"guarded_action"` tool name.
- **Improvement**:
  1. Wire `KrisisAdapter.verify_response` to construct a real `krisis.contracts.KrisisRequest` from multiple Hestia candidate completions (or Dianoia walk branches), populate per-claim support scores using LTHCM's `CrossEncoderReranker` / `SheafContradictionFilter`, and execute `KrisisSelector.evaluate()` with deterministic 6-decimal Half-Even scoring.
  2. Upgrade `PlanRouter` to query Pinax Level-0/Level-1 FTS5 tool summaries (`PinaxCatalogue.lookup`) and use constrained JSON schema / grammar-guided tool selection on Hestia.

### 7.3 Close the Loop Between `Paideia` Adapter Export and `Hestia` Hot-Swap Serving (`ModelSwapManager`)
- **Current State**: `paideia/export/registry.py` exports content-addressed `hestia_registry_entry.json` manifests, and `hestia/server/swap.py` implements `ModelSwapManager` with stream lease pinning and automatic rollback, but neither is wired into `hestia/server/__init__.py` or `LlamaCppBackend`. Furthermore, `QLoRATrainingRunner` only writes a `SIMULATED` marker file.
- **Improvement**:
  1. Integrate `ModelSwapManager` into `ServerManager` and expose `POST /manage/swap` and `POST /manage/rollback` in `hestia/api/app.py`.
  2. Add `--lora <adapter_path>` support to `LlamaCppBackend.build_command()` and allow `ModelRegistry` to dynamically ingest validated `hestia_registry_entry.json` bundles exported by Paideia.
  3. Add an optional real LoRA/GRPO training backend in `paideia/training/runner.py` (using `peft` / `trl` on the V100 / RTX 5060 Ti under the shared POSIX GPU lock) gated by Dokime's 10,000 paired bootstrap promotion test.

### 7.4 Wire `Kleroterion` Adaptive Entropy Feedback & Epistemic Disagreement into `Dianoia` Walk Controller
- **Current State**: `kleroterion` implements a full dimension-scaled ($1/\sqrt{d}$) masked-softmax proposal engine with marginal entropy tracking ($H(\bar{\pi})$) and Jensen-Shannon proposal disagreement ($J = H_{\text{marg}} - H_{\text{cond}}$), plus `DianoiaActionAdapter`, but `dianoia` never imports `kleroterion` and instead uses a stateless re-implementation in `dianoia/anadrome/policies.py`.
- **Improvement**:
  1. Register `DianoiaActionAdapter` as a first-class policy in `dianoia/walk.py`.
  2. Use Kleroterion's `proposal_disagreement` ($J$) diagnostic as an intrinsic epistemic uncertainty trigger: when feature perturbations disagree strongly on the next hop ($J > \theta$), automatically trigger a bounded `Anadrome` shadow branch expansion (`is_shadow=True`) or `BACKTRACK` without waiting for an expensive LLM synthesis step.

### 7.5 Atomic Generational Graph Enrichment & Incremental Indexing in `Stoma` + `LTHCM`
- **Current State**: `StomaIndexer` only commits new generations on file ingest/delete/move, emitting zero graph edges by default, while `stoma graph build` mutates a legacy `.stoma/index.pkl` file outside the generational ledger.
- **Improvement**:
  1. Add `StomaIndexer.rebuild_graph_edges(method="learned" | "knn", threshold=0.75)` as a first-class transactional operation that loads the active generation, runs `LTHCM`'s `LearnedEdgeScorer` (or k-NN builder) + Leiden community summarization (`CommunitySummarizer`), and commits `gen_{N+1}` with updated SHA-256 manifests and SQLite-vec indices.
  2. Add incremental delta-edge updates during `StomaWatcher` background flushes so newly ingested documents are immediately linked into the wormhole graph without a full-corpus rebuild.

### 7.6 Edge-Conditioned Cellular Sheaf Cohomology ($H^1 / H^2$) & Sign-Preserving Quantum Retrieval in `LTHCM`
- **Current State**: `CellularSheaf.compute_higher_obstruction_h2` uses vertex-only maps that telescope to zero on every triangle, and `SuperpositionMemoryStore.retrieve` squares raw dot products so $\cos = -1$ scores `1.0`.
- **Improvement**:
  1. Replace vertex-only restriction maps in `CellularSheaf` with edge-conditioned orthogonal transport maps $F_{u \to e}(u, v) \in O(d_{\text{stalk}})$ trained via `ContrastiveSheafTrainer` on NLI entailment vs. contradiction pairs, making both 1-cohomology ($H^1$ pairwise conflict) and 2-cohomology ($H^2$ 3-way circular paradoxes) mathematically non-trivial.
  2. Use rectified Born fidelity $\max(0, \langle q, \psi_k \rangle)^2$ (or density-matrix POVM trace $\text{Tr}(\rho_{\text{doc}} \Pi_q)$ constructed from positive-semidefinite outer products with signed orientation gating) so negated propositions are never confused with positive matches.

### 7.7 Connect Prosopon's 8-Tab Settings Drawer & VS Code Control Plane to Kybernetes Transactional `ChangeSetManager`
- **Current State**: Kybernetes already implements a rich transactional `ChangeSetManager` (`/v1/admin/modules/{module_id}/stage`, `/apply`, `/rollback`) with optimistic concurrency (`base_hash`), unified diffs, and pre-write snapshots, but Prosopon's `settings.js` only writes to browser `localStorage` and VS Code's `MetisSovereignSettings.tsx` hardcodes paths and tokens.
- **Improvement**:
  1. Connect Prosopon's Settings UI to `/v1/admin/modules/*` so staging a change displays the server-validated unified diff and applying it writes atomically with one-click rollback.
  2. Provide a single auto-generated `.metis/runtime_endpoints.json` discovery manifest (with `0600` permissions) read by `prosopon`, `kybernetes` CLI, `dokime`, and the `vscode` Void extension so ports, active model aliases, and service tokens never drift across tools.
