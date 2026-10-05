# LTHCM: Round 1 Defect Analysis, Gaps & Remediation Plan

This document details the defect findings, causality contract issues, and memory integration gaps identified in `metis-repo-wide-round1-fixes-6752222c.txt` and `metis-dashboard-implementation-spec-ae93d36c.txt` for **LTHCM**, along with the exact technical remediation plan and regression tests.

---

## 1. Defect Findings & Root Cause Analysis

### Item 11 & 12 (P1): Causal Timestamp Defaults & Light-Cone Horizons
- **Location**: `LTHCM/lthcm/engine/pipeline.py:168,262-266`, `core/lorentzian.py`.
- **Observed Behavior**: `LTHCMEngine.retrieve_and_verify` defaults to `query_timestamp = 100.0` unless explicitly overridden. When real-world callers (such as Stoma or Dianoia) index documents with real epoch seconds (`time.time()`), a query timestamp of `100.0` places all indexed documents far into the future relative to the query ($dt = 100.0 - \text{epoch} \ll 0$). This violates the timelike past light cone condition ($\Delta s^2 \le 0, \Delta t > 0$), which silently activates `causal_fallback = True`, bypassing Lorentzian causality filtering completely.
- **Root Cause**: Hardcoded synthetic test default (`100.0`) in the public pipeline entrypoint.
- **Remediation**:
  1. In integration adapters (e.g. `kybernetes/adapters/lthcm.py` and `dianoia`), always pass `query_timestamp = time.time()` explicitly.
  2. Document the timestamp contract prominently in all consumer modules.
  3. Ensure that whenever `causal_fallback` is triggered, an explicit telemetry warning is emitted in the retrieval response dict.
- **Regression Testing**:
  - Test `test_epoch_timestamp_retrieval_does_not_trigger_causal_fallback`: Index document with current epoch timestamp; query with current epoch; assert `causal_fallback == False`.

---

### Item 13 (P1): Untrusted Checkpoint Loading
- **Location**: `LTHCM/lthcm/engine/pipeline.py`.
- **Observed Behavior**: LTHCM checkpoint restores use `torch.load()` or `pickle.load()` without cryptographic integrity checks.
- **Root Cause**: Relying on unverified pickle-based serialization formats.
- **Remediation**:
  1. Enforce file path whitelisting and owner-only permissions.
  2. Require callers to provide or verify SHA-256 digests against known manifests before invoking checkpoint restoration.
- **Regression Testing**:
  - Test `test_tampered_checkpoint_rejected`: Altered checkpoint bytes trigger integrity validation failure.
