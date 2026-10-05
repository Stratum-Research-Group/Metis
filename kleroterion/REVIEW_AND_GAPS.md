# Kleroterion: Round 1 Defect Analysis, Gaps & Remediation Plan

This document details the defect findings, integration status, and profile boundary verifications identified in `metis-repo-wide-round1-fixes-6752222c.txt` and `metis-dashboard-implementation-spec-ae93d36c.txt` for **Kleroterion**, along with the exact technical remediation plan and regression tests.

---

## 1. Defect Findings & Root Cause Analysis

### Item 10 & 12 (P1): Opt-In Integration & Live Wiring Status
- **Location**: `kleroterion/adapters/dianoia_adapter.py`, `dianoia/dianoia/walk.py:349-410`.
- **Observed Behavior**: Kleroterion contains complete standalone stochastic proposal mechanisms and adapter interfaces, but Dianoia's default walk loop does not currently invoke Kleroterion. The documentation previously implied that Kleroterion is an active stochastic decision engine in production deliberation turns.
- **Root Cause**: Describing an opt-in component as an active default pipeline dependency.
- **Remediation**:
  1. Clearly document Kleroterion's status as an opt-in stochastic decision submodule.
  2. In `dianoia`, provide an explicit flag (`--use-kleroterion`) to enable stochastic action sampling via `DianoiaKleroterionAdapter`.
  3. Ensure that when running under the `exact_fact` policy profile, noise injection is strictly forced to 0.0 and selection mode is locked to `ARGMAX`, preserving exact factual determinism.
- **Regression Testing**:
  - Test `test_exact_fact_profile_zero_noise_guarantee`: Assert noise budget is 0.0 under `exact_fact`.
  - Test `test_dianoia_adapter_respects_authoritative_mask`: Masked actions are never selected.
