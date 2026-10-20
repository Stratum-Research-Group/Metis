# Krisis: Round 1 Defect Analysis, Gaps & Remediation Plan

This document details the defect findings, safety gate gaps, and snapshot handling issues identified in `metis-repo-wide-round1-fixes-6752222c.txt` and `metis-dashboard-implementation-spec-ae93d36c.txt` for **Krisis**, along with the exact technical remediation plan and regression tests.

---

## 1. Defect Findings & Root Cause Analysis

### Item 8 (P0): Required Eligibility Gates Defaulting to True
- **Location**: `krisis/krisis/contracts/candidate.py:28-59`, `krisis/contracts/request.py:9-16`, `gates/evaluator.py:29-35`.
- **Observed Behavior**: In `CandidateGateResults`, missing gate fields (`access`, `provenance`, `contradiction`, `citation_resolution`) default to `True`. As a result, a grounded request containing unverified claims, missing cited evidence passages, and completely empty gate results was selected and marked eligible.
- **Root Cause**: Permissive defaults in Pydantic schema contracts, allowing caller assertions to bypass independent verification.
- **Remediation**:
  1. Remove default `True` values for required gates under grounded evaluation profiles (`grounded_v1`, `grounded_support_only`).
  2. For grounded requests, if gate results are omitted or null, treat them as `False` (or raise validation errors).
  3. Validate bracketed citation markers (e.g. `[1]`, `[doc:123]`) against the provided list of evidence items; if an unknown citation ID is referenced, fail the `citation_resolution` gate immediately.
  4. Ensure claims with confidence below `per_claim_min` or status other than `supported` fail eligibility.
  5. Preserve unverified baseline selection *only* when `requires_grounding=False` under `eligibility_baseline`, accompanied by an explicit `unverified` status label.
- **Regression Testing**:
  - Test `test_omitted_gate_results_fail_closed_under_grounded_profile`: Request with missing gate results fails eligibility with `NO_VALID_CANDIDATE`.
  - Test `test_unresolved_citation_fails_gate`: Candidate citing `[999]` with only `[1]` in evidence fails `citation_resolution`.

---

### Item 9 (P1): Selecting Old Snapshot When Rebuild is Possible
- **Location**: `krisis/krisis/engine/selector.py:46-54`, `contracts/request.py:26-31`.
- **Observed Behavior**: When a snapshot change occurs (`evidence_changed=True`) and `rebuilds_remaining > 0`, the selector still selects from existing candidates rather than requesting a fresh rebuild. It only abstains with `STALE` when `rebuilds_remaining == 0`.
- **Root Cause**: Treating available rebuild budget as permission to serve stale candidates rather than triggering an upstream rebuild request.
- **Remediation**:
  1. If `evidence_changed=True`, return a typed `STALE_REBUILD_REQUIRED` outcome so the caller (Kybernetes) can perform a bounded rebuild and submit fresh candidates.
  2. Recheck snapshot identity before releasing a selected answer candidate.
- **Regression Testing**:
  - Test `test_evidence_changed_with_budget_requests_rebuild`: Assert `evidence_changed=True` returns rebuild-required status when `rebuilds_remaining=1`.
