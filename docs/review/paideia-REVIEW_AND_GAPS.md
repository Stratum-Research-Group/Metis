# Paideia: Round 1 Defect Analysis, Gaps & Remediation Plan

This document details the defect findings, dataset labeling gaps, and adapter export safety issues identified in `metis-repo-wide-round1-fixes-6752222c.txt` and `metis-dashboard-implementation-spec-ae93d36c.txt` for **Paideia**, along with the exact technical remediation plan and regression tests.

---

## 1. Defect Findings & Root Cause Analysis

### Item 18 (P0): Require Real Verification Before Gold Labels
- **Location**: `paideia/paideia/contracts/dianoia_schema.py:24-39,63-79`, `paideia/data/walk_loader.py:84-125`.
- **Observed Behavior**: `is_defensible_gold_trajectory()` returned `True` for walk records where the `verification` field was entirely missing. The default `conflicts_found=False` in Pydantic models conflated "not verified" with "checked and found free of contradictions".
- **Root Cause**: Permissive schema validation and default boolean parameters in the gold trajectory filter.
- **Remediation**:
  1. Require an explicit, non-null `verification` object in `DianoiaWalkRecord`.
  2. If verification was omitted, unexecuted, or failed, `is_defensible_gold_trajectory()` must return `False`.
  3. Exclude simulated walks and parse-failed trajectories from gold action datasets.
  4. Track and report excluded record counts in dataset audit logs.
- **Regression Testing**:
  - Test `test_missing_verification_rejected_from_gold_data`: Walk without verification field is excluded from gold dataset.
  - Test `test_unresolved_rewalk_rejected`: Conflicted walk where re-walk also failed is excluded.

---

### Item 19 (P0): Fail Closed on Invalid Adapter Exports & Artifact Identity
- **Location**: `paideia/paideia/export/registry.py:47-95`, `training/runner.py:46-75`.
- **Observed Behavior**: The export routine swallowed missing/invalid manifest exceptions. A run with a corrupt manifest and only `adapter_config.json` (no trained weights) exported successfully. Empty directories generated non-empty SHA-256 digests of empty strings. Fallback digests hashed alias string literals rather than true artifact byte contents.
- **Root Cause**: Inadequate post-condition validation on exported files and fallback hashing of alias strings.
- **Remediation**:
  1. Require valid, completed run manifests with status `TRAINED` (never `SIMULATED` or `FAILED`).
  2. Assert actual weight files (`adapter_model.safetensors` or `adapter_model.bin`) exist and are non-empty before computing digests.
  3. Compute digests strictly over file bytes; label unresolved model identities as `unknown` rather than hashing alias strings.
- **Regression Testing**:
  - Test `test_export_fails_closed_on_missing_weights`: Directory with only config files raises `ArtifactValidationError`.
  - Test `test_export_rejects_simulated_manifest`: Run marked `SIMULATED` cannot be exported to production registry.

---

### Item 20 (P1): Scope Documentation of Training Runner as Marker-Only
- **Location**: `paideia/paideia/training/runner.py:164-195`, `contracts/hestia_schema.py:8-25`.
- **Observed Behavior**: The training runner writes a run configuration marker and marks the run `SIMULATED` without performing actual PyTorch/CUDA optimizer steps, even when run in GPU environments. The documentation implied genuine QLoRA training occurred when GPU dependencies were installed.
- **Root Cause**: Disconnect between scaffolding implementation and advertised functionality.
- **Remediation**:
  1. Clearly label the current training runner as a `SIMULATED` marker-only pipeline in docs and CLI outputs.
  2. When real training is requested, fail with a typed `NotImplementedError` until the actual PyTorch/PEFT optimizer loop is integrated.
- **Regression Testing**:
  - Test `test_training_runner_records_simulated_status`: Runner output manifest explicitly contains `status: "SIMULATED"`.
