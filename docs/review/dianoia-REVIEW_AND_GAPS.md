# Dianoia: Round 1 Defect Analysis, Gaps & Remediation Plan

This document details the defect findings, causal walk constraints, and serialization issues identified in `metis-repo-wide-round1-fixes-6752222c.txt` and `metis-dashboard-implementation-spec-ae93d36c.txt` for **Dianoia**, along with the exact technical remediation plan and regression tests.

---

## 1. Defect Findings & Root Cause Analysis

### Item 12 (P1): Enforce Causal & Provenance Constraints in Deliberative Walks
- **Location**: `dianoia/dianoia/walk.py:349-410,535-572,633-650`, `LTHCM/lthcm/engine/pipeline.py:262-266`.
- **Observed Behavior**: Dianoia seeds candidate nodes from LTHCM and iterates through graph neighbors without passing an explicit query timestamp (`as_of`) or access filter context. Adjacency checks verify graph topology but fail to enforce Lorentzian past light-cone filtering on expanded nodes.
- **Root Cause**: Reliance on low-level LTHCM defaults (`query_timestamp=100.0`) which can silently trigger causal fallbacks or admit future-dated documents.
- **Remediation**:
  1. Pass the caller's real epoch timestamp (`time.time()`) explicitly into every retrieval, seed evaluation, and neighborhood expansion.
  2. Verify that expanded candidate nodes fall strictly within the causal light cone ($\Delta s^2 \le 0, \Delta t > 0$) prior to presenting them to the LLM prompt.
  3. Validate access boundaries on graph nodes to ensure restricted documents are not visited during deliberation.
- **Regression Testing**:
  - Test `test_walk_rejects_future_dated_neighbor`: Future-dated node neighbor is filtered from the expanded candidate set.
  - Test `test_walk_enforces_epoch_timestamps`: Assert real epoch timestamp is propagated to all LTHCM calls.

---

### Item 13 (P1): Elimination of Untrusted Pickle Deserialization
- **Location**: `dianoia/dianoia/cli.py:156-181`.
- **Observed Behavior**: The `--index` option deserializes pre-built indices directly via `pickle.load()` before verifying object types or internal shapes. Deserializing untrusted pickle archives creates arbitrary code execution vulnerabilities.
- **Root Cause**: Relying on Python `pickle` for model and index persistence without signature verification.
- **Remediation**:
  1. Restrict index loading to explicit, user-specified local files; enforce owner-only read permissions (`0600`).
  2. Transition to a structured, data-only interchange format (e.g. SafeTensors or JSON manifest + binary arrays) for shared graph snapshots.
  3. Verify cryptographic SHA-256 digests against known index manifests before deserialization.
- **Regression Testing**:
  - Test `test_corrupt_index_fails_gracefully`: Malformed or tampered index files fail validation prior to consumption.

---

### Item 18 (P0): Explicit Verification Reporting for Paideia Gold Trajectories
- **Location**: `dianoia/dianoia/log.py:34,85-99`.
- **Observed Behavior**: If verification is omitted or fails to run, downstream consumers (Paideia) might interpret the absence of recorded conflicts as proof of factual correctness (`conflicts_found=False`).
- **Root Cause**: Conflating "not verified" with "verified without contradictions".
- **Remediation**:
  1. Record explicit verification metadata: `verified: bool`, `verifier_mode: "sheaf"|"none"`, `conflicts_found: bool`, `obstruction_energy: float`.
  2. Ensure walk records conforming to schema `1.0.0` clearly distinguish unverified walks from checked-and-consistent trajectories.
- **Regression Testing**:
  - Test `test_walk_log_records_explicit_verification_mode`: Verify generated logs explicitly indicate whether NLI or Laplacian verification was executed.
