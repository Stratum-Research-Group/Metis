# Dokime: Round 1 Defect Analysis, Gaps & Remediation Plan

This document details the defect findings, release gate verifications, and model provenance gaps identified in `metis-repo-wide-round1-fixes-6752222c.txt` and `metis-dashboard-implementation-spec-ae93d36c.txt` for **Dokime**, along with the exact technical remediation plan and regression tests.

---

## 1. Defect Findings & Root Cause Analysis

### Item 26 (P1): Record Real Served Model & Release Identity
- **Location**: `dokime/src/dokime/manifest.py:102-162`, `dokime/tasks/arc.py:1-17`.
- **Observed Behavior**: Dokime records model aliases and benchmark configuration parameters, but does not capture the cryptographic SHA-256 digests of the actual served weights file, tokenizer configuration, or prompt templates. Reusing a model alias with completely different weights cannot be detected from Dokime's run manifest alone. In addition, the built-in 25-item ARC task is a synthetic smoke test rather than an end-to-end stack release gate.
- **Root Cause**: Reliance on nominal alias strings rather than content-addressed artifact digests.
- **Remediation**:
  1. Update `RunManifest` to query Hestia's `/v1/models` and record:
     - `weights_sha256`: Digest of model GGUF/checkpoint file.
     - `tokenizer_sha256`: Digest of tokenizer model/vocab.
     - `prompt_template_sha256`: Digest of system prompt formatting template.
     - `code_git_commit`: Full 40-character commit hash of active repository code and `dirty_tree` boolean.
  2. In release gate evaluations, require complete model identity matches.
  3. Clearly label synthetic fixtures (e.g. ARC-25) as smoke benchmarks, keeping them distinct from full held-out dataset release gates.
- **Regression Testing**:
  - Test `test_manifest_captures_served_model_digests`: Run benchmark and assert manifest contains non-empty cryptographic digests.
  - Test `test_mismatched_model_digest_invalidates_comparison`: Comparing two runs with identical aliases but different weights digests raises `ModelIdentityMismatchError`.
