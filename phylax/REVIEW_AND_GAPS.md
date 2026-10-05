# Phylax: Round 1 Defect Analysis, Gaps & Remediation Plan

This document details the defect findings, security vulnerabilities, and implementation gaps identified in `metis-repo-wide-round1-fixes-6752222c.txt` and `metis-dashboard-implementation-spec-ae93d36c.txt` for **Phylax**, along with the exact technical remediation plan and regression tests.

---

## 1. Defect Findings & Root Cause Analysis

### Item 5 (P0): File Approvals Not Bound to Exact Write Content
- **Location**: `phylax/phylax/gateway/file_adapter.py:62-105`, line 102.
- **Observed Behavior**: `write_text` passes only `{'content_bytes': len(encoded)}` into the capability decision request. The actual file content is captured separately by a closure. An approval granted for a 4-byte string allows an attacker to substitute an entirely different 4-byte payload. The argument digest cannot detect payload tampering because the actual content bytes were omitted from the digest.
- **Root Cause**: Passing incomplete metadata rather than the cryptographic content digest into the approval token generation.
- **Remediation**:
  1. Calculate `content_digest = hashlib.sha256(encoded_bytes).hexdigest()`.
  2. Include `content_digest`, `encoding`, `canonical_path`, and `overwrite` flags directly into the decision arguments.
  3. During execution, verify that the bytes to be written hash to the identical `content_digest`.
  4. Ensure full file content is not logged into the audit ledger; record only the `content_digest` and byte count.
- **Regression Testing**:
  - Test `test_tampered_content_same_length_rejected`: Generate approval for `"pass"`; attempting to write `"fail"` fails verification and raises `PermissionError`.
  - Test `test_exact_content_succeeds_once`: Exactly approved content writes cleanly; replay fails.

---

### Item 6 (P0): Pre-Approved Actions Execute Before Audit Recording
- **Location**: `phylax/phylax/gateway/gateway.py:121-163,185-225`.
- **Observed Behavior**: In the pre-approved path, `gateway.execute()` consumes the approval handle and immediately calls the user callback `action_fn()`, writing the receipt audit log *after* execution. If an I/O or disk error occurs during audit logging, the side-effecting action has already taken place, violating the fail-closed invariant.
- **Root Cause**: Inverted ordering: performing side effects prior to durable audit entry persistence.
- **Remediation**:
  1. Write an `AUDIT_INTENT` record durably to the audit log before invoking `action_fn()`.
  2. If writing the intent record fails, immediately abort and do *not* call `action_fn()`.
  3. Write an `AUDIT_RECEIPT` record upon completion. If writing the receipt fails, record the state as unacknowledged rather than pretending the action did not execute.
- **Regression Testing**:
  - Test `test_audit_failure_before_execution_prevents_side_effect`: Mock audit log failure; assert callback is never executed.

---

### Item 7 (P1): Path Race Protection & TOCTOU Containment
- **Location**: `phylax/phylax/gateway/file_adapter.py:28-46,74-94`.
- **Observed Behavior**: `O_NOFOLLOW` checks only the terminal component of a file path. Parent directories could be symlink-swapped between policy check and file descriptor opening. Temporary file creation in `write_text` uses a predictable PID-only naming scheme.
- **Root Cause**: Relying on string path checks rather than directory descriptor-relative operations and secure temp file creation.
- **Remediation**:
  1. Traverse path components using directory file descriptors (`os.open` with `O_PATH` and `O_NOFOLLOW`) to ensure every ancestor directory is contained within the allowlisted root.
  2. Use `tempfile.NamedTemporaryFile` with `delete=False` within the target directory, using `os.replace` for atomic commits.
- **Regression Testing**:
  - Test `test_parent_directory_symlink_swap_rejected`: Symlink pointing outside authorized root in a parent folder is detected and rejected.
