# Build Phylax — production-grade local policy and guarded capabilities

You are building **Phylax from scratch** as a new Python project. Read `STACK_BUILD_CONTRACT.md`, the Phylax v2 `.txt` brief and the existing stack docs. Deliver a complete, runnable first release with an explicit threat model. Preserve the owner's hand-written YAML policy and default-deny rule. Do not claim OS sandboxing when only a library decision is being made. Make routine choices independently and document any dependency/API mismatch.

## Product outcome and trust model

Phylax decides whether a named component may perform an exact local read/write/execute or outbound/send operation, may perform it after owner approval, or must be denied. The trusted execution path uses a **guarded capability gateway** that authorizes at the point of action. The policy engine alone cannot stop a component that bypasses the gateway; document that limit and provide OS permissions/sandbox guidance for untrusted plugins.

## Non-negotiable constraints

- Policy is data in validated YAML, never executable policy code. Unknown keys, unsupported verbs and ambiguous rules fail startup. Default deny is explicit and cannot be overridden accidentally.
- Add read-only LTHCM and Dianoia submodules at the exact v2 commits; verify pins without modifying or copying their contents.
- Every attempted decision is auditable with time, subject, verb, canonical resource, matched rule ID, policy version, effect, trace ID and execution receipt. Redact sensitive names/content according to a documented mode; required audit failure denies the action.
- `ASK` expires to deny. Approval is one-use and bound to the exact subject, verb, canonical target, argument digest, policy version and session. No blanket approval from a nearby or earlier request.
- Do not permit direct file/tool/network operations through Phylax adapters without an authorization decision. Clarify that standalone code outside those adapters is outside Phylax's enforcement boundary.

## Build in this order

### 1. Threat model and operation inventory

Enumerate Stoma ingestion/index reads/writes, LTHCM retrieval, Dianoia model calls/logs, Hestia management/inference, Kybernetes tools, Prosopon approvals, Dokime artifacts and Paideia training/export. Define subjects, trusted identities, delegated authority and resources. Separate user content from tool instructions. Include symlink, path traversal, argument swap, policy race, replay and audit tampering threats.

### 2. Typed policy schema and linter

Define rule IDs, subject identities, verbs, resource classes, path roots/globs or tool IDs, conditions, effect, optional approval TTL and explicit priority. Choose and document conflict semantics; deny must win over a conflicting broad allow, and the linter must reveal shadowed/unreachable rules. Canonicalize paths and validate roots before matching; do not treat string-prefix path checks as containment. Keep a final default-deny rule. Provide a reference policy with least privilege for the demo, not a permissive catch-all.

### 3. Evaluator and guarded executor

Expose typed `decide` and `authorize_and_execute` APIs. Bind subject identity from a trusted caller context, not a caller-supplied free-text string. For a file operation, validate the resolved target and protect against symlink/rename races as far as the host API permits; document residual race conditions. Tool adapters must declare nested capabilities and cannot inherit unrestricted Kybernetes authority. Load policy versions atomically so one operation cannot mix rules.

### 4. Approval lifecycle

Return a pending handle with exact action summary, rule, expiry, nonce and argument digest. Store it durably enough for a UI/CLI decision; consume approval atomically just before execution. Deny on timeout, denial, changed arguments, changed policy where revocation applies, duplicate use or lost owner session. Provide an interface adapter for Prosopon and a headless CLI flow. Never turn an unanswered request into allow.

### 5. Audit and operations

Use a structured append-only local audit with rotation/retention, tamper-evidence where practical, correlation to Kybernetes traces and an execution outcome. Provide CLI filters by time, component, effect, resource and rule; include policy validate/doctor commands. Avoid logging raw private file contents or unredacted send payloads by default. Document backup, policy edit and rollback procedures.

## Required tests and demo

- Policy tests: default deny, broad allow plus narrow deny, unknown/duplicate rules, canonical path, symlink escape, relative path, `..`, tool/resource class mismatch, policy hot reload, concurrency and audit-write failure.
- Approval tests: timeout, replay, changed arguments, changed target, changed policy, cross-session use, double-click and crash/restart.
- Integration tests: one Kybernetes local file search adapter, one outbound/send adapter, and one direct-adapter bypass attempt. Verify exact action is executed once and each attempt has a decision/execution trace.
- Demo: allowed docs read, denied private read, send requiring approval, an unanswered send that times out to deny, and an approved request whose arguments change before execution and is therefore denied. Show the audit output and matched rules.

## Definition of done

A clean checkout installs, validates a YAML policy, denies unknown operations, protects the guarded adapters, resolves approvals correctly, survives restart without replaying an action, and produces a redacted, traceable audit. The handoff states exactly which operations are mediated and which require OS-level isolation. No “secure” claim rests on a preflight lookup alone.
