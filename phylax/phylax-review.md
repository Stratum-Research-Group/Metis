# Phylax: permissions layer review and improved build brief

Reviewed 2026-09-29. Sources: both `7 phylax*.txt` prompts and the supplied stack documents. This is a design review, not an audit of an existing Phylax implementation.

## Verdict

The prompt states default deny but also says enforcement is entirely the caller's responsibility. That creates a useful policy library, **not a reliable security boundary**. Make the trust model explicit and put enforcement at the actual file/tool/network operation; use OS permissions or sandboxing when an untrusted component can bypass the wrapper.

## Highest-impact gaps and changes

| Priority | Gap in the prompt | Concrete change | Acceptance gate |
| --- | --- | --- | --- |
| P0 | A Kybernetes preflight can be skipped by another caller or bypassed by a tool that opens files itself. | Put a mandatory reference monitor in each capability adapter, with a common `authorize_and_execute` path. Document what remains outside the boundary; add OS isolation for untrusted code. | A direct tool call without authorization is rejected in integration tests. |
| P0 | First-match YAML rules can allow a broad path before a later deny. Path globs do not handle symlinks, relative paths, `..`, file replacement, or nested tool actions. | Use explicit priority with deny precedence, canonical resource identities and permitted roots, rule linting for shadowed/conflicting rules, and handle-based or post-open validation where possible. | Tests cover traversal, symlink escape, renamed files, and broad-allow/narrow-deny ordering. |
| P0 | `ASK` has no binding to the exact operation being approved. | Pending request includes subject identity, verb, canonical target, argument digest, policy version, expiry, one-use nonce, and session/user scope. Approval must be consumed atomically immediately before execution. | Changed arguments or expired/replayed approval are denied; timeout is logged as deny. |
| P0 | “Every decision logged” can leak private paths and message content, and a failed log write can erase auditability. | Define append-only structured audit with correlation ID, decision, rule ID/version, safe resource representation, and redaction; specify fail-closed behavior for required audit writes. | Every executed sensitive operation has one traceable decision and log-write failure blocks it. |
| P1 | Subject names are caller-supplied strings with no identity or delegation model. | Bind subjects to authenticated process/component identity; model delegated calls so a tool cannot gain Kybernetes's full authority. | A lower-privilege plugin cannot impersonate another component or expand its rights. |
| P1 | Rules cannot distinguish local read, index, train, export, delete, remote send, or model inference with sensitive content. | Define typed verbs/resource classes and data labels; propagate source restrictions through Stoma passages, LTHCM retrieval, logs, Paideia datasets, and Prosopon display. | A restricted source remains restricted in retrieval, training, and UI flows. |
| P1 | Policy edits and concurrent requests lack semantics. | Validate schema before activation; switch atomically to an immutable policy version; record that version with every decision; revoke outstanding asks on rule changes where necessary. | Requests cannot mix two policy versions within an operation. |

## Revised build order

1. Write a threat model and inventory operations across Stoma, LTHCM, Dianoia, Hestia, Kybernetes, and Prosopon. Define which operations Phylax can actually enforce.
2. Specify a typed policy schema, conflict resolution, canonicalization, subject identity, and audit schema. Ship a policy linter and a default-deny sample.
3. Implement the reference monitor and one guarded adapter for local file search and one outbound/action adapter. Verify nested calls and approval replay handling.
4. Add the Prosopon approval UI only after pending-request identity and expiry are stable. Keep the CLI approval path for headless use.
5. Test bypasses and denial modes; document where OS sandboxing is required before accepting untrusted plugins.

## First milestone

The three-request demo should include a fourth case: a tool tries to read through a symlink or with a changed approved argument, and the operation is blocked. The audit must link authorization to execution, not only to an earlier policy lookup.

See [stack integration plan](stack-integration.md) for the shared capability and data-handling contracts.
