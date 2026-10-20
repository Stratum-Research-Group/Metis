# Phylax (Φύλαξ)

**Phylax** (Greek: *guardian* / *watcher*) is the local policy engine, capability authorization gateway, and security guardrail for the Metis ecosystem.

It provides point-of-action capability mediation for files, tools, system processes, and network operations, verifying that callers possess authorized permissions according to hand-written, human-auditable YAML policies with default-deny semantics.

---

## 1. Product Outcome & Threat Model Summary

- **Point-of-Action Capability Gateway**: `GuardedGateway`, `GuardedFileAdapter`, `GuardedToolAdapter`, `GuardedSendAdapter`. Permissions are checked and operations executed atomically inside the guarded gateway.
- **Action-Bound Single-Use Approvals**: Sensitive operations (`ASK` effect) emit single-use cryptographic approval handles. Once approved, execution transitions state atomically from `APPROVED` to `EXECUTED`. Token replay, double-execution, or concurrent reuse are strictly impossible.
- **Anti-Argument-Swap Verification**: Approvals bind to the exact cryptographic digest of their arguments (`argument_digest = sha256(canonical_json(args))`). Any attempt to alter arguments between owner approval and execution triggers immediate abort with `DENY`.
- **Fail-Closed Audit Hash Chain**: Append-only cryptographic hash chain where each entry links `record_hash = sha256(prev_hash + record_fields)`. Authorized execution intent is logged durably before any side effect is invoked; if intent logging fails, execution aborts immediately. If post-execution receipt logging fails, the action is surfaced as `executed/audit-unacknowledged` to strictly prevent unsafe retries.
- **Symlink & Path Traversal Mitigations**: File access uses `O_NOFOLLOW`, descriptor regular-file verification (`S_ISREG`), dual device/inode (`st_dev`/`st_ino`) validation, unpredictable exclusive temporary files (`O_EXCL`), and atomic renames to mitigate leaf symlink swaps and path traversal (`../`) attacks within configured roots (library-level defense-in-depth, not a kernel namespace sandbox).

---

## 2. Policy Specification & Default Deny

Policies are written strictly in YAML and validated against a schema with `extra="forbid"`. No executable code is permitted.

### Policy Structure (`policies/default_policy.yaml`)
```yaml
version: "1.0.0"
owner: "metis-owner"
description: "Default least-privilege policy"

redaction:
  redact_content: true
  mask_patterns: ["token", "secret", "password", "api_key", "pin"]
  mask_replacement: "[REDACTED]"

rules:
  # Allow Kybernetes to read public docs
  - id: "kybernetes_read_docs"
    subject: "kybernetes"
    verb: "read"
    resource:
      class: "file"
      path_root: "docs"
      path_glob: "**/*"
    effect: "allow"
    priority: 10

  # Require owner approval for outbound network sends
  - id: "ask_outbound_send"
    subject: "*"
    verb: "send"
    resource:
      class: "network"
      destination: "*"
    effect: "ask"
    priority: 20
    ask_ttl_seconds: 300

  # Explicit mandatory default deny
  - id: "default_deny_all"
    subject: "*"
    verb: "*"
    resource:
      class: "*"
      path_root: "*"
    effect: "deny"
    priority: 1000
```

### Static Policy Linting
The linter detects:
- Syntax and schema errors
- Unknown verbs or resource classes
- Duplicate rule IDs
- Shadowed / unreachable rules
- Path traversal sequences (`..`) in roots
- Missing final default-deny rule

```bash
phylax validate policies/default_policy.yaml
```

---

## 3. Approval Lifecycle & State Machine

```
[Request Operation]
        │
        ├── Rule Matches 'allow' ──► [Execute Atomically & Audit Log]
        │
        ├── Rule Matches 'deny'  ──► [Abort Operation & Audit Log]
        │
        └── Rule Matches 'ask'   ──► [Generate Approval Handle]
                                            │
                                            ▼
                                     State: PENDING (with TTL)
                                            │
                        ┌───────────────────┴───────────────────┐
                        │                                       │
                  [User Approves]                        [User Denies / TTL Expires]
                        │                                       │
                        ▼                                       ▼
                 State: APPROVED                         State: DENIED / EXPIRED
                        │                                       │
                [Execute Gateway]                               ▼
                        │                               [Abort Execution]
                        ▼
                 State: EXECUTED (Consumed)
                        │
                [Replay Attempt -> DENY]
```

---

## 4. Integration Examples

### Kybernetes Orchestrator Shim
```python
from phylax.engine.manager import PolicyManager
from phylax.audit.logger import AuditLogger
from phylax.gateway.gateway import GuardedGateway
from phylax.adapters.kybernetes import KybernetesShim

# Initialize gateway
policy_mgr = PolicyManager("policies/default_policy.yaml")
audit_logger = AuditLogger(".phylax/audit.jsonl")
gateway = GuardedGateway(policy_mgr, audit_logger)
kybernetes = KybernetesShim(gateway)

# Read file through point-of-action check with O_NOFOLLOW / fstat protection
content = kybernetes.read_file("docs/manual.md")

# Outbound send (triggers ASK if configured)
def send_http(**payload):
    ...

try:
    kybernetes.send("https://api.external/notify", {"msg": "hello"}, send_http)
except PermissionError as exc:
    print(f"Approval required: {exc}")
```

### Prosopon Approval Integration
Pending requests conform to Prosopon's `ApprovalCard` contract:
```python
from phylax.adapters.prosopon import ProsoponApprovalAdapter

adapter = ProsoponApprovalAdapter(gateway.approval_manager)
cards = adapter.get_pending_cards()

# Owner approves in Prosopon UI
adapter.resolve("appr-12345", approved=True, reason="Owner approved")
```

---

## 5. CLI Commands

| Command | Description |
| :--- | :--- |
| `phylax doctor` | Health check on policy files, audit chain, and submodules |
| `phylax validate <path>` | Lint policy YAML for errors, warnings, and shadowed rules |
| `phylax decide --subject ... --verb ... --target ...` | Test authorization decision from CLI and log to audit |
| `phylax audit verify` | Cryptographically verify the append-only hash chain |
| `phylax audit list` | View audit entries with filters (`--subject`, `--effect`, `--limit`) |
| `phylax approvals list` | List pending approvals awaiting owner confirmation |
| `phylax approvals approve <id>` | Approve a pending request |
| `phylax approvals reject <id>` | Reject a pending request |

---

## 6. Running Tests

```bash
uv run pytest phylax/tests -v --cov=phylax
```
All tests validate contract schemas, static linter warnings, symlink containment, deny-precedence, file-level atomic swap protections (see [THREAT_MODEL.md §4.2](THREAT_MODEL.md#42) for residual parent-directory race considerations), argument digests, single-use token consumption, fail-closed audit writing, and CLI execution.
