# Phylax

**Phylax** (Greek: *guardian* / *watcher*) is the local policy engine and guarded capability gateway for the Metis ecosystem.

It provides point-of-action capability mediation for files, tools, system processes, and network operations, verifying that callers possess authorized permissions according to hand-written, human-auditable YAML policies with default-deny semantics.

---

## 1. Product Outcome & Threat Model Summary

- **Enforcement Model**: Point-of-action capability gateway (`GuardedGateway`, `GuardedFileAdapter`, `GuardedToolAdapter`, `GuardedSendAdapter`). Permissions are checked and operations executed atomically inside the guarded gateway.
- **Fail-Closed Audit**: Append-only cryptographic hash chain where each entry links `record_hash = sha256(prev_hash + record_fields)`. Any failure to flush to the audit log immediately aborts the operation and returns `DENY`.
- **Approval Lifecycle**: `ASK` decisions produce single-use, cryptographically bound approval handles (`argument_digest = sha256(args)`). Expired approvals, replays, or argument-tampering attempts are strictly denied.
- **Trust Boundary**: Phylax is an in-process Python capability gateway. It does not provide kernel sandboxing. Untrusted plugins must be isolated using OS-level mechanisms (e.g., `bubblewrap`, containers, or unprivileged UIDs).

For complete threat analysis and operation inventory across all Metis components (Stoma, LTHCM, Dianoia, Hestia, Kybernetes, Prosopon, Dokime, Paideia), see [THREAT_MODEL.md](file:///home/akarta/Projects/Metis/phylax/THREAT_MODEL.md).

---

## 2. Installation & Quickstart

Phylax requires Python `>=3.11` and uses `pydantic`, `pyyaml`, `click`, and `tabulate`.

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

Verify your environment using the Phylax CLI:
```bash
phylax doctor
```

---

## 3. Policy Specification & Linter

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

## 4. Guarded Capabilities Usage

### 4.1 Kybernetes Orchestrator Shim
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

### 4.2 Prosopon Approval Integration
Pending requests conform to Prosopon's `ApprovalCard` contract:
```python
from phylax.adapters.prosopon import ProsoponApprovalAdapter

adapter = ProsoponApprovalAdapter(gateway.approval_manager)
cards = adapter.get_pending_cards()
# [{ "approval_id": "appr-...", "turn_id": "...", "action": "send", ... }]

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

## 6. End-to-End Demo

Run the automated live demonstration:
```bash
python demo.py
```

The script demonstrates:
1. **Allowed docs read**: Safe file access within authorized directories.
2. **Denied private read**: Blocked file access into sensitive folders.
3. **Approval flow**: Outbound message triggers `ASK`, owner approves via Prosopon/CLI adapter, executed once, replay denied.
4. **Timeout denial**: Unanswered approval times out to `DENY` after TTL expires.
5. **Anti-Argument-Swap**: Arguments altered before execution trigger SHA-256 digest mismatch denial.
6. **Audit log verification**: Cryptographic verification of hash chain and redacted secrets.

---

## 7. Test Suite

Run unit and integration tests with coverage:
```bash
pytest --cov=phylax
```
All 32 tests validate contract schemas, static linter warnings, symlink containment, deny-precedence, TOCTOU races, argument digests, fail-closed audit writing, and CLI execution.
