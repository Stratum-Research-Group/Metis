# Phylax Threat Model & Operation Inventory

## 1. System Overview & Trust Model

Phylax (Greek: *guardian* / *watcher*) is the local policy engine and guarded capability gateway for the Metis ecosystem. It mediates access to local files, external tools, system command execution, and outbound network communication across all Metis stack components.

### 1.1 Enforcement Boundary & OS Sandboxing Disclaimer
> [!IMPORTANT]
> **Trust Boundary Notice**: Phylax is an application-level capability gateway and policy mediator, implemented in Python. It evaluates permissions at the point of action when callers invoke guarded adapters (`authorize_and_execute`, `GuardedFileAdapter`, `GuardedToolAdapter`, `GuardedSendAdapter`).
>
> Phylax **does NOT** provide OS-level kernel sandboxing (e.g. `seccomp-bpf`, `landlock`, Linux network/mount namespaces, cgroups, or separate UID/GID isolation). If an untrusted or hostile native binary, C extension, or uninstrumented Python script executes in the same process space or with identical user privileges, it can bypass in-memory Python gateways by calling libc syscalls directly (`open`, `execve`, `socket`).
>
> For third-party untrusted plugins, sandboxing **must** be enforced at the OS level (e.g. using `bubblewrap`, `firejail`, Docker containers, or unprivileged child processes). Phylax provides the declarative authorization, TOCTOU-resistant capability adapters, and fail-closed audit log for the cooperative Metis components.

---

## 2. Component Operation Inventory

| Component | Subject ID | Role & Delegated Authority | Verbs | Target Resources | Risk Level |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stoma** | `stoma` | Ingestion & document indexer | `read`, `write` | Documents dir, raw files, vector indices | Medium (file reading, index overwrite) |
| **LTHCM** | `lthcm` | Long-term hierarchical memory | `read`, `write` | Memory graphs, embeddings, checkpoints | Medium (memory leakage/poisoning) |
| **Dianoia** | `dianoia` | Deliberation engine & walk logs | `read`, `write`, `execute` | Reasoning graphs, local LLM calls | High (hallucinated execution) |
| **Hestia** | `hestia` | Model registry & inference runner | `read`, `execute` | Weights dir, llama.cpp / vLLM runtime | High (model weights manipulation) |
| **Kybernetes**| `kybernetes` | Autonomous agent orchestrator | `read`, `write`, `execute`, `send` | File search, tool invocations, outbound dispatch | Critical (broadest capability set) |
| **Prosopon** | `prosopon` | Web/Voice user interaction gateway | `read`, `send` | UI assets, TTS/STT sockets, SSE streams | High (user prompt injection/exposure) |
| **Dokime** | `dokime` | Evaluation & benchmarking runner | `read`, `write`, `execute` | Test suites, metric dumps, reports | Medium (benchmark code execution) |
| **Paideia** | `paideia` | Self-improvement & QLoRA trainer | `read`, `write`, `execute` | Walk logs, checkpoints, GPU device locks | High (GPU compute exhaustion, export) |

---

## 3. Resource & Verb Taxonomy

### 3.1 Verbs
- `read`: Reading local file contents, directory listings, or internal memory states.
- `write`: Writing, creating, appending, or deleting local files and directories.
- `execute`: Invoking local CLI tools, child processes, or internal model inference runners.
- `send`: Outbound network communication (HTTP requests, webhooks, sockets, SMS/email).

### 3.2 Resource Classes
- `file`: Local filesystem paths, validated against canonical roots and globs.
- `tool`: Kybernetes registered tools (e.g. `web_search`, `calculate`, `calendar`).
- `network`: Outbound network destinations (URLs, domain names, IP addresses, ports).
- `system`: System-level commands and hardware interfaces (e.g. GPU, shell execution).

### 3.3 Separation of User Content vs. Tool Instructions
To prevent Prompt Injection (Indirect / Direct), untrusted user text or ingested documents must never be evaluated as policy conditions or executable tool names. Tool names and actions are strictly constrained by the policy YAML schema.

---

## 4. Specific Threats & Phylax Mitigations

### 4.1 Path Traversal & Symlink Attacks
- **Threat**: Caller requests access to `/workspace/docs/../../etc/shadow` or creates a symlink inside `/workspace/docs/symlink` pointing to `/home/owner/.ssh/id_rsa`.
- **Phylax Mitigation**:
  1. `Path.resolve()` canonicalization removes relative `..` segments and resolves symlinks prior to rule evaluation.
  2. Canonical path root validation ensures the final resolved target resides strictly within an authorized base directory.
  3. Symlinks pointing outside permitted roots are denied immediately.

### 4.2 Time-Of-Check to Time-Of-Use (TOCTOU) File Race
- **Threat**: An attacker changes a checked regular file into a symlink to a sensitive file immediately after `decide()` returns `ALLOW` but before `open()` executes.
- **Phylax Mitigation**:
  1. Guarded capability gateway: callers cannot open files directly using the decision token; they must use `GuardedFileAdapter.read_text()` or `GuardedFileAdapter.write_text()`.
  2. POSIX flag `os.O_NOFOLLOW`: File descriptors are opened without following trailing symlinks.
  3. Descriptor inode verification: `fstat` is run on the opened file descriptor and cross-referenced with `stat` on the target path to guarantee identity.
  4. Residual race window documentation: Renaming of parent directories during traversal cannot be fully prevented in pure user-space without kernel namespaces; directory hierarchies must have owner-only write permissions.

### 4.3 Argument Swapping & Replay Attacks
- **Threat**: Caller requests approval for `send` with payload `{"to": "alice", "msg": "hello"}` (`ASK`). Owner approves in Prosopon. Before execution, caller swaps payload to `{"to": "eve", "msg": "exfiltrate private keys"}`.
- **Phylax Mitigation**:
  1. Every `ASK` approval handle includes a deterministic SHA-256 digest of canonicalized arguments (`argument_digest = sha256(json.dumps(args, sort_keys=True))`).
  2. At execution time, `GuardedGateway` recalculates the digest of actual arguments. If they differ by even 1 bit, execution is denied.
  3. Single-use nonce: Approval is consumed atomically upon execution. Attempting to replay the approval token fails with `APPROVAL_ALREADY_CONSUMED`.

### 4.4 Policy Concurrency & Atomicity
- **Threat**: Policy is reloaded during an ongoing multi-step operation, resulting in inconsistent decisions.
- **Phylax Mitigation**:
  1. In-memory policy document is an immutable Pydantic model swapped via an atomic reference update.
  2. Evaluator copies the active `policy_version` and pins it to the execution receipt and approval handle.

### 4.5 Audit Tampering & Fail-Closed Guarantee
- **Threat**: Adversary deletes or edits audit logs to hide unauthorized actions, or induces an I/O error (e.g. disk quota or read-only filesystem) to execute actions silently.
- **Phylax Mitigation**:
  1. Append-only cryptographic hash chain: each audit entry contains `record_hash = sha256(prev_hash + record_fields)`. The genesis record starts with a known seed. The CLI provides `phylax audit verify` to validate chain continuity.
  2. **Fail-Closed Gate**: If the audit logger encounters an exception while persisting the decision record (e.g. disk full, permission denied), the operation is aborted and returns `DENY`. Under no circumstances is an un-audited action allowed to proceed.
  3. Sensitive Data Redaction: Confidential keys, passwords, and raw file contents are masked in the audit log according to the policy's redaction configuration.
