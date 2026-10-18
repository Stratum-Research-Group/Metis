# Metis Sovereign Stack (MSS) & Autonomous Coding Agent

**Document ID:** `METIS-DOCS-MSS-001`  
**File:** `docs/MSS.md`  
**Status:** Canonical User Manual & Operational Reference  
**Subsystems:** `kybernetes` (Coding Agent, Lochos, Pinax), `phylax` (Guardrails), `hestia` (Inference), `prosopon` (Workbench & UI)

---

## 1. Executive Overview

The **Metis Sovereign Stack (MSS)** is an autonomous, local-first coding environment and multi-agent execution framework built into the Metis ecosystem. Designed to operate **100% offline with zero external API dependencies, telemetry, or recurring cloud costs**, MSS transforms local Large Language Models (LLMs) into reliable, self-verifying software engineers.

Rather than relying on naive prompt loops or unverified diffs, MSS synthesizes the proven architectural principles of the industry's premier coding harnesses:
- **Pi Agent**: A minimalist, high-signal tool surface (`read_file_range`, `edit_file_block`, `write_file`, `sandboxed_bash`) preserving $>95\%$ of context tokens for source code and compiler outputs.
- **Nous Research Hermes Agent**: Dynamic, self-evolving skills written to `.metis/skills/<domain>/SKILL.md` upon passing tests and automatically indexed for subsequent runs.
- **Odyssey**: Git milestone state checkpointing and automatic rollback upon test failure, completely preventing runaway workspace corruption.
- **Aider / SWE-agent**: Exact-anchor search/replace block editing with FNV-1a hashline validation and in-process AST syntax diagnostics, eliminating broken unified patch failures.
- **Metis Native Superpowers**: Action-bound single-use cryptographic approvals via **Phylax**, multi-agent squad orchestration via **Lochos**, and system tool discovery via **Pinax**.

---

## 2. Core Architecture

```
                                    +-----------------------------+
                                    |        Operator / CLI       |
                                    +-----------------------------+
                                                   |
                                                   v
+---------------------------------------------------------------------------------------------------+
| Kybernetes Engine                                                                                 |
|                                                                                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | TerminalCodingAgent (kybernetes code)                                                       |  |
|  |                                                                                             |  |
|  |  +--------------------+   +-----------------------+   +----------------------------------+  |  |
|  |  | DynamicSkillLearner|   |GitMilestoneCheckpointer|  | AgentStateMachine                |  |  |
|  |  | (.metis/skills)    |   | (Pre-turn git commit) |  | (IDLE -> ACTING -> VERIFYING)    |  |  |
|  |  +--------------------+   +-----------------------+   +----------------------------------+  |  |
|  |                                                                                             |  |
|  |  +---------------------------------------------------------------------------------------+  |  |
|  |  | ToolRegistry & Sandboxed Coding Tools                                                 |  |  |
|  |  |  - read_file_range   (1-indexed, hashline anchors)                                    |  |  |
|  |  |  - edit_file_block   (Exact match, AST syntax check, unified diff preview)            |  |  |
|  |  |  - write_file        (Atomic write with overwrite confirmation)                       |  |  |
|  |  |  - sandboxed_bash    (Path-restricted, timeout-bounded shell runner)                  |  |  |
|  |  |  - code_mode         (In-process AST sandboxed batch Python script executor)          |  |  |
|  |  |  - file_search       (Scoped workspace file search)                                   |  |  |
|  |  |  - calculator        (AST-whitelisted safe math evaluator)                           |  |  |
|  |  |  - catalogue_lookup  (Pinax Level 0/1 tool catalogue search)                         |  |  |
|  |  |  - catalogue_docs    (Pinax Level 2 size-capped tool documentation slicer)            |  |  |
|  |  +---------------------------------------------------------------------------------------+  |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                                                                   |
|  +------------------------------------+          +---------------------------------------------+  |
|  | Lochos Multi-Agent Squad           |          | Pinax Tool Catalogue Subsystem              |  |
|  |  - Durable Queue (lochos.sqlite)   |          |  - Passive PATH Scanner (No execution)      |  |
|  |  - Agent Seats (coder, reviewer)   |          |  - SQLite FTS5 Index & Category Classifier  |  |
|  |  - Baton Handover State Machine    |          |  - Risk Tier Classifier (T0 to T3)          |  |
|  |  - Attention Blocks & Resolutions  |          |  - Policy Override Engine & Probes          |  |
|  +------------------------------------+          +---------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
             |                                              |                             |
             v                                              v                             v
+--------------------------+                   +--------------------------+  +--------------------------+
| Hestia Model Server      |                   | Phylax Capability Gateway|  | Local Git Workspace      |
| (Local OpenAI /v1 API)   |                   | (Action-bound approvals) |  | (.git snapshot rollback) |
+--------------------------+                   +--------------------------+  +--------------------------+
```

---

## 3. Configuration & Settings Reference

MSS provides granular configuration via CLI options, environment variables, configuration files, and the Prosopon GUI Settings Panel.

### 3.1 CLI Options (`kybernetes code`)

| Option | Short Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--goal` | `-g` | String | **Required** | The engineering goal, feature request, or bug description to resolve. |
| `--workspace` | `-w` | Path | `.` (CWD) | Absolute or relative path to the target workspace root directory. |
| `--model` | `-m` | String | *From Hestia* | LLM model identifier on the Hestia server (e.g., `qwen2.5-coder-32b-instruct`). |
| `--base-url` | `-u` | String | `http://127.0.0.1:8080/v1` | Base URL for the Hestia OpenAI-compatible endpoint. |
| `--temperature` | | Float | `0.15` | Sampling temperature for code generation. Lower values provide more deterministic output. |
| `--test-cmd` | `-t` | String | `None` | Automated verification command (e.g. `pytest tests/`). **Triggers git rollback on failure.** |
| `--max-steps` | `-s` | Integer | `20` | Maximum autonomous turns before halting. |
| `--auto-approve` | `-y` | Flag | `False` | Auto-approve mutating file/shell operations without interactive prompt cards. |

### 3.2 Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HESTIA_BASE_URL` | `http://127.0.0.1:8080/v1` | Endpoint URL for the local Hestia model server. |
| `HESTIA_DEFAULT_MODEL` | `llama-3.1-8b-instruct` | Fallback model name when not specified via `--model`. |
| `HESTIA_SERVICE_TOKEN` | `None` | Optional Bearer token for authenticated Hestia endpoints. |
| `PHYLAX_ALLOWED_ROOTS` | Current Workspace | Colon-separated directory paths permitted for file access. |
| `LOCHOS_DB_PATH` | `lochos.sqlite` | Path to the SQLite database storing the Lochos work queue and seat state. |
| `PINAX_INDEX_PATH` | `~/.metis/pinax_catalogue.db` | Path to the SQLite FTS5 database storing the Pinax tool catalogue. |

### 3.3 Prosopon Workbench Settings Panel

In the Prosopon Web Workbench (`http://127.0.0.1:8000`), pressing `Ctrl+,` opens the centralized Settings Panel containing categories aligned with the MSS runtime:
- **General & Workbench**: Theme (`theme-obsidian`, `theme-linear`), editor font size, minimap, and breadcrumbs.
- **Hestia (Local Models)**: Active coding model, context length (4k–128k tokens), VRAM budget limit (MB), and idle unload timeout.
- **Kybernetes (Engine & Budget)**: Kalman filter budget smoothing $\alpha$, tool timeout bounds, max Dianoia tree depth, and milestone checkpoint toggle.
- **Phylax (Guardrails & Policies)**: Strict default-deny toggle, approval token TTL (seconds), auto-allow safe reads, and cryptographic audit hash-chaining.
- **Memory (LTHCM & Stoma)**: Light-cone horizon (seconds), Poincaré manifold dimension (default 384), and Stoma AST chunk token bound ($\le 254$).

---

## 4. The Tool Surface

All agent operations are mediated through the `ToolRegistry` with strict root bounding:

### 1. `read_file_range`
Reads lines from a file with 1-indexed line numbers and line-level FNV-1a hashline anchors:
```json
{
  "tool_name": "read_file_range",
  "arguments": {
    "path": "kybernetes/engine/terminal_agent.py",
    "start_line": 80,
    "end_line": 120
  }
}
```
*Output format*: `line_num:hash| content`. Hashline anchors ensure subsequent edits reference the exact lines inspected.

### 2. `edit_file_block`
Precisely replaces a search block with a replacement block:
```json
{
  "tool_name": "edit_file_block",
  "arguments": {
    "path": "kybernetes/engine/terminal_agent.py",
    "search_block": "temperature: float = 0.2",
    "replace_block": "temperature: float = 0.15",
    "allow_syntax_errors": false
  }
}
```
*Invariants*:
- Performs an in-process AST syntax check on Python files. Rejects invalid syntax with detailed line/column diagnostic messages before writing to disk.
- Renders an interactive colorized unified diff preview in the terminal before applying changes.

### 3. `write_file`
Atomically creates a new file or completely replaces an existing file:
```json
{
  "tool_name": "write_file",
  "arguments": {
    "path": "docs/NEW_FEATURE.md",
    "content": "# New Feature\nDocumentation...",
    "overwrite": true
  }
}
```

### 4. `sandboxed_bash`
Executes non-interactive shell commands confined to the workspace root:
```json
{
  "tool_name": "sandboxed_bash",
  "arguments": {
    "command": "pytest tests/test_agent.py -v",
    "timeout_seconds": 30
  }
}
```
*Guards*:
- Prevents commands attempting directory traversal or modifying files outside permitted roots.
- Head and tail truncation bounds prevent runaway terminal output from exhausting context windows.

### 5. `code_mode`
Executes in-process sandboxed Python scripts for high-speed multi-file analysis:
```json
{
  "tool_name": "code_mode",
  "arguments": {
    "script": "import glob\nfiles = glob.glob('**/*.py', recursive=True)\nprint(f'Total python files: {len(files)}')",
    "timeout_seconds": 5.0
  }
}
```

### 6. `catalogue_lookup` & `catalogue_docs` (Pinax Integration)
Enables the coding agent to query the system tool catalogue and retrieve tool documentation without consuming excessive context tokens:
```json
{
  "tool_name": "catalogue_lookup",
  "arguments": {
    "query": "ripgrep file search",
    "category": "search",
    "limit": 5
  }
}
```

---

## 5. Milestone Checkpoints & Automated Rollback

MSS incorporates the Odyssey state checkpointing pattern via `GitMilestoneCheckpointer`:

1. **Pre-Turn Snapshot**: Before the agent executes any mutating tool (`edit_file_block`, `write_file`, or mutating `sandboxed_bash`), a Git milestone commit is recorded on a hidden reference.
2. **Post-Turn Verification (`--test-cmd`)**: If a verification command is specified (e.g. `--test-cmd "pytest kybernetes/tests"`), MSS executes the command immediately after the turn completes.
3. **Automatic Rollback**:
   - If the test command exits with code `0`, the milestone is preserved as a verified checkpoint.
   - If the test command fails (non-zero exit), MSS **instantly resets the working tree** back to the pre-turn milestone commit.
   - The test failure output is injected into the model's context as a constraint for the next step, preventing runaway regressions.

---

## 6. Dynamic Self-Improving Skills

MSS learns from successful task resolutions using the Nous Hermes skill synthesis pattern:

1. **Storage Location**: `.metis/skills/<domain>/SKILL.md` within the workspace root.
2. **Structure**:
   ```markdown
   ---
   name: pytest-asyncio-configuration
   description: Guidelines for running pytest with asyncio fixtures in Kybernetes
   tags: [testing, asyncio, pytest]
   ---
   # Pytest Asyncio Configuration
   When running async tests in Kybernetes, always ensure `anyio` or `asyncio_mode = auto` is configured...
   ```
3. **Retrieval**: At the start of every MSS session, `DynamicSkillLearner` indexes available skill markdown files and injects relevant skills into the system prompt based on semantic keyword matching against the `--goal`.

---

## 7. Lochos: Multi-Agent Squad Orchestration

Lochos manages collaborative, multi-seat agent workflows backed by a durable SQLite queue (`lochos.sqlite`):

### 7.1 Agent Seats & Roles

| Seat Address | Role | Default Model | Permissions |
| :--- | :--- | :--- | :--- |
| `squad-coder` | Software Engineer | `qwen2.5-coder-32b-instruct` | Full workspace read/write, test execution. |
| `squad-reviewer` | Code Reviewer | `llama-3.1-8b-instruct` | Read-only analysis, diff inspection. |

### 7.2 Lochos CLI Commands

```bash
# 1. View all seats, current states, and active batons
kybernetes lochos ps

# 2. List queued, in-progress, and completed tasks
kybernetes lochos queue

# 3. Dispatch a new task slice to a seat
kybernetes lochos dispatch --mission "v0.6.0" --slice "001" --title "Add temperature setting" --seat coder

# 4. Transfer the task baton from coder to reviewer
kybernetes lochos transfer --task-id "task-abc123" --from-seat coder --to-seat reviewer --note "Ready for review"

# 5. Execute the claimed task using the terminal agent
kybernetes lochos run --seat coder --auto-approve

# 6. Resolve attention blocks (e.g. manual operator approvals)
kybernetes lochos resolve --task-id "task-abc123" --seat coder --resolution "approved_by_operator"
```

---

## 8. Pinax: System Tool Catalogue Subsystem

Pinax maintains an inspectable, risk-tiered inventory of all system binaries and utilities:

### 8.1 Risk Tiers

- **T0 (Read-Only / Safe)**: `cat`, `ls`, `grep`, `file`, `wc`. Pre-approved without prompts.
- **T1 (Workspace Mutating)**: `git commit`, `sed`, `touch`, `cp`. Requires local approval.
- **T2 (System Wide Mutating)**: `apt install`, `systemctl`, `pip install`. Requires explicit confirmation.
- **T3 (Dangerous / Network / Unsafe)**: `rm -rf`, `curl`, `ssh`, `dd`. Requires strict operator authorization.

### 8.2 Pinax CLI Commands

```bash
# 1. Run passive PATH scan (discovers executables without executing them)
kybernetes pinax scan

# 2. Search tool inventory by keyword, category, or tier
kybernetes pinax lookup "search" --category search --tier T0

# 3. Retrieve size-capped documentation slice for a tool
kybernetes pinax docs git --max-bytes 2048

# 4. View Level 1 specification, capabilities, and overrides
kybernetes pinax detail rg

# 5. Evaluate invocation arguments against risk override policies
kybernetes pinax evaluate git commit -m "feat: new feature"
```

---

## 9. Practical Usage Recipes

### Recipe 1: Standard Interactive Coding Session
```bash
kybernetes code \
  --goal "Fix division by zero in calculator tool and add unit tests" \
  --workspace . \
  --test-cmd "pytest kybernetes/tests/test_tools.py"
```

### Recipe 2: Unattended Automated CI Run
```bash
kybernetes code \
  --goal "Update deprecated HTTP_422_UNPROCESSABLE_ENTITY constants to HTTP_422_UNPROCESSABLE_CONTENT" \
  --auto-approve \
  --max-steps 10 \
  --test-cmd "pytest kybernetes/tests/"
```

### Recipe 3: Custom Model & Temperature
```bash
kybernetes code \
  --goal "Refactor Hestia model swap state machine" \
  --model "qwen2.5-coder-32b-instruct" \
  --base-url "http://127.0.0.1:8080/v1" \
  --temperature 0.1 \
  --test-cmd "pytest hestia/tests/test_swap.py"
```

---

## 10. Operational & Security Checklist

Before deploying MSS on production repositories:
- [ ] Ensure `git status` is clean before launching `kybernetes code` so milestone rollbacks have an immutable baseline.
- [ ] Always provide a reliable `--test-cmd` so the agent cannot exit successfully while leaving breaking changes.
- [ ] For sensitive projects, run without `--auto-approve` to interactively inspect every unified diff before it is applied to disk.
- [ ] Check `.phylax/audit.jsonl` to inspect the complete cryptographic audit trail of all actions executed by the agent.
