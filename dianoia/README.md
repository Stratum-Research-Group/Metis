# Dianoia

**LTHCM remembers, Dianoia thinks.**

Dianoia is a deliberative cognition layer that sits on top of [LTHCM](https://gitlab.com/akarta-315h/LTHCM), a topological memory substrate for LLM retrieval. Where LTHCM provides non-Euclidean memory storage and retrieval, Dianoia adds a reasoning walk: an LLM-directed traversal of the knowledge graph that deliberates over which passages to explore before synthesising an answer.

## How It Works

1. **Seed** — Retrieve initial candidate passages from LTHCM's stochastic resonance retrieval.
2. **Walk** — A local LLM (Qwen2.5-7B-Instruct via llama.cpp) examines the current passage and its graph neighbours, choosing at each step to **EXPAND** to a neighbour, **BACKTRACK** to the previous node, or **ANSWER**.
3. **Verify** — After answering, LTHCM's sheaf/NLI contradiction filter checks the answer against the retrieved evidence. If conflicts are found, one re-walk is triggered.
4. **Log** — Every walk is recorded as JSONL for future training of a learned walk policy.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   Dianoia                        │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  Walk    │──▶│  Verify  │──▶│  Walk Logger │ │
│  │Controller│   │ (Sheaf/  │   │  (JSONL)     │ │
│  │  (LLM)  │   │  NLI)    │   │              │ │
│  └────┬─────┘   └──────────┘   └──────────────┘ │
│       │                                          │
│       ▼                                          │
│  ┌──────────────────────────────────────────┐    │
│  │        LTHCM (git submodule)             │    │
│  │  Stochastic Resonance · Wormhole Graph   │    │
│  │  Sheaf Cohomology · Superposition Memory │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

## License

MIT — see [LICENSE](LICENSE).

Same license as [LTHCM](https://gitlab.com/akarta-315h/LTHCM).

## LTHCM Submodule

Dianoia depends on LTHCM as a git submodule pinned to commit `8a71605379ee32059e714feb3648cf74f4b42dc6`.

```bash
git clone --recurse-submodules https://gitlab.com/akarta-315h/dianoia.git
# or, if already cloned:
git submodule update --init
```

The submodule pin is updated deliberately, never automatically.

## Installation

```bash
pip install -e ".[nli]"
```

Requires Python ≥ 3.10.

## Hardware Requirements

- 16 GB VRAM (targets: RTX 5060 Ti 16 GB, Tesla V100 SXM2 16 GB)
- Deliberation model: Qwen2.5-7B-Instruct GGUF (Q4_K_M) served by llama.cpp

## Usage

### Prerequisites

Start the local OpenAI-compatible LLM server.

> **Operational Fallback Note**: Dianoia supports dynamic fallback (defaulting to `--fallback-model llama-3.1-8b-instruct`) when the primary model encounters repeated parse failures or GPU Out of Memory (OOM). To enable operational fallback, the server endpoint (`--llm-url`) must serve both models under a single OpenAI-compatible base URL. At startup, Dianoia queries `/v1/models` to verify that both models are listed. If the fallback model is missing, Dianoia warns at startup that operational fallback is disabled; if fallback is subsequently triggered, the walk aborts with a clear error.

#### Option A: Multi-Model Setup with Proxy (Enables Operational Fallback)

Run each model on a distinct port and expose them through an OpenAI-compatible proxy (e.g. [LiteLLM](https://github.com/BerriAI/litellm)):

```bash
# 1. Start primary deliberation model on port 8081
llama-server \
  -m Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --alias qwen2.5-7b-instruct \
  --host 0.0.0.0 \
  --port 8081 \
  -c 8192

# 2. Start fallback model on port 8082
llama-server \
  -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --alias llama-3.1-8b-instruct \
  --host 0.0.0.0 \
  --port 8082 \
  -c 8192

# 3. Create config.yaml and start LiteLLM proxy on port 8000:
cat << 'EOF' > config.yaml
model_list:
  - model_name: qwen2.5-7b-instruct
    litellm_params:
      model: openai/qwen2.5-7b-instruct
      api_base: http://localhost:8081/v1
  - model_name: llama-3.1-8b-instruct
    litellm_params:
      model: openai/llama-3.1-8b-instruct
      api_base: http://localhost:8082/v1
EOF

litellm --config config.yaml --port 8000

```

Dianoia connects to `--llm-url http://localhost:8000/v1` and verifies both models are available via `GET /v1/models`.

#### Option B: Single-Model Setup (Primary Model Only)

If only serving the primary model without a proxy:

```bash
llama-server \
  -m Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --alias qwen2.5-7b-instruct \
  --host 0.0.0.0 \
  --port 8080 \
  -c 8192
```

Dianoia connects to `--llm-url http://localhost:8080/v1`. It will detect that `llama-3.1-8b-instruct` is not served, log an honest warning at startup that operational fallback is disabled, and abort cleanly if fallback is triggered.

### CLI

```bash
# Build a fresh index from documents and ask a question
dianoia --docs ./my_documents "What caused the 2008 financial crisis?"

# Use a pre-built index
dianoia --index ./my_index.pkl "What is the mechanism of action of imatinib?"

# With dual-model proxy setup (Option A)
dianoia --docs ./docs \
  --llm-url http://localhost:8000/v1 \
  --model qwen2.5-7b-instruct \
  --fallback-model llama-3.1-8b-instruct \
  --max-steps 12 \
  --top-k 8 \
  --log-path ./walks.jsonl \
  --verbose \
  "Explain the relationship between BRCA1 and DNA repair."

# With single-model setup (Option B)
dianoia --docs ./docs \
  --llm-url http://localhost:8080/v1 \
  --model qwen2.5-7b-instruct \
  "Explain the relationship between BRCA1 and DNA repair."
```

### Output

The CLI prints:
1. The walk path step by step (action, node, reason; plus initial walk path if re-walked)
2. The verification report (what was checked, actual NLI status, obstruction energy, conflicts)
3. The final answer

### Verification & NLI Modes

Answer-time verification runs LTHCM's sheaf cohomological contradiction filter on answer-vs-evidence:
- By default, the CLI wires the LTHCM benchmark convention: `cross-encoder/nli-deberta-v3-xsmall` with `contradiction_threshold = 2.5`.
- When loading a pre-built index via `--index`, these spec defaults are always applied, overriding any stale or unconfigured settings stored in the saved filter, and the applied configuration is reported.
- Verification reports **actual** NLI usage (`sheaf_filter.nli_model is not None`), never mere intent:
  - When NLI model is loaded: `Method: sheaf cohomological contradiction filter (NLI active: cross-encoder/nli-deberta-v3-xsmall)`.
  - In environments without cross-encoder weights: `Method: sheaf cohomological contradiction filter (NLI inactive: vector sheaf Laplacian only)`.
- When answer-evidence conflicts are detected: prepends the conflict report, logs the initial walk path, and triggers at most one re-walk.

### Walk Logs & Training Signal

Walk logs are appended to `~/.dianoia/walks.jsonl` by default (override with `--log-path`). Each line is a self-contained JSON record conforming to `schema_version = "1.0.0"`.

#### JSON Record Schema (`1.0.0`)
```json
{
  "schema_version": "1.0.0",
  "walk_id": "4b684cb6-09ec-45c1-9278-6e5414d7a7c0",
  "question": "What is hippocampal replay?",
  "seeded_nodes": ["node_1", "node_2"],
  "path": [
    {
      "node": "node_1",
      "action": "EXPAND",
      "raw_output": "ACTION: EXPAND node_2\nREASON: Relevant to hippocampal replay",
      "reason": "Relevant to hippocampal replay",
      "parse_failed": false
    },
    {
      "node": "node_2",
      "action": "ANSWER",
      "raw_output": "ACTION: ANSWER\nREASON: Sufficient context retrieved",
      "reason": "Sufficient context retrieved",
      "parse_failed": false
    }
  ],
  "verification": {
    "conflicts_found": false,
    "rewalked": false,
    "report": "Consistent with evidence",
    "conflict_detail": null
  },
  "final_answer": "Hippocampal replay is the replay of neural activity patterns...",
  "max_steps": 12,
  "steps_used": 2,
  "model": "qwen2.5-7b-instruct",
  "fallback_used": false
}
```

#### Conflict & Re-Walk Two-Record Pattern
When answer verification detects a contradiction between the synthesized answer and the evidence graph:
1. **Initial Walk Record**: The controller logs the initial walk trajectory with `verification.conflicts_found = true`, `verification.rewalked = false`, and the specific contradiction obstruction report in `verification.conflict_detail`.
2. **Re-Walk Record**: The controller executes a re-walk informed by the conflict report. When completed, it logs the re-walk outcome with `verification.rewalked = true`, its own verification report, its own `conflict_detail`, and `initial_path` preserving the first trajectory. If the re-walk fails to complete, `verification.rewalk_failed = true` is logged.
3. **Log Linkage**: Both records in the pair share the same `walk_id` (a UUIDv4 string), enabling downstream analysis and policy training pipelines to join the failing attempt with the corrective re-walk. Normal walks without contradictions emit a single record with its own unique `walk_id`.

#### Paideia Gold Defensible Trajectory Criteria
Downstream training harnesses ([Paideia](../paideia/README.md)) consume these walk logs to train learned walk policies. Trajectories are filtered strictly according to defensibility criteria:
- **Parse-Failed Trajectories**: Trajectories containing parse failures or truncated steps are excluded from supervised policy training sets.
- **Unresolved Contradictions**: Trajectories where verification found conflicts and either no re-walk occurred or the re-walk failed to resolve the contradiction are excluded from gold datasets.
- **Resolved Re-Walks**: Re-walk records that successfully eliminated evidence contradictions are paired via `walk_id` with their initial trajectory, forming high-value contrastive and corrective supervisory pairs.

## Hard Constraints (v1)

- Everything runs locally: no paid APIs, no closed models.
- Read-only over the LTHCM memory graph (no write-back / edge-weight learning).
- Deliberation model defaults to `qwen2.5-7b-instruct`; fallback model defaults to `llama-3.1-8b-instruct`.
- Operational fallback scope: triggers automatically after 3 consecutive unparseable controller actions, or upon Out of Memory (OOM) errors during any phase of walk step execution (neighbour retrieval, prompt formatting, LLM call, action parsing) or answer synthesis. Fallback clears GPU cache and retries with the fallback model if registered on the server.
- Strict action output validation: controller output must strictly contain exactly two lines:
  - `ACTION: (EXPAND <neighbour_id>|BACKTRACK|ANSWER)`
  - `REASON: <justification>`
  Any extra prose lines, missing lines, unrecognised actions, or trailing tokens trigger immediate parse failure handling (log raw output, `BACKTRACK`, decrement step budget).
- Documentation standard: mechanisms only, no absolute claims.

## Out of Scope (v1)

- Learned question-type routing (belongs in LTHCM)
- Edge-weight learning / write-back
- Log rotation, prompt optimisation, multi-question interactive sessions, concurrent walks
