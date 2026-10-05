# Dokime (Δοκιμή)

> **Measurement layer, benchmark harness, and statistical regression gate for the Metis ecosystem.**

**Dokime** (Greek: *δοκιμή* — test, trial, proof by assay) is the empirical evaluation engine for the Metis local AI stack. It operationalizes reproducible benchmark discipline: every model, prompt variant, adapter checkpoint, and retrieval pipeline change receives real numbers with paired confidence intervals, never subjective impression.

---

## 1. Core Principles & Methodology

1. **Zero-Vibes Statistical Discipline**:
   - Point estimates are always paired with **10,000-resample paired bootstrap confidence intervals** (default seed 42).
   - Paired diffs ($y_{\text{candidate}} - y_{\text{baseline}}$ per query item) are preserved, enabling exact paired hypothesis testing and statistical power without distributional assumptions.
2. **Strict Hardware & Artifact Provenance**:
   - Every run generates an immutable `RunManifest` capturing:
     - Run ID (content-addressable 12-char hex derived from config & timestamp)
     - Full git commit SHA and dirty tree flag
     - Hardware metrics (GPU model, driver, VRAM, CPU architecture)
     - Complete model endpoint and revision info
     - Input configuration and prompt template SHA-256
3. **Automated Regression Release Gate**:
   - `dokime compare` evaluates a candidate run against a baseline.
   - If a statistically significant regression ($CI_{\text{upper}} < 0$) is detected on a primary task metric, `dokime compare` exits with non-zero status (`1`), preventing broken models or prompt degradation from merging to production.
4. **Hestia Integration**:
   - Directly exercises Hestia's OpenAI-compatible HTTP API (`/v1/chat/completions` and `/v1/models`).
   - Supports streaming and non-streaming verification, latency tracking (TTFT and end-to-end), and token generation rates.
5. **Matrix Evaluation**:
   - Simultaneously evaluates multiple models across multiple task suites in a single invocation, with per-item streaming outputs to `results.jsonl`.

---

## 2. Architecture & Data Flow

```text
       +----------------------------------------------------------------+
       |             Dokime CLI / TOML Configuration                    |
       |  Models + Tasks + Prompts + Sampling Parameters (Seed, Temp)   |
       +-------------------------------+--------------------------------+
                                       |
                                       v
       +----------------------------------------------------------------+
       |             Evaluation Runner (`dokime.runner`)                |
       |                                                                |
       |  ┌──────────────────────────────────────────────────────────┐  |
       |  │ 1. Capture Environment & Hardware Spec (`RunManifest`)   │  |
       |  └────────────────────────────┬─────────────────────────────┘  |
       |                               │                                |
       |                               v                                |
       |  ┌──────────────────────────────────────────────────────────┐  |
       |  │ 2. Task Item Loader (ARC, HotpotQA, GSM8k, Custom)        │  |
       |  └────────────────────────────┬─────────────────────────────┘  |
       |                               │                                |
       |                               v                                |
       |  ┌──────────────────────────────────────────────────────────┐  |
       |  │ 3. Hestia Client (`/v1/chat/completions`)                │  |
       |  │    Streaming response, TTFT, token counter, latency      │  |
       |  └────────────────────────────┬─────────────────────────────┘  |
       |                               │                                |
       |                               v                                |
       |  ┌──────────────────────────────────────────────────────────┐  |
       |  │ 4. Deterministic Task Scorers (Exact Match, Accuracy, F1)│  |
       |  └────────────────────────────┬─────────────────────────────┘  |
       |                               │                                |
       |                               v                                |
       |  ┌──────────────────────────────────────────────────────────┐  |
       |  │ 5. Atomic Storage: results.jsonl, summary.json, manifest │  |
       |  └──────────────────────────────────────────────────────────┘  |
       +-------------------------------+--------------------------------+
                                       |
                                       v
       +----------------------------------------------------------------+
       |             Comparative Analysis (`dokime.compare`)            |
       |  Paired Bootstrap CI (10k draws) -> Significant Delta Check    |
       |  Exit 0: No regressions | Exit 1: Statistically significant drop|
       +----------------------------------------------------------------+
```

---

## 3. CLI Commands & Options

### `dokime tasks`
Lists all discovered evaluation tasks and their dataset item counts.

```bash
dokime tasks
```

### `dokime run`
Executes an evaluation run across models and tasks.

```bash
# Evaluate a single model on a task
dokime run --model llama-3.1-8b-instruct --task arc_easy_25 --seed 42

# Matrix evaluation across multiple models and tasks
dokime run \
  --model llama-3.1-8b-instruct \
  --model qwen2.5-7b-instruct \
  --task arc_easy_25 \
  --task hotpotqa_10 \
  --temperature 0.0 \
  --seed 42 \
  --output-dir runs/

# Run from a versioned TOML configuration
dokime run --config config.toml
```

**Key Flags**:
- `--model` / `--models`: Model identifier(s) (can be passed multiple times).
- `--task` / `--tasks`: Task suite(s) (can be passed multiple times).
- `--prompt-template`: Path to a `.txt` prompt template or raw template string with `{prompt}`.
- `--seed`: Integer seed for deterministic generation and shuffling (default: `42`).
- `--temperature`: Sampling temperature (default: `0.0`).
- `--max-tokens`: Max response tokens.
- `--url`: Base URL for Hestia serving instance (defaults to `HESTIA_URL` env or `http://127.0.0.1:8080`).
- `--output-dir`: Path to store run artifacts (default: `runs/`).

### `dokime compare`
Compares two evaluation runs (or two models within a single matrix run) and computes paired bootstrap confidence intervals.

```bash
# Compare candidate run against baseline run
dokime compare <baseline_run_id> <candidate_run_id> --model llama-3.1-8b-instruct

# Cross-model comparison within a single matrix run
dokime compare <run_id> --cross-model llama-3.1-8b-instruct qwen2.5-7b-instruct --task arc_easy_25
```

---

## 4. Run Artifacts & Structure

Each evaluation run creates a self-contained, reproducible bundle under `runs/run_<run_id>/`:

```text
runs/run_1d4c30b91d06/
├── manifest.json       # Git SHA, hardware spec, model endpoints, and parameters
├── results.jsonl       # Streaming per-item prompt, completion, latency, and score
└── summary.json        # Point estimates, accuracy, and bootstrap confidence intervals
```

---

## 5. Configuration File Format (`config.toml`)

```toml
[run]
name = "nightly_eval"
output_dir = "runs"
seed = 42

[server]
url = "http://127.0.0.1:8080"
timeout_seconds = 60.0

[models]
targets = [
    "llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct",
]

[tasks]
suites = [
    "arc_easy_25",
]

[parameters]
temperature = 0.0
max_tokens = 512
```

---

## 6. Installation & Testing

```bash
# Within the Metis workspace
uv sync

# Standalone development install
pip install -e ".[dev]"

# Run test suite
pytest tests -v
```
