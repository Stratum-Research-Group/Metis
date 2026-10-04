# Dokime

Measurement layer and evaluation harness for a local AI stack.

Dokime generalizes LTHCM's benchmark discipline: every model, prompt, or pipeline change gets real numbers with confidence intervals, never vibes.

## Key Features

- **Serving Layer Integration**: Communicates via Hestia's OpenAI-compatible HTTP API.
- **Reproducibility Provenance**: Fixed seeds, run IDs, manifest with SHA-256 checksums, hardware capture.
- **LTHCM Statistics**: Point estimates + 10k paired bootstrap confidence intervals (seed 42), preserving per-query diffs.
- **Matrix Evaluation**: Evaluates multiple models across multiple tasks in a single run.
- **Regression Detection**: `dokime compare` diffs runs and flags statistically significant regressions or improvements.
- **Task Plugin Architecture**: Extensible tasks via entry points (`dokime.tasks`).

## Quick Start

### Installation

```bash
pip install -e .
```

### Discover Tasks

```bash
dokime tasks
```

### Run Evaluation

```bash
# Using CLI parameters
dokime run --model llama-3.1-8b-instruct --task arc_easy_25 --seed 42

# Using a TOML config file
dokime run --config config.toml
```

### Compare Runs

```bash
dokime compare <run_id_baseline> <run_id_candidate> --model llama-3.1-8b-instruct
```
