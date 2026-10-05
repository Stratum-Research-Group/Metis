# Paideia (Παιδεία)

**Paideia** (Greek: *education* / *training*) is the production-grade local training, dataset auditing, and policy evaluation pipeline of the Metis ecosystem.

Paideia turns validated, logged local outcomes (LTHCM router benchmark results and Dianoia deliberative walk logs) into deployable routing and deliberative walk-policy artifacts. It treats fine-tuning a QLoRA adapter as a candidate output evaluated against a ladder of cheap baselines, not an assumed default definition of success.

---

## 1. Core Architecture & Evidence-to-Policy Pipeline

```
                +-----------------------------------------+
                |           Logged Outcomes               |
                |  (LTHCM Benchmarks / Dianoia Walks)     |
                +--------------------+--------------------+
                                     |
                                     v
                +-----------------------------------------+
                |       1. Schema & Data Ingestion        |
                | - Pydantic contract validation          |
                | - Attempt linkage (walk_id grouping)   |
                | - Exclude conflicts / parse failures    |
                +--------------------+--------------------+
                                     |
                                     v
                +-----------------------------------------+
                |     2. Leakage-Safe Group Splitting     |
                | - Deduplication before split            |
                | - Query family stem grouping            |
                | - Cryptographic split checksums (SHA256)|
                +--------------------+--------------------+
                                     |
                                     v
                +-----------------------------------------+
                |           3. Baseline Ladder            |
                | Majority -> Regex -> Lexical -> Embed   |
                | (Identical folds, p50/p95 latency, mem) |
                +--------------------+--------------------+
                                     |
                                     v
                +-----------------------------------------+
                |         4. QLoRA Runner (Opt-in)        |
                | - VRAM headroom admission (>=12GB free) |
                | - Exclusive GPU lock (/tmp/metis_gpu)   |
                | - SIMULATED status for mock runs        |
                +--------------------+--------------------+
                                     |
                                     v
                +-----------------------------------------+
                |        5. Evaluation & Promotion        |
                | - Paired bootstrap 95% CI (10k, seed 42)|
                | - Paired sign-permutation / McNemar     |
                | - Operational guardrails (lat/mem/fail) |
                | - Dokime-compatible replay bundle       |
                +--------------------+--------------------+
                                     |
                                     v
                +-----------------------------------------+
                |        6. Hestia Registry Export        |
                | - Deterministic content hashing         |
                | - Exclude transient metadata timestamps |
                | - Rollback target specification         |
                +-----------------------------------------+
```

---

## 2. Key Responsibilities & Invariants

### 1. Separation of Task Classifier vs. Best-Route Policy
- Question-type classification (e.g. HotpotQA `bridge` vs `comparison`) is kept distinct from best-route decisions.
- Best-route labels are derived strictly from counterfactual paired outcomes under a declared utility function:
  $$\text{Utility} = \text{quality} - \lambda_{\text{lat}} \cdot \text{latency} - \lambda_{\text{cost}} \cdot \text{cost}$$
  Ties and missing counterfactuals are explicitly excluded.

### 2. Defensible Dianoia Trajectories
- Dianoia walk logs are joined by `walk_id`.
- Unresolved contradictions or parse failures are excluded from gold policy training data.
- Re-walks that successfully resolved conflicts are retained, forming paired contrastive supervision.

### 3. Strict Promotion Gates
An adapter is promoted to production candidate **only** if:
1. Primary held-out metric delta $> 0$ over the strongest baseline on the ladder.
2. Paired 95% bootstrap confidence interval strictly excludes zero ($CI_{\text{lower}} > 0$).
3. Paired null hypothesis test is statistically significant ($p < 0.05$).
4. Latency ($p95$), peak memory, and failure rate guardrails pass.

### 4. Honest Training Artifacts (`SIMULATED` vs `TRAINED`)
When running in environments without genuine GPU hardware, mock/dry-run executions record an explicit `SIMULATED` status in the run manifest. Mock passes cannot be passed off as trained adapter weights.

### 5. Content-Addressed Adapter Registry
When exporting adapters to Hestia's adapter registry (`models.yaml`), cryptographic checksums are computed strictly over immutable adapter weights and architecture configuration files, excluding transient metadata timestamps to ensure deterministic reproducibility.

---

## 3. Quick Start & CLI Usage

### Setup Virtual Environment
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[cli,dev]"
```

For GPU training dependencies:
```bash
uv pip install -e ".[gpu]"
```

### System Health & Hardware Doctor
Run diagnostic checks on CUDA, VRAM headroom, and installed libraries:
```bash
paideia doctor
```

### Ingest and Audit Datasets
Validate LTHCM router benchmarks and examine counterfactual route statistics:
```bash
paideia dataset validate-router tests/fixtures/lthcm_sample.json -r flat -r mined
```

Validate Dianoia walk JSONL logs, linking re-walk attempts by `walk_id`:
```bash
paideia dataset validate-walk tests/fixtures/dianoia_sample.jsonl
```

### Run the Baseline Ladder
Evaluate Majority, Regex Heuristic, Lexical (TF-IDF + Logistic Regression), and Small Embedding models on frozen folds:
```bash
paideia baseline tests/fixtures/lthcm_sample.json --task type --seed 42
```

### Verify an Execution Manifest
Check an immutable run manifest:
```bash
paideia verify-run --manifest runs/test_run/manifest.json
```

---

## 4. Running Tests

```bash
uv run pytest paideia/tests -v --cov=paideia
```
All unit and integration tests validate Pydantic data schemas, leakage-safe splits, baseline ladder evaluation, bootstrap statistical significance testing, and registry export formatting.
