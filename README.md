# Metis: Sovereign Local AI Cognitive & Multi-Agent Stack

**Metis** is a modular, single-owner, sovereign local AI ecosystem designed to ingest documents, maintain hierarchical topological memory, deliberate over graph-structured evidence, serve local open-weights models, enforce strict security and capability guardrails, and provide unified orchestration and interfaces—without paid third-party APIs or cloud lock-in.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |         Prosopon (UI & Voice)         |
                      +-------------------+-------------------+
                                          | Turn & Approval Events
                                          v
                      +---------------------------------------+
                      |       Kybernetes (Orchestrator)       |
                      |       (Planner & Budget Engine)       |
                      +---------+-------------------+---------+
                                |                   |
                 +--------------+                   +---------------+
                 | Evaluates & Selects                              | Policy Gates
                 v                                                  v
    +-------------------------+                       +-------------------------+
    | Krisis (Answer Selector)|                       |   Phylax (Guardrails)   |
    +-------------------------+                       +-------------------------+
                 |                                                  |
                 v                                                  v
    +-------------------------+                       +-------------------------+
    |   Hestia (Model Server) |                       |     External Tools      |
    +-------------------------+                       +-------------------------+
                 ^
                 | Reasoning & Verification
                 v
    +-------------------------+
    |  Dianoia (Deliberation) | <---> Kleroterion (Stochastic Action Selector)
    +-------------------------+
                 ^
                 | Graph Walks & Topological Memory
                 v
    +-------------------------+
    | LTHCM (Context Memory)  |
    +-------------------------+
                 ^
                 | Ingestion & Indexing
                 v
    +-------------------------+
    |     Stoma (Ingestion)   |
    +-------------------------+

    Cross-cutting:
    - Dokime: Reproducible evaluation harness & release gates
    - Paideia: Dataset auditing, LoRA fine-tuning & adapter registry
```

---

## Component Stacks

| Stack | Role | Key Capabilities |
| :--- | :--- | :--- |
| [**stoma**](stoma) | Ingestion & Parsing | Document parsing (PDF, text, code), chunking, sentence-transformers embeddings, directory watching, index generation. |
| [**LTHCM**](LTHCM) | Topological Memory | Long-Term Hierarchical Context Memory with topological retrieval, superposition, wormhole routing, and sheaf verification. |
| [**dianoia**](dianoia) | Deliberation & Walk | Deliberative cognition layer over LTHCM topological memory; bounded graph walks, counterfactual hypothesis testing, and verifier synthesis. |
| [**hestia**](hestia) | Model Serving | OpenAI-compatible local model server wrapping llama.cpp/vLLM/transformers, GPU admission control, `/v1/models` and streaming completions. |
| [**dokime**](dokime) | Evaluation Harness | Reproducible benchmark harness for local models, held-out task suites, statistical significance testing, and regression release gates. |
| [**kleroterion**](kleroterion) | Stochastic Selector | Entropy-aware, masked-softmax/Gaussian action selection among eligible Dianoia walk actions. |
| [**krisis**](krisis) | Candidate Selector | Policy-governed answer evaluation & candidate ranking with hard gates (access, provenance, contradiction, citation resolution). |
| [**kybernetes**](kybernetes) | Central Orchestrator | Central request planner, multi-turn session lifecycle, adapter framework, and strict token/time/cost budget engine. |
| [**paideia**](paideia) | Training Harness | Dataset auditing, trajectory filtering, LoRA adapter registry, and training provenance. |
| [**phylax**](phylax) | Security Guardrails | Capability execution gateway, RBAC policy enforcement, one-time approval handles, and sandboxed tool execution. |
| [**prosopon**](prosopon) | User Interface | Authenticated text & voice client interface, session history, and human-in-the-loop approval management. |
| [**docs**](docs) | System Contracts | Architecture specs, integration contracts ([`STACK_BUILD_CONTRACT.md`](docs/STACK_BUILD_CONTRACT.md)), and integration roadmap ([`stack-integration.md`](docs/stack-integration.md)). |

---

## Getting Started

Metis is managed as a unified [`uv`](https://github.com/astral-sh/uv) workspace across all 11 component stacks.

### 1. Workspace Bootstrap

To bootstrap all dependencies and initialize the workspace under Python 3.12:

```bash
./scripts/setup_workspace.sh
# or directly with uv:
uv sync --python 3.12
```

### 2. Running Stack Integration & Component Tests

To run the end-to-end multi-stack smoke test suite:

```bash
PYTHONPATH="kybernetes:prosopon:hestia/src:stoma/src" uv run pytest tests/test_stack_smoke.py -v
```

To run individual stack unit tests:

```bash
# Kybernetes orchestrator
uv run --package kybernetes pytest kybernetes/tests -v

# Prosopon UI & Session store
uv run --package prosopon pytest prosopon/tests -v
```

### 3. Launching the Local Stack

Start the full sovereign local AI stack (Hestia on port `8080`, Kybernetes on port `8081`, and Prosopon on port `8000`):

```bash
./scripts/run_stack.sh
```

Then navigate to `http://127.0.0.1:8000` to interact with Prosopon.

---

## Hardware & Architecture Guides

- **Heterogeneous GPU & CUDA Architecture**: [`docs/HARDWARE_AND_CUDA_GUIDE.md`](docs/HARDWARE_AND_CUDA_GUIDE.md) (covers dual NVIDIA Tesla V100 Volta `sm_70` and RTX 5060 Ti Blackwell `sm_120` environment partitioning).
- **Stack Build Contract**: [`docs/STACK_BUILD_CONTRACT.md`](docs/STACK_BUILD_CONTRACT.md)
- **Integration Roadmap**: [`docs/stack-integration.md`](docs/stack-integration.md)
