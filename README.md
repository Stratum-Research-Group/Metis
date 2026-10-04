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
| [**stoma**](file:///home/akarta/Projects/Metis/stoma) | Ingestion & Parsing | Document parsing (PDF, text, code), chunking, sentence-transformers embeddings, directory watching, index generation. |
| [**LTHCM**](file:///home/akarta/Projects/Metis/LTHCM) | Topological Memory | Long-Term Hierarchical Context Memory with topological retrieval, superposition, wormhole routing, and sheaf verification. |
| [**dianoia**](file:///home/akarta/Projects/Metis/dianoia) | Deliberation & Walk | Deliberative cognition layer over LTHCM topological memory; bounded graph walks, counterfactual hypothesis testing, and verifier synthesis. |
| [**hestia**](file:///home/akarta/Projects/Metis/hestia) | Model Serving | OpenAI-compatible local model server wrapping llama.cpp/vLLM/transformers, GPU admission control, `/v1/models` and streaming completions. |
| [**dokime**](file:///home/akarta/Projects/Metis/dokime) | Evaluation Harness | Reproducible benchmark harness for local models, held-out task suites, statistical significance testing, and regression release gates. |
| [**kleroterion**](file:///home/akarta/Projects/Metis/kleroterion) | Stochastic Selector | Entropy-aware, masked-softmax/Gaussian action selection among eligible Dianoia walk actions. |
| [**krisis**](file:///home/akarta/Projects/Metis/krisis) | Candidate Selector | Policy-governed answer evaluation & candidate ranking with hard gates (access, provenance, contradiction, citation resolution). |
| [**kybernetes**](file:///home/akarta/Projects/Metis/kybernetes) | Central Orchestrator | Central request planner, multi-turn session lifecycle, adapter framework, and strict token/time/cost budget engine. |
| [**paideia**](file:///home/akarta/Projects/Metis/paideia) | Training Harness | Dataset auditing, trajectory filtering, LoRA adapter registry, and training provenance. |
| [**phylax**](file:///home/akarta/Projects/Metis/phylax) | Security Guardrails | Capability execution gateway, RBAC policy enforcement, one-time approval handles, and sandboxed tool execution. |
| [**prosopon**](file:///home/akarta/Projects/Metis/prosopon) | User Interface | Authenticated text & voice client interface, session history, and human-in-the-loop approval management. |
| [**docs**](file:///home/akarta/Projects/Metis/docs) | System Contracts | Architecture specs, integration contracts ([`STACK_BUILD_CONTRACT.md`](docs/STACK_BUILD_CONTRACT.md)), and integration roadmap ([`stack-integration.md`](docs/stack-integration.md)). |

---

## Getting Started

Each stack contains its own `pyproject.toml` and test suite. You can set up virtual environments individually or manage them via [`uv`](https://github.com/astral-sh/uv).

### Running Component Tests

To run tests for a specific stack (e.g., `kybernetes`):

```bash
cd kybernetes
uv run --extra test pytest
# or using its existing virtualenv:
.venv/bin/pytest tests
```

### Stack Contracts & Integration Roadmap

For cross-component contracts (identity, provenance, evidence snapshot formats, and selection boundaries), refer to:
- [`docs/STACK_BUILD_CONTRACT.md`](docs/STACK_BUILD_CONTRACT.md)
- [`docs/stack-integration.md`](docs/stack-integration.md)
