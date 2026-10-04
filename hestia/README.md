# Hestia

Local AI model serving layer — shared model hosting for the local stack.

Hestia makes models a shared utility: one service hosts them on GPU, everything else (LTHCM, Dianoia) calls an OpenAI-compatible API.

## Quick Start

```bash
# Install in development mode
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Start the server
hestia serve

# In another terminal
hestia list
hestia load llama-3.1-8b-instruct
hestia status

# Chat request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# Query models (OpenAI-compatible)
curl http://localhost:8080/v1/models

# Embedding request
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "Hello world"
  }'
```

## Configuration

- **`hestia.yaml`** — Service config (port, loopback host, llama.cpp binary path, idle/load timeouts, `max_concurrent_requests_per_model`, `max_queue_depth_per_model`, `queue_timeout_seconds`)
- **`models.yaml`** — Model registry (GGUF paths, VRAM estimates, sampling params: `temperature`, `top_p`, `top_k`, `repeat_penalty`)

## Architecture

```
Client (curl / LTHCM / Dianoia)
    │
    ▼
Hestia API (FastAPI)           ← OpenAI-compatible endpoints
    │
    ▼
Server Manager                 ← VRAM budgets, lazy load, idle unload
    │
    ▼
Backend Interface (ABC)        ← Swappable: llama.cpp now, vLLM later
    │
    ▼
llama.cpp server processes     ← One per loaded model, own port
    │
    ▼
RTX 5060 Ti 16 GB             ← Single GPU, co-load when VRAM allows
```

## Hardware

- **Active**: RTX 5060 Ti 16 GB (Blackwell)
- **Planned**: Tesla V100 SXM2 16 GB (Volta via PCIe adapter) — not yet online

## Running Tests

```bash
pytest
```
