# Hestia (Ἑστία)

> **Sovereign local AI model serving layer — shared, OpenAI-compatible model hosting for the Metis ecosystem.**

**Hestia** (Greek: *Ἑστία* — goddess of the hearth and home, keeper of the central flame) makes open-weights models a shared local utility. It hosts LLMs and embedding models on local GPU/CPU hardware and exposes them through a high-performance, strictly local OpenAI-compatible HTTP API. All other components in Metis—including **Kybernetes**, **Dianoia**, **Dokime**, and **Prosopon**—communicate through Hestia rather than running disjoint model runtimes.

---

## 1. Key Architectural Features

1. **Strictly Local & OpenAI-Compatible**:
   - Implements standard OpenAI endpoints:
     - `GET /v1/models`: Enumerates loaded and registered models.
     - `POST /v1/chat/completions`: Full streaming Server-Sent Events (SSE) and non-streaming text generation.
     - `POST /v1/embeddings`: Local text embedding generation.
     - `GET /health` & `GET /v1/health`: Subsystem health and VRAM telemetry.
2. **VRAM Admission & Memory Budgeting**:
   - Tracks GPU VRAM capacity before loading models to prevent CUDA Out-of-Memory crashes.
   - Enforces configurable idle unload timeouts (`idle_timeout_seconds`), automatically reclaiming GPU memory when a model is unused.
3. **Queue Depth & Concurrency Safeguards**:
   - Configurable per-model concurrency limits (`max_concurrent_requests_per_model`) and request queue caps (`max_queue_depth_per_model`).
   - Automatically returns HTTP 429 / 503 with informative retry hints when capacity is saturated.
4. **Backend Engine Architecture**:
   - Built around an extensible `BackendInterface` abstract base class.
   - Default backend: high-performance `llama.cpp` server processes isolated per port.
   - Pluggable support for vLLM, TensorRT-LLM, and PyTorch transformers.

---

## 2. Architecture & Data Flow

```text
       +----------------------------------------------------------------+
       |             Consumers: Kybernetes, Dianoia, Dokime             |
       +-------------------------------+--------------------------------+
                                       |
                   HTTP / SSE (/v1/chat/completions, /v1/models)
                                       |
                                       v
       +----------------------------------------------------------------+
       |             Hestia API Gateway (FastAPI)                       |
       |  - Request Validation, Authentication (Optional Bearer Token)  |
       |  - Concurrency Limiter & Queue Depth Gate                      |
       +-------------------------------+--------------------------------+
                                       |
                                       v
       +----------------------------------------------------------------+
       |             Server & VRAM Manager (`ServerManager`)            |
       |  - VRAM Accounting (Inspect free VRAM before spawn)            |
       |  - Model Life-cycle (Lazy load on demand, unload on idle)       |
       |  - Process Supervisor (llama-server child processes)           |
       +-------------------------------+--------------------------------+
                                       |
                     Child Process Standard I/O / Local Port
                                       |
                                       v
       +----------------------------------------------------------------+
       |             Backend Process: `llama-server`                   |
       |  - GGUF Quantization (Q4_K_M, Q8_0, BF16)                      |
       |  - GPU Offload (CUDA `sm_70` / `sm_120` / Apple Metal / CPU)   |
       +----------------------------------------------------------------+
```

---

## 3. Quick Start & CLI Usage

### Installation

```bash
# Within the Metis workspace
uv sync

# Standalone development installation
pip install -e ".[dev]"
```

### Starting the Server

```bash
# Start server with default host (127.0.0.1) and port (8080)
hestia serve

# Start on custom host, port, and config paths
hestia serve --host 0.0.0.0 --port 8080 --config hestia.yaml --models models.yaml
```

### Model Management CLI Commands

```bash
# List all registered models and their load status
hestia list

# Pre-load a model into GPU memory
hestia load llama-3.1-8b-instruct

# Inspect active VRAM usage and server health
hestia status

# Explicitly unload a model to free VRAM
hestia unload llama-3.1-8b-instruct
```

---

## 4. API Endpoints & Curl Examples

### List Models (`GET /v1/models`)
```bash
curl http://127.0.0.1:8080/v1/models
```

### Chat Completion (`POST /v1/chat/completions`)
```bash
# Streaming completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain topological memory in one sentence."}
    ],
    "stream": true,
    "temperature": 0.2
  }'
```

### Embeddings (`POST /v1/embeddings`)
```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "Topological memory representations."
  }'
```

---

## 5. Configuration Files

### `hestia.yaml` (Service Configuration)
```yaml
host: "127.0.0.1"
port: 8080
llama_cpp_binary: "/usr/local/bin/llama-server"
idle_timeout_seconds: 600
load_timeout_seconds: 120
max_concurrent_requests_per_model: 4
max_queue_depth_per_model: 16
queue_timeout_seconds: 30
log_level: "info"
```

### `models.yaml` (Model Registry)
```yaml
models:
  llama-3.1-8b-instruct:
    type: "chat"
    path: "/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    context_length: 8192
    vram_mb_estimate: 5800
    ngl: 99  # GPU layer offload
    sampling:
      temperature: 0.1
      top_p: 0.95
      repeat_penalty: 1.1

  qwen2.5-7b-instruct:
    type: "chat"
    path: "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    context_length: 8192
    vram_mb_estimate: 5200
    ngl: 99

  nomic-embed-text:
    type: "embedding"
    path: "/models/nomic-embed-text-v1.5.Q8_0.gguf"
    context_length: 2048
    vram_mb_estimate: 800
```

---

## 6. Testing & Quality Verification

Run unit and integration tests:

```bash
uv run pytest hestia/tests -v
```
