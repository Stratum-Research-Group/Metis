# Hestia improvement review and benchmark plan

Reviewed 2026-09-29. Scope: Hestia's tracked service code, configuration, tests, and the documented Dianoia integration. The `deps/` directories are separate Git submodules; this is not an audit of their internals.

## Current baseline

- `pytest -q`: **47 passed, 1 skipped** in 0.91 s. The skipped test is the physical GPU integration scenario.
- `ruff check src tests`: **passed** (all checks clean).
- Implemented OpenAI-compatible `GET /v1/models` and `GET /v1/models/{model_name}` alongside operational `/models`.
- Implemented bounded admission concurrency control (`max_concurrent_requests_per_model`, `max_queue_depth_per_model`, `queue_timeout_seconds`) with HTTP 429/504 and `Retry-After` headers.
- Implemented strict Pydantic request/response schema validation and fallback to model sampling defaults.
- Implemented clean async task cancellation and awaiting during graceful shutdown.
- Implemented shared lifespan-scoped HTTP client for backend proxying.

## Highest impact gaps

| Priority | Gap and evidence | Consequence | Concrete improvement and acceptance check | Status |
| --- | --- | --- | --- | --- |
| P0 | Hestia exposes `/models` but no `/v1/models` (`src/hestia/api/app.py`). Dianoia's startup queries `/v1/models` (`deps/dianoia/README.md`, `deps/dianoia/dianoia/walk.py`). | A named client of this shared service cannot verify its models or enable its documented fallback. | Implement OpenAI-shaped `GET /v1/models` with stable IDs and object metadata; retain `/models` for operational state. | **DONE** |
| P0 | Every inference request calls `ensure_loaded` before forwarding, and each backend gets its own llama.cpp process. There is no request queue, per-model concurrency limit, admission timeout, or overload response (`src/hestia/server/__init__.py`, `src/hestia/api/app.py`). | Concurrent clients can overwhelm a loaded model; latency and memory use become unpredictable, especially with long streams. | Add bounded per-model admission with a documented maximum queue depth and wait time; return 429/504 with retry guidance on saturation. | **DONE** |
| P0 | VRAM management relies on a fixed estimate and current free memory; launches request full GPU offload (`-ngl 999`). `gpu_target` is stored but never used to select a GPU (`src/hestia/registry.py`, `src/hestia/server/__init__.py`, `src/hestia/backend/llamacpp.py`). | The budget can admit configurations that fail at startup, and a future second GPU cannot be targeted as advertised. | Implement explicit device mapping and per-device accounting, configurable safety headroom, and measured post-load VRAM calibration. | Planned |
| P1 | Only the configured host is restricted to loopback; `/manage/load` and `/manage/unload` have no authorization, and request origin is not checked (`src/hestia/config.py`, `src/hestia/api/app.py`). | Any local process can consume GPU resources or stop active models. Browser-origin access should also be assessed before broader exposure. | Add an optional local API token or Unix socket permissions for management, separate management from inference access. | Planned |
| P1 | API bodies are parsed as unrestricted dictionaries; config and registry values have little validation. Duplicate model names overwrite silently, paths need not exist, sampling settings are read but never applied (`src/hestia/api/app.py`, `src/hestia/config.py`, `src/hestia/registry.py`). | Errors surface late or as 500s, configuration typos go unnoticed, and model defaults do not match declared behavior. | Add strict request/config schemas, startup validation; apply sampling defaults. Assert deterministic 4xx responses for malformed input. | **DONE** |
| P1 | Non-streaming proxy assumes every backend response is JSON and does not map connection/timeout errors (`src/hestia/api/app.py:_proxy_request`). Streaming creates a client per request and may leave a request counted as in-flight if a response is never iterated. | Backend failure can become an opaque 500; connection churn and stranded in-flight counts can block unloads. | Reuse a lifespan-scoped HTTP client, normalize upstream errors, and make stream cleanup cancellation-safe. | **DONE** |
| P1 | `ServerManager` state is process-local and the API stores manager/registry in module globals (`src/hestia/api/app.py`). Port allocation only tracks Hestia's own map. | Multiple app instances or workers can contend for ports/GPU; tests and embedded apps can interfere with one another. | Put state on `app.state`, define a single-worker deployment rule until coordination exists, and probe or reserve ports before launch. | Planned |
| P1 | Health and idle tasks are started, then cancelled on shutdown without awaiting their completion; backend stop may spend up to 35 seconds per process (`src/hestia/server/__init__.py`, `src/hestia/backend/llamacpp.py`). | Shutdown and restart behavior is hard to predict; background exceptions can be missed. | Await cancelled tasks, bound total shutdown time, surface task failures, and add lifecycle tests for shutdown during load, stream, and restart. | **DONE** |
| P2 | `GET /health` reports healthy when backends have not been probed during that request; it also performs several GPU queries. There is no readiness endpoint or machine-readable metrics (`src/hestia/api/app.py`). | Operators cannot distinguish a live API from a ready model or observe queueing, failures, and latency. | Separate liveness/readiness and expose request, load, failure, queue, latency, token, and VRAM metrics with bounded label cardinality. | Planned |
| P2 | Quick start points at example paths that are absent here; no setup check, pinned llama.cpp revision, service unit, or operations guide (`README.md`, `hestia.yaml`, `models.yaml`). | The default installation cannot serve a request, and results cannot be reproduced across llama.cpp versions. | Add `hestia doctor`, explicit install/model instructions, version recording, and a reproducible deployment example. | Planned |

## Benchmark program

The benchmark should measure **the serving layer**, not model quality. Use the same GGUF, llama.cpp build, generation settings, context length, prompt set, GPU, and background load for both Hestia and a direct `llama-server` baseline. Keep LTHCM retrieval scores separate. Start with a small benchmark runner under `benchmarks/` and save machine-readable JSON plus a short Markdown summary for each run.

### Workloads

1. **Warm chat, non-streaming and streaming:** fixed prompts with short and long input lengths (for example 128, 2,048 and 8,192 tokens where the model context permits) and fixed output caps (64 and 256 tokens). Set temperature to zero and a fixed seed if supported by the pinned backend.
2. **Cold load:** fully unload, then time first request to readiness and first token. Repeat enough times to report a distribution, including failures and timeouts.
3. **Embedding:** single input and batches of 8/32 at short and medium token lengths; validate vector dimensions and batching semantics.
4. **Concurrency:** 1, 2, 4, 8 and 16 clients at fixed offered load. Include sustained 10-minute runs, mixed chat/embedding traffic, and a saturation point. Report errors and rejected requests rather than omitting them.
5. **Lifecycle and fault injection:** load/unload cycles, co-load, forced VRAM pressure, backend kill/restart, occupied port, client disconnect, idle eviction and recovery.
6. **Real client compatibility:** an OpenAI SDK chat/embedding smoke test and Dianoia startup/model-list/fallback smoke test. Record exact dependency revisions.

### Metrics and protocol

- Capture p50/p95/p99 end-to-end latency, time to first token, inter-token latency, output tokens/second, requests/second, embedding items/second, cold load time, GPU VRAM peak, CPU/RAM peak, queue wait, error rate, and successful recovery time. Distinguish backend time from Hestia overhead.
- Warm up the model before warm measurements. Use at least 30 successful observations per single-request case and three repeated load runs; longer saturation runs should report duration and request count. Preserve raw per-request samples and count all failures.
- Record GPU model and driver, llama.cpp commit/build flags, Hestia commit, model checksum/quantization, context and batch settings, temperature/seed, prompt-set checksum, concurrency, power limit, ambient GPU activity, and exact commands. Run direct and Hestia cases in alternating order to reduce drift.
- Produce a comparison table for each workload: direct backend, Hestia, absolute values, and percentage overhead. Attach raw JSON so comparisons can be recalculated. Do not claim a speedup from the passing unit tests.

### Initial target gates (proposals, not measured results)

- Hestia's warm median and p95 latency overhead versus direct llama.cpp: **under 5%** at concurrency 1, using the same backend settings. Revisit the target after measuring network and client variability.
- No lost in-flight counts, port leaks, or orphaned subprocesses after 100 load/unload and disconnect cycles.
- All overload requests either complete within a published timeout or receive a documented 429/503; zero silently hung requests.
- A backend crash is detected and recovered or clearly marked failed within a published service-level time; no false success responses.
- OpenAI SDK and Dianoia compatibility smoke tests pass on the pinned versions.

## Suggested delivery order

1. Fix `/v1/models` and prove the Dianoia path works. Add strict request/config validation and clear error mapping.
2. Add admission control and robust lifecycle/stream cleanup. Keep deployment to one worker until resource coordination is designed.
3. Build the benchmark runner and obtain direct-versus-Hestia baseline results on the target GPU. Use those measurements to set queue size, headroom, and latency targets.
4. Implement measured per-GPU placement and VRAM accounting, then stress co-load, eviction, and crash recovery.
5. Add security, operations, and observability before treating the service as a shared utility for more clients.

The first milestone should be judged by a working Dianoia integration and recorded real-hardware benchmark, not by a larger unit-test count alone.
