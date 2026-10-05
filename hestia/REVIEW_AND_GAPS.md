# Hestia: Round 1 Defect Analysis, Gaps & Remediation Plan

This document details the defect findings, resource concurrency issues, and configuration conflicts identified in `metis-repo-wide-round1-fixes-6752222c.txt` and `metis-dashboard-implementation-spec-ae93d36c.txt` for **Hestia**, along with the exact technical remediation plan and regression tests.

---

## 1. Defect Findings & Root Cause Analysis

### Item 14 (P1): Backend Port Pool Collision with Kybernetes (F5)
- **Location**: `hestia/hestia.yaml:14-16`, `hestia/src/hestia/config.py:16-18`, `hestia/server/__init__.py:350-356`, `scripts/run_stack.sh:13,102-105`.
- **Observed Behavior**: Hestia's `backend_port_range` defaults to `8081-8099`. When spawning local `llama-server` instances, Hestia picks port 8081 first. However, Kybernetes also defaults to port 8081. Hestia's port allocator tracks only internal assignments and does not verify OS socket availability before spawning.
- **Root Cause**: Hardcoded overlapping port ranges and missing OS socket occupancy probing (`socket.bind`).
- **Remediation**:
  1. Reassign Hestia's backend port pool to non-overlapping port range `9000-9019`.
  2. Implement active OS port availability probing before assigning a port to a backend process:
     ```python
     def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
             try:
                 s.bind((host, port))
                 return True
             except OSError:
                 return False
     ```
  3. Raise a typed `PortPoolExhaustedError` if all ports in the pool are occupied.
- **Regression Testing**:
  - Test `test_default_ports_do_not_overlap`: Assert Kybernetes port (8081) is not within Hestia's backend pool range.
  - Test `test_occupied_port_skipped_by_allocator`: Bind a dummy socket to 9000 and verify Hestia allocates 9001.

---

### Item 15 (P1): Queue Depth Leak on Request Cancellation
- **Location**: `hestia/src/hestia/server/__init__.py:166-190`.
- **Observed Behavior**: `acquire_admission` increments the pending queue depth and decrements it upon successful admission or timeout, but fails to decrement when `asyncio.CancelledError` is raised. In low-concurrency settings, cancelled client requests permanently leak queue depth counters, eventually triggering spurious `ServerOverloadedError` rejections.
- **Root Cause**: Queue depth decrement was not enclosed in an unconditional `finally` block covering cancellation.
- **Remediation**:
  1. Enclose queue accounting in a structured context manager or `try...finally`:
     ```python
     self._queue_depth[model_id] += 1
     try:
         await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
         acquired = True
     finally:
         self._queue_depth[model_id] -= 1
     ```
  2. Ensure counters never drop below zero using assertions or clamping.
- **Regression Testing**:
  - Test `test_cancelled_request_releases_queue_depth`: Cancel a task waiting in admission; assert `queue_depth` returns to 0 and subsequent requests succeed.

---

### Item 16 (P1): Concurrency Hardening in Model Load/Unload & Eviction
- **Location**: `hestia/src/hestia/server/__init__.py:118-134,200-223`, `backend/llamacpp.py:68-123`.
- **Observed Behavior**: Model fit checking and reservation transitions are decoupled. If two concurrent requests trigger eviction, race conditions can lead to over-committing GPU VRAM. Unload removes the process handle before `stop()` completes. Shutdown has a 5-second deadline while backend process termination can take up to 30 seconds, risking orphaned zombie processes.
- **Root Cause**: Fine-grained per-model locks without a global admission transaction lock, and mismatched termination timeouts.
- **Remediation**:
  1. Guard model admission, eviction decisions, and VRAM reservation with a single global `asyncio.Lock` (`_admission_lock`).
  2. Retain process references in the active registry until `backend.stop()` has fully reaped the subprocess.
  3. Align shutdown timeouts: allow up to 30 seconds for graceful SIGTERM, followed by SIGKILL.
- **Regression Testing**:
  - Test `test_concurrent_model_admission_serializes_cleanly`: Two requests triggering eviction of different models do not over-allocate VRAM.
  - Test `test_shutdown_waits_for_process_reaping`: Terminating the server does not leave orphan llama-server processes.

---

### Item 17 (P1): GPU Device Assignment & Multi-GPU Coordination
- **Location**: `hestia/src/hestia/registry.py:60,77`, `backend/llamacpp.py:36-57`.
- **Observed Behavior**: Models configure `gpu_target` in metadata, but the backend launcher does not set `CUDA_VISIBLE_DEVICES` or pass `--gpu-layers` pinned to a specific GPU device.
- **Root Cause**: Device targeting was declarative metadata without runtime process environment enforcement.
- **Remediation**:
  1. When spawning backend processes, inject `CUDA_VISIBLE_DEVICES=<gpu_index>` into the subprocess environment matching the model's assigned `gpu_target`.
  2. Respect shared file locks (`/tmp/metis_gpu_{id}.lock`) to prevent collisions with Paideia training runs.
- **Regression Testing**:
  - Test `test_backend_spawns_with_device_environment`: Verify subprocess launch arguments and environment include assigned GPU target.
