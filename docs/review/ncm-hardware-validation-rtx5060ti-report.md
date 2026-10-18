# NCM Hardware & Memory Validation Report (RTX 5060 Ti 16GB)

**Date:** 2026-10-18  
**Subsystems:** Schedion (NCM Generation) & Symbolon (VQ-VAE Multiscale Tokenizer)  
**Upstream Target:** `https://github.com/HazyResearch/ncm` @ `c70274273f489abc73832b2b98584a8483b6b203`  
**Host Hardware:** NVIDIA GeForce RTX 5060 Ti (15.56 GB usable VRAM, Driver 615.78.08, CUDA 13.0/13.4, PyTorch 2.14.1+cu130)  
**Status:** **GATES G-m0a & G-m0b PASSED**

---

## 1. Executive Summary

Empirical measurement of upstream Next-Concept Modeling (NCM) checkpoints on local hardware demonstrates exceptional efficiency, comfortably clearing all memory and concurrency thresholds:

1. **Peak Memory Footprint (Standalone)**:
   - **0.78 GB allocated** (0.80 GB reserved).
   - Threshold Gate G-m0a ($\le 6.5\text{ GB}$): **PASSED** by a wide safety margin ($8.3\times$ below ceiling).
2. **Co-loaded Memory Footprint (with GPT-2 Large Scorer)**:
   - **3.67 GB allocated** (3.77 GB reserved).
   - Gate G-m0b ($\le 16.0\text{ GB}$): **PASSED**.
3. **Concurrency with Hestia**:
   - Because NCM consumes $< 1.0\text{ GB}$ in standalone mode, NCM can safely co-exist in VRAM concurrently with the primary Hestia inference model without requiring sequential GPU memory unloading.
4. **Latency & Throughput**:
   - VQ-VAE multiscale decode (48-token context window): **6.74 ms** (148.4 windows/sec).
   - NCM next-concept generation (coarse-to-fine hierarchy, 512-token prefix): **~122 – 126 ms per document** (~8 docs/sec).
   - Full evaluation path (`evaluate.py --n 5` on XSum) ran end-to-end without errors:
     - ROUGE-1: 41.37
     - ROUGE-2: 12.24
     - ROUGE-L: 33.28
     - Generative PPL: 45.39

---

## 2. Parameter & Model Profile

| Component | Architecture | Total Params | Non-Embedding Params | Checkpoint File |
| :--- | :--- | :--- | :--- | :--- |
| **VQ-VAE Tokenizer** | 6-layer Multiscale Residual VQ | 60.63 M | 41.30 M | `tokenizer.pt` (286 MB) |
| **Next-Concept Generator** | 12-layer, 12-head Transformer | 88.02 M | 87.71 M | `generator.pt` (338 MB) |
| **Generative Scorer** | GPT-2 Large (36-layer) | 774 M | 774 M | HuggingFace `gpt2-large` (3.25 GB) |

---

## 3. Detailed Benchmark Telemetry

### A. Memory Utilization Progression
```
[Baseline]          Allocated:   0.0 MB | Reserved:   0.0 MB
[Model Load]        Allocated: 690.7 MB | Reserved: 734.0 MB (in 2.92s)
[Peak Generation]   Allocated: 794.3 MB | Reserved: 824.0 MB (0.78 GB / 0.80 GB)
[Peak + Scorer]     Allocated: 3.67 GB  | Reserved: 3.77 GB
```

### B. Sample Latencies (Single-Document Batches)
- Sample 1 (warmup + CUDA graph setup): `555.5 ms`
- Sample 2: `121.3 ms`
- Sample 3: `122.2 ms`
- Sample 4: `122.4 ms`
- Sample 5: `123.3 ms`
- **Steady-state generation average**: `122.3 ms/doc`

### C. Multiscale Reconstruction Decode
- Context window size: 48 tokens
- Test iterations: 100 passes
- Mean latency: `6.74 ms`
- Throughput: `148.37 windows/sec`

---

## 4. Architectural Decisions for Schedion & Symbolon

Based on these local measurements:

1. **Schedion Candidate Generator**:
   - Retain batch generation on GPU; single-pass candidate generation latency (~125 ms) is low enough to fit into Krisis proposal deadlines without blocking interactive agent turns.
2. **Symbolon Multiscale Tokenizer**:
   - The 6.74 ms decode time allows Krisis to generate per-level preview expansions on-demand during candidate evaluation.
3. **No Sequential Unloading Needed**:
   - Peak allocation of 0.78 GB allows NCM to remain memory-resident alongside Hestia (assuming Hestia allocates $\le 12\text{ GB}$).
