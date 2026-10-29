"""
ANN (Approximate Nearest Neighbor) Vector Index Benchmark
=========================================================
Evaluates vector retrieval backends (Exact Brute-Force, Google ScaNN Tree-AH,
ScaNN Brute-Force, FAISS Flat/IVF) on empirical recall, query latency,
throughput, and build time across configurable embedding dimensions and dataset sizes.

Addresses Google Research Takeaways (LTHCM Item 1 & Dokime yardstick):
  - Quantitative empirical Recall@k against exact ground truth
  - Latency percentiles (mean, p50, p95, p99 ms)
  - Throughput (QPS)
  - Build and indexing duration
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ANNAlgorithmResult:
    """Metrics for a single vector retrieval algorithm."""
    algorithm_name: str
    build_time_s: float
    total_query_time_s: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    qps: float
    recall_at_1: float
    recall_at_k: float
    speedup_vs_exact: float
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ANNBenchmarkConfig:
    """Configuration for an ANN benchmark execution."""
    num_documents: int = 1000
    d_model: int = 64
    num_queries: int = 50
    top_k: int = 10
    seed: int = 42
    distance_measure: str = "dot_product"
    num_leaves: Optional[int] = None
    leaves_to_search: Optional[int] = None
    anisotropic_quantization_threshold: float = 0.2
    reorder_num_neighbors: Optional[int] = None


@dataclass
class ANNBenchmarkReport:
    """Consolidated benchmark report across all evaluated backends."""
    num_documents: int
    d_model: int
    num_queries: int
    top_k: int
    distance_measure: str
    algorithms: Dict[str, ANNAlgorithmResult]

    def format_table(self) -> str:
        """Render a formatted markdown comparison table."""
        lines = [
            f"# ANN Vector Index Benchmark (N={self.num_documents}, D={self.d_model}, Queries={self.num_queries}, Top-K={self.top_k})",
            "",
            "| Algorithm | Recall@1 | Recall@K | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | QPS | Speedup | Build Time (s) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for name, res in self.algorithms.items():
            lines.append(
                f"| **{res.algorithm_name}** | {res.recall_at_1:.4f} | {res.recall_at_k:.4f} | "
                f"{res.mean_latency_ms:.3f} | {res.p50_latency_ms:.3f} | {res.p95_latency_ms:.3f} | "
                f"{res.p99_latency_ms:.3f} | {res.qps:.1f} | {res.speedup_vs_exact:.2f}x | {res.build_time_s:.4f} |"
            )
        return "\n".join(lines)


class ANNBenchmarkSuite:
    """
    Executes comparative benchmarks across vector index backends.
    """

    def __init__(self, config: Optional[ANNBenchmarkConfig] = None):
        self.config = config or ANNBenchmarkConfig()

    def generate_synthetic_data(
        self,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Generate normalized synthetic corpus and query embeddings."""
        rng = np.random.default_rng(self.config.seed)
        n = self.config.num_documents
        d = self.config.d_model
        q_count = self.config.num_queries

        corpus = rng.normal(size=(n, d)).astype(np.float32)
        corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
        corpus_norms = np.maximum(corpus_norms, 1e-12)
        corpus = (corpus / corpus_norms).astype(np.float32)

        doc_ids = [f"doc_{i}" for i in range(n)]

        queries = rng.normal(size=(q_count, d)).astype(np.float32)
        queries_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        queries_norms = np.maximum(queries_norms, 1e-12)
        queries = (queries / queries_norms).astype(np.float32)

        return corpus, doc_ids, queries

    def run(
        self,
        corpus: Optional[np.ndarray] = None,
        doc_ids: Optional[List[str]] = None,
        queries: Optional[np.ndarray] = None,
    ) -> ANNBenchmarkReport:
        """
        Execute full benchmark suite against exact ground truth.
        """
        if corpus is None or queries is None:
            corpus, doc_ids, queries = self.generate_synthetic_data()
        elif doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(corpus))]

        k = min(self.config.top_k, len(corpus))

        # 1. Exact Ground Truth (Linear Dot-Product)
        t0_build_exact = time.perf_counter()
        # Exact index build time is trivial array allocation
        build_exact = time.perf_counter() - t0_build_exact

        exact_latencies: List[float] = []
        exact_results: List[List[str]] = []

        for q in queries:
            t0_q = time.perf_counter()
            sims = np.dot(corpus, q)
            top_indices = np.argsort(sims)[::-1][:k]
            exact_latencies.append((time.perf_counter() - t0_q) * 1000.0)
            exact_results.append([doc_ids[idx] for idx in top_indices])

        exact_total_time = sum(exact_latencies) / 1000.0
        exact_mean = float(np.mean(exact_latencies))
        exact_p50 = float(np.percentile(exact_latencies, 50))
        exact_p95 = float(np.percentile(exact_latencies, 95))
        exact_p99 = float(np.percentile(exact_latencies, 99))
        exact_qps = float(len(queries) / exact_total_time) if exact_total_time > 0 else 0.0

        algorithms: Dict[str, ANNAlgorithmResult] = {
            "exact": ANNAlgorithmResult(
                algorithm_name="Exact Linear Dot-Product",
                build_time_s=build_exact,
                total_query_time_s=exact_total_time,
                mean_latency_ms=exact_mean,
                p50_latency_ms=exact_p50,
                p95_latency_ms=exact_p95,
                p99_latency_ms=exact_p99,
                qps=exact_qps,
                recall_at_1=1.0,
                recall_at_k=1.0,
                speedup_vs_exact=1.0,
            )
        }

        # 2. ScaNN Tree-AH Anisotropic Vector Quantization
        try:
            import scann

            num_leaves = self.config.num_leaves or max(int(np.sqrt(len(corpus))), 4)
            num_leaves = min(num_leaves, max(len(corpus) // 4, 4))
            leaves_to_search = self.config.leaves_to_search or max(num_leaves // 4, 2)

            t0_build_scann = time.perf_counter()
            builder = scann.scann_ops_pybind.builder(corpus, k, self.config.distance_measure)
            builder = builder.tree(
                num_leaves=num_leaves,
                num_leaves_to_search=leaves_to_search,
                training_sample_size=min(len(corpus), 100000),
            )
            dim_per_block = 2 if self.config.d_model % 2 == 0 else 1
            builder = builder.score_ah(
                dimensions_per_block=dim_per_block,
                anisotropic_quantization_threshold=self.config.anisotropic_quantization_threshold,
            )
            reorder_k = self.config.reorder_num_neighbors or max(k * 2, 20)
            builder = builder.reorder(min(reorder_k, len(corpus)))
            scann_searcher = builder.build(docids=doc_ids)
            scann_build_time = time.perf_counter() - t0_build_scann

            scann_latencies: List[float] = []
            scann_results: List[List[str]] = []

            for q in queries:
                t0_q = time.perf_counter()
                ids, _ = scann_searcher.search(q, final_num_neighbors=k, leaves_to_search=leaves_to_search)
                scann_latencies.append((time.perf_counter() - t0_q) * 1000.0)
                cleaned_ids = [did.decode("utf-8") if isinstance(did, bytes) else str(did) for did in ids]
                scann_results.append(cleaned_ids)

            # Compute empirical recall against exact ground truth
            r1_hits = sum(1 for ex, sc in zip(exact_results, scann_results) if ex and sc and ex[0] == sc[0])
            rk_hits = sum(len(set(ex).intersection(set(sc))) for ex, sc in zip(exact_results, scann_results))

            scann_total_time = sum(scann_latencies) / 1000.0
            scann_mean = float(np.mean(scann_latencies))
            scann_p50 = float(np.percentile(scann_latencies, 50))
            scann_p95 = float(np.percentile(scann_latencies, 95))
            scann_p99 = float(np.percentile(scann_latencies, 99))
            scann_qps = float(len(queries) / scann_total_time) if scann_total_time > 0 else 0.0

            algorithms["scann_tree_ah"] = ANNAlgorithmResult(
                algorithm_name="ScaNN Tree-AH (Anisotropic)",
                build_time_s=scann_build_time,
                total_query_time_s=scann_total_time,
                mean_latency_ms=scann_mean,
                p50_latency_ms=scann_p50,
                p95_latency_ms=scann_p95,
                p99_latency_ms=scann_p99,
                qps=scann_qps,
                recall_at_1=float(r1_hits / len(queries)),
                recall_at_k=float(rk_hits / (len(queries) * k)),
                speedup_vs_exact=float(exact_mean / scann_mean) if scann_mean > 0 else 1.0,
                extra_metadata={"num_leaves": num_leaves, "leaves_to_search": leaves_to_search},
            )

            # 3. ScaNN Brute Force (SIMD-accelerated exact)
            t0_build_bf = time.perf_counter()
            scann_bf_searcher = scann.scann_ops_pybind.builder(
                corpus, k, self.config.distance_measure
            ).score_brute_force().build(docids=doc_ids)
            scann_bf_build_time = time.perf_counter() - t0_build_bf

            scann_bf_latencies: List[float] = []
            scann_bf_results: List[List[str]] = []
            for q in queries:
                t0_q = time.perf_counter()
                ids, _ = scann_bf_searcher.search(q, final_num_neighbors=k)
                scann_bf_latencies.append((time.perf_counter() - t0_q) * 1000.0)
                cleaned_ids = [did.decode("utf-8") if isinstance(did, bytes) else str(did) for did in ids]
                scann_bf_results.append(cleaned_ids)

            r1_bf_hits = sum(1 for ex, sc in zip(exact_results, scann_bf_results) if ex and sc and ex[0] == sc[0])
            rk_bf_hits = sum(len(set(ex).intersection(set(sc))) for ex, sc in zip(exact_results, scann_bf_results))

            scann_bf_total_time = sum(scann_bf_latencies) / 1000.0
            scann_bf_mean = float(np.mean(scann_bf_latencies))
            scann_bf_p50 = float(np.percentile(scann_bf_latencies, 50))
            scann_bf_p95 = float(np.percentile(scann_bf_latencies, 95))
            scann_bf_p99 = float(np.percentile(scann_bf_latencies, 99))
            scann_bf_qps = float(len(queries) / scann_bf_total_time) if scann_bf_total_time > 0 else 0.0

            algorithms["scann_brute_force"] = ANNAlgorithmResult(
                algorithm_name="ScaNN Brute-Force (SIMD)",
                build_time_s=scann_bf_build_time,
                total_query_time_s=scann_bf_total_time,
                mean_latency_ms=scann_bf_mean,
                p50_latency_ms=scann_bf_p50,
                p95_latency_ms=scann_bf_p95,
                p99_latency_ms=scann_bf_p99,
                qps=scann_bf_qps,
                recall_at_1=float(r1_bf_hits / len(queries)),
                recall_at_k=float(rk_bf_hits / (len(queries) * k)),
                speedup_vs_exact=float(exact_mean / scann_bf_mean) if scann_bf_mean > 0 else 1.0,
            )

        except ImportError as e:
            logger.info(f"ScaNN not available for ANN benchmark: {e}")

        # 4. FAISS Flat Inner-Product (if available)
        try:
            import faiss
            t0_build_faiss = time.perf_counter()
            faiss_index = faiss.IndexFlatIP(self.config.d_model)
            faiss_index.add(corpus)
            faiss_build_time = time.perf_counter() - t0_build_faiss

            faiss_latencies: List[float] = []
            faiss_results: List[List[str]] = []

            for q in queries:
                t0_q = time.perf_counter()
                _, I = faiss_index.search(q.reshape(1, -1), k)
                faiss_latencies.append((time.perf_counter() - t0_q) * 1000.0)
                faiss_results.append([doc_ids[idx] for idx in I[0]])

            r1_faiss = sum(1 for ex, sc in zip(exact_results, faiss_results) if ex and sc and ex[0] == sc[0])
            rk_faiss = sum(len(set(ex).intersection(set(sc))) for ex, sc in zip(exact_results, faiss_results))

            faiss_total_time = sum(faiss_latencies) / 1000.0
            faiss_mean = float(np.mean(faiss_latencies))
            faiss_p50 = float(np.percentile(faiss_latencies, 50))
            faiss_p95 = float(np.percentile(faiss_latencies, 95))
            faiss_p99 = float(np.percentile(faiss_latencies, 99))
            faiss_qps = float(len(queries) / faiss_total_time) if faiss_total_time > 0 else 0.0

            algorithms["faiss_flat"] = ANNAlgorithmResult(
                algorithm_name="FAISS IndexFlatIP",
                build_time_s=faiss_build_time,
                total_query_time_s=faiss_total_time,
                mean_latency_ms=faiss_mean,
                p50_latency_ms=faiss_p50,
                p95_latency_ms=faiss_p95,
                p99_latency_ms=faiss_p99,
                qps=faiss_qps,
                recall_at_1=float(r1_faiss / len(queries)),
                recall_at_k=float(rk_faiss / (len(queries) * k)),
                speedup_vs_exact=float(exact_mean / faiss_mean) if faiss_mean > 0 else 1.0,
            )

        except ImportError:
            pass

        return ANNBenchmarkReport(
            num_documents=len(corpus),
            d_model=self.config.d_model,
            num_queries=len(queries),
            top_k=k,
            distance_measure=self.config.distance_measure,
            algorithms=algorithms,
        )
