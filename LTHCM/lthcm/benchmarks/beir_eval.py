"""
Legacy alias for Synthetic BEIR / SciFact Benchmark.
Redirects to lthcm.benchmarks.synthetic.demo_scifact.
"""
from lthcm.benchmarks.synthetic.demo_scifact import (
    SyntheticSciFactDemo,
    BEIRBenchmark,
    BM25Baseline,
)

__all__ = ["SyntheticSciFactDemo", "BEIRBenchmark", "BM25Baseline"]
