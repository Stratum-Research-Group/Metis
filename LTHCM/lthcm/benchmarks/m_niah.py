"""
Legacy alias for Synthetic Multi-Needle Haystack Benchmark.
Redirects to lthcm.benchmarks.synthetic.demo_niah.
"""
from lthcm.benchmarks.synthetic.demo_niah import (
    SyntheticNeedleDemo,
    MultiNeedleHaystackBenchmark,
)

__all__ = ["SyntheticNeedleDemo", "MultiNeedleHaystackBenchmark"]
