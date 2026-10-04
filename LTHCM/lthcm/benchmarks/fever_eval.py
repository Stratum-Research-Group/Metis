"""
Legacy alias for Synthetic Contradiction / Sheaf Benchmark.
Redirects to lthcm.benchmarks.synthetic.demo_sheaf.
"""
from lthcm.benchmarks.synthetic.demo_sheaf import (
    SyntheticSheafDemo,
    FEVERContradictionBenchmark,
)

__all__ = ["SyntheticSheafDemo", "FEVERContradictionBenchmark"]
