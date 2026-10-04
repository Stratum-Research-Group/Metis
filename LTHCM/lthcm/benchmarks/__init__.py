"""
LTHCM Benchmark Suite
=====================
Provides dual-regime evaluation:
  1. Synthetic Demos (`lthcm.benchmarks.synthetic`): Fast smoke fixtures testing core mechanics.
  2. Real Community Benchmarks (`lthcm.benchmarks.real`): Full-scale evaluation across community datasets
     (HotpotQA, 2WikiMultiHopQA, MuSiQue, FEVER, VitaminC, ANLI, SciFact, NFCorpus, TREC-COVID, RULER, WordNet, MeSH).
"""

from lthcm.benchmarks.synthetic.demo_niah import SyntheticNeedleDemo, MultiNeedleHaystackBenchmark
from lthcm.benchmarks.synthetic.demo_multihop import SyntheticMultiHopDemo
from lthcm.benchmarks.synthetic.demo_sheaf import SyntheticSheafDemo, FEVERContradictionBenchmark
from lthcm.benchmarks.synthetic.demo_scifact import SyntheticSciFactDemo, BEIRBenchmark, BM25Baseline

# Legacy alias for backward compatibility
HotpotQABenchmark = SyntheticMultiHopDemo

__all__ = [
    "SyntheticNeedleDemo",
    "SyntheticMultiHopDemo",
    "SyntheticSheafDemo",
    "SyntheticSciFactDemo",
    "MultiNeedleHaystackBenchmark",
    "HotpotQABenchmark",
    "FEVERContradictionBenchmark",
    "BEIRBenchmark",
    "BM25Baseline",
]
