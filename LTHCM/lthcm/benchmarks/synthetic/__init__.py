"""
Synthetic Demonstrations and Fast Smoke Tests for LTHCM.
========================================================
These modules provide quick verification of core mathematical components
(Wormhole Relational Graph, Chebyshev Spectral Refinement, Cellular Sheaf Dirichlet Energy,
Multi-Facet POVM, Stochastic Resonance, and Fisher-Rao Metric).

NOTE: These are synthetic smoke demos, NOT the official community benchmark fleet.
For full-scale evaluation across community datasets (HotpotQA, 2WikiMultiHopQA, MuSiQue,
FEVER, VitaminC, ANLI, SciFact, NFCorpus, TREC-COVID, RULER, WordNet, MeSH),
see `lthcm.benchmarks.real`.
"""

from lthcm.benchmarks.synthetic.demo_multihop import SyntheticMultiHopDemo
from lthcm.benchmarks.synthetic.demo_scifact import SyntheticSciFactDemo, BM25Baseline
from lthcm.benchmarks.synthetic.demo_sheaf import SyntheticSheafDemo
from lthcm.benchmarks.synthetic.demo_niah import SyntheticNeedleDemo

__all__ = [
    "SyntheticMultiHopDemo",
    "SyntheticSciFactDemo",
    "SyntheticSheafDemo",
    "SyntheticNeedleDemo",
    "BM25Baseline",
]
