"""
LTHCM Core Mathematical and Geometric Modules
"""

from .superposition import SuperpositionMemoryStore
from .lorentzian import LorentzianCausalAttention, LorentzianMetric
from .sheaf import CellularSheaf, SheafContradictionFilter
from .gauge_bundle import FibreBundle, LieAlgebraGaugeConnection
from .titans_memory import TitansNeuralMemory
from .wormhole import WormholeGraphIndex
from .diffeomorphic import HamiltonianMetricFlow
from .fisher_rao import FisherRaoMetric
from .stochastic_resonance import StochasticResonanceQueryLayer
from .matryoshka import MatryoshkaCompressor
from .optimal_transport import WassersteinDocumentAligner, sinkhorn_knopp
from .hdc_vsa import VectorSymbolicArchitecture
from .hyperbolic import PoincareBall, HyperbolicHierarchicalStore
from .spectral_wormhole import ChebyshevSpectralGraphConv, SpectralWormholeRefiner, BidirectionalWormholeEngine
from .persistent_homology import PersistentHomologyGuard
from .riemannian_langevin import RiemannianLangevinMemory, RiemannianMetricTensor
from .cross_encoder import CrossEncoderReranker
from .query_expansion import HyDEExpander, QueryDecomposer, MultiHopQueryExpander
from .contrastive_sheaf import (
    ContrastiveStalkProjector,
    LearnedRestrictionSheaf,
    ContrastiveSheafTrainer,
    ContrastiveSheafVerifier,
)
from .categorical import (
    Morphism,
    IdentityMorphism,
    CompositeMorphism,
    TensorMorphism,
    BraidingMorphism,
    AssociatorMorphism,
    CategoricalLens,
    AdjointFunctorPair,
    FunctorialPipeline,
    PipelineState,
    CausalFilterMorphism,
    FisherRaoRankingMorphism,
    PersistentHomologyAuditMorphism,
)
from .scann_index import ScannConfig, ScannIndex

__all__ = [
    "SuperpositionMemoryStore",
    "LorentzianCausalAttention",
    "LorentzianMetric",
    "CellularSheaf",
    "SheafContradictionFilter",
    "FibreBundle",
    "LieAlgebraGaugeConnection",
    "TitansNeuralMemory",
    "WormholeGraphIndex",
    "HamiltonianMetricFlow",
    "FisherRaoMetric",
    "StochasticResonanceQueryLayer",
    "MatryoshkaCompressor",
    "WassersteinDocumentAligner",
    "sinkhorn_knopp",
    "VectorSymbolicArchitecture",
    "PoincareBall",
    "HyperbolicHierarchicalStore",
    "ChebyshevSpectralGraphConv",
    "SpectralWormholeRefiner",
    "BidirectionalWormholeEngine",
    "PersistentHomologyGuard",
    "RiemannianLangevinMemory",
    "RiemannianMetricTensor",
    "CrossEncoderReranker",
    "HyDEExpander",
    "QueryDecomposer",
    "MultiHopQueryExpander",
    "ContrastiveStalkProjector",
    "LearnedRestrictionSheaf",
    "ContrastiveSheafTrainer",
    "ContrastiveSheafVerifier",
    "Morphism",
    "IdentityMorphism",
    "CompositeMorphism",
    "TensorMorphism",
    "BraidingMorphism",
    "AssociatorMorphism",
    "CategoricalLens",
    "AdjointFunctorPair",
    "FunctorialPipeline",
    "PipelineState",
    "CausalFilterMorphism",
    "FisherRaoRankingMorphism",
    "PersistentHomologyAuditMorphism",
    "ScannConfig",
    "ScannIndex",
]


