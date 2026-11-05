"""
Stoma — Ingestion pipeline and mouth for LTHCM topological memory.
"""

from stoma.config import DependencyOriginError, StomaConfig, verify_runtime_dependencies
from stoma.provenance import PassageProvenance, ProvenanceManifest, SourceFileRecord

try:
    from stoma.chunker import Chunker
except (ImportError, ModuleNotFoundError):
    Chunker = None  # type: ignore

try:
    from stoma.adapter import StomaLTHCMAdapter
except (ImportError, ModuleNotFoundError):
    StomaLTHCMAdapter = None  # type: ignore

try:
    from stoma.embedder import StomaEmbedder
except (ImportError, ModuleNotFoundError):
    StomaEmbedder = None  # type: ignore

try:
    from stoma.indexer import StomaIndexer
except (ImportError, ModuleNotFoundError):
    StomaIndexer = None  # type: ignore

from stoma.manifest_staging import (
    ManifestStagingGate,
    SecurityPathError,
    StagedFileRecord,
    StagingGate,
    StagingManifest,
    StagingValidationError,
)
from stoma.redaction import (
    ContentAssessment,
    PrivacyRedactor,
)
from stoma.safe_extract import (
    ArchiveBombError,
    ArchiveSafetyError,
    ArchiveSafetyGuard,
    SafeExtractor,
)

from stoma.filters import DeterministicSpanFilter
from stoma.alias_reconciler import (
    CanonicalEntity,
    EntityAliasReconciler,
)
from stoma.table_normalizer import (
    ParsedTable,
    TableProseNormalizer,
    normalize_markdown_tables,
)
from stoma.crawlers.openreview import (
    CommunityAvailabilityIndex,
    OpenReviewBoundaryCrawler,
    OpenReviewCrawlerConfig,
    PeerReviewForumRecord,
    ReviewerObjectionRecord,
)
from stoma.identity import (
    EmbeddingSpaceSpec,
    IdentityValidationError,
    IncompatibleSpaceError,
    IndexBuildSpec,
    InvalidVectorPayloadError,
    ProducerBuildSpec,
    VectorBatchEnvelope,
    canonical_jcs_serialize,
    create_vector_envelope,
    domain_separated_hash,
)

# Enforce exact submodule origins and commit pins on runtime import
verify_runtime_dependencies()

__version__ = "0.1.0"

__all__ = [
    "StomaConfig",
    "StomaEmbedder",
    "Chunker",
    "StomaLTHCMAdapter",
    "StomaIndexer",
    "PassageProvenance",
    "SourceFileRecord",
    "ProvenanceManifest",
    "StagingGate",
    "ManifestStagingGate",
    "StagingManifest",
    "StagedFileRecord",
    "SecurityPathError",
    "StagingValidationError",
    "PrivacyRedactor",
    "ContentAssessment",
    "ArchiveSafetyGuard",
    "ArchiveBombError",
    "ArchiveSafetyError",
    "SafeExtractor",
    "EmbeddingSpaceSpec",
    "ProducerBuildSpec",
    "IndexBuildSpec",
    "VectorBatchEnvelope",
    "canonical_jcs_serialize",
    "domain_separated_hash",
    "create_vector_envelope",
    "IdentityValidationError",
    "IncompatibleSpaceError",
    "InvalidVectorPayloadError",
    "DeterministicSpanFilter",
    "CanonicalEntity",
    "EntityAliasReconciler",
    "ParsedTable",
    "TableProseNormalizer",
    "normalize_markdown_tables",
    "OpenReviewBoundaryCrawler",
    "OpenReviewCrawlerConfig",
    "PeerReviewForumRecord",
    "ReviewerObjectionRecord",
    "CommunityAvailabilityIndex",
]
