"""
Stoma — Ingestion pipeline and mouth for LTHCM topological memory.
"""

from stoma.config import DependencyOriginError, StomaConfig, verify_runtime_dependencies
from stoma.adapter import StomaLTHCMAdapter
from stoma.chunker import Chunker
from stoma.embedder import StomaEmbedder
from stoma.indexer import StomaIndexer
from stoma.provenance import PassageProvenance, ProvenanceManifest, SourceFileRecord

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
]
