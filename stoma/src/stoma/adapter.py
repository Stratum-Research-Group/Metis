"""
Stoma Narrow LTHCM Adapter.
Provides a strictly bounded, validated boundary over the pinned LTHCMEngine (commit 336f4af).
Enforces:
- Locked single-facet tensor shapes: [1, d_model], rejecting mismatches loudly.
- Real epoch-second timestamps for ingestion and retrieval to keep candidates within the Lorentzian light cone.
- Strict provenance resolution from manifest: fails closed on missing or unresolved provenance.
- Clean persistence/restoration matching dianoia serialization conventions.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import torch

from lthcm.engine.pipeline import LTHCMEngine

from stoma.embedder import StomaEmbedder
from stoma.provenance import PassageProvenance, ProvenanceManifest

logger = logging.getLogger(__name__)


class ShapeMismatchError(ValueError):
    """Raised when tensor or facet dimensions do not match the locked configuration."""
    pass


class UnresolvedProvenanceError(RuntimeError):
    """Raised when a candidate passage lacks valid matching provenance in the manifest."""
    pass


class IndexIntegrityError(RuntimeError):
    """Raised when underlying index stores contain duplicate IDs or internal store mismatches."""
    pass


class StomaLTHCMAdapter:
    """
    Narrow adapter wrapping LTHCMEngine.
    Enforces shape guarantees, timestamp contracts, and provenance resolution.
    """

    def __init__(
        self,
        engine: LTHCMEngine,
        embedder: StomaEmbedder,
        manifest: Optional[ProvenanceManifest] = None,
    ):
        self.engine = engine
        self.embedder = embedder
        self.manifest = manifest or ProvenanceManifest(
            model_meta=embedder.get_metadata(),
            d_model=embedder.d_model,
            num_facets=1,
        )
        self.validate_shapes()

    @classmethod
    def create_fresh_engine(cls, embedder: StomaEmbedder) -> LTHCMEngine:
        """Create a fresh LTHCMEngine configured with single-facet and matching d_model."""
        return LTHCMEngine(
            d_model=embedder.d_model,
            num_facets=1,
            c_speed=1.0,
        )

    def validate_shapes(self) -> None:
        """
        Validate that the underlying engine conforms strictly to locked decisions:
        - engine.d_model == embedder.d_model (384)
        - engine.num_facets == 1
        """
        engine_d = getattr(self.engine, "d_model", None)
        if engine_d != self.embedder.d_model:
            raise ShapeMismatchError(
                f"LOUD MISMATCH: Engine d_model {engine_d} does not match "
                f"Embedder d_model {self.embedder.d_model}."
            )

        engine_facets = getattr(self.engine, "num_facets", None)
        if engine_facets != 1:
            raise ShapeMismatchError(
                f"LOUD MISMATCH: Engine num_facets {engine_facets} does not match "
                f"locked single-facet configuration (1)."
            )

    def get_store_doc_ids(self) -> Tuple[Set[str], Set[str]]:
        """
        Inspect and return (hyperbolic_ids, superposition_ids).
        Checks for duplicate IDs within the hyperbolic store.
        """
        hyp_ids: List[str] = []
        if hasattr(self.engine, "hyperbolic") and hasattr(self.engine.hyperbolic, "documents"):
            for doc in self.engine.hyperbolic.documents:
                if isinstance(doc, dict) and "id" in doc:
                    hyp_ids.append(doc["id"])

        # Check for duplicates in hyperbolic store
        if len(hyp_ids) != len(set(hyp_ids)):
            seen = set()
            dups = [x for x in hyp_ids if x in seen or seen.add(x)]
            raise IndexIntegrityError(f"Duplicate document IDs detected in hyperbolic store: {dups}")

        sup_ids: Set[str] = set()
        if hasattr(self.engine, "superposition") and hasattr(self.engine.superposition, "doc_index"):
            sup_ids = set(self.engine.superposition.doc_index.keys())

        return set(hyp_ids), sup_ids

    def get_indexed_doc_ids(self) -> Set[str]:
        """
        Return verified document IDs from both hyperbolic and superposition stores.
        Fails if stores are inconsistent with each other, including if one is empty and peer is not.
        """
        hyp_ids, sup_ids = self.get_store_doc_ids()
        if hyp_ids != sup_ids:
            diff1 = hyp_ids - sup_ids
            diff2 = sup_ids - hyp_ids
            raise IndexIntegrityError(
                f"Dual-store mismatch: hyperbolic has {len(hyp_ids)} IDs, superposition has {len(sup_ids)} IDs. "
                f"In hyperbolic only: {diff1}. In superposition only: {diff2}."
            )
        return hyp_ids

    def validate_generation_integrity(self) -> None:
        """
        Validate generation-wide ID equality across both engine stores and manifest.
        Fails closed if any indexed passage lacks provenance or vice-versa, or if stores disagree.
        Requires hyp_ids == sup_ids == manifest_ids unconditionally.
        """
        if self.manifest is None:
            raise UnresolvedProvenanceError("Engine adapter has no manifest attached.")

        hyp_ids, sup_ids = self.get_store_doc_ids()
        manifest_ids = set(self.manifest.passages.keys())

        if hyp_ids != sup_ids:
            diff1 = hyp_ids - sup_ids
            diff2 = sup_ids - hyp_ids
            raise IndexIntegrityError(
                f"Store mismatch between hyperbolic and superposition stores: "
                f"in hyperbolic only: {diff1}; in superposition only: {diff2}"
            )

        if hyp_ids != manifest_ids:
            missing_in_manifest = hyp_ids - manifest_ids
            missing_in_engine = manifest_ids - hyp_ids
            raise UnresolvedProvenanceError(
                f"Generation ID mismatch between engine and manifest! "
                f"In engine but missing from manifest: {len(missing_in_manifest)} IDs. "
                f"In manifest but missing from engine: {len(missing_in_engine)} IDs."
            )

    def index_passage(
        self,
        doc_id: str,
        content: str,
        embedding: torch.Tensor,
        domain: str = "general",
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Index a passage into LTHCM topological memory.
        Validates embedding dimensionality before submission.
        """
        self.validate_shapes()

        facet_vectors = self.embedder.create_facet_vector(embedding)
        if facet_vectors.shape != (1, self.embedder.d_model):
            raise ShapeMismatchError(
                f"Invalid facet_vectors shape {facet_vectors.shape}; expected (1, {self.embedder.d_model})"
            )

        # Enforce real epoch-second timestamp (default: current epoch seconds)
        ts = timestamp if timestamp is not None else time.time()

        self.engine.index_document(
            doc_id=doc_id,
            content=content,
            facet_vectors=facet_vectors,
            domain=domain,
            timestamp=ts,
            wormhole_links=None,  # Spec line 50: no wormhole edge mining in v1
        )

        if self.manifest is not None and ":" not in doc_id and doc_id not in self.manifest.passages:
            from stoma.provenance import SourceFileRecord
            src_id = f"direct:{doc_id}"
            if src_id not in self.manifest.sources:
                self.manifest.sources[src_id] = SourceFileRecord(
                    source_id=src_id,
                    root_id="direct",
                    root_path="/direct",
                    rel_path=f"{doc_id}.txt",
                    content_hash="",
                    size_bytes=len(content.encode("utf-8")),
                    mtime=ts,
                    last_ingested_at=ts,
                    passage_ids=[doc_id],
                    plugin_name="direct",
                )
            self.manifest.register_passage(
                PassageProvenance(
                    passage_id=doc_id,
                    source_id=src_id,
                    root_id="direct",
                    root_path="/direct",
                    rel_path=f"{doc_id}.txt",
                    file_content_hash="",
                    chunk_index=0,
                    chunk_hash="",
                    location={"passage_id": doc_id},
                    ingested_at=ts,
                    token_count=len(content.split()),
                    char_count=len(content),
                )
            )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        domain: str = "general",
        query_timestamp: Optional[float] = None,
        enforce_causality: bool = True,
    ) -> Dict[str, Any]:
        """
        Query LTHCM topological memory and attach Stoma provenance to every returned candidate.
        Fails closed with UnresolvedProvenanceError if any candidate lacks matching provenance.
        Crucial: passes query_timestamp = current epoch seconds (time.time()) so past documents
        are inside the Lorentzian light cone and causal_fallback is False!
        """
        self.validate_shapes()

        q_vec = self.embedder.encode_single(query_text)
        q_time = query_timestamp if query_timestamp is not None else time.time()

        raw_result = self.engine.retrieve_and_verify(
            query_vector=q_vec,
            query_domain=domain,
            query_timestamp=q_time,
            top_k=top_k,
            enforce_causality=enforce_causality,
            enforce_sheaf_verification=False,  # Sheaf verification without NLI/edges is bypassed
            use_stochastic_resonance=False,    # Direct retrieval
            use_spectral_wormhole=False,       # No graph edges in v1
        )

        results = raw_result.get("results", [])
        enriched_results = []

        for cand in results:
            doc_id = cand.get("doc_id", "")
            if not self.manifest:
                raise UnresolvedProvenanceError("Cannot resolve provenance: no manifest attached to adapter.")
            prov = self.manifest.get_provenance(doc_id)
            if prov is None or prov.passage_id != doc_id:
                raise UnresolvedProvenanceError(
                    f"Candidate passage '{doc_id}' has unresolved provenance in manifest."
                )
            if prov.source_id not in self.manifest.sources:
                raise UnresolvedProvenanceError(
                    f"Candidate passage '{doc_id}' references unknown source '{prov.source_id}'."
                )
            source_rec = self.manifest.sources[prov.source_id]
            if prov.file_content_hash != source_rec.content_hash:
                raise UnresolvedProvenanceError(
                    f"Candidate passage '{doc_id}' content hash '{prov.file_content_hash}' "
                    f"does not match source '{prov.source_id}' content hash '{source_rec.content_hash}'."
                )
            if (
                prov.root_id != source_rec.root_id
                or prov.root_path != source_rec.root_path
                or prov.rel_path != source_rec.rel_path
            ):
                raise UnresolvedProvenanceError(
                    f"Candidate passage '{doc_id}' path metadata ({prov.root_id}:{prov.rel_path}) "
                    f"does not match source '{prov.source_id}' ({source_rec.root_id}:{source_rec.rel_path})."
                )

            cand_dict = {
                "doc_id": doc_id,
                "content": cand.get("content", ""),
                "domain": cand.get("domain", domain),
                "timestamp": cand.get("timestamp", 0.0),
                "gauge_sim": float(cand.get("gauge_sim", 0.0)) if "gauge_sim" in cand else None,
                "provenance": prov.to_dict(),
            }
            enriched_results.append(cand_dict)

        return {
            "query": query_text,
            "query_timestamp": q_time,
            "top_k": top_k,
            "candidates_count": raw_result.get("candidates_count", 0),
            "causally_valid_count": raw_result.get("causally_valid_count", 0),
            "causal_fallback": raw_result.get("causal_fallback", False),
            "results": enriched_results,
        }

    def save(self, index_path: Path) -> None:
        """
        Serialize engine using pickle.dump, matching dianoia's persistence pattern.
        Atomically writes via temporary file and replace.
        """
        index_path = Path(index_path).resolve()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = index_path.with_suffix(f".tmp.{os.getpid()}_{int(time.time()*1000)}")
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(self.engine, f)
            os.replace(tmp_path, index_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        index_path: Path,
        embedder: StomaEmbedder,
        manifest: Optional[ProvenanceManifest] = None,
        expected_index_hash: Optional[str] = None,
        allow_trusted_pickle: bool = True,
    ) -> StomaLTHCMAdapter:
        """
        Load a persisted engine and validate d_model and num_facets before serving.
        Validates generation-wide ID equality if manifest is provided.
        Fails closed before loading if untrusted pickle is forbidden or if expected_index_hash mismatches.
        """
        if not allow_trusted_pickle:
            raise PermissionError("Untrusted pickle deserialization is prohibited without explicit authorization.")

        index_path = Path(index_path).resolve()
        if not index_path.is_file():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        if index_path.stat().st_size > 2 * 1024 * 1024 * 1024:
            raise ValueError(f"Index file exceeds maximum allowed size: {index_path}")

        if expected_index_hash:
            import hashlib
            hasher = hashlib.sha256()
            with open(index_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            actual_h = hasher.hexdigest()
            if actual_h != expected_index_hash:
                raise ValueError(
                    f"Index file SHA-256 digest ({actual_h}) does not match expected hash ({expected_index_hash}); refusing to deserialize."
                )

        with open(index_path, "rb") as f:
            engine = pickle.load(f)

        if not isinstance(engine, LTHCMEngine):
            raise TypeError(
                f"Loaded object is not an LTHCMEngine (got {type(engine).__name__})"
            )

        adapter = cls(engine=engine, embedder=embedder, manifest=manifest)
        adapter.validate_shapes()
        if manifest is not None:
            adapter.validate_generation_integrity()
        return adapter
