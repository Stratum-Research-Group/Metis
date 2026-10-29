"""
Stoma Indexer and Memory Coordinator.
Manages the lifecycle of document extraction, chunking, embedding, indexing,
durable manifest tracking, generational storage, incremental updates, and atomic swap.
Guarantees transactional recoverable generations (Finding 1), move/rename integrity (Finding 2),
bounded memory streaming (Finding 5), and distinct root identities (Finding 6).
"""
from __future__ import annotations

from enum import Enum
import collections
import copy
import json
import logging
import os
import pickle
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from stoma.adapter import StomaLTHCMAdapter
from stoma.chunker import Chunker, PassageChunk
from stoma.config import StomaConfig
from stoma.embedder import StomaEmbedder
from stoma.plugins import (
    CorruptedDocumentError,
    EmptyExtractionError,
    ExtractionError,
    ExtractionResult,
    FileModifiedDuringExtractionError,
    ImageOnlyPDFError,
    PluginRegistry,
    UnsupportedDocumentError,
    ZeroPagePDFError,
)
from stoma.provenance import (
    LegacyManifestError,
    ModelIdentityMismatchError,
    PassageProvenance,
    ProvenanceManifest,
    SourceFileRecord,
    compute_file_sha256,
)

logger = logging.getLogger(__name__)


class CommitState(str, Enum):
    """Explicit state of a generation commit."""
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"
    BEFORE_REPLACE = "BEFORE_REPLACE"
    REPLACED_DURABILITY_UNKNOWN = "REPLACED_DURABILITY_UNKNOWN"
    DURABLY_ACKNOWLEDGED = "DURABLY_ACKNOWLEDGED"


class CorruptedStorageError(RuntimeError):
    """Raised when the storage directory or generation pointer is corrupt or missing paired files."""
    pass


class RebuildAbortedError(RuntimeError):
    """Raised when rebuild is safely aborted to keep the last good generation."""
    pass


class IndeterminateCommitError(RuntimeError):
    """Raised when pointer replacement took effect on disk but post-replace durability or ledger append was unacknowledged."""
    pass


@dataclass
class IngestionReport:
    """Report for an individual file ingestion attempt."""
    path: Path
    rel_path: str
    status: str  # "indexed", "skipped_unchanged", "skipped_unsupported", "failed", "moved", "deleted"
    passages_count: int
    duration_sec: float
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    source_id: Optional[str] = None


@dataclass
class FolderIngestSummary:
    """Summary of a batch/folder ingestion run with strictly bounded reporting."""
    total_files_scanned: int
    files_indexed: int
    files_skipped_unchanged: int
    files_skipped_unsupported: int
    files_failed: int
    total_passages_indexed: int
    duration_sec: float
    recent_reports: List[IngestionReport] = field(default_factory=list)

    @property
    def reports(self) -> List[IngestionReport]:
        """Backward compatibility for callers expecting 'reports'."""
        return self.recent_reports


def _durable_fsync_file(file_path: Path) -> None:
    """Flush and fsync a file path to durable disk storage."""
    try:
        with open(file_path, "r+b") as f:
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass


def _durable_fsync_dir(dir_path: Path) -> None:
    """Fsync directory to ensure file additions / renames are durable."""
    try:
        dfd = os.open(dir_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


class StomaIndexer:
    """
    Coordinates ingestion pipeline with generational transactions:
    - Source discovery & extraction (bounded streaming)
    - Structure-aware token-bounded chunking
    - Dynamic single-facet embedding
    - LTHCM memory storage with epoch-second timestamps
    - Generational durable manifest and index commits via atomic current.json pointer
    - Verifiable commit-chain and durable commit-markers (Finding 1)
    - Clean transactional move/rename and edit/delete without stale IDs
    """

    def __init__(
        self,
        config: Optional[StomaConfig] = None,
        embedder: Optional[StomaEmbedder] = None,
        plugin_registry: Optional[PluginRegistry] = None,
        redactor: Optional[Any] = None,
        filter_boilerplate: bool = False,
    ):
        from stoma.config import verify_runtime_dependencies
        verify_runtime_dependencies()

        self.config = config or StomaConfig()
        self.embedder = embedder or StomaEmbedder(model_name=self.config.model_name)
        self.plugins = plugin_registry or PluginRegistry()
        self.redactor = redactor
        self.filter_boilerplate = filter_boilerplate
        self.chunker = Chunker(
            embedder=self.embedder,
            target_tokens=self.config.max_chunk_tokens,
            overlap_tokens=self.config.chunk_overlap_tokens,
            redactor=self.redactor,
            filter_boilerplate=self.filter_boilerplate,
        )

        self.manifest: ProvenanceManifest
        self.adapter: StomaLTHCMAdapter
        self.current_generation: int = 0

        self.config.storage_dir.mkdir(parents=True, exist_ok=True)
        self.config.generations_dir.mkdir(parents=True, exist_ok=True)

        self._load_or_initialize()

    def _read_commit_chain(self) -> List[Dict[str, Any]]:
        """Read verifiable chain of committed generations from commit_chain.json."""
        chain_file = self.config.commit_chain_path
        if not chain_file.is_file():
            return []
        try:
            with open(chain_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            logger.warning("Failed to read commit chain at %s: %s", chain_file, e)
            return []

    def _record_commit_chain_entry(self, entry: Dict[str, Any]) -> None:
        """Atomically append a committed generation record to commit_chain.json."""
        chain = self._read_commit_chain()
        chain.append(entry)
        chain_tmp = self.config.commit_chain_path.with_suffix(f".tmp.{os.getpid()}_{int(time.time()*1000)}")
        with open(chain_tmp, "w", encoding="utf-8") as f:
            json.dump(chain, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(chain_tmp, self.config.commit_chain_path)
        _durable_fsync_dir(self.config.storage_dir)

    def _read_pointer_generation_safely(self) -> int:
        """Safely read generation number from current.json without raising."""
        pointer_file = self.config.current_pointer_path
        if not pointer_file.is_file():
            return 0
        try:
            with open(pointer_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return int(data.get("generation", 0))
        except Exception:
            return 0

    def _validate_and_get_proven_chain_prefix(self) -> List[Dict[str, Any]]:
        """
        Validate the cryptographic commit chain from commit_chain.json (Finding 1).
        Verifies:
        1. Monotonic generation increment.
        2. Strict predecessor link (prev_generation == last_gen, prev_manifest_hash == last_manifest_hash).
        3. Matching on-disk commit_marker.json in each generation directory.
        4. Matching index.pkl and manifest.json file hashes on disk.
        Stops at the first broken link and returns only the provably committed prefix.
        """
        raw_chain = self._read_commit_chain()
        valid_prefix: List[Dict[str, Any]] = []

        for i, entry in enumerate(raw_chain):
            if not isinstance(entry, dict):
                logger.warning("Commit chain entry %d is not a dict; stopping chain validation.", i)
                break

            g = int(entry.get("generation", 0))
            if g <= 0:
                logger.warning("Commit chain entry %d has invalid generation %d; stopping.", i, g)
                break

            # Predecessor link and monotonic check
            if i == 0:
                prev_g = int(entry.get("prev_generation", 0))
                if prev_g != 0 and g != 1:
                    logger.warning("Genesis commit chain entry has invalid prev_generation %d; stopping.", prev_g)
                    break
            else:
                prev_entry = valid_prefix[i - 1]
                prev_g = int(entry.get("prev_generation", -1))
                prev_mh = entry.get("prev_manifest_hash")
                expected_prev_g = int(prev_entry.get("generation", 0))
                expected_prev_mh = prev_entry.get("manifest_hash")

                if g <= expected_prev_g:
                    logger.warning("Non-monotonic generation in chain: %d <= %d; stopping.", g, expected_prev_g)
                    break
                if prev_g != expected_prev_g:
                    logger.warning("Predecessor generation mismatch at %d: prev_g=%d != %d; stopping.", g, prev_g, expected_prev_g)
                    break
                if prev_mh != expected_prev_mh:
                    logger.warning("Predecessor manifest hash mismatch at %d: prev_mh=%s != %s; stopping.", g, prev_mh, expected_prev_mh)
                    break

            # Verify commit_marker.json on disk matches ledger entry
            gen_dir = self.config.get_generation_dir(g)
            marker_file = gen_dir / "commit_marker.json"
            if not marker_file.is_file():
                logger.warning("Generation %d missing commit marker on disk; stopping chain validation.", g)
                break

            try:
                with open(marker_file, "r", encoding="utf-8") as f:
                    marker = json.load(f)
                if (
                    int(marker.get("generation", -1)) != g
                    or marker.get("manifest_hash") != entry.get("manifest_hash")
                    or marker.get("index_hash") != entry.get("index_hash")
                    or int(marker.get("prev_generation", -1)) != int(entry.get("prev_generation", -1))
                    or marker.get("prev_manifest_hash") != entry.get("prev_manifest_hash")
                ):
                    logger.warning("Generation %d marker does not match ledger entry; stopping chain.", g)
                    break
            except Exception as e:
                logger.warning("Failed to verify marker for generation %d: %s; stopping chain.", g, e)
                break

            # Verify files exist on disk and match stored hashes
            idx_p, man_p = self.config.get_generation_paths(g)
            if not (idx_p.is_file() and man_p.is_file()):
                logger.warning("Generation %d files missing on disk; stopping chain.", g)
                break

            if compute_file_sha256(idx_p) != entry.get("index_hash"):
                logger.warning("Generation %d index.pkl hash on disk differs from ledger; stopping chain.", g)
                break
            if compute_file_sha256(man_p) != entry.get("manifest_hash"):
                logger.warning("Generation %d manifest.json hash on disk differs from ledger; stopping chain.", g)
                break

            valid_prefix.append(entry)

        return valid_prefix

    def _reconcile_chain_with_active(self, active_gen: int, valid_prefix: List[Dict[str, Any]]) -> bool:
        """
        Safely reconcile commit_chain.json if a crash occurred immediately after pointer commit (Finding 1).
        Defined precisely:
        Only reconciles if:
        1. active_gen marker exists on disk and matches on-disk files.
        2. active_gen marker legitimately extends valid_prefix (prev_generation and prev_manifest_hash match the tail).
        Rejects arbitrary orphans, forged markers, or uncommitted directories.
        """
        prefix_gens = [int(e.get("generation", 0)) for e in valid_prefix]
        if active_gen in prefix_gens:
            return True

        gen_dir = self.config.get_generation_dir(active_gen)
        marker_file = gen_dir / "commit_marker.json"
        if not marker_file.is_file():
            logger.warning("Active generation %d lacks commit marker; cannot reconcile into ledger.", active_gen)
            return False

        try:
            with open(marker_file, "r", encoding="utf-8") as f:
                marker_data = json.load(f)

            if int(marker_data.get("generation", -1)) != active_gen:
                return False

            idx_p, man_p = self.config.get_generation_paths(active_gen)
            if not (idx_p.is_file() and man_p.is_file()):
                return False

            if compute_file_sha256(idx_p) != marker_data.get("index_hash"):
                return False
            if compute_file_sha256(man_p) != marker_data.get("manifest_hash"):
                return False

            # Check predecessor link to valid_prefix
            if valid_prefix:
                tail = valid_prefix[-1]
                expected_tail_gen = int(tail.get("generation", 0))
                expected_tail_mh = tail.get("manifest_hash")
                if (
                    active_gen <= expected_tail_gen
                    or int(marker_data.get("prev_generation", -1)) != expected_tail_gen
                    or marker_data.get("prev_manifest_hash") != expected_tail_mh
                ):
                    logger.warning(
                        "Active generation %d marker does not legitimately extend proven chain tail (tail_gen=%d); "
                        "rejecting as orphan/forged.",
                        active_gen, expected_tail_gen
                    )
                    return False
            else:
                if int(marker_data.get("prev_generation", 0)) != 0 and active_gen != 1:
                    logger.warning("Active generation %d without prior chain has invalid prev_generation %s; rejecting.", active_gen, marker_data.get("prev_generation"))
                    return False

            self._record_commit_chain_entry(marker_data)
            logger.info("Legitimately reconciled commit chain with active generation %d from commit marker.", active_gen)
            return True
        except Exception as e:
            logger.warning("Could not reconcile commit chain with active generation %d: %s", active_gen, e)
            return False

    def _load_or_initialize(self) -> None:
        """
        Load active verified generation or recover previous valid generation.
        Never silently initializes an empty index if partial files are present.
        Never promotes uncommitted orphans (Finding 1).
        """
        pointer_file = self.config.current_pointer_path
        active_gen = 0
        pointer_idx_hash = None
        pointer_man_hash = None

        valid_chain_entries = self._validate_and_get_proven_chain_prefix()

        if pointer_file.is_file():
            try:
                with open(pointer_file, "r", encoding="utf-8") as f:
                    pointer_data = json.load(f)
                active_gen = int(pointer_data.get("generation", 0))
                pointer_idx_hash = pointer_data.get("index_hash")
                pointer_man_hash = pointer_data.get("manifest_hash")
            except Exception as e:
                logger.error("Failed to read generation pointer %s: %s", pointer_file, e)
                active_gen = 0

            # Check if active_gen is in valid_chain_entries OR legitimately extends it
            active_is_proven = (
                any(int(e.get("generation", 0)) == active_gen for e in valid_chain_entries)
                or self._reconcile_chain_with_active(active_gen, valid_chain_entries)
            )

            if active_gen > 0 and active_is_proven:
                if self._try_load_generation(
                    active_gen,
                    expected_index_hash=pointer_idx_hash,
                    expected_manifest_hash=pointer_man_hash,
                    require_commit_marker=True,
                ):
                    return
                logger.warning("Active generation %d failed verification; attempting commit-chain recovery...", active_gen)
            else:
                logger.warning("Active pointer generation %d is unproven or orphan; attempting commit-chain recovery...", active_gen)

        # RECOVERY PROTOCOL (Finding 1):
        # Fallback recovery is strictly restricted to independently verified committed ledger entries.
        # An uncommitted orphan or valid-looking directory outside the ledger must NEVER become current.
        candidate_gens_to_try: List[int] = []
        for entry in reversed(valid_chain_entries):
            c_gen = int(entry.get("generation", 0))
            if c_gen > 0 and c_gen != active_gen and c_gen not in candidate_gens_to_try:
                candidate_gens_to_try.append(c_gen)

        for num in candidate_gens_to_try:
            c_entry = next((e for e in valid_chain_entries if int(e.get("generation", 0)) == num), None)
            exp_idx_h = c_entry.get("index_hash") if c_entry else None
            exp_man_h = c_entry.get("manifest_hash") if c_entry else None
            try:
                if self._try_load_generation(
                    num,
                    expected_index_hash=exp_idx_h,
                    expected_manifest_hash=exp_man_h,
                    require_commit_marker=True,
                ):
                    logger.warning("Successfully recovered prior provably committed generation %d from ledger.", num)
                    self._write_current_pointer(
                        num,
                        candidate_manifest=self.manifest,
                        index_hash=exp_idx_h,
                        manifest_hash=exp_man_h,
                    )
                    return
            except (ModelIdentityMismatchError, LegacyManifestError):
                # Direct import/model mismatch must be a loud error, NOT swallowed
                raise
            except Exception as e:
                logger.warning("Candidate generation %d recovery rejected: %s", num, e)
                continue

        # If pointer file existed or generation directories exist or chain exists but recovery failed: fail closed
        if (
            pointer_file.is_file()
            or valid_chain_entries
            or (self.config.generations_dir.exists() and any(self.config.generations_dir.iterdir()))
        ):
            raise CorruptedStorageError(
                f"Storage at {self.config.storage_dir} is corrupt. Could not load generation {active_gen} "
                "or recover any provably committed prior generation from ledger. Uncommitted orphans rejected."
            )

        # No pointer file: check if legacy files exist in storage_dir
        idx_exists = self.config.index_path.is_file()
        man_exists = self.config.manifest_path.is_file()

        if idx_exists and not man_exists:
            raise CorruptedStorageError(
                f"Corrupted storage: index.pkl exists at {self.config.index_path} but manifest.json is missing. "
                "Refusing to silently initialize empty state."
            )
        if man_exists and not idx_exists:
            raise CorruptedStorageError(
                f"Corrupted storage: manifest.json exists at {self.config.manifest_path} but index.pkl is missing. "
                "Refusing to silently initialize empty state."
            )

        if idx_exists and man_exists:
            # Legacy files exist: attempt migration to generation 1
            logger.info("Migrating legacy un-generational storage to generation 1...")
            try:
                manifest = ProvenanceManifest.load(self.config.manifest_path)
                manifest.validate_against_embedder(
                    model_name=self.embedder.model_name,
                    d_model=self.embedder.d_model,
                    revision=self.embedder.revision,
                    weights_checksum=self.embedder.weights_checksum,
                    num_facets=1,
                    tokenizer_name=self.embedder.tokenizer_name,
                    max_seq_length=self.embedder.max_seq_length,
                    special_tokens_overhead=self.embedder.special_tokens_overhead,
                    effective_max_tokens=self.embedder.effective_max_tokens,
                )
                adapter = StomaLTHCMAdapter.load(
                    index_path=self.config.index_path,
                    embedder=self.embedder,
                    manifest=manifest,
                    allow_trusted_pickle=True,
                )
                # Save as generation 1
                self.manifest = manifest
                self.adapter = adapter
                self._commit_generation(1, adapter, manifest)
                return
            except Exception as e:
                raise CorruptedStorageError(f"Failed to migrate legacy storage: {e}") from e

        # Completely fresh storage: initialize generation 1 with clean empty index
        logger.info("Initializing fresh Stoma index and manifest at %s (generation 1)", self.config.storage_dir)
        fresh_manifest = ProvenanceManifest(
            model_meta=self.embedder.get_metadata(),
            d_model=self.embedder.d_model,
            num_facets=1,
        )
        fresh_engine = StomaLTHCMAdapter.create_fresh_engine(self.embedder)
        fresh_adapter = StomaLTHCMAdapter(
            engine=fresh_engine,
            embedder=self.embedder,
            manifest=fresh_manifest,
        )
        self.manifest = fresh_manifest
        self.adapter = fresh_adapter
        self._commit_generation(1, fresh_adapter, fresh_manifest)

    def _try_load_generation(
        self,
        gen_id: int,
        expected_index_hash: Optional[str] = None,
        expected_manifest_hash: Optional[str] = None,
        require_commit_marker: bool = True,
    ) -> bool:
        """
        Attempt to load and strictly validate a generation.
        Validates commit marker, index hash, manifest hash, embedder identity, and dual-store integrity.
        """
        gen_dir = self.config.get_generation_dir(gen_id)
        idx_path, man_path = self.config.get_generation_paths(gen_id)
        if not (idx_path.is_file() and man_path.is_file()):
            return False

        marker_path = gen_dir / "commit_marker.json"
        marker_data: Optional[Dict[str, Any]] = None
        if marker_path.is_file():
            try:
                with open(marker_path, "r", encoding="utf-8") as f:
                    marker_data = json.load(f)
                if int(marker_data.get("generation", -1)) != gen_id:
                    logger.warning("Generation %d commit marker generation mismatch: %s", gen_id, marker_data)
                    return False
            except Exception as e:
                logger.warning("Generation %d commit marker corrupted: %s", gen_id, e)
                return False
        elif require_commit_marker:
            logger.warning("Generation %d lacks durable commit marker; treating as uncommitted orphan.", gen_id)
            return False

        target_idx_hash = expected_index_hash or (marker_data.get("index_hash") if marker_data else None)
        target_man_hash = expected_manifest_hash or (marker_data.get("manifest_hash") if marker_data else None)

        if target_idx_hash:
            actual_idx_hash = compute_file_sha256(idx_path)
            if actual_idx_hash != target_idx_hash:
                logger.warning("Generation %d index hash mismatch: expected %s, got %s", gen_id, target_idx_hash, actual_idx_hash)
                return False

        if target_man_hash:
            actual_man_hash = compute_file_sha256(man_path)
            if actual_man_hash != target_man_hash:
                logger.warning("Generation %d manifest hash mismatch: expected %s, got %s", gen_id, target_man_hash, actual_man_hash)
                return False

        try:
            manifest = ProvenanceManifest.load(man_path)
        except Exception as e:
            logger.warning("Generation %d manifest unreadable: %s", gen_id, e)
            return False

        # Model identity validation must NOT be swallowed: fail loudly if mismatched model
        manifest.validate_against_embedder(
            model_name=self.embedder.model_name,
            d_model=self.embedder.d_model,
            revision=self.embedder.revision,
            weights_checksum=self.embedder.weights_checksum,
            num_facets=1,
            tokenizer_name=self.embedder.tokenizer_name,
            max_seq_length=self.embedder.max_seq_length,
            special_tokens_overhead=self.embedder.special_tokens_overhead,
            effective_max_tokens=self.embedder.effective_max_tokens,
        )

        try:
            adapter = StomaLTHCMAdapter.load(
                index_path=idx_path,
                embedder=self.embedder,
                manifest=manifest,
                expected_index_hash=expected_index_hash,
                allow_trusted_pickle=True,
            )
            adapter.validate_generation_integrity()

            self.manifest = manifest
            self.adapter = adapter
            self.current_generation = gen_id

            # Keep compatibility copies/links at config paths
            self._sync_compatibility_paths(idx_path, man_path)
            return True
        except (ModelIdentityMismatchError, LegacyManifestError):
            raise
        except Exception as e:
            logger.warning("Failed validation on generation %d: %s", gen_id, e)
            return False

    def _sync_compatibility_paths(self, idx_path: Path, man_path: Path) -> None:
        """Update top-level index.pkl and manifest.json so external tools see current state."""
        try:
            shutil.copy2(idx_path, self.config.index_path)
            shutil.copy2(man_path, self.config.manifest_path)
        except Exception:
            pass

    def _write_current_pointer(
        self,
        gen_id: int,
        candidate_manifest: Optional[ProvenanceManifest] = None,
        index_hash: Optional[str] = None,
        manifest_hash: Optional[str] = None,
    ) -> None:
        """Atomically commit current.json pointer via temporary file and os.replace."""
        idx_path, man_path = self.config.get_generation_paths(gen_id)
        pointer_tmp = self.config.current_pointer_path.with_suffix(f".tmp.{os.getpid()}_{int(time.time()*1000)}")

        manifest_obj = candidate_manifest if candidate_manifest is not None else getattr(self, "manifest", None)
        idx_h = index_hash or compute_file_sha256(idx_path)
        man_h = manifest_hash or compute_file_sha256(man_path)

        meta = {
            "generation": gen_id,
            "updated_at": time.time(),
            "manifest_hash": man_h,
            "index_hash": idx_h,
            "sources_count": len(manifest_obj.sources) if manifest_obj else 0,
            "passages_count": len(manifest_obj.passages) if manifest_obj else 0,
            "model_name": manifest_obj.model_meta.model_name if manifest_obj else self.embedder.model_name,
            "weights_checksum": manifest_obj.model_meta.weights_checksum if manifest_obj else self.embedder.weights_checksum,
        }

        with open(pointer_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        _durable_fsync_dir(self.config.storage_dir)
        os.replace(pointer_tmp, self.config.current_pointer_path)
        _durable_fsync_dir(self.config.storage_dir)

    def _commit_generation(
        self,
        gen_id: int,
        candidate_adapter: StomaLTHCMAdapter,
        candidate_manifest: ProvenanceManifest,
    ) -> None:
        """
        Durable crash-consistent generation commit protocol (Finding 1):
        1. Validate candidate adapter shapes and generation-wide ID equality.
        2. Validate candidate manifest integrity.
        3. Write staged files to generations/gen_{N}.staged/ with fsync.
        4. Atomic directory rename: gen_{N}.staged/ -> gen_{N}/.
        5. Write durable commit_marker.json inside gen_{N}/ with fsync.
        6. Explicit commit state tracking:
           - PREPARING
           - PREPARED (files & marker durable on disk)
           - BEFORE_REPLACE (pointer temp written & fsynced, storage_dir fsynced)
           - REPLACED_DURABILITY_UNKNOWN (os.replace executed)
           - DURABLY_ACKNOWLEDGED (storage_dir fsynced post-replace, ledger entry appended)
        7. On failure: never delete a generation that a live pointer could reference.
           If os.replace succeeded or may have succeeded, preserve both old and candidate,
           reconcile in-memory state, and disclose indeterminate outcome.
        """
        commit_state = CommitState.PREPARING

        candidate_manifest.validate_integrity()
        candidate_adapter.validate_shapes()
        candidate_adapter.validate_generation_integrity()

        gen_dir = self.config.get_generation_dir(gen_id)
        staged_dir = self.config.generations_dir / f"gen_{gen_id:06d}.staged"

        if staged_dir.exists():
            shutil.rmtree(staged_dir, ignore_errors=True)
        staged_dir.mkdir(parents=True, exist_ok=True)

        staged_idx = staged_dir / "index.pkl"
        staged_man = staged_dir / "manifest.json"

        committed_idx = gen_dir / "index.pkl"
        committed_man = gen_dir / "manifest.json"

        try:
            # 1. Persist both into staged generation
            candidate_adapter.save(staged_idx)
            candidate_manifest.save(staged_man)

            _durable_fsync_file(staged_idx)
            _durable_fsync_file(staged_man)
            _durable_fsync_dir(staged_dir)

            # Validate serialized staged files
            if not staged_idx.is_file() or staged_idx.stat().st_size == 0:
                raise CorruptedStorageError(f"Staged index serialization failed for generation {gen_id}.")
            if not staged_man.is_file() or staged_man.stat().st_size == 0:
                raise CorruptedStorageError(f"Staged manifest serialization failed for generation {gen_id}.")

            # 2. Replace target generation directory atomically
            if gen_dir.exists():
                shutil.rmtree(gen_dir, ignore_errors=True)
            os.replace(staged_dir, gen_dir)
            _durable_fsync_dir(self.config.generations_dir)

            # Compute durable hashes of committed files
            idx_hash = compute_file_sha256(committed_idx)
            man_hash = compute_file_sha256(committed_man)

            # Compute prev_manifest_hash from previous generation if present
            prev_man_hash = None
            if self.current_generation > 0:
                prev_man = self.config.get_generation_dir(self.current_generation) / "manifest.json"
                if prev_man.exists():
                    prev_man_hash = compute_file_sha256(prev_man)

            # 3. Write durable commit_marker.json inside gen_dir
            marker_data = {
                "generation": gen_id,
                "prev_generation": self.current_generation,
                "prev_manifest_hash": prev_man_hash,
                "committed_at": time.time(),
                "manifest_hash": man_hash,
                "index_hash": idx_hash,
                "sources_count": len(candidate_manifest.sources),
                "passages_count": len(candidate_manifest.passages),
                "model_name": candidate_manifest.model_meta.model_name,
                "weights_checksum": candidate_manifest.model_meta.weights_checksum,
            }
            marker_tmp = gen_dir / f"commit_marker.tmp.{os.getpid()}_{int(time.time()*1000)}"
            with open(marker_tmp, "w", encoding="utf-8") as f:
                json.dump(marker_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(marker_tmp, gen_dir / "commit_marker.json")
            _durable_fsync_dir(gen_dir)

            commit_state = CommitState.BEFORE_REPLACE

            # Delegate pointer write to hookable helper with durability guarantees
            self._write_current_pointer(
                gen_id=gen_id,
                candidate_manifest=candidate_manifest,
                index_hash=idx_hash,
                manifest_hash=man_hash,
            )
            commit_state = CommitState.DURABLY_ACKNOWLEDGED

            # Record in commit chain ledger
            try:
                self._record_commit_chain_entry(marker_data)
                _durable_fsync_dir(self.config.storage_dir)
            except Exception as e:
                logger.warning(
                    "Generation %d committed to pointer, but recording to commit chain failed: %s. "
                    "Pointer and candidate preserved; ledger will be reconciled.",
                    gen_id,
                    e,
                )
                self.adapter = candidate_adapter
                self.manifest = candidate_manifest
                self.current_generation = gen_id
                self._sync_compatibility_paths(committed_idx, committed_man)
                raise IndeterminateCommitError(
                    f"Generation {gen_id} pointer committed, but recording to commit chain failed: {e}. "
                    "Candidate preserved to avoid dangling pointer."
                ) from e

            # Update serving pair
            self.adapter = candidate_adapter
            self.manifest = candidate_manifest
            self.current_generation = gen_id

            # Update top-level compatibility copies
            self._sync_compatibility_paths(committed_idx, committed_man)
            logger.info(
                "Committed generation %d atomically (%d sources, %d passages)",
                gen_id,
                len(self.manifest.sources),
                len(self.manifest.passages),
            )
        except Exception as e:
            if commit_state in (CommitState.PREPARING, CommitState.PREPARED, CommitState.BEFORE_REPLACE):
                # Check whether pointer was actually updated despite exception
                pointer_on_disk = self._read_pointer_generation_safely()
                if pointer_on_disk == gen_id:
                    # os.replace took effect! Never delete gen_dir!
                    logger.warning(
                        "Commit error occurred but pointer on disk was replaced with generation %d: %s. "
                        "Preserving generation %d to avoid dangling pointer.",
                        gen_id, e, gen_id
                    )
                    self.adapter = candidate_adapter
                    self.manifest = candidate_manifest
                    self.current_generation = gen_id
                    self._sync_compatibility_paths(committed_idx, committed_man)
                    raise IndeterminateCommitError(
                        f"Generation {gen_id} pointer replace took effect on disk, but operation encountered an error: {e}. "
                        "Generation was preserved."
                    ) from e
                else:
                    # Pointer was NOT replaced. Safe to rollback candidate!
                    logger.error("Commit failed before pointer replacement for generation %d: %s; rolling back.", gen_id, e)
                    if staged_dir.exists():
                        shutil.rmtree(staged_dir, ignore_errors=True)
                    if gen_dir.exists():
                        shutil.rmtree(gen_dir, ignore_errors=True)
                    _durable_fsync_dir(self.config.generations_dir)
                    raise
            else:
                # commit_state is REPLACED_DURABILITY_UNKNOWN or DURABLY_ACKNOWLEDGED:
                # The pointer file already points to gen_dir! NEVER delete gen_dir!
                logger.warning(
                    "Commit error after pointer replacement for generation %d (commit_state=%s): %s. "
                    "Preserving generation %d to avoid dangling pointer.",
                    gen_id, commit_state.value, e, gen_id
                )
                self.adapter = candidate_adapter
                self.manifest = candidate_manifest
                self.current_generation = gen_id
                self._sync_compatibility_paths(committed_idx, committed_man)
                raise IndeterminateCommitError(
                    f"Generation {gen_id} pointer replacement took effect, but post-commit step failed: {e}. "
                    "Generation was preserved."
                ) from e

    def _clone_candidate_engine(self) -> LTHCMEngine:
        """Clone current serving engine in memory without touching serving state."""
        return copy.deepcopy(self.adapter.engine)

    def _normalize_rel_path(self, path: Path) -> str:
        """Convert a path into a stable normalized relative path."""
        loc = self.config.locate_path(path)
        if loc:
            return loc[2]
        return path.name

    def ingest_file(self, path: Path, force: bool = False) -> IngestionReport:
        """
        Ingest a file into Stoma topological memory transactionally under a new generation.
        Builds a candidate pair and commits atomically. If extraction or embedding fails,
        the serving generation remains 100% untouched.
        """
        t0 = time.time()
        loc_raw = self.config.locate_path(path)
        resolved_path = path.resolve()
        loc_resolved = self.config.locate_path(resolved_path)

        if not loc_resolved or not self.config.is_path_allowed(resolved_path) or not loc_raw:
            return IngestionReport(
                path=path,
                rel_path=path.name if not loc_raw else loc_raw[2],
                status="skipped_unsupported",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message="Path or symlink target is not within permitted ingestion roots or contains forbidden components.",
                warning_message="Path or symlink target is not within permitted ingestion roots or contains forbidden components.",
            )

        root_id, root_path, rel_path = loc_resolved
        source_id = f"{root_id}:{rel_path}"
        path = resolved_path

        if not path.is_file():
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message="Target is not a file.",
            )

        plugin = self.plugins.get_plugin_for_file(path)
        if plugin is None:
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="skipped_unsupported",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=f"No source plugin supports format '{path.suffix}'.",
            )

        # Compute content hash
        current_hash = compute_file_sha256(path)
        existing_source = self.manifest.sources.get(source_id)

        # Check if unchanged
        if not force and existing_source is not None:
            if existing_source.content_hash == current_hash:
                return IngestionReport(
                    path=path,
                    rel_path=rel_path,
                    source_id=source_id,
                    status="skipped_unchanged",
                    passages_count=len(existing_source.passage_ids),
                    duration_sec=time.time() - t0,
                )

        # Check safe size cap on non-streaming AST/JSON plugins first
        stat = path.stat()
        pre_size = stat.st_size
        pre_mtime = stat.st_mtime
        if getattr(plugin, "MAX_SAFE_SIZE_BYTES", None) and stat.st_size > plugin.MAX_SAFE_SIZE_BYTES:
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="skipped_unsupported",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=f"File size {stat.st_size} bytes exceeds safe 25MB limit. Memory-unbounded parsing unsupported; source BLOCKED to protect process memory.",
                warning_message="File size exceeds safe parser limit.",
            )

        # Use streaming extraction and streaming chunking without eager preflight
        try:
            raw_stream = self.plugins.extract_file_stream(path, rel_path)
            chunks_stream = self.chunker.chunk_raw_passages_stream(raw_stream)
        except ImageOnlyPDFError as e:
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="skipped_unsupported",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
                warning_message=str(e),
            )
        except UnsupportedDocumentError as e:
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="skipped_unsupported",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
            )
        except (CorruptedDocumentError, ExtractionError) as e:
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
            )

        # If file previously existed: perform clean atomic rebuild transaction
        if existing_source is not None:
            logger.info("File '%s' modified; rebuilding clean index via transactional swap.", source_id)
            try:
                return self._rebuild_clean_index_transaction(
                    updated_files={
                        source_id: (
                            root_id,
                            str(root_path),
                            rel_path,
                            current_hash,
                            stat.st_size,
                            stat.st_mtime,
                            plugin.name,
                            chunks_stream,
                            path,
                        )
                    },
                    original_path=path,
                    rel_path=rel_path,
                    source_id=source_id,
                    t0=t0,
                )
            except (CorruptedDocumentError, ExtractionError) as e:
                logger.error("Transactional rebuild extraction failed for '%s': %s", source_id, e)
                return IngestionReport(
                    path=path,
                    rel_path=rel_path,
                    source_id=source_id,
                    status="failed",
                    passages_count=0,
                    duration_sec=time.time() - t0,
                    error_message=str(e),
                )
            except (ImageOnlyPDFError, UnsupportedDocumentError) as e:
                return IngestionReport(
                    path=path,
                    rel_path=rel_path,
                    source_id=source_id,
                    status="skipped_unsupported",
                    passages_count=0,
                    duration_sec=time.time() - t0,
                    error_message=str(e),
                    warning_message=str(e),
                )

        # Fresh file: build candidate adapter without touching serving engine
        candidate_manifest = ProvenanceManifest.from_dict(self.manifest.to_dict())
        candidate_engine = self._clone_candidate_engine()
        candidate_adapter = StomaLTHCMAdapter(
            engine=candidate_engine,
            embedder=self.embedder,
            manifest=candidate_manifest,
        )

        batch_size = self.config.batch_size
        batch: List[PassageChunk] = []
        now = time.time()
        new_provs: List[PassageProvenance] = []
        passage_ids: List[str] = []
        total_chunks = 0

        def _process_batch(b: List[PassageChunk]) -> None:
            nonlocal total_chunks
            if not b:
                return
            texts = [c.content for c in b]
            embs = self.embedder.encode(texts)
            for j, c in enumerate(b):
                doc_id = f"{root_id}:{rel_path}#{c.chunk_index:04d}"
                passage_ids.append(doc_id)
                candidate_adapter.index_passage(
                    doc_id=doc_id,
                    content=c.content,
                    embedding=embs[j],
                    domain="general",
                    timestamp=now,
                )
                new_provs.append(
                    PassageProvenance(
                        passage_id=doc_id,
                        source_id=source_id,
                        root_id=root_id,
                        root_path=str(root_path),
                        rel_path=rel_path,
                        file_content_hash=current_hash,
                        chunk_index=c.chunk_index,
                        chunk_hash=c.chunk_hash,
                        location=c.location,
                        ingested_at=now,
                        token_count=c.token_count,
                        char_count=c.char_count,
                    )
                )
            total_chunks += len(b)

        try:
            for chunk in chunks_stream:
                batch.append(chunk)
                if len(batch) >= batch_size:
                    _process_batch(batch)
                    batch = []

            if batch:
                _process_batch(batch)
        except ImageOnlyPDFError as e:
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="skipped_unsupported",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
                warning_message=str(e),
            )
        except UnsupportedDocumentError as e:
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="skipped_unsupported",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
            )
        except (CorruptedDocumentError, ExtractionError) as e:
            logger.error("Document extraction for %s failed: %s", path, e)
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
            )
        except Exception as e:
            logger.error("Streaming ingestion for %s failed during batch processing: %s", path, e)
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=f"Stream processing failed: {e}",
            )

        # Verify non-empty file produced passages
        if total_chunks == 0:
            if stat.st_size == 0 and plugin.name in ("text", "code"):
                logger.info("File '%s' is an empty supported text/code file (0 bytes). Recording empty source.", path)
            else:
                return IngestionReport(
                    path=path,
                    rel_path=rel_path,
                    source_id=source_id,
                    status="failed",
                    passages_count=0,
                    duration_sec=time.time() - t0,
                    error_message=f"Extraction produced 0 passages for non-empty file '{path}'.",
                )

        # Verify file consistency post-extraction (Item 9)
        post_stat = path.stat()
        post_hash = compute_file_sha256(path)
        if post_stat.st_size != pre_size or post_stat.st_mtime != pre_mtime or post_hash != current_hash:
            logger.error("File '%s' modified during extraction; aborting commit to prevent partial/inconsistent record.", path)
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=f"File '{path}' modified during extraction.",
            )

        new_source = SourceFileRecord(
            source_id=source_id,
            root_id=root_id,
            root_path=str(root_path),
            rel_path=rel_path,
            content_hash=current_hash,
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
            last_ingested_at=now,
            passage_ids=passage_ids,
            plugin_name=plugin.name,
        )
        candidate_manifest.record_source_and_passages(new_source, new_provs)

        # Commit candidate generation atomically
        next_gen = self.current_generation + 1
        try:
            self._commit_generation(next_gen, candidate_adapter, candidate_manifest)
        except IndeterminateCommitError as e:
            logger.warning("Commit generation %d indeterminate: %s", next_gen, e)
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="indeterminate",
                passages_count=total_chunks,
                duration_sec=time.time() - t0,
                error_message=str(e),
                warning_message="Generation pointer replaced on disk but post-replace durability or ledger append was unacknowledged.",
            )
        except Exception as e:
            logger.error("Commit generation %d failed: %s", next_gen, e)
            return IngestionReport(
                path=path,
                rel_path=rel_path,
                source_id=source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=f"Generation commit failed: {e}",
            )

        return IngestionReport(
            path=path,
            rel_path=rel_path,
            source_id=source_id,
            status="indexed",
            passages_count=total_chunks,
            duration_sec=time.time() - t0,
        )

    def delete_file(self, target: Path | str) -> bool:
        """
        Remove a deleted file's passages transactionally under a new generation.
        Accepts Path, source_id ("root_id:rel_path"), or rel_path.
        Eliminates stale entries from hyperbolic store via atomic swap.
        """
        source_id = None
        if isinstance(target, Path):
            loc = self.config.locate_path(target)
            if loc:
                source_id = f"{loc[0]}:{loc[2]}"
            else:
                source_id = target.name
        else:
            source_id = target
            if ":" not in source_id:
                # Ambiguous rel_path: search in manifest
                for sid, src in self.manifest.sources.items():
                    if src.rel_path == source_id:
                        source_id = sid
                        break

        if source_id not in self.manifest.sources:
            return False

        logger.info("File '%s' deleted; rebuilding clean index via transactional swap.", source_id)
        try:
            self._rebuild_clean_index_transaction(deleted_source_ids={source_id})
            return True
        except RebuildAbortedError as e:
            logger.error("Deletion rebuild aborted for '%s': %s", source_id, e)
            return False

    def move_file(self, src: Path, dest: Path) -> IngestionReport:
        """
        Transactional Move/Rename Protocol (Finding 2):
        1. If dest is outside allowed roots (or external symlink): treat as deletion of src.
        2. If src was outside and dest is inside: treat as new ingest of dest.
        3. If both inside: atomic removal of old ID set + indexing new path under new ID set.
        4. Rebuilds cleanly in one atomic generation commit.
        """
        t0 = time.time()
        src_loc = self.config.locate_path(src)
        dest_loc = self.config.locate_path(dest)

        if not src_loc and not dest_loc:
            return IngestionReport(
                path=dest,
                rel_path=dest.name,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message="Neither source nor destination is inside allowed roots.",
            )

        if src_loc and not dest_loc:
            # Move out of allowed roots: treat as deletion
            src_source_id = f"{src_loc[0]}:{src_loc[2]}"
            deleted = self.delete_file(src_source_id)
            return IngestionReport(
                path=dest,
                rel_path=src_loc[2],
                source_id=src_source_id,
                status="deleted" if deleted else "failed",
                passages_count=0,
                duration_sec=time.time() - t0,
            )

        if not src_loc and dest_loc:
            # Move in from outside: treat as fresh ingest
            return self.ingest_file(dest)

        # Both inside allowed roots: transactional move
        assert src_loc is not None and dest_loc is not None
        src_root_id, src_root_path, src_rel = src_loc
        dest_root_id, dest_root_path, dest_rel = dest_loc

        old_source_id = f"{src_root_id}:{src_rel}"
        new_source_id = f"{dest_root_id}:{dest_rel}"

        if not dest.is_file():
            return IngestionReport(
                path=dest,
                rel_path=dest_rel,
                source_id=new_source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=f"Destination file '{dest}' does not exist on disk.",
            )

        plugin = self.plugins.get_plugin_for_file(dest)
        if plugin is None:
            return IngestionReport(
                path=dest,
                rel_path=dest_rel,
                source_id=new_source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=f"No source plugin supports format '{dest.suffix}'.",
            )

        dest_stat = dest.stat()
        if getattr(plugin, "MAX_SAFE_SIZE_BYTES", None) and dest_stat.st_size > plugin.MAX_SAFE_SIZE_BYTES:
            return IngestionReport(
                path=dest,
                rel_path=dest_rel,
                source_id=new_source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=f"Destination file size {dest_stat.st_size} bytes exceeds safe 25MB limit.",
            )

        # Remove eager PDF preflight - streaming handles validation page by page
        try:
            raw_stream = self.plugins.extract_file_stream(dest, dest_rel)
            chunks_stream = self.chunker.chunk_raw_passages_stream(raw_stream)
        except (ImageOnlyPDFError, UnsupportedDocumentError) as e:
            return IngestionReport(
                path=dest,
                rel_path=dest_rel,
                source_id=new_source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
            )
        except (CorruptedDocumentError, ExtractionError) as e:
            return IngestionReport(
                path=dest,
                rel_path=dest_rel,
                source_id=new_source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
            )

        dest_hash = compute_file_sha256(dest)

        logger.info("Executing transactional move: %s -> %s", old_source_id, new_source_id)
        try:
            return self._rebuild_clean_index_transaction(
                deleted_source_ids={old_source_id, new_source_id},
                updated_files={
                    new_source_id: (
                        dest_root_id,
                        str(dest_root_path),
                        dest_rel,
                        dest_hash,
                        dest_stat.st_size,
                        dest_stat.st_mtime,
                        plugin.name,
                        chunks_stream,
                        dest,
                    )
                },
                original_path=dest,
                rel_path=dest_rel,
                source_id=new_source_id,
                t0=t0,
                report_status="moved",
            )
        except (CorruptedDocumentError, ExtractionError) as e:
            logger.error("Transactional move extraction failed for '%s' -> '%s': %s", old_source_id, new_source_id, e)
            return IngestionReport(
                path=dest,
                rel_path=dest_rel,
                source_id=new_source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
            )
        except (ImageOnlyPDFError, UnsupportedDocumentError) as e:
            return IngestionReport(
                path=dest,
                rel_path=dest_rel,
                source_id=new_source_id,
                status="skipped_unsupported",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
                warning_message=str(e),
            )
        except RebuildAbortedError as e:
            logger.error("Transactional move rebuild aborted for '%s' -> '%s': %s", old_source_id, new_source_id, e)
            return IngestionReport(
                path=dest,
                rel_path=dest_rel,
                source_id=new_source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
            )

    def _rebuild_clean_index_transaction(
        self,
        deleted_source_ids: Optional[Set[str]] = None,
        updated_files: Optional[Dict[str, Tuple[Any, ...]]] = None,
        original_path: Optional[Path] = None,
        rel_path: Optional[str] = None,
        source_id: Optional[str] = None,
        t0: Optional[float] = None,
        report_status: str = "indexed",
    ) -> IngestionReport:
        """
        Transactional Rebuild Protocol (Finding 1, 2, 5, 6):
        Builds a completely fresh LTHCMEngine from scratch using candidate state.
        Safe Policy: If any unchanged file is missing on disk or fails extraction,
        ABORTS transaction immediately, discards candidate state, and preserves the serving generation.
        """
        t0 = t0 or time.time()
        deleted_source_ids = deleted_source_ids or set()
        updated_files = updated_files or {}

        try:
            # Clone candidate manifest
            candidate_manifest = ProvenanceManifest.from_dict(self.manifest.to_dict())

            # Remove deleted sources
            for sid in deleted_source_ids:
                candidate_manifest.remove_source(sid)

            # Create fresh candidate engine
            candidate_engine = StomaLTHCMAdapter.create_fresh_engine(self.embedder)
            candidate_adapter = StomaLTHCMAdapter(
                engine=candidate_engine,
                embedder=self.embedder,
                manifest=candidate_manifest,
            )

            all_target_sources = set(candidate_manifest.sources.keys()).union(updated_files.keys())

            # Collect and index chunks in bounded batches
            batch_size = self.config.batch_size
            chunks_batch: List[Tuple[str, str, float]] = []  # (doc_id, content, timestamp)

            def flush_batch() -> None:
                if not chunks_batch:
                    return
                texts = [item[1] for item in chunks_batch]
                embs = self.embedder.encode(texts)
                for j, (doc_id, content, ts) in enumerate(chunks_batch):
                    candidate_adapter.index_passage(
                        doc_id=doc_id,
                        content=content,
                        embedding=embs[j],
                        domain="general",
                        timestamp=ts,
                    )
                chunks_batch.clear()

            now = time.time()

            for sid in sorted(all_target_sources):
                if sid in updated_files:
                    entry = updated_files[sid]
                    u_root_id = entry[0]
                    u_root_path = entry[1]
                    u_rel = entry[2]
                    u_hash = entry[3]
                    u_size = entry[4]
                    u_mtime = entry[5]
                    u_plugin = entry[6]
                    u_chunks = entry[7]
                    u_file_path = entry[8] if len(entry) > 8 else None

                    u_provs: List[PassageProvenance] = []
                    u_pids: List[str] = []

                    try:
                        for c in u_chunks:
                            doc_id = f"{u_root_id}:{u_rel}#{c.chunk_index:04d}"
                            u_pids.append(doc_id)
                            chunks_batch.append((doc_id, c.content, now))
                            if len(chunks_batch) >= batch_size:
                                flush_batch()

                            u_provs.append(
                                PassageProvenance(
                                    passage_id=doc_id,
                                    source_id=sid,
                                    root_id=u_root_id,
                                    root_path=u_root_path,
                                    rel_path=u_rel,
                                    file_content_hash=u_hash,
                                    chunk_index=c.chunk_index,
                                    chunk_hash=c.chunk_hash,
                                    location=c.location,
                                    ingested_at=now,
                                    token_count=c.token_count,
                                    char_count=c.char_count,
                                )
                            )
                    except (ImageOnlyPDFError, UnsupportedDocumentError, CorruptedDocumentError, ExtractionError):
                        raise

                    # Post-extraction verification for updated file
                    if u_file_path is not None and u_file_path.exists():
                        post_stat = u_file_path.stat()
                        post_hash = compute_file_sha256(u_file_path)
                        if post_stat.st_size != u_size or post_stat.st_mtime != u_mtime or post_hash != u_hash:
                            raise RebuildAbortedError(f"File '{u_file_path}' modified during rebuild extraction.")

                    u_rec = SourceFileRecord(
                        source_id=sid,
                        root_id=u_root_id,
                        root_path=u_root_path,
                        rel_path=u_rel,
                        content_hash=u_hash,
                        size_bytes=u_size,
                        mtime=u_mtime,
                        last_ingested_at=now,
                        passage_ids=u_pids,
                        plugin_name=u_plugin,
                    )
                    candidate_manifest.record_source_and_passages(u_rec, u_provs)

                else:
                    # Existing unchanged file: re-read from disk to preserve 100% fidelity
                    src_rec = candidate_manifest.sources[sid]
                    file_path = self.config.resolve_source_path(src_rec.root_id, src_rec.rel_path)

                    if file_path is None or not file_path.is_file():
                        raise RebuildAbortedError(
                            f"Rebuild aborted: source file '{src_rec.rel_path}' in root '{src_rec.root_id}' "
                            f"is missing from disk. Aborting rebuild to preserve last good generation {self.current_generation}."
                        )

                    plugin = self.plugins.get_plugin_for_file(file_path)
                    if plugin is None:
                        raise RebuildAbortedError(
                            f"Rebuild aborted: source file '{src_rec.rel_path}' format unsupported. "
                            f"Aborting rebuild to preserve last good generation {self.current_generation}."
                        )

                    pre_unchanged_stat = file_path.stat()
                    if pre_unchanged_stat.st_size != src_rec.size_bytes:
                        raise RebuildAbortedError(
                            f"Rebuild aborted: source file '{src_rec.rel_path}' was modified on disk prior to rebuild. "
                            f"Aborting rebuild to preserve last good generation {self.current_generation}."
                        )

                    pre_unchanged_hash = compute_file_sha256(file_path)
                    if pre_unchanged_hash != src_rec.content_hash:
                        raise RebuildAbortedError(
                            f"Rebuild aborted: source file '{src_rec.rel_path}' content hash mismatch before extraction "
                            f"(expected {src_rec.content_hash}, got {pre_unchanged_hash}). "
                            f"Aborting rebuild to preserve last good generation {self.current_generation}."
                        )

                    try:
                        raw_stream = self.plugins.extract_file_stream(file_path, src_rec.rel_path)
                        chunks_stream = self.chunker.chunk_raw_passages_stream(raw_stream)
                        u_pids = []
                        u_provs = []
                        for c in chunks_stream:
                            doc_id = f"{src_rec.root_id}:{src_rec.rel_path}#{c.chunk_index:04d}"
                            u_pids.append(doc_id)
                            chunks_batch.append((doc_id, c.content, src_rec.last_ingested_at))
                            if len(chunks_batch) >= batch_size:
                                flush_batch()

                            u_provs.append(
                                PassageProvenance(
                                    passage_id=doc_id,
                                    source_id=sid,
                                    root_id=src_rec.root_id,
                                    root_path=src_rec.root_path,
                                    rel_path=src_rec.rel_path,
                                    file_content_hash=src_rec.content_hash,
                                    chunk_index=c.chunk_index,
                                    chunk_hash=c.chunk_hash,
                                    location=c.location,
                                    ingested_at=src_rec.last_ingested_at,
                                    token_count=c.token_count,
                                    char_count=c.char_count,
                                    )
                            )
                    except Exception as e:
                        raise RebuildAbortedError(
                            f"Rebuild aborted: extraction failed for unchanged file '{src_rec.rel_path}': {e}. "
                            f"Aborting rebuild to preserve last good generation {self.current_generation}."
                        )

                    post_unchanged_stat = file_path.stat()
                    if post_unchanged_stat.st_size != pre_unchanged_stat.st_size or post_unchanged_stat.st_mtime != pre_unchanged_stat.st_mtime:
                        raise RebuildAbortedError(
                            f"Rebuild aborted: source file '{src_rec.rel_path}' modified during rebuild extraction."
                        )
                    post_unchanged_hash = compute_file_sha256(file_path)
                    if post_unchanged_hash != pre_unchanged_hash:
                        raise RebuildAbortedError(
                            f"Rebuild aborted: source file '{src_rec.rel_path}' content hash changed during rebuild extraction."
                        )

                    src_rec.passage_ids = u_pids
                    candidate_manifest.record_source_and_passages(src_rec, u_provs)

            # Flush any remaining chunks
            flush_batch()

            # Atomic durable commit to new generation
            next_gen = self.current_generation + 1
            self._commit_generation(next_gen, candidate_adapter, candidate_manifest)

            report_p = original_path or Path(rel_path or "unknown")
            passages_cnt = 0
            if source_id and source_id in candidate_manifest.sources:
                passages_cnt = len(candidate_manifest.sources[source_id].passage_ids)

            return IngestionReport(
                path=report_p,
                rel_path=rel_path or report_p.name,
                source_id=source_id,
                status=report_status,
                passages_count=passages_cnt,
                duration_sec=time.time() - t0,
            )

        except IndeterminateCommitError as e:
            logger.warning("Clean index rebuild transaction indeterminate: %s", e)
            report_p = original_path or Path(rel_path or "unknown")
            return IngestionReport(
                path=report_p,
                rel_path=rel_path or report_p.name,
                source_id=source_id,
                status="indeterminate",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=str(e),
                warning_message="Rebuild commit succeeded on disk pointer but post-commit durability or ledger append was unacknowledged.",
            )
        except RebuildAbortedError as e:
            logger.error("Clean index rebuild transaction aborted: %s", e)
            raise
        except Exception as e:
            logger.error("Commit generation rebuild transaction failed: %s", e)
            report_p = original_path or Path(rel_path or "unknown")
            return IngestionReport(
                path=report_p,
                rel_path=rel_path or report_p.name,
                source_id=source_id,
                status="failed",
                passages_count=0,
                duration_sec=time.time() - t0,
                error_message=f"Generation commit rebuild failed: {e}",
            )

    def scan_and_ingest_directory(self, dir_path: Path) -> FolderIngestSummary:
        """
        Recursively scan a directory and ingest supported files in bounded batches (Finding 5).
        Keeps memory bounded: reports are stored in a ring buffer, and folder traversal is bounded.
        """
        t0 = time.time()
        dir_path = dir_path.resolve()

        if not self.config.is_path_allowed(dir_path):
            return FolderIngestSummary(
                total_files_scanned=0,
                files_indexed=0,
                files_skipped_unchanged=0,
                files_skipped_unsupported=0,
                files_failed=1,
                total_passages_indexed=0,
                duration_sec=time.time() - t0,
                recent_reports=[
                    IngestionReport(
                        path=dir_path,
                        rel_path=dir_path.name,
                        status="failed",
                        passages_count=0,
                        duration_sec=0.0,
                        error_message="Directory is outside permitted ingestion roots.",
                    )
                ],
            )

        total_scanned = 0
        indexed_count = 0
        skipped_unchanged = 0
        skipped_unsupported = 0
        failed_count = 0
        total_passages = 0

        # Ring buffer of max 100 recent reports to keep RAM strictly O(1)
        recent_reports: collections.deque = collections.deque(maxlen=100)

        for root_str, dirs, files in os.walk(dir_path, followlinks=False):
            # Prune forbidden directories in-place
            dirs[:] = [
                d for d in dirs
                if d not in self.config.forbidden_dir_names
                and self.config.is_path_allowed(Path(root_str) / d)
            ]

            for fname in files:
                fpath = Path(root_str) / fname
                if not self.config.is_path_allowed(fpath):
                    continue

                total_scanned += 1
                try:
                    rep = self.ingest_file(fpath)
                    recent_reports.append(rep)
                    if rep.status == "indexed":
                        indexed_count += 1
                        total_passages += rep.passages_count
                    elif rep.status == "skipped_unchanged":
                        skipped_unchanged += 1
                    elif rep.status == "skipped_unsupported":
                        skipped_unsupported += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error("Error ingesting %s: %s", fpath, e)
                    failed_count += 1
                    recent_reports.append(
                        IngestionReport(
                            path=fpath,
                            rel_path=fpath.name,
                            status="failed",
                            passages_count=0,
                            duration_sec=0.0,
                            error_message=str(e),
                        )
                    )

        return FolderIngestSummary(
            total_files_scanned=total_scanned,
            files_indexed=indexed_count,
            files_skipped_unchanged=skipped_unchanged,
            files_skipped_unsupported=skipped_unsupported,
            files_failed=failed_count,
            total_passages_indexed=total_passages,
            duration_sec=time.time() - t0,
            recent_reports=list(recent_reports),
        )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        domain: str = "general",
        query_timestamp: Optional[float] = None,
        enforce_causality: bool = True,
        allow_causal_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Query Stoma topological memory through the serving adapter."""
        return self.adapter.query(
            query_text=query_text,
            top_k=top_k,
            domain=domain,
            query_timestamp=query_timestamp,
            enforce_causality=enforce_causality,
            allow_causal_fallback=allow_causal_fallback,
        )

    def save(self) -> None:
        """Manual save: commits current serving adapter and manifest."""
        self._commit_generation(self.current_generation + 1, self.adapter, self.manifest)
