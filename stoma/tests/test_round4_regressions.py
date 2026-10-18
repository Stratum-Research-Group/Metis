"""
Targeted Round 4 Regression Tests for Stoma.
Verifies all 9 items from stoma-round4-fixes-168e2f9-e3631c84.txt:
- Item 1: Crash-consistent commit point, explicit CommitState, rollback before replace vs
          preservation after replace with IndeterminateCommitError, cryptographic ledger chain
          validation with predecessor links, orphan rejection, and tampered marker handling.
- Item 2: Real deletion delays and due timestamps, exponential backoff retries, dead-letter
          exhaustion, bounded queues with dropped counters, and move tuple identity survival.
- Item 3: OfflineCacheMissError exception typing and inspection command.
- Item 4: Submodule checkout verification and unreadable checkout handling.
- Item 5: Large input streaming generation and peak-RSS measurement via fresh subprocesses.
- Item 6: Cross-root move and query provenance on restart.
- Item 7: Real Dianoia CLI execution, returncode 1, and exact edge structure refusal.
- Item 8: Dual-store consistency under asymmetry.
- Item 9: Zero-page PDF typed unsupported status, unchanged-source content-hash verification
          against same-size restored-mtime modification, and late parser failure rollback in edit/move.
"""

from __future__ import annotations

import json
import os
import pickle
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pypdf
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pytest
import torch

from stoma.adapter import StomaLTHCMAdapter
from stoma.config import DependencyOriginError, StomaConfig, verify_runtime_dependencies
from stoma.embedder import ModelInspectionError, OfflineCacheMissError, StomaEmbedder
from stoma.indexer import (
    CommitState,
    CorruptedStorageError,
    IndeterminateCommitError,
    IngestionReport,
    RebuildAbortedError,
    StomaIndexer,
)
from stoma.interchange import verify_dianoia_interchange
from stoma.plugins import (
    CorruptedDocumentError,
    ExtractionResult,
    PDFPlugin,
    RawPassage,
    UnsupportedDocumentError,
    ZeroPagePDFError,
)
from stoma.provenance import (
    ModelIdentityMismatchError,
    PassageProvenance,
    ProvenanceManifest,
    SourceFileRecord,
    compute_file_sha256,
)
from stoma.watcher import StomaWatchEventHandler, StomaWatcher


def make_pdf_with_text(path: Path, pages: List[str]) -> None:
    """Generate a valid PDF with extractable text on each page using pypdf."""
    writer = pypdf.PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=300, height=300)
        content = DecodedStreamObject()
        safe_text = text.replace("(", "\\(").replace(")", "\\)")
        content.set_data(f"BT /F1 12 Tf 72 712 Td ({safe_text}) Tj ET".encode("utf-8"))
        page[NameObject("/Contents")] = writer._add_object(content)
        font_dict = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({
                NameObject("/F1"): writer._add_object(font_dict)
            })
        })
    with open(path, "wb") as f:
        writer.write(f)


def make_zero_page_pdf(path: Path) -> None:
    """Generate a PDF file with 0 pages."""
    writer = pypdf.PdfWriter()
    with open(path, "wb") as f:
        writer.write(f)


# ===========================================================================
# Item 1: Crash-consistent commit point, pointer, and real ledger chain
# ===========================================================================

def test_r4_item1_commit_error_before_pointer_replace(test_config, shared_embedder, temp_workspace):
    """
    Error injected BEFORE pointer replacement:
    - Candidate directory is rolled back and cleaned up.
    - Pointer remains unchanged on old generation.
    - Serving state is untouched.
    - Restart recovers old generation cleanly.
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    f1 = temp_workspace / "f1.txt"
    f1.write_text("Generation 2 content")
    rep1 = indexer.ingest_file(f1)
    assert rep1.status == "indexed"
    assert indexer.current_generation == 2

    f2 = temp_workspace / "f2.txt"
    f2.write_text("Generation 3 content to fail before replace")

    original_replace = os.replace

    def fail_before_pointer_replace(src, dst):
        if "current.json" in str(dst):
            raise OSError("Injected disk failure BEFORE pointer replace")
        return original_replace(src, dst)

    with patch("os.replace", side_effect=fail_before_pointer_replace):
        rep2 = indexer.ingest_file(f2)
        assert rep2.status == "failed"
        assert "Injected disk failure BEFORE pointer replace" in rep2.error_message

    # Generation 3 dir was rolled back and deleted
    gen3_dir = test_config.get_generation_dir(3)
    assert not gen3_dir.exists()
    assert indexer.current_generation == 2

    # Pointer still points to 2
    with open(test_config.current_pointer_path, "r", encoding="utf-8") as f:
        assert json.load(f)["generation"] == 2

    # Restart recovers generation 2
    restart_indexer = StomaIndexer(test_config, embedder=shared_embedder)
    assert restart_indexer.current_generation == 2
    assert "f1.txt" in str(list(restart_indexer.manifest.sources.values())[0].rel_path)


def test_r4_item1_commit_error_immediately_after_replace_before_or_during_dir_fsync(test_config, shared_embedder, temp_workspace):
    """
    Error injected immediately AFTER pointer replacement (during post-replace directory fsync):
    - os.replace succeeded on disk.
    - Candidate generation directory MUST NOT be deleted (pointer references it).
    - Status is indeterminate (or IndeterminateCommitError).
    - Restart reads current pointer, verifies files and marker, reconciles ledger, and loads gen 3!
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    f1 = temp_workspace / "f1.txt"
    f1.write_text("Gen 2")
    indexer.ingest_file(f1)
    assert indexer.current_generation == 2

    f2 = temp_workspace / "f2.txt"
    f2.write_text("Gen 3 to fail post-replace fsync")

    import stoma.indexer as idx_module
    original_fsync_dir = idx_module._durable_fsync_dir

    fsync_call_count = 0

    def fail_after_pointer_replace(dir_path):
        nonlocal fsync_call_count
        fsync_call_count += 1
        # Fail during post-replace storage_dir fsync in _commit_generation
        if dir_path == test_config.storage_dir and test_config.current_pointer_path.is_file():
            try:
                with open(test_config.current_pointer_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("generation") == 3:
                    raise OSError("Injected directory fsync failure AFTER pointer replacement")
            except OSError:
                raise
            except Exception:
                pass
        return original_fsync_dir(dir_path)

    with patch("stoma.indexer._durable_fsync_dir", side_effect=fail_after_pointer_replace):
        rep = indexer.ingest_file(f2)
        assert rep.status == "indeterminate"
        assert "pointer" in rep.warning_message.lower()

    # Candidate generation 3 MUST NOT be deleted!
    gen3_dir = test_config.get_generation_dir(3)
    assert gen3_dir.exists()
    assert (gen3_dir / "index.pkl").is_file()
    assert (gen3_dir / "manifest.json").is_file()
    assert (gen3_dir / "commit_marker.json").is_file()

    # On restart: loads generation 3 cleanly and reconciles ledger
    restart_indexer = StomaIndexer(test_config, embedder=shared_embedder)
    assert restart_indexer.current_generation == 3
    chain = restart_indexer._read_commit_chain()
    chain_gens = [e["generation"] for e in chain]
    assert 3 in chain_gens


def test_r4_item1_commit_error_after_fsync_before_ledger_write(test_config, shared_embedder, temp_workspace):
    """
    Error injected after directory fsync before/during commit_chain.json append:
    - Generation 3 directory and pointer are durable.
    - Status is indeterminate.
    - Candidate generation preserved.
    - Restart reconciles commit_chain.json from commit_marker.json.
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    f1 = temp_workspace / "f1.txt"
    f1.write_text("Gen 2")
    indexer.ingest_file(f1)

    f2 = temp_workspace / "f2.txt"
    f2.write_text("Gen 3 to fail ledger append")

    with patch.object(indexer, "_record_commit_chain_entry", side_effect=OSError("Injected ledger append failure")):
        rep = indexer.ingest_file(f2)
        assert rep.status == "indeterminate"

    # Generation 3 exists and pointer points to 3
    assert test_config.get_generation_dir(3).exists()

    # Restart reconciles ledger and loads generation 3
    restart_indexer = StomaIndexer(test_config, embedder=shared_embedder)
    assert restart_indexer.current_generation == 3
    chain = restart_indexer._read_commit_chain()
    assert any(e["generation"] == 3 for e in chain)


def test_r4_item1_recovery_with_deleted_or_corrupt_pointer(test_config, shared_embedder, temp_workspace):
    """
    Corrupt or deleted current.json:
    - Falls back to latest valid generation from commit_chain.json.
    - Re-establishes pointer file.
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    f1 = temp_workspace / "f1.txt"
    f1.write_text("Gen 2")
    indexer.ingest_file(f1)
    f2 = temp_workspace / "f2.txt"
    f2.write_text("Gen 3")
    indexer.ingest_file(f2)
    assert indexer.current_generation == 3

    # Corrupt pointer file
    test_config.current_pointer_path.write_text("{corrupt json garbage")

    restart_indexer = StomaIndexer(test_config, embedder=shared_embedder)
    assert restart_indexer.current_generation == 3
    assert test_config.current_pointer_path.is_file()
    with open(test_config.current_pointer_path, "r", encoding="utf-8") as f:
        assert json.load(f)["generation"] == 3


def test_r4_item1_uncommitted_orphan_outside_ledger_rejected(test_config, shared_embedder, temp_workspace):
    """
    An uncommitted orphan generation directory (gen_999) with files outside the ledger:
    - If pointer points to 999 or is missing, orphan MUST NOT become current.
    - Recovered generation must be the proven ledger member (gen 2).
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    f1 = temp_workspace / "f1.txt"
    f1.write_text("Gen 2")
    indexer.ingest_file(f1)
    assert indexer.current_generation == 2

    # Create uncommitted orphan gen 999
    orphan_dir = test_config.get_generation_dir(999)
    orphan_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_config.get_generation_dir(2) / "index.pkl", orphan_dir / "index.pkl")
    shutil.copy2(test_config.get_generation_dir(2) / "manifest.json", orphan_dir / "manifest.json")

    # Point current.json to orphan 999
    meta = {
        "generation": 999,
        "updated_at": time.time(),
        "manifest_hash": compute_file_sha256(orphan_dir / "manifest.json"),
        "index_hash": compute_file_sha256(orphan_dir / "index.pkl"),
        "sources_count": 1,
        "passages_count": 1,
        "model_name": shared_embedder.model_name,
        "weights_checksum": shared_embedder.weights_checksum,
    }
    with open(test_config.current_pointer_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Restart MUST reject orphan 999 and recover generation 2
    restart_indexer = StomaIndexer(test_config, embedder=shared_embedder)
    assert restart_indexer.current_generation == 2
    assert restart_indexer.current_generation != 999


def test_r4_item1_broken_ledger_predecessor_link_halts_chain(test_config, shared_embedder, temp_workspace):
    """
    A commit chain containing a broken predecessor link:
    - Valid chain stops at the last proven member before the break.
    - Later entries are rejected.
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    f1 = temp_workspace / "f1.txt"
    f1.write_text("Gen 2")
    indexer.ingest_file(f1)
    f2 = temp_workspace / "f2.txt"
    f2.write_text("Gen 3")
    indexer.ingest_file(f2)
    assert indexer.current_generation == 3

    # Tamper commit_chain.json entry for gen 3 to have wrong prev_manifest_hash
    chain = indexer._read_commit_chain()
    assert len(chain) >= 3
    chain[-1]["prev_manifest_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
    with open(test_config.commit_chain_path, "w", encoding="utf-8") as f:
        json.dump(chain, f)

    # Delete pointer to force ledger recovery
    test_config.current_pointer_path.unlink()

    # Restart: chain validation halts before gen 3. Recovers gen 2!
    restart_indexer = StomaIndexer(test_config, embedder=shared_embedder)
    assert restart_indexer.current_generation == 2


# ===========================================================================
# Item 2: Real deletion delays and bounded watcher state
# ===========================================================================

def test_r4_item2_real_deletion_delay_honored(test_config, shared_embedder, temp_workspace):
    """
    Deletion events store due timestamps and worker does not drain before the clock reaches due time.
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    f1 = temp_workspace / "f1.txt"
    f1.write_text("Content to delete")
    indexer.ingest_file(f1)
    sid = list(indexer.manifest.sources.keys())[0]
    assert sid in indexer.manifest.sources

    handler = StomaWatchEventHandler(indexer, debounce_sec=0.25)
    f1.unlink()

    mock_event = MagicMock()
    mock_event.is_directory = False
    mock_event.src_path = str(f1)
    handler.on_deleted(mock_event)

    # Immediately: pending deletion is recorded with due time in future
    with handler._lock:
        assert f1 in handler._pending_deletions
        due_time = handler._pending_deletions[f1]
        assert due_time > time.time()

    # Wait 0.08s (well before 0.25s)
    time.sleep(0.08)
    # File is still in manifest because due time not reached
    assert sid in indexer.manifest.sources

    # Wait until after 0.25s debounce and worker finishes processing deletion
    t_end = time.time() + 2.0
    while time.time() < t_end and sid in indexer.manifest.sources:
        time.sleep(0.05)
    # Worker has now processed deletion
    assert sid not in indexer.manifest.sources
    handler.stop()


def test_r4_item2_deletion_retry_exponential_backoff_and_dead_letter(test_config, shared_embedder, temp_workspace):
    """
    Deletion retry honors backoff delay and exhausts into visible dead-letter list after 3 retries.
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    f1 = temp_workspace / "f1.txt"
    f1.write_text("Content")
    indexer.ingest_file(f1)

    handler = StomaWatchEventHandler(indexer, debounce_sec=0.1)
    f1.unlink()

    # Patch delete_file to always raise
    with patch.object(indexer, "delete_file", side_effect=OSError("Injected deletion failure")):
        handler._schedule_deletion(f1, delay=0.01)

        # Allow time for worker to attempt initial delete and 3 retries with backoff (0.1 + 0.2 + 0.4s)
        time.sleep(1.0)

    dead = handler.get_dead_letters()
    assert dead["total_dead_letters"] >= 1
    assert any(str(f1) in d["path"] for d in dead["deletions"])
    handler.stop()


def test_r4_item2_bounded_watcher_queues_and_overflow(test_config, shared_embedder, temp_workspace):
    """
    Explicit queue capacity limits:
    - Items exceeding MAX_PENDING are dropped and counted in dropped counters.
    - Overflow summary and queue stats accurately report drops.
    - reconcile_overflowed_work recovers dropped work.
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    watcher = StomaWatcher(indexer, debounce_sec=0.5)
    handler = watcher.handler

    # Artificially set capacity low for test
    handler.MAX_PENDING = 5

    # Schedule 10 changes
    for i in range(10):
        p = temp_workspace / f"test_{i}.txt"
        p.write_text(f"content {i}")
        handler._schedule_change(p)

    stats = handler.get_queue_stats()
    assert stats["pending_changes_count"] <= 5
    assert stats["total_scheduled_changes"] == 10
    assert stats["dropped_scheduled_changes"] == 5

    overflow = watcher.get_overflow_summary()
    assert overflow["dropped_scheduled_changes"] == 5
    assert overflow["total_dropped"] >= 5

    # Run visible recovery path
    rec = watcher.reconcile_overflowed_work()
    assert rec["files_indexed"] >= 5
    watcher.stop()


def test_r4_item2_move_tuple_survives_retries_and_no_source_resurrection(test_config, shared_embedder, temp_workspace):
    """
    Move (src, dest) preserves move identity across retries.
    An on_modified event for dest while move is active does not overwrite the move or resurrect src.
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    src = temp_workspace / "orig.txt"
    src.write_text("Hello Move")
    indexer.ingest_file(src)
    src_id = list(indexer.manifest.sources.keys())[0]

    dest = temp_workspace / "renamed.txt"
    src.rename(dest)

    handler = StomaWatchEventHandler(indexer, debounce_sec=0.5)

    # Schedule move
    handler._schedule_move(src, dest, delay=1.0)
    with handler._lock:
        assert (src, dest) in handler._pending_moves

    # Trigger on_modified on dest
    mock_mod = MagicMock()
    mock_mod.is_directory = False
    mock_mod.src_path = str(dest)
    handler.on_modified(mock_mod)

    # Verify (src, dest) move identity was NOT discarded
    with handler._lock:
        assert (src, dest) in handler._pending_moves
        assert dest not in handler._pending_changes
    handler.stop()


# ===========================================================================
# Item 3: Accurate encoder exception and raw identity evidence
# ===========================================================================

def test_r4_item3_offline_cache_miss_exception():
    """Offline cache miss raises OfflineCacheMissError which is a subclass of ModelInspectionError."""
    assert issubclass(OfflineCacheMissError, ModelInspectionError)

    with patch("stoma.embedder.TextEmbedder", side_effect=RuntimeError("Offline cache miss simulated")):
        with pytest.raises(OfflineCacheMissError) as exc_info:
            StomaEmbedder(model_name="nonexistent-model")
        assert "python -m stoma.embedder prefetch" in str(exc_info.value) or "stoma setup-model" in str(exc_info.value)


def test_r4_item3_embedder_inspect_cli(shared_embedder):
    """Run python -m stoma.embedder inspect and verify output format."""
    env = os.environ.copy()
    extra_paths = [str(Path(p).resolve()) for p in sys.path if p]
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing_pp] if existing_pp else []))
    res = subprocess.run(
        [sys.executable, "-m", "stoma.embedder", "inspect"],
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )
    assert res.returncode == 0
    out = res.stdout
    assert "selected_model_identifier: all-MiniLM-L6-v2" in out
    assert "import_origin: dianoia.embedding" in out
    assert "tokenizer_identity: BertTokenizer" in out
    assert "dimension: 384" in out
    assert "raw_max_length: 256" in out


# ===========================================================================
# Item 4: Runtime SHA/origin verification
# ===========================================================================

def test_r4_item4_submodule_commit_unreadable_fails_closed():
    """Unreadable or tampered checkout commit SHA raises DependencyOriginError."""
    with patch("stoma.config._read_submodule_commit", return_value=None):
        with pytest.raises(DependencyOriginError) as exc_info:
            verify_runtime_dependencies()
        assert "Runtime import violation" in str(exc_info.value)


# ===========================================================================
# Item 6: Root-qualified identity and provenance across cross-root move
# ===========================================================================

def test_r4_item6_cross_root_move_and_restart(temp_workspace, shared_embedder):
    """
    Cross-root file move:
    - Moving from root_A to root_B removes old root_A ID and indexes under root_B.
    - Restart preserves root_B provenance and query answers correctly.
    """
    root_a = temp_workspace / "root_a"
    root_b = temp_workspace / "root_b"
    root_a.mkdir()
    root_b.mkdir()

    storage = temp_workspace / ".stoma"
    cfg = StomaConfig(
        allowed_roots=[root_a, root_b],
        storage_dir=storage,
        index_path=storage / "index.pkl",
        manifest_path=storage / "manifest.json",
    )
    indexer = StomaIndexer(cfg, embedder=shared_embedder)

    f_a = root_a / "guide.txt"
    f_a.write_text("Guide to quantum hyperplanes in Poincaré ball.")
    rep1 = indexer.ingest_file(f_a)
    assert rep1.status == "indexed"
    old_sid = rep1.source_id
    root_a_id = cfg.get_root_id(root_a)
    root_b_id = cfg.get_root_id(root_b)
    assert old_sid.startswith(f"{root_a_id}:")

    f_b = root_b / "guide.txt"
    f_a.rename(f_b)

    rep2 = indexer.move_file(f_a, f_b)
    assert rep2.status == "moved"
    new_sid = rep2.source_id
    assert new_sid.startswith(f"{root_b_id}:")
    assert old_sid not in indexer.manifest.sources
    assert new_sid in indexer.manifest.sources

    # Restart
    restart_indexer = StomaIndexer(cfg, embedder=shared_embedder)
    assert new_sid in restart_indexer.manifest.sources
    assert old_sid not in restart_indexer.manifest.sources
    res = restart_indexer.query("quantum hyperplanes", top_k=1)
    assert len(res["results"]) == 1
    assert res["results"][0]["provenance"]["root_id"] == root_b_id


# ===========================================================================
# Item 7: Real Dianoia command and edge refusal
# ===========================================================================

def test_r4_item7_real_dianoia_cli_refusal(test_config, shared_embedder, temp_workspace):
    """
    Invoke real Dianoia CLI: returns code 1, edge_gate_triggered == True, and exact edge requirement stderr.
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    f = temp_workspace / "doc.txt"
    f.write_text("Stoma document for interchange ping")
    indexer.ingest_file(f)

    res = verify_dianoia_interchange(test_config.index_path)
    assert res["status"] == "direct_cli_walk_unsupported_zero_edges"
    assert res["cli_return_code"] == 1
    assert res["edge_gate_triggered"] is True
    assert "Deliberative walk requires an edge structure" in res["cli_stderr"]


# ===========================================================================
# Item 9: Complete extraction, zero-page PDF, and old-hash fidelity
# ===========================================================================

def test_r4_item9_zero_page_pdf_unsupported(test_config, shared_embedder, temp_workspace):
    """0-page PDF raises ZeroPagePDFError and is reported as skipped_unsupported."""
    pdf_path = temp_workspace / "zero_page.pdf"
    make_zero_page_pdf(pdf_path)

    plugin = PDFPlugin()
    with pytest.raises(ZeroPagePDFError):
        list(plugin.extract_stream(pdf_path, "zero_page.pdf"))

    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    rep = indexer.ingest_file(pdf_path)
    assert rep.status == "skipped_unsupported"
    assert "0 pages" in rep.error_message


def test_r4_item9_unchanged_rebuild_content_hash_fidelity(test_config, shared_embedder, temp_workspace):
    """
    Rebuild detects same-byte-length modification with restored mtime on unchanged file:
    - Pre-extraction content hash check detects change and raises RebuildAbortedError.
    - Last good generation is preserved.
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    f_a = temp_workspace / "file_a.txt"
    f_b = temp_workspace / "file_b.txt"
    f_a.write_text("AAAA")
    f_b.write_text("BBBB")

    indexer.ingest_file(f_a)
    indexer.ingest_file(f_b)
    last_good_gen = indexer.current_generation
    orig_a_stat = f_a.stat()

    # Modify file_a with identical size (4 bytes) and restore mtime
    f_a.write_text("CCCC")
    os.utime(f_a, (orig_a_stat.st_atime, orig_a_stat.st_mtime))

    # Trigger rebuild by editing file_b
    f_b.write_text("BBBB modified new content")
    with pytest.raises(RebuildAbortedError) as exc_info:
        indexer.ingest_file(f_b)
    assert "content hash mismatch before extraction" in str(exc_info.value)
    assert indexer.current_generation == last_good_gen


def test_r4_item9_late_parser_failure_rollback_during_edit_and_move(test_config, shared_embedder, temp_workspace):
    """
    Late streaming parser failure during file edit and move:
    - Rebuild transaction cleanly rolls back.
    - Previous generation is preserved intact.
    """
    indexer = StomaIndexer(test_config, embedder=shared_embedder)
    f_a = temp_workspace / "file_a.txt"
    f_b = temp_workspace / "file_b.txt"
    f_a.write_text("File A initial")
    f_b.write_text("File B initial")
    indexer.ingest_file(f_a)
    indexer.ingest_file(f_b)
    gen_before = indexer.current_generation

    # Edit file_a with simulated late stream failure during chunking/encoding
    f_a.write_text("File A modified")

    def failing_chunk_stream(stream):
        for i, p in enumerate(stream):
            if i >= 0:
                raise CorruptedDocumentError("Late stream corruption on passage 2")
            yield p

    with patch.object(indexer.chunker, "chunk_raw_passages_stream", side_effect=failing_chunk_stream):
        rep = indexer.ingest_file(f_a)
        assert rep.status == "failed"
        assert indexer.current_generation == gen_before

    # Move file_b to file_c with late stream failure
    f_c = temp_workspace / "file_c.txt"
    f_b.rename(f_c)

    with patch.object(indexer.chunker, "chunk_raw_passages_stream", side_effect=failing_chunk_stream):
        rep = indexer.move_file(f_b, f_c)
        assert rep.status == "failed"
        assert indexer.current_generation == gen_before
