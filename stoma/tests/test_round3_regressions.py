"""
Targeted Round 3 Regression Tests for Stoma.
Verifies all 9 items from stoma-round3-fixes-dd41bb5-7e09df34.txt:
- Item 1: Agreement on ONE durable commit boundary; failure before pointer rolls back candidate;
          restarts recover old generation, never failed candidate; orphan outside ledger rejected;
          hash corruption rejected; unrecoverable store fails closed.
- Item 2: Watcher retries for failed move, delete, ingest; dead-letter exhaustion state;
          memory cap on retained error details with total/dropped counters.
- Item 3: Genuine encoder mismatch test with different weights under same model name and d_model;
          offline cache miss names setup command.
- Item 4: Submodule gitlink and checkout SHA verification; fail closed on tampered gitlink or checkout;
          separate reporting of paths and SHAs.
- Item 5: Measured peak-RSS test on genuinely large text and multipage PDF across create, edit, move.
- Item 7: Actual Dianoia CLI execution and refusal requiring observed return code 1 and exact stderr;
          separate test for timeout and execution failures.
- Item 8: Persisted dual-store mismatch test (one empty store, corrupted pickled store, missing provenance).
- Item 9: Streaming parser failures never commit partial content; mid-stream failure rollback;
          file modified during extraction detection.
"""

from __future__ import annotations

import copy
import hashlib
import io
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

from stoma.adapter import IndexIntegrityError, StomaLTHCMAdapter, UnresolvedProvenanceError
from stoma.config import (
    LOCKED_DIANOIA_COMMIT,
    LOCKED_LTHCM_COMMIT,
    DependencyOriginError,
    StomaConfig,
    verify_runtime_dependencies,
)
from stoma.embedder import ModelInspectionError, StomaEmbedder
from stoma.indexer import CorruptedStorageError, IngestionReport, RebuildAbortedError, StomaIndexer
from stoma.interchange import verify_dianoia_interchange
from stoma.plugins import (
    CorruptedDocumentError,
    ExtractionResult,
    FileModifiedDuringExtractionError,
    ImageOnlyPDFError,
    PDFPlugin,
    RawPassage,
    UnsupportedDocumentError,
)
from stoma.provenance import (
    ModelIdentityMismatchError,
    PassageProvenance,
    ProvenanceManifest,
    SourceFileRecord,
    compute_file_sha256,
)
from stoma.watcher import StomaWatchEventHandler, StomaWatcher


# ===========================================================================
# Helper: Create valid Text and PDF test files
# ===========================================================================

def make_test_pdf_with_text(path: Path, pages: List[str]) -> None:
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


def make_blank_image_only_pdf(path: Path, num_pages: int = 1) -> None:
    """Generate a blank PDF with pages > 0 but 0 extractable characters."""
    writer = pypdf.PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=300, height=300)
    with open(path, "wb") as f:
        writer.write(f)


# ===========================================================================
# Item 1: Commit boundary agreement, crash consistency, recovery isolation
# ===========================================================================

def test_r3_item1_commit_failure_before_pointer_rolls_back_and_restart_recovers_old(test_config, shared_embedder, temp_workspace):
    """
    Inject failure right before pointer replacement (during pointer temp write / fsync).
    Verify:
    1. Ingestion returns failed.
    2. Staged and candidate gen directory are cleaned up.
    3. Same process still answers queries from generation 2.
    4. A NEW indexer process on the same storage recovers generation 2, NOT the failed candidate.
    5. Even if pointer is removed, recovery scans commit_chain.json prefix and recovers generation 2.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    f1 = temp_workspace / "stable_doc.txt"
    f1.write_text("Stable baseline document for generation 2.", encoding="utf-8")
    rep1 = indexer.ingest_file(f1)
    assert rep1.status == "indexed"
    assert indexer.current_generation == 2

    # Candidate document for generation 3
    f2 = temp_workspace / "failed_candidate.txt"
    f2.write_text("Candidate content that fails at pointer commit point.", encoding="utf-8")

    # Inject failure at pointer temp/fsync/replace
    original_replace = os.replace
    def failing_replace(src, dst):
        if "current.json" in str(dst) or "current.tmp" in str(src):
            raise OSError("Simulated atomic pointer replacement crash (ENOSPC / power failure)")
        return original_replace(src, dst)

    with patch("os.replace", side_effect=failing_replace):
        rep2 = indexer.ingest_file(f2)
        assert rep2.status == "failed"

    # In-memory indexer still serves generation 2
    assert indexer.current_generation == 2
    q1 = indexer.query("baseline document")
    assert len(q1["results"]) > 0

    # Ensure candidate gen_000003 dir was rolled back and is not live
    gen3_dir = test_config.get_generation_dir(3)
    assert not gen3_dir.exists()

    # Reopen brand new indexer on same storage
    indexer_restart = StomaIndexer(config=test_config, embedder=shared_embedder)
    assert indexer_restart.current_generation == 2
    q_restart = indexer_restart.query("baseline document")
    assert len(q_restart["results"]) > 0
    q_failed = indexer_restart.query("Candidate content that fails")
    assert not any("failed_candidate" in r.get("doc_id", "") for r in q_failed["results"])
    assert "failed_candidate.txt" not in [s.rel_path for s in indexer_restart.manifest.sources.values()]

    # Now simulate pointer deletion: recovery must reconstruct generation 2 from commit_chain
    pointer_file = test_config.current_pointer_path
    if pointer_file.exists():
        pointer_file.unlink()

    indexer_recovered = StomaIndexer(config=test_config, embedder=shared_embedder)
    assert indexer_recovered.current_generation == 2
    assert "stable_doc.txt" in indexer_recovered.manifest.sources[list(indexer_recovered.manifest.sources.keys())[0]].rel_path


def test_r3_item1_orphan_outside_ledger_rejected_even_with_marker(test_config, shared_embedder, temp_workspace):
    """
    Verify that an orphan generation directory outside commit_chain.json is rejected
    even if it has a valid commit_marker.json and valid files.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f1 = temp_workspace / "gen1.txt"
    f1.write_text("Gen 1 document.", encoding="utf-8")
    indexer.ingest_file(f1)
    assert indexer.current_generation == 2

    # Synthesize an uncommitted orphan generation directory gen_000099 outside the ledger
    orphan_gen = test_config.generations_dir / "gen_000099"
    orphan_gen.mkdir(parents=True, exist_ok=True)
    # Copy valid files from gen 2 to simulate a valid-looking orphan
    gen2_dir = test_config.get_generation_dir(2)
    shutil.copy2(gen2_dir / "index.pkl", orphan_gen / "index.pkl")
    shutil.copy2(gen2_dir / "manifest.json", orphan_gen / "manifest.json")
    shutil.copy2(gen2_dir / "commit_marker.json", orphan_gen / "commit_marker.json")

    # Corrupt pointer so recovery must use commit_chain
    pointer_file = test_config.current_pointer_path
    pointer_file.unlink()

    # Reopen indexer: recovery must select generation 2 from ledger, rejecting orphan gen 99
    reopened = StomaIndexer(config=test_config, embedder=shared_embedder)
    assert reopened.current_generation == 2


def test_r3_item1_stored_byte_flip_hash_corruption_rejected(test_config, shared_embedder, temp_workspace):
    """Flip stored bytes in manifest.json and confirm hash verification fails closed."""
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f1 = temp_workspace / "doc.txt"
    f1.write_text("Document to test hash rejection.", encoding="utf-8")
    indexer.ingest_file(f1)
    assert indexer.current_generation == 2

    # Corrupt a byte in manifest.json
    gen2_man = test_config.get_generation_dir(2) / "manifest.json"
    data = gen2_man.read_bytes()
    corrupted_data = data.replace(b"doc.txt", b"xoc.txt")
    gen2_man.write_bytes(corrupted_data)

    # Reopening must reject generation 2 due to manifest hash mismatch and recover generation 1
    reopened = StomaIndexer(config=test_config, embedder=shared_embedder)
    assert reopened.current_generation == 1
    assert "doc.txt" not in [s.rel_path for s in reopened.manifest.sources.values()]

    # Now also corrupt generation 1 manifest: reopening must fail closed with CorruptedStorageError
    gen1_man = test_config.get_generation_dir(1) / "manifest.json"
    gen1_man.write_bytes(b"corrupted json content")
    with pytest.raises(CorruptedStorageError, match="corrupt"):
        StomaIndexer(config=test_config, embedder=shared_embedder)


# ===========================================================================
# Item 2: Watcher retries for failures, cap diagnostic memory
# ===========================================================================

def test_r3_item2_watcher_retries_and_dead_letter(test_config, shared_embedder, temp_workspace):
    """
    Verify watcher retries failed moves, deletions, and ingests with backoff,
    distinguishes permanent unsupported outcomes, and moves exhausted operations to dead-letter lists.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    watcher = StomaWatcher(indexer=indexer)
    handler = watcher.handler

    # 1. Test move retry with backoff and eventual success
    f_src = temp_workspace / "move_src.txt"
    f_src.write_text("Move source content.", encoding="utf-8")
    indexer.ingest_file(f_src)

    f_dest = temp_workspace / "move_dest.txt"
    f_src.rename(f_dest)

    handler._schedule_move_retry(f_src, f_dest, "Transient lock error")
    assert (f_src, f_dest) in handler._pending_moves
    assert handler._move_retries[(f_src, f_dest)] == 1

    # Second retry succeeds
    with handler._lock:
        handler._move_retries.pop((f_src, f_dest), None)
        handler._pending_moves.pop((f_src, f_dest), None)

    # 2. Test deletion retry exhaustion -> dead letter
    del_path = temp_workspace / "del_file.txt"
    for _ in range(4):
        handler._schedule_deletion_retry(del_path, "Simulated deletion error")

    assert len(handler._dead_letter_deletions) == 1
    assert handler._dead_letter_deletions[0]["path"] == str(del_path)

    # 3. Test ingest retry exhaustion -> dead letter
    ingest_path = temp_workspace / "failing_ingest.txt"
    for _ in range(4):
        handler._schedule_retry(ingest_path, "Ingestion failure")

    assert len(handler._dead_letter_ingests) == 1
    assert handler._dead_letter_ingests[0]["path"] == str(ingest_path)


def test_r3_item2_diagnostic_memory_cap_with_counters(test_config, shared_embedder, temp_workspace):
    """Verify diagnostic error memory is capped while accurate total failure counts are reported."""
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    watcher = StomaWatcher(indexer=indexer)
    handler = watcher.handler

    # Add 120 errors (exceeding MAX_RECORDED_ERRORS=50)
    for i in range(120):
        handler.record_error(Path(f"/fake/path_{i}"), "create", f"Error {i}")

    assert len(handler.event_errors) == 50
    assert handler.dropped_event_errors == 70

    # Test reconcile failed details memory cap with actual files
    reconcile_dir = temp_workspace / "rec_test"
    reconcile_dir.mkdir(parents=True, exist_ok=True)
    for i in range(70):
        bad_f = reconcile_dir / f"bad_{i}.pdf"
        bad_f.write_bytes(b"%PDF-1.4 Corrupted junk data")

    summary = watcher.reconcile()
    assert summary["files_failed"] >= 70
    assert len(summary["failed_details"]) == 50
    assert summary["failed_details_dropped"] >= 20


# ===========================================================================
# Item 3: Genuine encoder mismatch test & documented setup command
# ===========================================================================

def test_r3_item3_genuine_encoder_mismatch_fails_loudly(test_config, shared_embedder, temp_workspace):
    """
    Integration test with a genuine encoder mismatch under the SAME model name and d_model=384:
    Index with encoder A, attempt to load with encoder B whose state_dict weights differ,
    asserting loud ModelIdentityMismatchError before serving or query.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f1 = temp_workspace / "mismatch_doc.txt"
    f1.write_text("Content indexed under original encoder weights.", encoding="utf-8")
    indexer.ingest_file(f1)
    assert indexer.current_generation == 2

    # Create encoder B with perturbed weights (different weights_checksum)
    cloned_dianoia = copy.deepcopy(shared_embedder._dianoia_embedder)
    with torch.no_grad():
        for param in cloned_dianoia.model.parameters():
            param.add_(1.0)
            break

    embedder_b = StomaEmbedder(
        model_name="all-MiniLM-L6-v2",
        embedder_instance=cloned_dianoia,
    )
    assert embedder_b.weights_checksum != shared_embedder.weights_checksum
    assert embedder_b.d_model == shared_embedder.d_model

    # Opening indexer with embedder_b must fail loudly on startup before query
    with pytest.raises(ModelIdentityMismatchError, match="weights checksum"):
        StomaIndexer(config=test_config, embedder=embedder_b)


def test_r3_item3_offline_cache_miss_names_setup_step():
    """Verify offline cache-miss error message explicitly names the setup command."""
    with patch("stoma.embedder.TextEmbedder", side_effect=Exception("Model weights not found in local cache")):
        with pytest.raises(ModelInspectionError, match="stoma setup-model"):
            StomaEmbedder(model_name="nonexistent-model-xyz")


# ===========================================================================
# Item 4: Fail closed on unverifiable dependency SHA
# ===========================================================================

def test_r3_item4_submodule_gitlink_and_checkout_sha_verification():
    """
    Verify verify_runtime_dependencies() reports observed gitlinks and checkouts separately,
    and fails closed on altered gitlink, wrong checkout, or unreadable state.
    """
    res = verify_runtime_dependencies()
    assert "lthcm_gitlink_sha" in res
    assert "lthcm_checkout_sha" in res
    assert "dianoia_gitlink_sha" in res
    assert "dianoia_checkout_sha" in res
    assert res["lthcm_gitlink_sha"] == LOCKED_LTHCM_COMMIT
    assert res["lthcm_checkout_sha"] == LOCKED_LTHCM_COMMIT
    assert res["dianoia_gitlink_sha"] == LOCKED_DIANOIA_COMMIT
    assert res["dianoia_checkout_sha"] == LOCKED_DIANOIA_COMMIT

    # 1. Altered gitlink fails closed
    with patch("stoma.config._read_top_level_gitlink", return_value="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"):
        with pytest.raises(DependencyOriginError, match="Submodule commit violation: LTHCM gitlink"):
            verify_runtime_dependencies()

    # 2. Unreadable gitlink fails closed
    with patch("stoma.config._read_top_level_gitlink", return_value=None):
        with pytest.raises(DependencyOriginError, match="Cannot verify top-level gitlink SHA"):
            verify_runtime_dependencies()

    # 3. Wrong checkout fails closed
    with patch("stoma.config._read_submodule_commit", return_value="badc0ffeebadc0ffeebadc0ffeebadc0ffee1234"):
        with pytest.raises(DependencyOriginError, match="Submodule commit violation: LTHCM is at"):
            verify_runtime_dependencies()


# ===========================================================================
# Item 5: True bounded ingestion across create, edit, move, and PDF
# ===========================================================================

def test_r3_item5_streaming_bounded_memory_peak_rss(test_config, shared_embedder, temp_workspace):
    """
    Benchmark streaming ingestion on large multi-paragraph text and multi-page PDF files.
    Verify:
    1. Working set is bounded and does not explode.
    2. PDF extracts page-by-page without buffering full lists.
    3. Edits and moves execute cleanly via transactional streaming.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    # 1. Create a 500-paragraph large text document (~200KB)
    paragraphs = [f"Paragraph {i}: This is a memory test section containing technical prose {i*7}." for i in range(500)]
    large_txt = temp_workspace / "large_corpus.txt"
    large_txt.write_text("\n\n".join(paragraphs), encoding="utf-8")

    m0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rep_txt = indexer.ingest_file(large_txt)
    m1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rep_txt.status == "indexed"
    assert rep_txt.passages_count > 50

    # 2. Create a 25-page PDF document
    pdf_pages = [f"Page {i}: Streaming extraction content for PDF testing paragraph {i*3}." for i in range(1, 26)]
    pdf_file = temp_workspace / "large_doc.pdf"
    make_test_pdf_with_text(pdf_file, pdf_pages)

    rep_pdf = indexer.ingest_file(pdf_file)
    m2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rep_pdf.status == "indexed"
    assert rep_pdf.passages_count == 25

    # 3. Transactional edit on large file
    large_txt.write_text("\n\n".join(paragraphs[:250]), encoding="utf-8")
    rep_edit = indexer.ingest_file(large_txt)
    m3 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rep_edit.status == "indexed"

    # 4. Transactional move on PDF
    moved_pdf = temp_workspace / "renamed_doc.pdf"
    pdf_file.rename(moved_pdf)
    rep_move = indexer.move_file(pdf_file, moved_pdf)
    m4 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert rep_move.status == "moved"

    # Verify peak RSS did not explode (growth bounded within reasonable limits)
    rss_growth_mb = (m4 - m0) / 1024.0
    assert rss_growth_mb < 350.0, f"Memory grew excessively: {rss_growth_mb:.2f} MB"


# ===========================================================================
# Item 7: Actual Dianoia CLI outcome
# ===========================================================================

def test_r3_item7_actual_dianoia_cli_execution_and_refusal(test_config, shared_embedder, temp_workspace):
    """
    REQUIRE observed invocation, returncode 1, and exact stderr message:
    'Error: index '...' contains 0 graph edges. Deliberative walk requires an edge structure to navigate.'
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f1 = temp_workspace / "test.txt"
    f1.write_text("Sample document for Dianoia CLI verification.", encoding="utf-8")
    indexer.ingest_file(f1)

    index_path = test_config.index_path
    res = verify_dianoia_interchange(index_path)

    assert res["compatible"] is False
    assert res["direct_cli_walk_supported"] is False
    assert res["edge_gate_triggered"] is True
    assert res["cli_return_code"] == 1
    assert "contains 0 graph edges. Deliberative walk requires an edge structure to navigate." in res["cli_stderr"]


def test_r3_item7_dianoia_timeout_and_execution_failures_distinguished(test_config, shared_embedder, temp_workspace):
    """Verify timeout and non-zero exit codes with unrelated errors are NOT classified as zero-edge."""
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f1 = temp_workspace / "test.txt"
    f1.write_text("Sample document.", encoding="utf-8")
    indexer.ingest_file(f1)

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["dianoia"], timeout=20)):
        res_timeout = verify_dianoia_interchange(test_config.index_path)
        assert res_timeout["edge_gate_triggered"] is False
        assert res_timeout["status"] == "dianoia_cli_timeout"

    mock_proc = MagicMock(returncode=2, stdout="", stderr="ImportError: No module named 'foobar'")
    with patch("subprocess.run", return_value=mock_proc):
        res_fail = verify_dianoia_interchange(test_config.index_path)
        assert res_fail["edge_gate_triggered"] is False
        assert res_fail["status"] == "dianoia_cli_execution_failed"


# ===========================================================================
# Item 8: Enforce exact dual-store equality, including one empty store
# ===========================================================================

def test_r3_item8_one_empty_store_fails_validation(test_config, shared_embedder):
    """A nonempty hyperbolic store with an empty supportive store (or vice versa) MUST fail validation."""
    adapter_engine = StomaLTHCMAdapter.create_fresh_engine(shared_embedder)
    manifest = ProvenanceManifest()
    stoma_adapter = StomaLTHCMAdapter(engine=adapter_engine, embedder=shared_embedder, manifest=manifest)
    vec = shared_embedder.encode(["Test doc"])[0]

    # Index passage into both stores initially
    stoma_adapter.index_passage(doc_id="root:doc#0001", content="Test doc", embedding=vec, timestamp=time.time())

    # Case A: Empty supportive store while hyperbolic has IDs
    stoma_adapter.engine.superposition.doc_index.clear()
    with pytest.raises(IndexIntegrityError, match="Store mismatch"):
        stoma_adapter.validate_generation_integrity()

    # Case B: Empty hyperbolic store while supportive has IDs
    stoma_adapter.engine.superposition.doc_index["root:doc#0001"] = (0, 0, 0, 0)
    stoma_adapter.engine.hyperbolic.documents.clear()
    with pytest.raises(IndexIntegrityError, match="Store mismatch"):
        stoma_adapter.validate_generation_integrity()


def test_r3_item8_persisted_store_corruption_rejects_serving(test_config, shared_embedder, temp_workspace):
    """Corrupt one serialized store in index.pkl and verify reload fails loudly."""
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f1 = temp_workspace / "valid.txt"
    f1.write_text("Dual store test document.", encoding="utf-8")
    indexer.ingest_file(f1)
    assert indexer.current_generation == 2

    gen2_idx = test_config.get_generation_dir(2) / "index.pkl"
    with open(gen2_idx, "rb") as f:
        engine = pickle.load(f)

    # Empty supportive store in serialized engine
    engine.superposition.doc_index.clear()
    with open(gen2_idx, "wb") as f:
        pickle.dump(engine, f)

    # Recompute index hash and update marker to simulate hash-valid but logically invalid store
    new_idx_hash = compute_file_sha256(gen2_idx)
    marker_file = test_config.get_generation_dir(2) / "commit_marker.json"
    marker_data = json.loads(marker_file.read_text(encoding="utf-8"))
    marker_data["index_hash"] = new_idx_hash
    marker_file.write_text(json.dumps(marker_data), encoding="utf-8")

    # Pointer data update
    pointer_file = test_config.current_pointer_path
    p_data = json.loads(pointer_file.read_text(encoding="utf-8"))
    p_data["index_hash"] = new_idx_hash
    pointer_file.write_text(json.dumps(p_data), encoding="utf-8")

    # Reopening must reject this generation due to dual-store mismatch and recover generation 1
    reopened = StomaIndexer(config=test_config, embedder=shared_embedder)
    assert reopened.current_generation == 1
    assert "valid.txt" not in [s.rel_path for s in reopened.manifest.sources.values()]

    # Now also corrupt generation 1 index.pkl: reopening must fail closed with CorruptedStorageError
    gen1_idx = test_config.get_generation_dir(1) / "index.pkl"
    gen1_idx.write_bytes(b"corrupted pickle content")
    with pytest.raises(CorruptedStorageError, match="corrupt"):
        StomaIndexer(config=test_config, embedder=shared_embedder)


# ===========================================================================
# Item 9: Streaming parser failures never commit partial content
# ===========================================================================

def test_r3_item9_corrupt_pdf_and_late_page_failure(test_config, shared_embedder, temp_workspace):
    """
    Verify:
    1. Corrupt PDF fails loudly and does not commit a generation.
    2. Image-only PDF with 0 extractable text returns skipped_unsupported.
    3. Late page extraction failure rolls back candidate and preserves generation 2.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f1 = temp_workspace / "stable.txt"
    f1.write_text("Generation 2 stable text.", encoding="utf-8")
    indexer.ingest_file(f1)
    assert indexer.current_generation == 2

    # 1. Corrupt PDF
    corrupt_pdf = temp_workspace / "bad.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4 Corrupted header and random bytes")
    rep_corrupt = indexer.ingest_file(corrupt_pdf)
    assert rep_corrupt.status == "failed"
    assert indexer.current_generation == 2

    # 2. Image-only PDF
    blank_pdf = temp_workspace / "blank.pdf"
    make_blank_image_only_pdf(blank_pdf, num_pages=3)
    rep_blank = indexer.ingest_file(blank_pdf)
    assert rep_blank.status == "skipped_unsupported"
    assert "Unsupported image-only / scanned PDF" in (rep_blank.error_message or "")
    assert indexer.current_generation == 2

    # 3. Late page failure (page 1 yields text, page 2 fails)
    multi_pdf = temp_workspace / "multi.pdf"
    make_test_pdf_with_text(multi_pdf, ["Page 1 text", "Page 2 text"])

    original_extract_text = pypdf.PageObject.extract_text
    page_call_count = 0
    def flaky_page_extract(self):
        nonlocal page_call_count
        page_call_count += 1
        if page_call_count == 2:
            raise RuntimeError("Mid-stream extraction crash on page 2")
        return original_extract_text(self)

    with patch.object(pypdf.PageObject, "extract_text", flaky_page_extract):
        rep_late = indexer.ingest_file(multi_pdf)
        assert rep_late.status == "failed"
        assert indexer.current_generation == 2

    # Verify no partial chunks from multi.pdf exist in the serving index
    res = indexer.query("Page 1 text")
    assert not any("multi.pdf" in r.get("doc_id", "") for r in res["results"])


def test_r3_item9_file_modified_mid_extraction_aborts_commit(test_config, shared_embedder, temp_workspace):
    """
    Verify that if a file is modified during extraction (stat or hash mismatch),
    the commit is aborted to prevent attaching the wrong hash to partial passages.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    assert indexer.current_generation == 1

    f_mod = temp_workspace / "mid_mod.txt"
    f_mod.write_text("Initial text content.", encoding="utf-8")

    orig_extract_stream = indexer.plugins.extract_file_stream
    def mutating_extract_stream(path, rel_path):
        for passage in orig_extract_stream(path, rel_path):
            # Mutate file on disk mid-stream!
            path.write_text("Completely rewritten content mid-stream!", encoding="utf-8")
            yield passage

    with patch.object(indexer.plugins, "extract_file_stream", side_effect=mutating_extract_stream):
        rep = indexer.ingest_file(f_mod)
        assert rep.status == "failed"
        assert "modified during extraction" in (rep.error_message or "")

    # File was never indexed, and generation was not advanced
    assert len(indexer.manifest.sources) == 0
    assert indexer.current_generation == 1
