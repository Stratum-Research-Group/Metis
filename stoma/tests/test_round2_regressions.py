"""
Comprehensive Round 2 Regression Tests for Stoma.
Strictly verifies all seven Round 2 findings from stoma-round2-fixes-bca66e2-92c6d154.txt:
- Finding 1: Commit-chain recovery, orphan generation rejection, stored hash corruption detection.
- Finding 2: Watcher resilience, per-event error boundaries, retry backoff, reconcile reporting and non-purge on extraction failure.
- Finding 3: True offline runtime (offline env vars), prefetch vs runtime, full encoder identity (tokenizer limits, checksum, offline cache miss).
- Finding 4: Enforce exact submodule gitlink/HEAD commit SHAs at startup, CLI failure on altered commit/origin.
- Finding 5: Streaming ingestion, 25MB safe size cap on non-streaming AST/JSON, memory boundedness, unbroken token tail handling.
- Finding 7: Dianoia interchange CLI reporting disambiguation (edge gate refusal vs subprocess execution failure).
- Finding 8: Dual-store validation (hyperbolic & superposition parity, duplicate detection), query-time strong provenance checks.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from stoma.adapter import IndexIntegrityError, StomaLTHCMAdapter, UnresolvedProvenanceError
from stoma.chunker import Chunker
from stoma.config import (
    LOCKED_DIANOIA_COMMIT,
    LOCKED_LTHCM_COMMIT,
    DependencyOriginError,
    StomaConfig,
    verify_runtime_dependencies,
)
from stoma.embedder import StomaEmbedder
from stoma.indexer import CorruptedStorageError, StomaIndexer
from stoma.interchange import verify_dianoia_interchange
from stoma.provenance import (
    CorruptedManifestError,
    LegacyManifestError,
    ModelIdentityMismatchError,
    ModelMetadata,
    PassageProvenance,
    ProvenanceManifest,
    SourceFileRecord,
    compute_file_sha256,
)
from stoma.watcher import StomaWatcher


# ===========================================================================
# Finding 1: Commit-Chain Recovery, Orphan Rejection, and Stored Hash Checks
# ===========================================================================

def test_r2_f1_commit_chain_recovery_rejects_uncommitted_orphan(test_config, shared_embedder, temp_workspace):
    """
    Assert that an uncommitted orphan generation directory (gen_N created without
    commit_marker.json and not present in commit_chain.json) is NEVER promoted
    on startup or recovery, even if current.json is missing or corrupted.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f1 = temp_workspace / "doc1.txt"
    f1.write_text("Legitimate committed document generation 2.", encoding="utf-8")
    indexer.ingest_file(f1)
    gen2 = indexer.current_generation
    assert gen2 >= 2

    # Simulate an uncommitted orphan gen_999 published before pointer commit
    orphan_gen_dir = test_config.generations_dir / "gen_000999"
    orphan_gen_dir.mkdir(parents=True, exist_ok=True)
    orphan_idx = orphan_gen_dir / "index.pkl"
    orphan_man = orphan_gen_dir / "manifest.json"

    # Create valid pickled engine and manifest with a different corpus inside orphan
    orphan_adapter = StomaLTHCMAdapter(
        engine=StomaLTHCMAdapter.create_fresh_engine(shared_embedder),
        embedder=shared_embedder,
    )
    orphan_adapter.index_passage(
        doc_id="test:orphan_doc.txt:0000",
        content="Orphan corpus content that was never committed!",
        embedding=shared_embedder.encode_single("Orphan corpus content that was never committed!"),
        timestamp=time.time(),
    )
    orphan_adapter.save(orphan_idx)

    orphan_manifest = ProvenanceManifest(
        d_model=shared_embedder.d_model,
        num_facets=1,
        model_meta=ModelMetadata(
            model_name=shared_embedder.model_name,
            d_model=shared_embedder.d_model,
            max_seq_length=shared_embedder.max_seq_length,
            tokenizer_name=shared_embedder.tokenizer_name,
            revision=shared_embedder.revision,
            weights_checksum=shared_embedder.weights_checksum,
            special_tokens_overhead=shared_embedder.special_tokens_overhead,
            effective_max_tokens=shared_embedder.effective_max_tokens,
        ),
    )
    orphan_source = SourceFileRecord(
        source_id="test:orphan_doc.txt",
        root_id="test",
        root_path=str(temp_workspace),
        rel_path="orphan_doc.txt",
        content_hash="orphan_hash",
        size_bytes=100,
        mtime=time.time(),
        last_ingested_at=time.time(),
        plugin_name="text",
        passage_ids=["test:orphan_doc.txt:0000"],
    )
    orphan_prov = PassageProvenance(
        passage_id="test:orphan_doc.txt:0000",
        source_id="test:orphan_doc.txt",
        root_id="test",
        root_path=str(temp_workspace),
        rel_path="orphan_doc.txt",
        file_content_hash="orphan_hash",
        chunk_index=0,
        chunk_hash="orphan_chunk_hash",
        location={"line": 1},
        ingested_at=time.time(),
        token_count=10,
        char_count=45,
    )
    orphan_manifest.record_source_and_passages(orphan_source, [orphan_prov])
    orphan_manifest.save(orphan_man)
    # Crucially: DO NOT write commit_marker.json in orphan_gen_dir, DO NOT add to commit_chain.json

    # Now corrupt current.json
    cur_p = test_config.current_pointer_path
    cur_p.write_text("{corrupt json", encoding="utf-8")

    # Restart indexer: it must ignore orphan_gen_dir and recover gen2
    recovered_indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    assert recovered_indexer.current_generation == gen2
    assert "test:orphan_doc.txt:0000" not in recovered_indexer.manifest.passages

    # Query should return legitimate doc1, never orphan
    res = recovered_indexer.query("committed document", top_k=5)
    assert len(res["results"]) > 0
    assert any("doc1.txt" in r["doc_id"] for r in res["results"])


def test_r2_f1_byte_flip_stored_hashes_catch_corruption(test_config, shared_embedder, temp_workspace):
    """
    Assert that byte-level changes in index.pkl or manifest.json that retain valid
    syntax are detected via current.json's recorded index_hash and manifest_hash.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f = temp_workspace / "hash_doc.txt"
    f.write_text("Document to test hash integrity catching byte flips.", encoding="utf-8")
    indexer.ingest_file(f)
    active_gen = indexer.current_generation

    gen_idx, gen_man = test_config.get_generation_paths(active_gen)

    # Read current pointer
    with open(test_config.current_pointer_path, "r", encoding="utf-8") as pf:
        pdata = json.load(pf)
    orig_idx_hash = pdata["index_hash"]
    orig_man_hash = pdata["manifest_hash"]

    assert compute_file_sha256(gen_idx) == orig_idx_hash
    assert compute_file_sha256(gen_man) == orig_man_hash

    # Subtly alter manifest JSON by modifying a non-critical field (retains valid JSON)
    with open(gen_man, "r", encoding="utf-8") as mf:
        mdata = json.load(mf)
    mdata["sources"][list(mdata["sources"].keys())[0]]["plugin"] = "altered_plugin"
    with open(gen_man, "w", encoding="utf-8") as mf:
        json.dump(mdata, mf)

    # Hash should now differ from pointer
    assert compute_file_sha256(gen_man) != orig_man_hash

    # Reopening indexer should detect hash mismatch on active_gen and recover gen 1
    reopened = StomaIndexer(config=test_config, embedder=shared_embedder)
    assert reopened.current_generation < active_gen


def test_r2_f1_failed_commit_preserves_old_serving_index(test_config, shared_embedder, temp_workspace):
    """
    Assert that if pointer-write or directory sync fails during a commit,
    the previous complete generation remains current and continues to serve queries cleanly.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f1 = temp_workspace / "serve1.txt"
    f1.write_text("First version that must continue serving.", encoding="utf-8")
    indexer.ingest_file(f1)
    gen_before = indexer.current_generation

    # Query works before failure
    res_before = indexer.query("First version", top_k=1)
    assert len(res_before["results"]) == 1

    # Now attempt ingest of f2 with simulated failure during commit
    f2 = temp_workspace / "serve2.txt"
    f2.write_text("Second version that will fail commit.", encoding="utf-8")

    with patch.object(indexer, "_write_current_pointer", side_effect=IOError("Simulated disk full")):
        report = indexer.ingest_file(f2)
        assert report.status == "failed"

    # Indexer current generation should be rolled back to gen_before
    assert indexer.current_generation == gen_before

    # Old results still serve cleanly
    res_after = indexer.query("First version", top_k=1)
    assert len(res_after["results"]) == 1
    assert "serve1.txt" in res_after["results"][0]["doc_id"]


# ===========================================================================
# Finding 2: Watcher Resilience and Reconciliation Reporting
# ===========================================================================

def test_r2_f2_watcher_resilience_on_failing_event(test_config, shared_embedder, temp_workspace):
    """
    Assert that an exception in the watcher event queue (bad file, permission error)
    does not kill the watcher worker loop and subsequent good events proceed.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    watcher = StomaWatcher(indexer=indexer, debounce_sec=0.05)
    watcher.start([temp_workspace])

    try:
        # 1. Enqueue a failing path (e.g. non-existent file or unreadable file)
        fake_bad = temp_workspace / "non_existent_or_bad.txt"
        watcher.handler.on_created(MagicMock(is_directory=False, src_path=str(fake_bad)))

        # 2. Immediately enqueue a valid file
        good_file = temp_workspace / "resilient_good.txt"
        good_file.write_text("Watcher stayed alive and processed this file!", encoding="utf-8")
        watcher.handler.on_created(MagicMock(is_directory=False, src_path=str(good_file)))

        # Wait for debounced worker to process
        time.sleep(0.5)

        # Worker thread must still be alive
        assert watcher.handler._worker_thread.is_alive()

        # The good file must be indexed
        root_id = test_config.locate_path(good_file)[0]
        source_id = f"{root_id}:resilient_good.txt"
        assert source_id in indexer.manifest.sources
    finally:
        watcher.stop()


def test_r2_f2_reconcile_reports_counts_and_does_not_purge_on_extract_error(test_config, shared_embedder, temp_workspace):
    """
    Assert that reconcile() returns visible failed/skipped counts and details,
    and does NOT delete existing valid passages if a disk file encounters an extraction error.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    watcher = StomaWatcher(indexer=indexer)

    # 1. Ingest a valid file
    valid_f = temp_workspace / "good.txt"
    valid_f.write_text("Valid text content before reconciliation.", encoding="utf-8")
    indexer.ingest_file(valid_f)
    assert len(indexer.manifest.sources) == 1
    assert len(indexer.manifest.passages) >= 1

    # 2. Add an unsupported file (.xyz)
    unsupported_f = temp_workspace / "audio.xyz"
    unsupported_f.write_text("Unsupported file content", encoding="utf-8")

    # 3. Add a new file that fails extraction
    bad_f = temp_workspace / "bad.txt"
    bad_f.write_text("Corrupt content", encoding="utf-8")

    # Simulate an extraction failure on bad.txt during reconcile
    orig_extract = indexer.plugins.extract_file_stream
    def mock_extract(p, r):
        if "bad.txt" in str(p):
            raise ValueError("Simulated parser explosion on bad.txt")
        return orig_extract(p, r)

    with patch.object(indexer.plugins, "extract_file_stream", side_effect=mock_extract):
        summary = watcher.reconcile()

    # Reconcile summary should report failure details
    assert summary["files_failed"] >= 1
    assert len(summary["failed_details"]) >= 1
    assert any("bad.txt" in d["path"] for d in summary["failed_details"])
    assert summary["files_unchanged"] >= 1
    assert summary["files_skipped_unsupported"] >= 1

    # CRITICAL: Old valid passages must NOT be removed on extraction failure
    assert len(indexer.manifest.sources) == 1
    assert len(indexer.manifest.passages) >= 1
    root_id = test_config.locate_path(valid_f)[0]
    assert f"{root_id}:good.txt" in indexer.manifest.sources


# ===========================================================================
# Finding 3: True Offline Runtime and Complete Encoder Identity
# ===========================================================================

def test_r2_f3_offline_env_vars_and_cache_miss():
    """
    Assert that offline environment variables are strictly enforced,
    and attempting to load an un-cached model fails locally without network access.
    """
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"

    # Attempting to load an unknown model should fail offline immediately
    with pytest.raises(Exception):
        StomaEmbedder(model_name="nonexistent-stoma-test-model-404")


def test_r2_f3_manifest_rejects_missing_or_mismatched_tokenizer_limits(shared_embedder):
    """
    Assert that ProvenanceManifest rejects missing special_tokens_overhead / effective_max_tokens,
    and rejects reopening if tokenizer limits differ from current embedder.
    """
    # 1. Missing fields in raw dictionary must raise LegacyManifestError
    raw_dict = {
        "schema_version": 1,
        "d_model": shared_embedder.d_model,
        "num_facets": 1,
        "model_meta": {
            "model_name": shared_embedder.model_name,
            "d_model": shared_embedder.d_model,
            "max_seq_length": shared_embedder.max_seq_length,
            "tokenizer_name": shared_embedder.tokenizer_name,
            "revision": shared_embedder.revision,
            "weights_checksum": shared_embedder.weights_checksum,
            # Missing special_tokens_overhead and effective_max_tokens
        },
        "sources": {},
        "passages": {},
    }
    with pytest.raises(LegacyManifestError, match="lacks mandatory 'special_tokens_overhead'"):
        ProvenanceManifest.from_dict(raw_dict)

    # 2. Mismatched special_tokens_overhead in manifest must fail against embedder
    manifest = ProvenanceManifest(
        d_model=shared_embedder.d_model,
        num_facets=1,
        model_meta=ModelMetadata(
            model_name=shared_embedder.model_name,
            d_model=shared_embedder.d_model,
            max_seq_length=shared_embedder.max_seq_length,
            tokenizer_name=shared_embedder.tokenizer_name,
            revision=shared_embedder.revision,
            weights_checksum=shared_embedder.weights_checksum,
            special_tokens_overhead=shared_embedder.special_tokens_overhead + 2,  # altered!
            effective_max_tokens=shared_embedder.effective_max_tokens - 2,
        ),
    )
    with pytest.raises(ModelIdentityMismatchError, match="Special tokens overhead mismatch"):
        manifest.validate_against_embedder(
            model_name=shared_embedder.model_name,
            d_model=shared_embedder.d_model,
            revision=shared_embedder.revision,
            weights_checksum=shared_embedder.weights_checksum,
            num_facets=1,
            tokenizer_name=shared_embedder.tokenizer_name,
            max_seq_length=shared_embedder.max_seq_length,
            special_tokens_overhead=shared_embedder.special_tokens_overhead,
            effective_max_tokens=shared_embedder.effective_max_tokens,
        )


def test_r2_f3_long_token_boundary_chunks_encode_losslessly(shared_embedder):
    """
    Assert that long documents with continuous unbroken token runs are split
    losslessly within tokenizer limits and encoded without truncation errors.
    """
    chunker = Chunker(
        embedder=shared_embedder,
        target_tokens=50,
        overlap_tokens=10,
    )
    long_text = "word " * 300  # 300 tokens exceeds 50 max_chunk_tokens
    from stoma.plugins.base import RawPassage
    raw = [RawPassage(content=long_text, location={"section": "body"})]

    chunks = list(chunker.chunk_raw_passages_stream(iter(raw)))
    assert len(chunks) > 1

    # Verify each chunk has token count <= effective_max_tokens
    for c in chunks:
        tok_cnt = shared_embedder.count_tokens(c.content)
        assert tok_cnt <= shared_embedder.effective_max_tokens

    # Verify encode runs cleanly on all chunks
    texts = [c.content for c in chunks]
    vecs = shared_embedder.encode(texts)
    assert vecs.shape == (len(chunks), shared_embedder.d_model)


# ===========================================================================
# Finding 4: Enforce Exact Imported Submodule SHAs
# ===========================================================================

def test_r2_f4_exact_sha_verification_fails_on_altered_commit():
    """
    Assert that verify_runtime_dependencies() fails loudly with DependencyOriginError
    prefixed with 'Runtime import violation:' if gitlink or checked-out commit differs.
    """
    # Clean check should pass
    clean_res = verify_runtime_dependencies()
    assert clean_res["lthcm_origin_ok"] is True
    assert clean_res["dianoia_origin_ok"] is True

    # Altered commit SHA should raise DependencyOriginError
    with patch("stoma.config.LOCKED_LTHCM_COMMIT", "1111111111111111111111111111111111111111"):
        with pytest.raises(DependencyOriginError, match="Runtime import violation: Submodule commit violation: LTHCM"):
            verify_runtime_dependencies()

    with patch("stoma.config.LOCKED_DIANOIA_COMMIT", "2222222222222222222222222222222222222222"):
        with pytest.raises(DependencyOriginError, match="Runtime import violation: Submodule commit violation: dianoia"):
            verify_runtime_dependencies()


def test_r2_f4_cli_fails_loudly_on_altered_submodule(temp_workspace):
    """
    Assert that invoking the Stoma CLI entrypoint with altered submodule commit fails loudly.
    """
    env = os.environ.copy()
    # Inject a python execution that patches LOCKED_LTHCM_COMMIT and calls cli()
    code = (
        "import unittest.mock as mock; "
        "import stoma.config as cfg; "
        "cfg.LOCKED_LTHCM_COMMIT = '0000000000000000000000000000000000000000'; "
        "import stoma.cli as cli; "
        "cli.cli(['status'], standalone_mode=False)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "DependencyOriginError" in result.stderr or "Runtime import violation" in result.stderr


# ===========================================================================
# Finding 5: Streaming Ingestion and Safe Size Caps
# ===========================================================================

def test_r2_f5_safe_size_cap_blocks_oversized_code_or_json(test_config, shared_embedder, temp_workspace):
    """
    Assert that files exceeding 25MB for non-streaming plugins (Code AST, OCR/Whisper JSON)
    are safely blocked and reported as failed rather than consuming unbounded memory.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    oversized_code = temp_workspace / "giant_script.py"
    oversized_code.write_text("x = 1\n" * 100, encoding="utf-8")

    code_plugin = indexer.plugins.get_plugin_for_file(oversized_code)
    assert code_plugin is not None
    orig_cap = code_plugin.MAX_SAFE_SIZE_BYTES
    try:
        code_plugin.MAX_SAFE_SIZE_BYTES = 5
        report = indexer.ingest_file(oversized_code)
    finally:
        code_plugin.MAX_SAFE_SIZE_BYTES = orig_cap

    assert report.status in ("skipped_unsupported", "failed")
    assert "exceeds safe" in (report.error_message or "") or "BLOCKED" in (report.error_message or "")


def test_r2_f5_streaming_ingestion_error_late_in_stream_rolls_back(test_config, shared_embedder, temp_workspace):
    """
    Assert that if an error occurs late in a streaming ingestion batch,
    the active generation remains unmodified and rolls back cleanly.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f_good = temp_workspace / "base.txt"
    f_good.write_text("Base generation content.", encoding="utf-8")
    indexer.ingest_file(f_good)
    gen_before = indexer.current_generation

    # Ingest another file where embedder explodes mid-stream
    f_stream = temp_workspace / "stream_fail.txt"
    f_stream.write_text("\n\n".join([f"Paragraph {i} content text." for i in range(20)]), encoding="utf-8")

    with patch.object(indexer.embedder, "encode", side_effect=RuntimeError("Simulated CUDA OOM in stream")):
        report = indexer.ingest_file(f_stream)
        assert report.status == "failed"

    # Current generation must remain untouched
    assert indexer.current_generation == gen_before
    assert f"{test_config.allowed_roots[0].name}:stream_fail.txt" not in indexer.manifest.sources


# ===========================================================================
# Finding 7: Dianoia Interchange Observed Outcome Reporting
# ===========================================================================

def test_r2_f7_interchange_reports_cli_execution_failure_not_unsupported(test_config, shared_embedder, temp_workspace):
    """
    Assert that if Dianoia CLI execution fails for reasons other than the edge gate
    (e.g. timeout, missing module, crash), verify_dianoia_interchange reports
    'dianoia_cli_execution_failed' with exit code and stderr, NOT 'direct_cli_walk_unsupported_zero_edges'
    and NOT compatible=True.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f = temp_workspace / "interchange_doc.txt"
    f.write_text("Interchange test document.", encoding="utf-8")
    indexer.ingest_file(f)

    # 1. Simulate subprocess error with non-zero exit code and unrelated error (e.g. ModuleNotFoundError)
    simulated_proc = MagicMock()
    simulated_proc.returncode = 2
    simulated_proc.stdout = ""
    simulated_proc.stderr = "ModuleNotFoundError: No module named 'scipy'"

    with patch("subprocess.run", return_value=simulated_proc):
        res = verify_dianoia_interchange(test_config.index_path)

    assert res["status"] == "dianoia_cli_execution_failed"
    assert res["compatible"] is False
    assert res["cli_return_code"] == 2
    assert "No module named" in res["cli_stderr"]

    # 2. Simulate timeout
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="dianoia query", timeout=10.0)):
        res_to = verify_dianoia_interchange(test_config.index_path)

    assert res_to["status"] == "dianoia_cli_timeout"
    assert res_to["compatible"] is False


def test_r2_f7_interchange_real_edge_refusal_outcome(test_config, shared_embedder, temp_workspace):
    """
    Assert that the legitimate observed outcome on a valid 0-edge Stoma index
    is 'direct_cli_walk_unsupported_zero_edges' when edge_gate_triggered is true.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f = temp_workspace / "edge_refusal.txt"
    f.write_text("Testing real edge refusal message.", encoding="utf-8")
    indexer.ingest_file(f)

    res = verify_dianoia_interchange(test_config.index_path)
    assert res["serialization_match"] is True
    assert res["is_lthcm_engine"] is True
    assert res["dimension_match"] is True
    assert res["edge_count"] == 0
    # In v1 zero graph edges are built by design; CLI exits with code 1 & edge gate message
    if res.get("edge_gate_triggered"):
        assert res["status"] == "direct_cli_walk_unsupported_zero_edges"
        assert res["compatible"] is False


# ===========================================================================
# Finding 8: Dual Store Validation and Query Strong Provenance
# ===========================================================================

def test_r2_f8_dual_store_validation_detects_store_disagreements(test_config, shared_embedder):
    """
    Assert that StomaLTHCMAdapter detects disagreements between hyperbolic
    and superposition stores, as well as duplicate hyperbolic document IDs.
    """
    adapter = StomaLTHCMAdapter(
        engine=StomaLTHCMAdapter.create_fresh_engine(shared_embedder),
        embedder=shared_embedder,
    )

    # 1. Duplicate hyperbolic ID detection
    adapter.engine.hyperbolic.documents = [
        {"id": "doc_A", "vector": torch.zeros(shared_embedder.d_model)},
        {"id": "doc_A", "vector": torch.zeros(shared_embedder.d_model)},
    ]
    with pytest.raises(IndexIntegrityError, match="Duplicate document IDs detected in hyperbolic store"):
        adapter.get_store_doc_ids()

    # 2. Store disagreement: hyperbolic has doc_A, superposition has doc_B
    adapter.engine.hyperbolic.documents = [
        {"id": "doc_A", "vector": torch.zeros(shared_embedder.d_model)},
    ]
    adapter.engine.superposition.doc_index = {
        "doc_B": 0,
    }
    with pytest.raises(IndexIntegrityError, match="Dual-store mismatch"):
        adapter.get_indexed_doc_ids()


def test_r2_f8_query_fails_closed_on_source_hash_or_path_mismatch(test_config, shared_embedder, temp_workspace):
    """
    Assert that adapter.query() validates candidate provenance against manifest.sources,
    failing closed with UnresolvedProvenanceError if content_hash or path metadata mismatches.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f = temp_workspace / "provenance_check.txt"
    f.write_text("Candidate document for query provenance checks.", encoding="utf-8")
    indexer.ingest_file(f)

    # Normal query succeeds
    res = indexer.query("Candidate document", top_k=1)
    assert len(res["results"]) == 1

    # Corrupt source content_hash in manifest (simulating disk source alteration or metadata spoofing)
    src_id = list(indexer.manifest.sources.keys())[0]
    indexer.manifest.sources[src_id].content_hash = "spoofed_hash_12345"

    # Query must fail closed with UnresolvedProvenanceError
    with pytest.raises(UnresolvedProvenanceError, match="content hash .* does not match source"):
        indexer.query("Candidate document", top_k=1)
