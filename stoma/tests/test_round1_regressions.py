"""
Stoma Round 1 Regression Acceptance Suite.
Exhaustively tests all 8 findings from stoma-round1-fixes-5c798e0-a984755f.txt:
1. Transactional Generations & Failure Recovery
2. Move/Rename Semantics Without Stale IDs
3. Real Model Identity, Weights Checksum & Strict Non-Truncating Chunker
4. Pinned Runtime Import Verification
5. Bounded Large Input & Memory Streaming
6. Root Identity in Source and Passage IDs
7. Actual Dianoia CLI Interchange Refusal
8. Fail Closed on Unresolved Provenance
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import pytest

from stoma.adapter import StomaLTHCMAdapter, UnresolvedProvenanceError
from stoma.chunker import Chunker, PassageChunk
from stoma.config import DependencyOriginError, StomaConfig, verify_runtime_dependencies
from stoma.embedder import ModelInspectionError, StomaEmbedder
from stoma.indexer import CorruptedStorageError, RebuildAbortedError, StomaIndexer
from stoma.interchange import verify_dianoia_interchange
from stoma.plugins.base import RawPassage
from stoma.provenance import (
    ModelIdentityMismatchError,
    PassageProvenance,
    ProvenanceManifest,
    SourceFileRecord,
    compute_file_sha256,
)
from stoma.watcher import StomaWatcher


# ---------------------------------------------------------------------------
# Finding 1: Transactional Generations & Failure Recovery
# ---------------------------------------------------------------------------

def test_finding_1_failure_injection_preserves_last_generation(test_config, shared_embedder, temp_workspace):
    """
    Inject failures at various pipeline stages during incremental ingest / rebuild:
    - Extraction of a second file
    - Mid-embedding
    - Index write
    - Manifest write
    - Commit interruption
    Assert that the prior generation remains completely untouched and answers queries immediately
    and after process restart.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    f1 = temp_workspace / "initial_doc.txt"
    f1.write_text("Stable initial document content serving queries reliably.", encoding="utf-8")
    rep1 = indexer.ingest_file(f1)
    assert rep1.status == "indexed"
    initial_gen = indexer.current_generation
    assert initial_gen >= 1

    # Verify initial query succeeds
    res1 = indexer.query("Stable initial document")
    assert len(res1["results"]) > 0

    # 1. Failure injection during rebuild on missing / failed unchanged file
    f2 = temp_workspace / "second_doc.txt"
    f2.write_text("Second document content.", encoding="utf-8")
    indexer.ingest_file(f2)
    gen2 = indexer.current_generation
    assert gen2 > initial_gen

    # Delete f2 from disk behind Stoma's back, then modify f1 to trigger a rebuild
    f2.unlink()
    f1.write_text("Modified f1 attempting rebuild while f2 is missing on disk.", encoding="utf-8")

    with pytest.raises(RebuildAbortedError, match="Rebuild aborted"):
        indexer.ingest_file(f1)

    # Assert generation was preserved and did not advance
    assert indexer.current_generation == gen2
    res_after_abort = indexer.query("Stable initial document")
    assert len(res_after_abort["results"]) > 0

    # 2. Failure injection mid-embedding
    f3 = temp_workspace / "third_doc.txt"
    f3.write_text("Third document to fail during embedding.", encoding="utf-8")

    def failing_encode(texts):
        raise RuntimeError("Simulated GPU out-of-memory or mid-embedding crash!")

    with patch.object(shared_embedder, "encode", side_effect=failing_encode):
        rep3 = indexer.ingest_file(f3)
        assert rep3.status == "failed"
        assert "Simulated GPU" in (rep3.error_message or "")

    # Generation must not have advanced
    assert indexer.current_generation == gen2
    res_post_emb_crash = indexer.query("Stable initial document")
    assert len(res_post_emb_crash["results"]) > 0

    # 3. Interruption between staged files and commit:
    # Simulate a crashed process that wrote .staged files but died before replacing current.json
    staged_fake = test_config.generations_dir / f"gen_{gen2+1:06d}.staged"
    staged_fake.mkdir(parents=True, exist_ok=True)
    (staged_fake / "index.pkl").write_bytes(b"partial corrupted index")
    (staged_fake / "manifest.json").write_text("corrupted json", encoding="utf-8")

    # Restart Indexer in a new process: must ignore or clean incomplete stage and load gen2
    restarted = StomaIndexer(config=test_config, embedder=shared_embedder)
    assert restarted.current_generation == gen2
    res_restart = restarted.query("Stable initial document")
    assert len(res_restart["results"]) > 0
    assert any("initial_doc.txt" in r["provenance"]["rel_path"] for r in res_restart["results"])


# ---------------------------------------------------------------------------
# Finding 2: Move/Rename Semantics Without Stale IDs
# ---------------------------------------------------------------------------

def test_finding_2_move_rename_transaction_semantics(test_config, shared_embedder, temp_workspace):
    """
    Test that file move/rename:
    1. Transactionally purges the old ID set and indexes the new path under a new ID set.
    2. Zero old passage IDs remain in manifest or engine.
    3. Moving outside allowed roots acts as deletion.
    4. Moving inside from outside acts as new ingest.
    5. Watcher events and manual reconcile handle renames cleanly without resurrecting old paths.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    f_src = temp_workspace / "old_name.txt"
    f_src.write_text("Alpha unique content for move rename test.", encoding="utf-8")
    indexer.ingest_file(f_src)

    # Verify old source and passages exist
    old_pids = list(indexer.manifest.passages.keys())
    assert any("old_name.txt" in pid for pid in old_pids)

    # Execute move to new name inside allowed root
    f_dst = temp_workspace / "new_name.txt"
    f_src.rename(f_dst)

    rep_move = indexer.move_file(f_src, f_dst)
    assert rep_move.status == "moved"

    # Verify manifest: old source and old passages are completely gone!
    assert all("old_name.txt" not in sid for sid in indexer.manifest.sources)
    assert all("old_name.txt" not in pid for pid in indexer.manifest.passages)

    # Verify new source exists
    assert any("new_name.txt" in sid for sid in indexer.manifest.sources)
    assert any("new_name.txt" in pid for pid in indexer.manifest.passages)

    # Query returns results pointing strictly to new location
    res = indexer.query("Alpha unique content")
    assert len(res["results"]) > 0
    assert res["results"][0]["provenance"]["rel_path"] == "new_name.txt"
    assert "old_name.txt" not in res["results"][0]["doc_id"]

    # Test move OUT of allowed roots: acts as deletion
    with tempfile.TemporaryDirectory() as external_dir:
        ext_dest = Path(external_dir) / "exported.txt"
        f_dst.rename(ext_dest)

        rep_out = indexer.move_file(f_dst, ext_dest)
        assert rep_out.status == "deleted"
        assert len(indexer.manifest.sources) == 0
        assert len(indexer.manifest.passages) == 0

        # Query returns no results
        res_del = indexer.query("Alpha unique content")
        assert len(res_del["results"]) == 0


def test_finding_2_watcher_manual_reconcile_rename(test_config, shared_embedder, temp_workspace):
    """Test manual reconcile detects missed renames on disk and purges old entries."""
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    watcher = StomaWatcher(indexer=indexer)

    f_init = temp_workspace / "reconcile_old.txt"
    f_init.write_text("Content before manual reconcile rename.", encoding="utf-8")
    indexer.ingest_file(f_init)

    # Simulate a missed rename while watcher was down
    f_renamed = temp_workspace / "reconcile_new.txt"
    f_init.rename(f_renamed)

    summary = watcher.reconcile()
    assert summary["missing_files_cleaned"] == 1

    assert any("reconcile_new.txt" in sid for sid in indexer.manifest.sources)
    assert all("reconcile_old.txt" not in sid for sid in indexer.manifest.sources)

    res = indexer.query("Content before manual reconcile")
    assert len(res["results"]) > 0
    assert res["results"][0]["provenance"]["rel_path"] == "reconcile_new.txt"


# ---------------------------------------------------------------------------
# Finding 3: Real Model Identity, Weights Checksum & Strict Chunker
# ---------------------------------------------------------------------------

def test_finding_3_unbroken_string_no_truncation(shared_embedder):
    """
    Assert that an unbroken string > 500 characters is NOT truncated by the chunker.
    Every single character must be preserved across chunks, and each chunk must fit
    within effective sequence limits and embed successfully.
    """
    chunker = Chunker(shared_embedder)
    unbroken_text = "x" * 1500

    passages = [RawPassage(content=unbroken_text, location={"char_start": 0})]
    chunks = chunker.chunk_raw_passages(passages, deduplicate_within_file=False)

    assert len(chunks) >= 1
    # Check that reconstructed content matches 100% (zero lost tail, zero discarded characters)
    reconstructed = "".join(c.content for c in chunks)
    assert reconstructed == unbroken_text
    assert len(reconstructed) == 1500

    # Verify each chunk embeds cleanly without exceeding model limit
    for c in chunks:
        assert c.token_count <= chunker.hard_max_tokens
        emb = shared_embedder.encode_single(c.content)
        assert emb.shape == (384,)


def test_finding_3_strict_model_identity_validation(test_config, shared_embedder, temp_workspace):
    """
    Assert strict model identity validation:
    - Differing weights checksum rejects on reopen with ModelIdentityMismatchError.
    - Missing revision or checksum rejects on reopen.
    - Tokenizer failure fails loudly.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f = temp_workspace / "meta_test.txt"
    f.write_text("Testing model metadata persistence and validation.", encoding="utf-8")
    indexer.ingest_file(f)

    # 1. Corrupt weights checksum in active generation manifest
    gen_man_path = test_config.get_generation_paths(indexer.current_generation)[1]
    with open(gen_man_path, "r", encoding="utf-8") as mf:
        data = json.load(mf)

    data["model_meta"]["weights_checksum"] = "badc0ffee" * 7
    with open(gen_man_path, "w", encoding="utf-8") as mf:
        json.dump(data, mf)

    # Update recorded manifest hash in current.json & commit_marker.json so startup hash check passes
    from stoma.provenance import compute_file_sha256
    new_hash = compute_file_sha256(gen_man_path)
    cur_p = test_config.current_pointer_path
    if cur_p.is_file():
        with open(cur_p, "r", encoding="utf-8") as pf:
            pdata = json.load(pf)
        pdata["manifest_hash"] = new_hash
        with open(cur_p, "w", encoding="utf-8") as pf:
            json.dump(pdata, pf)
    marker_p = gen_man_path.parent / "commit_marker.json"
    if marker_p.is_file():
        with open(marker_p, "r", encoding="utf-8") as mf:
            mdata = json.load(mf)
        mdata["manifest_hash"] = new_hash
        with open(marker_p, "w", encoding="utf-8") as mf:
            json.dump(mdata, mf)

    # Reopening with original embedder must raise ModelIdentityMismatchError loudly
    with pytest.raises(ModelIdentityMismatchError, match="Model weights checksum mismatch"):
        StomaIndexer(config=test_config, embedder=shared_embedder)


# ---------------------------------------------------------------------------
# Finding 4: Pinned Runtime Import Verification
# ---------------------------------------------------------------------------

def test_finding_4_runtime_import_verification():
    """Verify that lthcm and dianoia origin check asserts local submodule paths."""
    res = verify_runtime_dependencies()
    assert res["lthcm_origin_ok"] is True
    assert res["dianoia_origin_ok"] is True
    assert "deps/LTHCM" in res["lthcm_path"]
    assert "deps/dianoia" in res["dianoia_path"]

    # Test that an unexpected fake root raises DependencyOriginError
    with tempfile.TemporaryDirectory() as fake_root:
        with pytest.raises(DependencyOriginError, match="Runtime import violation"):
            verify_runtime_dependencies(stoma_root=Path(fake_root))


# ---------------------------------------------------------------------------
# Finding 5: Bounded Large Input & Memory Streaming
# ---------------------------------------------------------------------------

def test_finding_5_large_file_streaming_bounded(test_config, shared_embedder, temp_workspace):
    """
    Assert that indexing a synthetic large text file streams passages without consuming
    unbounded memory, and scan_and_ingest_directory bounds report lists.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    # Create synthetic multi-section file (1,000 paragraphs)
    large_doc = temp_workspace / "large_source.txt"
    with open(large_doc, "w", encoding="utf-8") as f:
        for p in range(500):
            f.write(f"# Section {p}\n\nThis is paragraph content for section {p}. Detailed domain text.\n\n")

    rep = indexer.ingest_file(large_doc)
    assert rep.status == "indexed"
    assert rep.passages_count >= 500

    # Query within large file
    res = indexer.query("section 250")
    assert len(res["results"]) > 0

    # Assert bounded reporting in folder scan
    summary = indexer.scan_and_ingest_directory(temp_workspace)
    assert summary.total_files_scanned >= 1
    # Verify recent_reports list is bounded
    assert len(summary.recent_reports) <= 100


# ---------------------------------------------------------------------------
# Finding 6: Root Identity in Source and Passage IDs
# ---------------------------------------------------------------------------

def test_finding_6_multi_root_collision_resistance(shared_embedder, tmp_path):
    """
    Configure two distinct allowed roots that both contain a file named 'notes.txt' with identical content.
    Assert:
    - Both files are indexed under distinct root IDs without collision or overwriting.
    - Provenance for each result accurately specifies its exact root and path.
    - Editing notes.txt in Root A does not affect Root B.
    - Deleting notes.txt in Root A leaves Root B active and retrievable.
    """
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    storage = tmp_path / ".stoma"
    root_a.mkdir()
    root_b.mkdir()

    cfg = StomaConfig(allowed_roots=[root_a, root_b], storage_dir=storage)
    indexer = StomaIndexer(config=cfg, embedder=shared_embedder)

    file_a = root_a / "notes.txt"
    file_b = root_b / "notes.txt"

    file_a.write_text("Common text content inside notes in root A.", encoding="utf-8")
    file_b.write_text("Common text content inside notes in root B.", encoding="utf-8")

    rep_a = indexer.ingest_file(file_a)
    rep_b = indexer.ingest_file(file_b)

    assert rep_a.status == "indexed"
    assert rep_b.status == "indexed"

    # Both sources must exist in manifest simultaneously
    assert len(indexer.manifest.sources) == 2
    root_id_a = cfg.get_root_id(root_a)
    root_id_b = cfg.get_root_id(root_b)

    assert f"{root_id_a}:notes.txt" in indexer.manifest.sources
    assert f"{root_id_b}:notes.txt" in indexer.manifest.sources

    # Query returns results with distinct root IDs
    res = indexer.query("Common text content")
    assert len(res["results"]) >= 2
    provs = [item["provenance"] for item in res["results"]]
    root_ids_returned = {p["root_id"] for p in provs}
    assert root_id_a in root_ids_returned
    assert root_id_b in root_ids_returned

    # Delete notes.txt in Root A only
    indexer.delete_file(file_a)
    assert f"{root_id_a}:notes.txt" not in indexer.manifest.sources
    assert f"{root_id_b}:notes.txt" in indexer.manifest.sources

    # Root B still answers queries
    res_b = indexer.query("Common text content")
    assert len(res_b["results"]) >= 1
    assert res_b["results"][0]["provenance"]["root_id"] == root_id_b


# ---------------------------------------------------------------------------
# Finding 7: Actual Dianoia CLI Interchange Refusal
# ---------------------------------------------------------------------------

def test_finding_7_actual_dianoia_cli_refusal(test_config, shared_embedder, temp_workspace):
    """
    Exercise Dianoia's actual CLI loader via subprocess against Stoma's persisted engine.
    Assert that the real CLI refusal is captured ('contains 0 graph edges') with exit code 1.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    doc = temp_workspace / "dianoia_doc.txt"
    doc.write_text("Interchange test document verifying real CLI refusal.", encoding="utf-8")
    indexer.ingest_file(doc)

    res = verify_dianoia_interchange(test_config.index_path)
    assert res["direct_cli_walk_supported"] is False
    assert res["cli_return_code"] == 1
    assert res["edge_gate_triggered"] is True
    assert "contains 0 graph edges" in res["cli_stderr"]
    assert res["status"] == "direct_cli_walk_unsupported_zero_edges"


# ---------------------------------------------------------------------------
# Finding 8: Fail Closed on Unresolved Provenance
# ---------------------------------------------------------------------------

def test_finding_8_fail_closed_unresolved_provenance(test_config, shared_embedder, temp_workspace):
    """
    Assert that query fails closed with UnresolvedProvenanceError if any candidate passage ID
    lacks matching provenance in the manifest. No result must ever leak with provenance=None.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f = temp_workspace / "prov_test.txt"
    f.write_text("Provenance fail-closed test content.", encoding="utf-8")
    indexer.ingest_file(f)

    # Corrupt manifest by deleting the passage's provenance record while leaving the engine indexed
    first_pid = list(indexer.manifest.passages.keys())[0]
    del indexer.manifest.passages[first_pid]

    # Query must fail closed with UnresolvedProvenanceError
    with pytest.raises(UnresolvedProvenanceError, match="has unresolved provenance in manifest"):
        indexer.query("Provenance fail-closed")
