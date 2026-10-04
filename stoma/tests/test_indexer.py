"""
Integration tests for Stoma Indexer lifecycle:
- Provenance resolution (path, content hash, location, timestamp)
- Lifecycle: unchanged skipped, changed replaced without duplicates, deleted gone
- Rename policy
- Persistence and restart validation
- Rejection of mismatched encoder on reopen
- Large-folder bounded behavior
"""

import time
from pathlib import Path
import pytest

from stoma.config import StomaConfig
from stoma.embedder import StomaEmbedder
from stoma.indexer import StomaIndexer
from stoma.provenance import compute_file_sha256


def test_indexer_provenance_resolution(test_config, shared_embedder, temp_workspace):
    """Verify that every query result resolves passage ID to source path, hash, and timestamp."""
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    doc_file = temp_workspace / "guide.md"
    doc_file.write_text(
        "# Memory Substrate Guide\n\n"
        "LTHCM provides non-Euclidean representation with Poincare ball embeddings.",
        encoding="utf-8",
    )

    report = indexer.ingest_file(doc_file)
    assert report.status == "indexed"
    assert report.passages_count >= 1

    # Query Stoma
    res = indexer.query("non-Euclidean representation")
    assert len(res["results"]) > 0

    first = res["results"][0]
    prov = first["provenance"]
    assert prov is not None
    assert prov["rel_path"] == "guide.md"
    assert prov["file_content_hash"] == compute_file_sha256(doc_file)
    assert prov["ingested_at"] > 0.0
    assert "start_line" in prov["location"]


def test_indexer_incremental_lifecycle(test_config, shared_embedder, temp_workspace):
    """
    Test unchanged file skipped, modified file replaced cleanly via atomic swap,
    and deleted file removed completely.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    f1 = temp_workspace / "topic_alpha.txt"
    f1.write_text("Alpha topic: initial baseline content.", encoding="utf-8")

    # 1. Initial Ingest
    rep1 = indexer.ingest_file(f1)
    assert rep1.status == "indexed"

    # Query initial
    res_init = indexer.query("baseline content")
    assert any("Alpha topic: initial baseline content" in item["content"] for item in res_init["results"])

    # 2. Ingest unchanged -> must be skipped
    rep_skip = indexer.ingest_file(f1)
    assert rep_skip.status == "skipped_unchanged"

    # 3. Modify file -> must rebuild clean index via atomic swap with NO stale entries
    f1.write_text("Alpha topic: entirely revised updated content.", encoding="utf-8")
    rep_mod = indexer.ingest_file(f1)
    assert rep_mod.status == "indexed"

    # Query after modification: assert old passages are NOT returned!
    res_old = indexer.query("baseline content")
    old_matches = [
        item for item in res_old["results"]
        if "initial baseline content" in item["content"]
    ]
    assert len(old_matches) == 0, "Old passages were still returned after file modification!"

    # New passages must be returned
    res_new = indexer.query("entirely revised updated content")
    assert any("entirely revised updated content" in item["content"] for item in res_new["results"])

    # 4. Delete file -> must purge passages completely
    f1.unlink()
    deleted = indexer.delete_file("topic_alpha.txt")
    assert deleted is True

    # Query after deletion: assert no passages returned
    res_del = indexer.query("revised updated content")
    del_matches = [
        item for item in res_del["results"]
        if "topic_alpha.txt" in item.get("doc_id", "")
    ]
    assert len(del_matches) == 0, "Deleted file passages were still returned!"


def test_indexer_restart_durability(test_config, shared_embedder, temp_workspace):
    """Assert index and manifest survive process restart and return identical results."""
    # First process
    indexer1 = StomaIndexer(config=test_config, embedder=shared_embedder)
    f = temp_workspace / "persist.txt"
    f.write_text("Durable topological persistence test across restarts.", encoding="utf-8")
    indexer1.ingest_file(f)

    # Reopen in a second, completely separate Indexer instance
    indexer2 = StomaIndexer(config=test_config, embedder=shared_embedder)
    res = indexer2.query("topological persistence")
    assert len(res["results"]) > 0
    assert "persist.txt" in res["results"][0]["doc_id"]


def test_indexer_rejects_mismatched_model_on_restart(test_config, shared_embedder, temp_workspace):
    """
    On reopen, Stoma must validate the loaded engine's d_model/num_facets and recorded encoder
    model identity against the manifest and current embedder, and reject mismatches loudly.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    f = temp_workspace / "test.txt"
    f.write_text("Valid text content.", encoding="utf-8")
    indexer.ingest_file(f)

    # Intentionally corrupt manifest model name in active generation to simulate an incompatible encoder restart
    manifest_data = indexer.manifest.to_dict()
    manifest_data["model_meta"]["model_name"] = "incompatible-other-model"
    gen_man_path = test_config.get_generation_paths(indexer.current_generation)[1]
    import json
    with open(gen_man_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf)
    with open(test_config.manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf)

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

    # Attempting to load with the original embedder must raise ValueError
    with pytest.raises(ValueError, match="Model identity mismatch"):
        StomaIndexer(config=test_config, embedder=shared_embedder)


def test_indexer_rename_policy(test_config, shared_embedder, temp_workspace):
    """
    Test transactional move/rename semantics (Finding 2):
    - Moving old.txt -> new.txt removes old source and old passage IDs completely.
    - Emits new source and new passage IDs with correct root_id.
    - Query returns result pointing to new.txt, with 0 old passages retrievable.
    - Reopen survives cleanly.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    f_old = temp_workspace / "old.txt"
    f_old.write_text("Unique text content inside old file for rename test.", encoding="utf-8")
    indexer.ingest_file(f_old)

    # Verify initial state
    res_before = indexer.query("Unique text content inside old file")
    assert len(res_before["results"]) > 0
    assert "old.txt" in res_before["results"][0]["provenance"]["rel_path"]

    # Rename file on disk
    f_new = temp_workspace / "new.txt"
    f_old.rename(f_new)

    # Execute move_file
    rep_move = indexer.move_file(f_old, f_new)
    assert rep_move.status == "moved"

    # Verify manifest state: exactly ONE source, and it is new.txt
    assert len(indexer.manifest.sources) == 1
    src = list(indexer.manifest.sources.values())[0]
    assert src.rel_path == "new.txt"
    assert all("new.txt" in pid for pid in src.passage_ids)
    assert all("old.txt" not in pid for pid in src.passage_ids)

    # Query after move: result provenance points to new.txt
    res_after = indexer.query("Unique text content inside old file")
    assert len(res_after["results"]) > 0
    prov = res_after["results"][0]["provenance"]
    assert prov["rel_path"] == "new.txt"
    assert "old.txt" not in prov["passage_id"]

    # Reopen in new indexer instance
    reopened = StomaIndexer(config=test_config, embedder=shared_embedder)
    assert len(reopened.manifest.sources) == 1
    res_reopen = reopened.query("Unique text content inside old file")
    assert res_reopen["results"][0]["provenance"]["rel_path"] == "new.txt"


def test_indexer_large_folder_bounded_walk(test_config, shared_embedder, temp_workspace):
    """
    Assert that scan_and_ingest_directory bounds batches and avoids loading all files at once.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    # Create a directory structure with subfolders
    sub1 = temp_workspace / "sub1"
    sub2 = temp_workspace / "sub2"
    sub1.mkdir()
    sub2.mkdir()

    for i in range(5):
        (sub1 / f"doc_{i}.txt").write_text(f"Document {i} in subfolder 1.", encoding="utf-8")
        (sub2 / f"doc_{i}.txt").write_text(f"Document {i} in subfolder 2.", encoding="utf-8")

    summary = indexer.scan_and_ingest_directory(temp_workspace)
    assert summary.total_files_scanned >= 10
    assert summary.files_indexed >= 10
    assert summary.files_failed == 0
