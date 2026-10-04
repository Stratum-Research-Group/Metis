"""
Unit and integration tests for Stoma Watcher:
- Symlink exclusion (symlinks outside allowed roots are never followed)
- Event debouncing
- Full manual reconciliation
"""

import os
import tempfile
import time
from pathlib import Path
import pytest

from stoma.indexer import StomaIndexer
from stoma.watcher import StomaWatcher


def test_symlink_outside_allowed_roots_excluded(test_config, shared_embedder, temp_workspace):
    """
    Assert that symlinks pointing outside allowed roots are NEVER followed or indexed.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    # Create an external directory completely outside temp_workspace
    with tempfile.TemporaryDirectory(prefix="outside_root_") as outside_dir:
        outside_path = Path(outside_dir)
        secret_file = outside_path / "secret.txt"
        secret_file.write_text("Confidential external text outside allowed roots.", encoding="utf-8")

        # Create a symlink inside temp_workspace pointing to external secret file
        symlink_file = temp_workspace / "symlink_to_secret.txt"
        try:
            os.symlink(secret_file, symlink_file)
        except OSError:
            pytest.skip("Symlinks not supported on this filesystem.")

        # Attempt to ingest the symlink file
        report = indexer.ingest_file(symlink_file)
        assert report.status == "skipped_unsupported"
        assert "not within permitted ingestion roots" in (report.warning_message or "")

        # Verify not indexed
        res = indexer.query("Confidential external text")
        assert len(res["results"]) == 0


def test_watcher_manual_reconcile(test_config, shared_embedder, temp_workspace):
    """
    Test that full manual reconciliation discovers files on disk and prunes
    files that were deleted from disk.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    watcher = StomaWatcher(indexer=indexer)

    f1 = temp_workspace / "keep.txt"
    f1.write_text("File that will stay on disk.", encoding="utf-8")
    f2 = temp_workspace / "to_delete.txt"
    f2.write_text("File that will be deleted directly on disk.", encoding="utf-8")

    # Ingest both
    indexer.ingest_file(f1)
    indexer.ingest_file(f2)
    assert len(indexer.manifest.sources) == 2

    # Remove f2 directly from disk without informing indexer
    f2.unlink()

    # Reconcile
    summary = watcher.reconcile()
    assert summary["missing_files_cleaned"] == 1
    assert "to_delete.txt" not in indexer.manifest.sources
    assert "keep.txt" in indexer.manifest.sources
