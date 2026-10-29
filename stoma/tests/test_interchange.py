"""
Dianoia Interchange Compatibility Tests.
Exercises Dianoia's actual CLI loader via controlled subprocess execution against Stoma's persisted engine.
Captures the real Dianoia edge-gate refusal and reports accurately that direct CLI walk is unsupported
because Stoma v1 builds zero wormhole graph edges.
"""

from pathlib import Path
import pytest

from stoma.indexer import StomaIndexer
from stoma.interchange import verify_dianoia_interchange


def test_dianoia_interchange_verification(test_config, shared_embedder, temp_workspace):
    """
    Build a Stoma index and verify interchange against Dianoia's actual CLI load requirements.
    Asserts:
    - Serialization is 100% matched (pickle)
    - Engine is an LTHCMEngine
    - Dimensions match (d_model=384, num_facets=1)
    - Dianoia actual CLI loader is invoked and refuses with exit code 1 due to 0 graph edges.
    - Status is reported as 'direct_cli_walk_unsupported_zero_edges'.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)

    doc = temp_workspace / "interchange_doc.txt"
    doc.write_text("Interchange test document verifying Dianoia compatibility.", encoding="utf-8")
    indexer.ingest_file(doc)

    res = verify_dianoia_interchange(test_config.index_path)

    # Assert format and type compatibility
    assert res["serialization_match"] is True
    assert res["is_lthcm_engine"] is True
    assert res["dimension_match"] is True
    assert res["d_model"] == shared_embedder.d_model
    assert res["num_facets"] == 1
    assert res["edge_count"] == 0

    # Assert real Dianoia CLI execution and refusal
    assert res["direct_cli_walk_supported"] is False
    assert res["cli_return_code"] == 1
    assert res["edge_gate_triggered"] is True
    assert "contains 0 graph edges" in res["cli_stderr"]
    assert res["status"] == "direct_cli_walk_unsupported_zero_edges"
    assert "0 graph edges" in res["reason"]


def test_dianoia_interchange_untrusted_pickle_rejected(test_config, shared_embedder, temp_workspace, monkeypatch):
    """
    Assert that verify_dianoia_interchange fails closed with 'untrusted_pickle_rejected'
    if allow_trusted_pickle is False and no authorization env var is set.
    """
    indexer = StomaIndexer(config=test_config, embedder=shared_embedder)
    doc = temp_workspace / "interchange_doc2.txt"
    doc.write_text("Interchange test document.", encoding="utf-8")
    indexer.ingest_file(doc)

    monkeypatch.delenv("STOMA_ALLOW_TRUSTED_PICKLE", raising=False)
    monkeypatch.delenv("DIANOIA_ALLOW_TRUSTED_PICKLE", raising=False)

    res = verify_dianoia_interchange(test_config.index_path, allow_trusted_pickle=False)
    assert res["compatible"] is False
    assert res["status"] == "untrusted_pickle_rejected"
    assert "without explicit authorization" in res["message"]

