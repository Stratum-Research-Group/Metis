"""Unit tests for Stoma offline knowledge graph edge builder."""

from pathlib import Path
import pickle
import pytest
import torch
from click.testing import CliRunner

from lthcm.engine.pipeline import LTHCMEngine
from stoma.cli import cli
from stoma.config import StomaConfig
from stoma.graph_builder import build_knn_knowledge_graph
from stoma.interchange import verify_dianoia_interchange


@pytest.fixture
def mock_index_file(tmp_path: Path) -> Path:
    """Create a minimal LTHCMEngine pickle with two highly aligned embeddings and 0 edges."""
    idx_path = tmp_path / "test_index.pkl"
    engine = LTHCMEngine(d_model=384, num_facets=1, device="cpu")

    # Two nearly identical vectors
    vec1 = torch.zeros(1, 384)
    vec1[0, 0] = 1.0
    vec2 = torch.zeros(1, 384)
    vec2[0, 0] = 0.99
    vec2[0, 1] = 0.05

    # One orthogonal vector
    vec3 = torch.zeros(1, 384)
    vec3[0, 2] = 1.0

    engine.index_document("doc_1", "Artificial intelligence in medicine", vec1, timestamp=100.0)
    engine.index_document("doc_2", "AI applications in clinical healthcare", vec2, timestamp=101.0)
    engine.index_document("doc_3", "Baking sourdough bread recipes", vec3, timestamp=102.0)

    with open(idx_path, "wb") as f:
        pickle.dump(engine, f)

    return idx_path


def test_build_knn_knowledge_graph(mock_index_file: Path):
    # Verify initially 0 edges
    interchange_before = verify_dianoia_interchange(mock_index_file)
    assert interchange_before.get("edge_count") == 0

    # Build graph with threshold 0.75
    res = build_knn_knowledge_graph(
        index_path=mock_index_file,
        similarity_threshold=0.75,
        max_k_neighbors=2,
    )

    assert res["nodes_count"] == 3
    assert res["edges_added"] >= 2  # doc_1 <-> doc_2 bidirectional
    assert res["total_edges"] >= 2

    # Verify updated index on disk has edges
    with open(mock_index_file, "rb") as f:
        updated_engine = pickle.load(f)
    assert updated_engine.wormhole.graph.number_of_edges() >= 2

    # Verify interchange sees the edges
    interchange_after = verify_dianoia_interchange(mock_index_file)
    assert interchange_after.get("edge_count") >= 2


def test_cli_graph_build(mock_index_file: Path, tmp_path: Path):
    cfg_path = tmp_path / "stoma_config.yaml"
    cfg = StomaConfig(
        storage_dir=tmp_path / ".stoma",
        allowed_roots=[tmp_path],
    )
    cfg.save(cfg_path)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--config", str(cfg_path), "graph", "build", "--index", str(mock_index_file), "--threshold", "0.70"],
    )
    assert result.exit_code == 0
    assert "Nodes in index: 3" in result.output
    assert "Edges added:" in result.output
