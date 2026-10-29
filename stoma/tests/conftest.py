"""
Pytest configuration and shared fixtures for Stoma test suite.
"""

import shutil
import tempfile
from pathlib import Path
import pytest

from stoma.config import StomaConfig
try:
    from stoma.embedder import StomaEmbedder
except (ImportError, ModuleNotFoundError):
    StomaEmbedder = None


@pytest.fixture(scope="session")
def shared_embedder():
    """Shared StomaEmbedder instance across test session to avoid repeated model loading."""
    if StomaEmbedder is None:
        pytest.skip("StomaEmbedder requires torch and sentence_transformers")
    return StomaEmbedder(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def temp_workspace():
    """Create an isolated temporary workspace directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="stoma_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_workspace, monkeypatch):
    """Create a StomaConfig pointing to isolated temporary directories."""
    monkeypatch.setenv("STOMA_ALLOW_TRUSTED_PICKLE", "1")
    storage = temp_workspace / ".stoma"
    storage.mkdir(parents=True, exist_ok=True)
    cfg = StomaConfig(
        allowed_roots=[temp_workspace],
        storage_dir=storage,
        index_path=storage / "index.pkl",
        manifest_path=storage / "manifest.json",
        max_chunk_tokens=220,
        chunk_overlap_tokens=30,
        batch_size=16,
    )
    return cfg
