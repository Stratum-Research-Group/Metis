import os
from pathlib import Path
import pytest

@pytest.fixture(autouse=True, scope="function")
def set_phylax_cwd(monkeypatch):
    phylax_dir = Path(__file__).parent.parent.resolve()
    monkeypatch.chdir(phylax_dir)
