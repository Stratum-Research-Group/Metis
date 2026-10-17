import os
from pathlib import Path
import pytest

@pytest.fixture(autouse=True, scope="function")
def set_krisis_cwd(monkeypatch):
    krisis_dir = Path(__file__).parent.parent.resolve()
    monkeypatch.chdir(krisis_dir)
