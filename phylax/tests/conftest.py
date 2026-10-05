import os
from pathlib import Path
import pytest

@pytest.fixture(autouse=True, scope="session")
def set_phylax_cwd():
    phylax_dir = Path(__file__).parent.parent.resolve()
    old_cwd = os.getcwd()
    os.chdir(phylax_dir)
    yield
    os.chdir(old_cwd)
