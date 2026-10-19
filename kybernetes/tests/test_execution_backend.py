"""Tests for sandboxed ExecutionBackend implementations (Bubblewrap and LocalProcess)."""

from pathlib import Path
import pytest

from kybernetes.execution.backend import (
    BubblewrapBackend,
    ExecutionResult,
    LocalProcessBackend,
    get_default_execution_backend,
)
from kybernetes.tools.sandboxed_bash import SafeSandboxedBash


def test_local_process_backend(tmp_path: Path):
    backend = LocalProcessBackend()
    res = backend.execute("echo 'local-ok'", cwd=tmp_path, env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert res.exit_code == 0
    assert "local-ok" in res.stdout
    assert res.backend == "local_process"


def test_bubblewrap_backend_workspace_isolation(tmp_path: Path):
    bwrap = BubblewrapBackend()
    if not bwrap.is_available():
        pytest.skip("Bubblewrap (bwrap) not available in this test environment")

    # 1. Writes within workspace cwd succeed
    res_ws = bwrap.execute(
        "echo 'payload' > file.txt && cat file.txt",
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert res_ws.exit_code == 0
    assert "payload" in res_ws.stdout
    assert (tmp_path / "file.txt").read_text().strip() == "payload"

    # 2. Writes outside workspace to root fail with Read-only file system
    res_ro = bwrap.execute(
        "touch /etc/metis_test_rogue_file.txt",
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert res_ro.exit_code != 0
    assert "Read-only file system" in res_ro.stderr or "Permission denied" in res_ro.stderr


def test_sandboxed_bash_home_isolation(tmp_path: Path):
    """Verify $HOME in sandboxed bash points into the workspace rather than leaking host user home."""
    runner = SafeSandboxedBash(allowed_roots=[tmp_path])
    res = runner.execute("echo $HOME", cwd=str(tmp_path))
    assert res["exit_code"] == 0
    home_val = res["stdout"].strip()
    assert home_val == str(tmp_path)
    assert home_val != "/home/akarta"


def test_sandboxed_bash_tmp_ephemeral_isolation(tmp_path: Path):
    """Verify /tmp operations run in isolated tmpfs and do not persist to host /tmp."""
    runner = SafeSandboxedBash(allowed_roots=[tmp_path])
    res = runner.execute("touch /tmp/metis_sandbox_isolated.flag && ls /tmp/metis_sandbox_isolated.flag", cwd=str(tmp_path))
    assert res["exit_code"] == 0
    assert "metis_sandbox_isolated.flag" in res["stdout"]

    # Verify flag file is NOT left behind on the actual host root /tmp
    assert not Path("/tmp/metis_sandbox_isolated.flag").exists()
