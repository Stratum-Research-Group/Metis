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


def test_bubblewrap_network_isolation(tmp_path: Path):
    """Verify bubblewrap isolates the network namespace by default unless explicitly granted."""
    bwrap = BubblewrapBackend()
    if not bwrap.is_available():
        pytest.skip("Bubblewrap (bwrap) not available in this test environment")

    # 1. By default, network is unshared: socket connect fails
    script = (
        "python3 -c \""
        "import socket, sys; "
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); "
        "s.settimeout(1.0); "
        "s.connect(('127.0.0.1', 8080)); "
        "print('CONNECTED')\""
    )
    res_isolated = bwrap.execute(
        script,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        allow_network=False,
    )
    # Network is unshared, connection refused / unreachable
    assert res_isolated.exit_code != 0
    assert "CONNECTED" not in res_isolated.stdout

    # 2. When allow_network=True, if host service (Hestia on 8080) is listening, connection succeeds
    res_allowed = bwrap.execute(
        script,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        allow_network=True,
    )
    # Hestia is running on port 8080 on the host
    assert res_allowed.exit_code == 0
    assert "CONNECTED" in res_allowed.stdout


def test_bubblewrap_confinement_env(tmp_path: Path):
    """Verify bubblewrap enforces HOME pointing into workspace and isolated TMPDIR."""
    bwrap = BubblewrapBackend()
    if not bwrap.is_available():
        pytest.skip("Bubblewrap (bwrap) not available in this test environment")

    script = "python3 -c \"import os; print(f'HOME={os.environ.get(\\\"HOME\\\")} TMPDIR={os.environ.get(\\\"TMPDIR\\\")}')\""
    res = bwrap.execute(
        script,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert res.exit_code == 0
    assert f"HOME={str(tmp_path)}" in res.stdout
    assert "TMPDIR=/tmp" in res.stdout


def test_get_default_execution_backend_explicit_bwrap_fail_closed(monkeypatch):
    """Verify Finding 3: If bwrap is explicitly requested but unavailable, fail closed."""
    from kybernetes.execution.backend import BubblewrapBackend, get_default_execution_backend

    monkeypatch.setattr(BubblewrapBackend, "is_available", lambda self: False)

    with pytest.raises(RuntimeError, match="Bubblewrap backend explicitly requested, but 'bwrap' is not available"):
        get_default_execution_backend("bwrap")


def test_get_default_execution_backend_fallback_warns(monkeypatch):
    """Verify Finding 3: Auto-detect fallback emits warning when bwrap unavailable."""
    import warnings
    from kybernetes.execution.backend import BubblewrapBackend, LocalProcessBackend, get_default_execution_backend

    monkeypatch.setattr(BubblewrapBackend, "is_available", lambda self: False)

    with pytest.warns(UserWarning, match="Bubblewrap \\('bwrap'\\) is not available"):
        backend = get_default_execution_backend()
        assert isinstance(backend, LocalProcessBackend)
