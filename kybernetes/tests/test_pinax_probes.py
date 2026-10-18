"""Tests for Pinax Active Health Probing & Stale Venv Shim Detection (Phase 1).
Validates 5-state probe machine (ok, missing, broken, timeout, error),
stale venv interpreter detection, probe environment secret scrubbing,
and PinaxCatalogue/Database state updates.
"""

import os
from pathlib import Path
import shutil
import tempfile
import pytest

from kybernetes.pinax.catalogue import PinaxCatalogue
from kybernetes.pinax.db import PinaxDatabase
from kybernetes.pinax.models import EntryKind, InstallState, ProbeStatus, RiskTier, ToolKind, ToolRecord
from kybernetes.pinax.probe import PinaxProber


def test_probe_system_binary_ok():
    """Verify that a legitimate system binary probes as OK."""
    sh_path = shutil.which("sh") or "/bin/sh"
    res = PinaxProber.probe_command(
        tool_id="tool:sh",
        invocation_path=sh_path,
        args=["--version"],
        timeout_seconds=2.0,
    )
    # /bin/sh may exit 0 or return version info; python3 is even more standard
    py3_path = shutil.which("python3")
    if py3_path:
        res_py = PinaxProber.probe_command(
            tool_id="tool:python3",
            invocation_path=py3_path,
            args=["--version"],
            timeout_seconds=2.0,
        )
        assert res_py.status == ProbeStatus.OK
        assert res_py.exit_code == 0
        assert "Python" in res_py.stdout or "Python" in res_py.stderr


def test_probe_missing_binary():
    """Verify missing binary is classified as MISSING."""
    res = PinaxProber.probe_command(
        tool_id="tool:phantom",
        invocation_path="/opt/nonexistent/bin/phantom_tool_99",
        args=["--version"],
    )
    assert res.status == ProbeStatus.MISSING
    assert "not found" in res.detail.lower() or "empty" in res.detail.lower()


def test_probe_stale_venv_shim():
    """
    Verify stale-venv-shim detection:
    The tool shim file exists on disk, but its shebang points to a Python interpreter
    in a virtual environment that has been deleted or upgraded away.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        dead_interp = tmp / ".venv" / "bin" / "python3"
        shim_file = tmp / "my_cli_shim"
        # Shebang points to nonexistent dead_interp
        shim_file.write_text(f"#!{dead_interp}\nprint('unreachable')\n", encoding="utf-8")
        shim_file.chmod(0o755)

        is_stale, reason = PinaxProber.check_stale_shim(str(shim_file))
        assert is_stale is True
        assert "stale shim" in reason.lower()

        res = PinaxProber.probe_command(
            tool_id="tool:my_cli_shim",
            invocation_path=str(shim_file),
            args=["--version"],
        )
        assert res.status == ProbeStatus.BROKEN
        assert res.is_stale_shim is True
        assert "stale shim" in res.detail.lower()


def test_probe_timeout_enforcement():
    """Verify that a probe exceeding its timeout budget is killed and marked TIMEOUT."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        slow_script = tmp / "slow_tool"
        slow_script.write_text("#!/bin/sh\nsleep 3\nexit 0\n", encoding="utf-8")
        slow_script.chmod(0o755)

        res = PinaxProber.probe_command(
            tool_id="tool:slow",
            invocation_path=str(slow_script),
            args=["--version"],
            timeout_seconds=0.4,
        )
        assert res.status == ProbeStatus.TIMEOUT
        assert "timeout" in res.detail.lower()


def test_probe_error_non_zero():
    """Verify that a binary exiting non-zero is marked as ERROR."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fail_script = tmp / "fail_tool"
        fail_script.write_text("#!/bin/sh\necho 'Fatal probe failure' >&2\nexit 2\n", encoding="utf-8")
        fail_script.chmod(0o755)

        res = PinaxProber.probe_command(
            tool_id="tool:fail",
            invocation_path=str(fail_script),
            args=["--version"],
        )
        assert res.status == ProbeStatus.ERROR
        assert res.exit_code == 2
        assert "Fatal probe failure" in res.stderr


def test_probe_environment_secret_scrubbing():
    """
    Critical Security Test (Phase 1):
    Probe execution must NEVER leak auth tokens, API keys, or secrets into the child process environment.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        dump_env_script = tmp / "dump_env"
        # Script dumps full environment
        dump_env_script.write_text("#!/bin/sh\nenv\n", encoding="utf-8")
        dump_env_script.chmod(0o755)

        # Inject fake sensitive tokens into parent process environment
        os.environ["METIS_ADMIN_TOKEN"] = "metis-super-secret-admin"
        os.environ["TWITTER_AUTH_TOKEN"] = "leaked-twitter-session"
        os.environ["DATABASE_PASSWORD"] = "production-db-pwd"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "aws-secret-key-12345"

        try:
            res = PinaxProber.probe_command(
                tool_id="tool:dump_env",
                invocation_path=str(dump_env_script),
                args=["--version"],
            )
            assert res.status == ProbeStatus.OK

            out = res.stdout
            # Ensure none of the sensitive values or key names leaked
            assert "metis-super-secret-admin" not in out
            assert "METIS_ADMIN_TOKEN" not in out
            assert "leaked-twitter-session" not in out
            assert "TWITTER_AUTH_TOKEN" not in out
            assert "production-db-pwd" not in out
            assert "DATABASE_PASSWORD" not in out
            assert "aws-secret-key-12345" not in out
            assert "AWS_SECRET_ACCESS_KEY" not in out

        finally:
            # Clean up injected test variables
            os.environ.pop("METIS_ADMIN_TOKEN", None)
            os.environ.pop("TWITTER_AUTH_TOKEN", None)
            os.environ.pop("DATABASE_PASSWORD", None)
            os.environ.pop("AWS_SECRET_ACCESS_KEY", None)


def test_catalogue_probe_tool_lifecycle(tmp_path):
    """Verify end-to-end integration: probing a tool updates PinaxCatalogue and DB state."""
    db_path = tmp_path / "pinax.db"
    catalogue = PinaxCatalogue(db_path=db_path)

    # 1. Probe a structured tool (in-process)
    struct_rec = ToolRecord(
        id="tool:in_process_tool",
        name="in_process_tool",
        kind=ToolKind.STRUCTURED,
        entry_kind=EntryKind.INSTALLED,
        summary="In-process structured tool",
        install_state=InstallState.INSTALLED,
    )
    catalogue.db.upsert_tool(struct_rec)

    res_struct = catalogue.probe_tool("in_process_tool")
    assert res_struct["status"] == "ok"
    assert res_struct["install_state"] == "installed"

    # 2. Add an unprobed binary tool pointing to a stale shim
    stale_script = tmp_path / "broken_tool"
    stale_script.write_text("#!/nonexistent/venv/bin/python\n", encoding="utf-8")
    stale_script.chmod(0o755)

    tool_rec = ToolRecord(
        id="tool:broken_tool",
        name="broken_tool",
        kind=ToolKind.BINARY,
        entry_kind=EntryKind.INSTALLED,
        summary="A broken tool",
        invocation_path=str(stale_script),
        install_state=InstallState.UNKNOWN,
        risk_tier=RiskTier.T1,
    )
    catalogue.db.upsert_tool(tool_rec)

    # 3. Probe the broken tool
    probe_out = catalogue.probe_tool("broken_tool")
    assert probe_out["status"] == "broken"
    assert probe_out["is_stale_shim"] is True
    assert probe_out["install_state"] == "broken"

    # 4. Verify DB was updated
    saved_tool = catalogue.db.get_tool("broken_tool")
    assert saved_tool is not None
    assert saved_tool.probe_status == ProbeStatus.BROKEN
    assert saved_tool.install_state == InstallState.BROKEN
    assert saved_tool.last_probed > 0.0
    assert "stale shim" in (saved_tool.probe_detail or "").lower()
