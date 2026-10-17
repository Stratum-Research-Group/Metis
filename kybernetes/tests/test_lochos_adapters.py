"""Unit tests for Lochos 5-method RuntimeAdapter contract."""

from pathlib import Path
import tempfile
import pytest

from kybernetes.lochos.adapters import (
    KybernetesCodingAdapter,
    StubRuntimeAdapter,
    TerminalSeatAdapter,
)
from kybernetes.lochos.queue import WorkItem
from kybernetes.lochos.spec import SeatSpec


def test_kybernetes_coding_adapter():
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path(tmpdir)
        seat = SeatSpec(
            id="coder",
            label="Coder",
            role="Autonomous Coder",
            skills=["python", "refactoring"],
            worktree_branch="squad/coder",
        )
        adapter = KybernetesCodingAdapter()

        # 1. Project
        skills_root = cwd / "skills_source"
        skill_dir = skills_root / "python" / "refactoring"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("# Python Refactoring", encoding="utf-8")

        proj = adapter.project(seat, cwd, skills_root=skills_root)
        assert Path(proj["manifest"]).exists()
        assert "refactoring" in proj["projected_skills"]

        # 2. List installed
        installed = adapter.list_installed(seat, cwd)
        assert len(installed) == 1
        assert installed[0]["name"] == "refactoring"

        # 3. Deliver startup
        startup = adapter.deliver_startup(seat, cwd, instructions="Focus on security")
        assert startup["delivered"] == 1
        assert Path(startup["path"]).exists()

        # 4. Check ready
        ready = adapter.check_ready(seat, cwd)
        assert ready.ready is True
        assert ready.attention_required is False


def test_terminal_seat_adapter():
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path(tmpdir)
        seat = SeatSpec(id="terminal", label="Terminal", role="Terminal Runner")
        adapter = TerminalSeatAdapter()

        ready = adapter.check_ready(seat, cwd)
        assert ready.ready is True

        task = WorkItem(
            task_id="t-cmd",
            rig_id="r1",
            mission="m1",
            slice_id="s1",
            title="Run Echo",
            description="",
            payload={"command": "echo 'LOCHOS_TERMINAL_READY'"},
            status="in_progress",
            baton_owner="terminal@r1",
            created_at=0.0,
            updated_at=0.0,
        )
        res = adapter.launch_harness(seat, task, cwd)
        assert res.ok is True
        assert "LOCHOS_TERMINAL_READY" in (res.evidence or "")


def test_stub_runtime_adapter():
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = Path(tmpdir)
        seat = SeatSpec(id="stub-seat", label="Stub", role="Stub Agent", skills=["mock-skill"])

        # 1. Normal run
        adapter = StubRuntimeAdapter()
        ready = adapter.check_ready(seat, cwd)
        assert ready.ready is True
        assert ready.attention_required is False

        task = WorkItem(
            task_id="t-stub",
            rig_id="r1",
            mission="m1",
            slice_id="s1",
            title="Stub Task",
            description="",
            payload={},
            status="in_progress",
            baton_owner="stub-seat@r1",
            created_at=0.0,
            updated_at=0.0,
        )
        res = adapter.launch_harness(seat, task, cwd)
        assert res.ok is True
        assert adapter.launch_count == 1

        # 2. Blocked on attention run
        blocked_adapter = StubRuntimeAdapter(script={"attention_required": True})
        ready_blocked = blocked_adapter.check_ready(seat, cwd)
        assert ready_blocked.ready is False
        assert ready_blocked.attention_required is True
        assert ready_blocked.code == "trust_gate"

        res_blocked = blocked_adapter.launch_harness(seat, task, cwd)
        assert res_blocked.ok is False
        assert "trust_gate" in (res_blocked.evidence or "")
