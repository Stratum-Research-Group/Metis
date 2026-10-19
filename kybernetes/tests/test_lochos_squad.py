import tempfile
from pathlib import Path
from kybernetes.contracts.enums import AgentSemanticState
from kybernetes.lochos.queue import LochosQueue
from kybernetes.lochos.spec import create_starter_rig
from kybernetes.lochos.squad import LochosSquadManager


def test_squad_manager_dispatch_and_transfer():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        db_path = root / ".metis" / "lochos.sqlite"
        rig = create_starter_rig("metis-test-squad")
        queue = LochosQueue(db_path)

        manager = LochosSquadManager(
            rig_spec=rig,
            workspace_root=root,
            queue=queue,
        )

        seats = manager.list_seats()
        assert len(seats) == 2
        coder = manager.get_seat("coder")
        reviewer = manager.get_seat("reviewer")
        assert coder is not None
        assert reviewer is not None
        assert coder.current_state == AgentSemanticState.IDLE

        # Dispatch work directly to coder
        task = manager.dispatch_work(
            mission="security-hardening",
            slice_id="SEC.001",
            title="Audit Token Consumption",
            target_seat="coder",
        )
        assert task.baton_owner == "coder@metis-test-squad"
        assert task.status == "in_progress"
        assert coder.current_state == AgentSemanticState.PLANNING

        # Coder transitions to MUTATING and finishes work
        coder.state_machine.transition(AgentSemanticState.MUTATING, reason="Writing patch")
        assert coder.current_state == AgentSemanticState.MUTATING

        # Coder passes baton to reviewer
        handed_off = manager.transfer_baton(
            task_id=task.task_id,
            from_seat="coder",
            to_seat="reviewer",
            note="Changes implemented, ready for Dokime verification",
        )
        assert handed_off.baton_owner == "reviewer@metis-test-squad"
        assert handed_off.status == "in_progress"
        assert coder.current_state == AgentSemanticState.IDLE
        assert reviewer.current_state == AgentSemanticState.PLANNING

        # Check status dictionary for UI
        status = manager.to_status_dict()
        assert status["rig_name"] == "metis-test-squad"
        assert len(status["seats"]) == 2
        assert len(status["active_tasks"]) == 1


def test_squad_manager_execute_seat_task_with_worktree_and_rollback():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        import subprocess
        subprocess.run(["git", "init"], cwd=str(root), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(root), check=True)
        subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(root), check=True)
        (root / "base.txt").write_text("initial base\n")
        subprocess.run(["git", "add", "base.txt"], cwd=str(root), check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(root), check=True)

        rig = create_starter_rig("metis-git-squad")
        manager = LochosSquadManager(
            rig_spec=rig,
            workspace_root=root,
        )

        coder = manager.get_seat("coder")
        assert coder is not None
        assert coder.worktree_path is not None
        assert coder.worktree_path.exists()

        # Dispatch task to coder
        manager.dispatch_work(
            mission="feature",
            slice_id="F.1",
            title="Implement feature",
            target_seat="coder",
        )

        # 1. Test failing execution rolls back in seat worktree
        def failing_action_generator(step, hist):
            if step == 1:
                return {
                    "tool_name": "write_file",
                    "arguments": {"path": "broken.py", "content": "broken code\n"},
                }
            return None

        def failing_verifier():
            return False

        res = manager.execute_seat_task(
            seat_id="coder",
            action_generator=failing_action_generator,
            verifier=failing_verifier,
        )
        assert res.success is False
        assert not (coder.worktree_path / "broken.py").exists()

        # 2. Test successful execution succeeds and persists in seat worktree
        manager.dispatch_work(
            mission="feature",
            slice_id="F.2",
            title="Implement verified feature",
            target_seat="coder",
        )

        def succeeding_action_generator(step, hist):
            if step == 1:
                return {
                    "tool_name": "write_file",
                    "arguments": {"path": "verified.py", "content": "print('ok')\n"},
                }
            return None

        res2 = manager.execute_seat_task(
            seat_id="coder",
            action_generator=succeeding_action_generator,
            verifier=lambda: True,
        )
        assert res2.success is True
        assert (coder.worktree_path / "verified.py").exists()


def test_squad_manager_subagent_delegation():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        import subprocess
        import time
        subprocess.run(["git", "init"], cwd=str(root), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Metis Test"], cwd=str(root), check=True)
        subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(root), check=True)
        (root / "base.txt").write_text("initial base\n")
        subprocess.run(["git", "add", "base.txt"], cwd=str(root), check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(root), check=True)

        rig = create_starter_rig("metis-subagent-squad")
        manager = LochosSquadManager(
            rig_spec=rig,
            workspace_root=root,
        )

        coder = manager.get_seat("coder")
        coder_dir = coder.worktree_path or root

        def sub_action(worktree_dir: Path):
            (worktree_dir / "sub_feat.py").write_text("subagent_result = 42\n")
            subprocess.run(["git", "add", "sub_feat.py"], cwd=str(worktree_dir), check=True)
            subprocess.run(["git", "commit", "-m", "Subagent commit"], cwd=str(worktree_dir), check=True)
            return {"success": True, "output": "subagent done"}

        subtask = manager.spawn_subagent(
            seat_id="coder",
            subagent_id="sub-1",
            description="Explore subtask",
            action_fn=sub_action,
            isolated=True,
        )

        for _ in range(50):
            if subtask.state_machine.is_terminal():
                break
            time.sleep(0.05)

        assert subtask.success is True
        assert not (coder_dir / "sub_feat.py").exists()

        merged = manager.merge_subagent("coder", "sub-1")
        assert merged is True
        assert (coder_dir / "sub_feat.py").exists()

