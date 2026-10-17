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
