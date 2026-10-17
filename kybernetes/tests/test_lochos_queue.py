import tempfile
from pathlib import Path
import pytest
from kybernetes.lochos.queue import LochosQueue


def test_queue_enqueue_and_claim():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "lochos.sqlite"
        queue = LochosQueue(db_path)

        task = queue.enqueue(
            rig_id="squad-1",
            mission="release-0.5.8",
            slice_id="OPR.0.5.8.14",
            title="Implement Lochos Queue",
            description="Build SQLite durable queue",
            payload={"files": ["queue.py"]},
            assigned_to="coder@squad-1",
        )

        assert task.status == "queued"
        assert task.baton_owner == "coder@squad-1"

        # Claim baton
        claimed = queue.claim_baton("coder@squad-1")
        assert claimed is not None
        assert claimed.task_id == task.task_id
        assert claimed.status == "in_progress"
        assert claimed.baton_owner == "coder@squad-1"

        # Active seat work lookup
        current = queue.get_seat_current_work("coder@squad-1")
        assert current is not None
        assert current.task_id == task.task_id


def test_queue_baton_handoff():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "lochos.sqlite"
        queue = LochosQueue(db_path)

        task = queue.enqueue(
            rig_id="squad-1",
            mission="m1",
            slice_id="s1",
            title="Code and Audit",
            assigned_to="coder@squad-1",
        )
        queue.claim_baton("coder@squad-1")

        # Transfer to reviewer
        transferred = queue.handoff_baton(
            task_id=task.task_id,
            from_seat="coder@squad-1",
            to_seat="reviewer@squad-1",
            note="Ready for audit",
        )
        assert transferred.baton_owner == "reviewer@squad-1"
        assert transferred.status == "queued"

        # Reviewer claims
        claimed = queue.claim_baton("reviewer@squad-1")
        assert claimed is not None
        assert claimed.baton_owner == "reviewer@squad-1"
        assert claimed.status == "in_progress"


def test_queue_baton_survives_restart():
    """
    OpenRig installed-scenario equivalent:
    Proves that unfinished work and baton ownership survive a simulated crash/restart.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "lochos.sqlite"
        
        # Phase 1: Instance 1 starts, enqueues and claims task
        queue1 = LochosQueue(db_path)
        task = queue1.enqueue(
            rig_id="squad-1",
            mission="m-restart",
            slice_id="s-restart",
            title="Critical Mission Task",
            assigned_to="coder@squad-1",
        )
        queue1.claim_baton("coder@squad-1")
        
        # Abrupt crash of process/queue1 (simulated by dropping queue1)
        del queue1

        # Phase 2: Kybernetes restarts, instantiates new LochosQueue from same SQLite DB
        queue2 = LochosQueue(db_path)
        
        # Verify task is still present
        recovered_task = queue2.get_task(task.task_id)
        assert recovered_task is not None
        assert recovered_task.task_id == task.task_id
        assert recovered_task.baton_owner == "coder@squad-1"
        assert recovered_task.status == "in_progress"

        # Run crash recovery protocol
        reconciled = queue2.recover_after_crash()
        assert len(reconciled) == 1
        assert reconciled[0].task_id == task.task_id
        assert reconciled[0].status == "attention_required"

        # Verify attention event was recorded
        attention_list = queue2.reconcile_attention()
        assert len(attention_list) == 1
        assert attention_list[0]["reason"] == "unclean_shutdown_detected"
        assert attention_list[0]["task_id"] == task.task_id

        # Human operator resolves attention
        queue2.resolve_attention(task.task_id, "coder@squad-1", resolution="resumed_safe")
        resumed = queue2.get_task(task.task_id)
        assert resumed.status == "in_progress"

        # Complete task
        queue2.complete_task(task.task_id, "coder@squad-1", result={"tests_passed": True})
        completed = queue2.get_task(task.task_id)
        assert completed.status == "completed"
        assert completed.result == {"tests_passed": True}
