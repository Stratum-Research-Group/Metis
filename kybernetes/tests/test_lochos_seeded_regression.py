"""Dokime-Style Seeded Regression Scenarios for Lochos Multi-Agent Squad.
Implements the seed contract from OpenRig / Dokime:
Each defect-class scenario pairs a healthy baseline run (GREEN) with a deliberately seeded
regression run (RED / defect caught). A scenario that cannot catch its seeded defect is not verified.
"""

from pathlib import Path
import tempfile
import pytest

from kybernetes.contracts.enums import AgentSemanticState
from kybernetes.lochos.queue import LochosQueue
from kybernetes.lochos.spec import create_starter_rig
from kybernetes.lochos.squad import LochosSquadManager


def test_scenario_baton_lifecycle_pair():
    """
    Scenario: BATON-LIFECYCLE-HANDOFF
    Defect Class: UNAUTHORIZED_BATON_THEFT
    Seed Contract:
      - Healthy: Coder claims task, initiates work, transfers baton to Reviewer. Reviewer claims and finishes.
      - Seeded: Rogue or unauthorized seat attempts to transfer or claim a baton owned by another seat.
        Lochos queue MUST reject the action with PermissionError, catching the defect before state corruption.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        db_path = tmp / "lochos.sqlite"
        queue = LochosQueue(db_path)
        rig = create_starter_rig("squad-alpha")
        squad = LochosSquadManager(rig_spec=rig, workspace_root=tmp, queue=queue)

        # ---------------------------------------------------------
        # 1. Healthy Run (GREEN)
        # ---------------------------------------------------------
        item = squad.dispatch_work(
            mission="m-healthy",
            slice_id="s-01",
            title="Implement feature A",
            target_seat="coder",
        )
        coder_seat = squad.get_seat("coder")
        assert coder_seat is not None
        assert coder_seat.current_state == AgentSemanticState.PLANNING
        assert coder_seat.current_task is not None
        assert coder_seat.current_task.baton_owner == "coder@squad-alpha"

        # Coder completes slice and transfers to reviewer
        transferred = squad.transfer_baton(
            task_id=item.task_id,
            from_seat="coder",
            to_seat="reviewer",
            note="Ready for peer review",
        )
        reviewer_seat = squad.get_seat("reviewer")
        assert reviewer_seat is not None
        assert reviewer_seat.current_task is not None
        assert reviewer_seat.current_task.baton_owner == "reviewer@squad-alpha"
        assert reviewer_seat.current_state == AgentSemanticState.PLANNING

        # Reviewer completes the task
        completed = queue.complete_task(
            task_id=item.task_id,
            seat_id="reviewer@squad-alpha",
            result={"review": "APPROVED"},
        )
        assert completed.status == "completed"

        # ---------------------------------------------------------
        # 2. Seeded Regression Run (RED / Defect Caught)
        # Defect Class: UNAUTHORIZED_BATON_THEFT
        # ---------------------------------------------------------
        task2 = squad.dispatch_work(
            mission="m-seeded",
            slice_id="s-02",
            title="Secure Financial Module",
            target_seat="coder",
        )

        # Seeded mutation: An unauthorized third seat attempts to handoff the baton
        defect_caught = False
        try:
            queue.handoff_baton(
                task_id=task2.task_id,
                from_seat="rogue_agent@squad-alpha",
                to_seat="reviewer@squad-alpha",
                note="Attempting to hijack baton",
            )
        except PermissionError as e:
            defect_caught = True
            assert "does not hold the baton" in str(e)

        assert defect_caught, "Seeded defect UNAUTHORIZED_BATON_THEFT was not caught by LochosQueue!"

        # Verify task state remained intact and owned by coder
        untouched = queue.get_task(task2.task_id)
        assert untouched.baton_owner == "coder@squad-alpha"
        assert untouched.status == "in_progress"


def test_scenario_unclean_restart_attention_pair():
    """
    Scenario: UNCLEAN-RESTART-ATTENTION-FENCE
    Defect Class: BYPASS_UNRESOLVED_CRASH_ATTENTION
    Seed Contract:
      - Healthy: Queue detects ungraceful crash, raises attention_required, blocks until operator resolves.
      - Seeded: Agent tries to bypass attention and complete task directly after crash without resolution.
        Queue MUST reject with PermissionError.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        db_path = tmp / "lochos.sqlite"

        # Phase 1: Task in progress
        q1 = LochosQueue(db_path)
        task = q1.enqueue(
            rig_id="squad-1",
            mission="m-crash",
            slice_id="s-03",
            title="Critical Data Migration",
            assigned_to="coder@squad-1",
        )
        q1.claim_baton("coder@squad-1")
        del q1  # Simulate crash

        # Phase 2: Restart and recover
        q2 = LochosQueue(db_path)
        reconciled = q2.recover_after_crash()
        assert len(reconciled) == 1
        assert reconciled[0].status == "attention_required"

        # ---------------------------------------------------------
        # Seeded Run: Try to complete task while blocked in attention
        # Defect Class: BYPASS_UNRESOLVED_CRASH_ATTENTION
        # ---------------------------------------------------------
        bypass_caught = False
        try:
            q2.complete_task(
                task_id=task.task_id,
                seat_id="coder@squad-1",
                result={"corrupted": True},
            )
        except PermissionError as e:
            bypass_caught = True
            assert "blocked on attention_required" in str(e)

        assert bypass_caught, "Seeded defect BYPASS_UNRESOLVED_CRASH_ATTENTION was not caught!"

        # ---------------------------------------------------------
        # Healthy Path: Operator inspects and resolves attention
        # ---------------------------------------------------------
        attention_items = q2.reconcile_attention()
        assert len(attention_items) == 1
        assert attention_items[0]["reason"] == "unclean_shutdown_detected"

        q2.resolve_attention(task.task_id, "coder@squad-1", resolution="verified_safe")
        resumed = q2.get_task(task.task_id)
        assert resumed.status == "in_progress"

        # Now completion succeeds
        finished = q2.complete_task(
            task_id=task.task_id,
            seat_id="coder@squad-1",
            result={"migration_verified": True},
        )
        assert finished.status == "completed"


def test_scenario_context_pack_and_brief():
    """
    Scenario: CONTEXT-PACK-AND-CONTINUITY-BRIEF
    Verifies that seat initialization creates the .metis/context-pack manifest
    and produces structured continuity and peer handoff briefs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        db_path = tmp / "lochos.sqlite"
        rig = create_starter_rig("squad-beta")
        squad = LochosSquadManager(rig_spec=rig, workspace_root=tmp, queue=LochosQueue(db_path))

        coder = squad.get_seat("coder")
        assert coder is not None
        brief = coder.generate_restore_brief()
        assert brief["id"] == "coder"
        assert brief["role"] == "Primary code implementation, AST navigation, and exact-anchor editing"
        assert "python" in brief["skills"]

        # Check context pack manifest
        manifest_path = tmp / ".metis" / "context-pack" / "manifest.yaml"
        assert manifest_path.exists()
        content = manifest_path.read_text(encoding="utf-8")
        assert "ref: seat:reviewer" in content or "ref: seat:coder" in content

        # Dispatch task and generate handoff packet
        item = squad.dispatch_work(
            mission="m-brief",
            slice_id="s-04",
            title="Refactor Parser",
            target_seat="coder",
        )
        handoff_brief = squad.generate_handoff_brief(
            from_seat="coder",
            to_seat="reviewer",
            task_id=item.task_id,
            note="Ready for peer check",
        )
        assert handoff_brief["task_id"] == item.task_id
        assert handoff_brief["handoff_note"] == "Ready for peer check"
        assert handoff_brief["source_seat"]["id"] == "coder"
        assert handoff_brief["destination_seat"]["id"] == "reviewer"
