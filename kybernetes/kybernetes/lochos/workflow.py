"""Lochos Autonomous Pipeline Orchestrator for Metis Sovereign Studio.
Coordinates end-to-end multi-agent execution pipelines:
1. Baseline Git checkpointing before mutations.
2. Automated task dispatch to the implementer seat (coder).
3. Coder autonomous execution loop.
4. Post-coding milestone checkpointing.
5. Structured baton transfer with continuity handoff brief to the reviewer seat.
6. Reviewer zero-trust verification gate.
7. Automatic sign-off completion on verification pass, or transactional Git rollback on rejection.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

from kybernetes.contracts.enums import AgentSemanticState
from kybernetes.engine.checkpoint import GitMilestoneCheckpointer, MilestoneCheckpoint
from kybernetes.lochos.queue import LochosQueue, WorkItem
from kybernetes.lochos.squad import LochosSquadManager, SeatRuntime


class PipelineStage(str, Enum):
    IDLE = "idle"
    DISPATCHED = "dispatched"
    CODING = "coding"
    MILESTONE_TAKEN = "milestone_taken"
    BATON_TRANSFERRED = "baton_transferred"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    ATTENTION_REQUIRED = "attention_required"


@dataclass
class PipelineResult:
    """Represents the end-to-end outcome of a Lochos automated pipeline run."""
    task_id: str
    success: bool
    stage: PipelineStage
    baton_owner: Optional[str] = None
    milestones: List[str] = field(default_factory=list)
    rolled_back: bool = False
    handoff_brief: Optional[Dict[str, Any]] = None
    verification_passed: bool = False
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "stage": self.stage.value if isinstance(self.stage, PipelineStage) else str(self.stage),
            "baton_owner": self.baton_owner,
            "milestones": list(self.milestones),
            "rolled_back": self.rolled_back,
            "handoff_brief": self.handoff_brief,
            "verification_passed": self.verification_passed,
            "error": self.error,
            "details": dict(self.details),
        }


class LochosWorkflowOrchestrator:
    """
    Coordinates multi-seat autonomous workflows with clean baton handoff and zero-trust verification gates.
    """

    def __init__(
        self,
        squad_manager: Optional[LochosSquadManager] = None,
        workspace_root: Optional[Path] = None,
    ):
        self.squad = squad_manager or LochosSquadManager(workspace_root=workspace_root)
        self.workspace_root = self.squad.workspace_root
        self.checkpointer = self.squad.checkpointer

    def run_pipeline(
        self,
        mission: str,
        slice_id: str,
        title: str,
        description: str = "",
        payload: Optional[Dict[str, Any]] = None,
        coder_seat: str = "coder",
        reviewer_seat: str = "reviewer",
        coder_action_generator: Optional[Callable[[int, List[Any]], Optional[Dict[str, Any]]]] = None,
        coder_runner: Optional[Callable[[SeatRuntime, WorkItem], Any]] = None,
        verifier: Optional[Callable[[], bool]] = None,
        test_command: Optional[str] = None,
        auto_approve: bool = False,
        auto_rollback_on_failure: bool = True,
        escalate_attention_on_failure: bool = False,
        max_steps: int = 15,
    ) -> PipelineResult:
        """
        Executes the full automated multi-agent pipeline:
        Dispatch -> Baseline Checkpoint -> Coder Execution -> Milestone -> Baton Transfer -> Reviewer Gate -> Sign-off / Rollback.
        """
        coder = self.squad.get_seat(coder_seat)
        reviewer = self.squad.get_seat(reviewer_seat)

        if not coder:
            raise ValueError(f"Coder seat '{coder_seat}' not found in squad '{self.squad.spec.name}'")
        if not reviewer:
            raise ValueError(f"Reviewer seat '{reviewer_seat}' not found in squad '{self.squad.spec.name}'")

        milestones: List[str] = []

        # 1. Baseline Git Milestone Checkpoint
        baseline_cp = self.checkpointer.create_checkpoint(f"Pipeline baseline: [{slice_id}] {title}")
        milestones.append(baseline_cp.id)

        # 2. Dispatch task to coder
        task = self.squad.dispatch_work(
            mission=mission,
            slice_id=slice_id,
            title=title,
            description=description,
            target_seat=coder_seat,
            payload=payload,
        )
        self.squad.emit_event("pipeline_started", {
            "task_id": task.task_id,
            "mission": mission,
            "slice_id": slice_id,
            "title": title,
            "coder": coder.full_address,
            "reviewer": reviewer.full_address,
            "baseline_checkpoint": baseline_cp.id,
            "stage": PipelineStage.DISPATCHED.value,
        })

        # 3. Coder Execution Phase
        coder_success = True
        coder_error: Optional[str] = None
        coder_result: Any = None

        if coder.current_state != AgentSemanticState.MUTATING:
            try:
                coder.state_machine.transition(AgentSemanticState.MUTATING, reason="Executing implementation actions")
            except Exception:
                pass

        try:
            if coder_runner:
                coder_result = coder_runner(coder, task)
                if hasattr(coder_result, "success"):
                    coder_success = bool(coder_result.success)
                    if not coder_success:
                        coder_error = getattr(coder_result, "final_message", "Coder runner reported failure")
                elif isinstance(coder_result, bool):
                    coder_success = coder_result
                    if not coder_success:
                        coder_error = "Coder runner returned False"
            elif coder_action_generator:
                res = self.squad.execute_seat_task(
                    seat_id=coder_seat,
                    action_generator=coder_action_generator,
                    enforce_approvals=not auto_approve,
                )
                coder_result = res
                coder_success = res.success
                if not coder_success:
                    coder_error = res.final_message or "Coder action sequence failed"
            else:
                # Default: no mutations requested, consider coder phase completed
                coder_result = {"status": "noop"}
                coder_success = True
        except Exception as exc:
            coder_success = False
            coder_error = f"Exception during coder execution: {exc}"

        if not coder_success:
            # Handle coder execution failure
            rolled_back = False
            if auto_rollback_on_failure:
                self.checkpointer.rollback(baseline_cp)
                rolled_back = True
                self.squad.emit_event("pipeline_rolled_back", {
                    "task_id": task.task_id,
                    "checkpoint": baseline_cp.id,
                    "reason": coder_error,
                })

            if escalate_attention_on_failure:
                self.squad.queue.raise_attention(
                    task_id=task.task_id,
                    seat_id=coder.full_address,
                    reason="coder_execution_failed",
                    details={"error": coder_error},
                )
                stage = PipelineStage.ATTENTION_REQUIRED
            else:
                self.squad.queue.fail_task(
                    task_id=task.task_id,
                    seat_id=coder.full_address,
                    error=coder_error or "Coder execution failed",
                )
                stage = PipelineStage.ROLLED_BACK if rolled_back else PipelineStage.FAILED

            if coder.current_state != AgentSemanticState.IDLE:
                if coder.current_state != AgentSemanticState.FAILED:
                    coder.state_machine.transition(AgentSemanticState.FAILED, reason="Execution failed")
                coder.state_machine.transition(AgentSemanticState.IDLE, reason="Returning to idle after failure")

            self.squad.emit_event("pipeline_failed", {
                "task_id": task.task_id,
                "stage": stage.value,
                "error": coder_error,
            })

            return PipelineResult(
                task_id=task.task_id,
                success=False,
                stage=stage,
                baton_owner=coder.full_address,
                milestones=milestones,
                rolled_back=rolled_back,
                verification_passed=False,
                error=coder_error,
                details={"phase": "coder", "error": coder_error},
            )

        # Coder successfully executed implementation
        post_coding_cp = self.checkpointer.create_checkpoint(f"Post-coding milestone: [{slice_id}] {title}")
        milestones.append(post_coding_cp.id)

        # 4. Baton Transfer to Reviewer with Handoff Brief
        handoff_brief = self.squad.generate_handoff_brief(
            from_seat=coder_seat,
            to_seat=reviewer_seat,
            task_id=task.task_id,
            note="Implementation complete; baton transferred for zero-trust review and verification.",
        )

        transferred = self.squad.transfer_baton(
            task_id=task.task_id,
            from_seat=coder_seat,
            to_seat=reviewer_seat,
            note="Requesting zero-trust Dokime/test verification",
        )

        self.squad.emit_event("pipeline_baton_transferred", {
            "task_id": task.task_id,
            "from_seat": coder.full_address,
            "to_seat": reviewer.full_address,
            "brief": handoff_brief,
            "stage": PipelineStage.BATON_TRANSFERRED.value,
        })

        # 5. Reviewer Zero-Trust Verification Gate
        try:
            reviewer.state_machine.transition(AgentSemanticState.VERIFYING, reason="Executing zero-trust verification gate")
        except Exception:
            pass

        verification_passed = True
        verification_error: Optional[str] = None

        if verifier:
            try:
                verification_passed = bool(verifier())
                if not verification_passed:
                    verification_error = "Independent verifier callback reported failure"
            except Exception as exc:
                verification_passed = False
                verification_error = f"Verifier exception: {exc}"
        elif test_command:
            try:
                target_cwd = reviewer.worktree_path or self.workspace_root
                proc = subprocess.run(
                    test_command,
                    shell=True,
                    cwd=str(target_cwd),
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    verification_passed = False
                    verification_error = (
                        f"Test command failed (code {proc.returncode}):\n"
                        f"{proc.stderr or proc.stdout}"
                    )
            except Exception as exc:
                verification_passed = False
                verification_error = f"Failed to execute verification test command: {exc}"

        # 6. Verdict Sign-Off or Rollback
        if verification_passed:
            self.squad.queue.complete_task(
                task_id=task.task_id,
                seat_id=reviewer.full_address,
                result={
                    "verification": "PASSED",
                    "milestones": milestones,
                    "brief": handoff_brief,
                },
            )
            reviewer.current_task = None
            if reviewer.current_state != AgentSemanticState.IDLE:
                reviewer.state_machine.transition(AgentSemanticState.COMPLETED, reason="Verification gate passed")
                reviewer.state_machine.transition(AgentSemanticState.IDLE, reason="Task completed")

            self.squad.emit_event("pipeline_completed", {
                "task_id": task.task_id,
                "reviewer": reviewer.full_address,
                "milestones": milestones,
                "stage": PipelineStage.COMPLETED.value,
            })

            return PipelineResult(
                task_id=task.task_id,
                success=True,
                stage=PipelineStage.COMPLETED,
                baton_owner=reviewer.full_address,
                milestones=milestones,
                rolled_back=False,
                handoff_brief=handoff_brief,
                verification_passed=True,
                details={"message": "All pipeline verification gates passed."},
            )
        else:
            # Verification failed: zero-trust rejection
            rolled_back = False
            if auto_rollback_on_failure:
                self.checkpointer.rollback(baseline_cp)
                rolled_back = True
                self.squad.emit_event("pipeline_rolled_back", {
                    "task_id": task.task_id,
                    "checkpoint": baseline_cp.id,
                    "rejection_reason": verification_error,
                })

            if escalate_attention_on_failure:
                self.squad.queue.raise_attention(
                    task_id=task.task_id,
                    seat_id=reviewer.full_address,
                    reason="reviewer_verification_failed",
                    details={"error": verification_error, "brief": handoff_brief},
                )
                stage = PipelineStage.ATTENTION_REQUIRED
            else:
                self.squad.queue.fail_task(
                    task_id=task.task_id,
                    seat_id=reviewer.full_address,
                    error=verification_error or "Reviewer rejected changes",
                )
                stage = PipelineStage.ROLLED_BACK if rolled_back else PipelineStage.FAILED

            reviewer.current_task = None
            if reviewer.current_state != AgentSemanticState.IDLE:
                if reviewer.current_state != AgentSemanticState.FAILED:
                    reviewer.state_machine.transition(AgentSemanticState.FAILED, reason="Verification gate rejected")
                reviewer.state_machine.transition(AgentSemanticState.IDLE, reason="Returning to idle after rejection")

            self.squad.emit_event("pipeline_rejected", {
                "task_id": task.task_id,
                "stage": stage.value,
                "error": verification_error,
                "rolled_back": rolled_back,
            })

            return PipelineResult(
                task_id=task.task_id,
                success=False,
                stage=stage,
                baton_owner=reviewer.full_address,
                milestones=milestones,
                rolled_back=rolled_back,
                handoff_brief=handoff_brief,
                verification_passed=False,
                error=verification_error,
                details={"phase": "reviewer", "rejection_reason": verification_error},
            )
