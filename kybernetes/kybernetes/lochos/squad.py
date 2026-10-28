"""Lochos Squad Coordinator for Metis Sovereign Studio.
Manages multi-seat teams, git worktree assignments, state machines, and task baton flows.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional

import yaml
from kybernetes.contracts.enums import AgentSemanticState
from kybernetes.engine.checkpoint import GitMilestoneCheckpointer
from kybernetes.engine.state import AgentStateMachine
from kybernetes.lochos.queue import LochosQueue, WorkItem
from kybernetes.lochos.spec import RigSpec, SeatSpec, create_starter_rig


@dataclass
class SeatRuntime:
    """Runtime instance for an addressed specialist seat in the squad."""
    spec: SeatSpec
    rig_name: str
    state_machine: AgentStateMachine = field(default_factory=AgentStateMachine)
    worktree_path: Optional[Path] = None
    current_task: Optional[WorkItem] = None

    @property
    def full_address(self) -> str:
        return self.spec.full_address(self.rig_name)

    @property
    def current_state(self) -> AgentSemanticState:
        return self.state_machine.current_state

    def generate_restore_brief(self) -> Dict[str, Any]:
        """
        Generates a structured handover/restore brief for continuity across restarts or peer handoffs.
        """
        return {
            "address": self.full_address,
            "id": self.spec.id,
            "role": self.spec.role,
            "model": self.spec.model,
            "current_state": self.current_state.value,
            "worktree_path": str(self.worktree_path) if self.worktree_path else None,
            "worktree_branch": self.spec.worktree_branch,
            "task_id": self.current_task.task_id if self.current_task else None,
            "task_title": self.current_task.title if self.current_task else None,
            "task_status": self.current_task.status if self.current_task else None,
            "skills": list(self.spec.skills),
            "permission_policy": self.spec.permission_policy,
        }

    def project_context_pack(self, skills_root: Optional[Path] = None) -> Path:
        """
        Projects seat context pack and skills manifest into seat workspace.
        Enables instant skill retrieval and runtime isolation without global leakage.
        """
        target_root = self.worktree_path or Path.cwd()
        pack_dir = target_root / ".metis" / "context-pack"
        pack_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = pack_dir / "manifest.yaml"
        manifest_data = {
            "ref": f"seat:{self.spec.id}",
            "address": self.full_address,
            "role": self.spec.role,
            "model": self.spec.model,
            "permission_policy": self.spec.permission_policy,
            "skills": list(self.spec.skills),
            "worktree_branch": self.spec.worktree_branch,
        }
        manifest_file.write_text(yaml.dump(manifest_data, sort_keys=False), encoding="utf-8")

        if skills_root and skills_root.exists():
            agents_skills_dir = target_root / ".agents" / "skills"
            agents_skills_dir.mkdir(parents=True, exist_ok=True)
            for skill_name in self.spec.skills:
                for match in skills_root.rglob(f"*{skill_name}*"):
                    if match.is_dir() and (match / "SKILL.md").exists():
                        dest = agents_skills_dir / match.name
                        dest.mkdir(parents=True, exist_ok=True)
                        (dest / "SKILL.md").write_text((match / "SKILL.md").read_text(encoding="utf-8"), encoding="utf-8")
        return manifest_file

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.full_address,
            "id": self.spec.id,
            "label": self.spec.label,
            "role": self.spec.role,
            "model": self.spec.model,
            "state": self.current_state.value,
            "worktree_path": str(self.worktree_path) if self.worktree_path else None,
            "worktree_branch": self.spec.worktree_branch,
            "current_task": self.current_task.to_dict() if self.current_task else None,
            "permission_policy": self.spec.permission_policy,
            "skills": self.spec.skills,
            "restore_brief": self.generate_restore_brief(),
        }


class LochosSquadManager:
    """
    Coordinates multi-seat agent squads in Metis Sovereign Studio.
    Provides direct Python execution, isolated git worktrees, and durable SQLite queue tracking.
    """

    def __init__(
        self,
        rig_spec: Optional[RigSpec] = None,
        workspace_root: Optional[Path] = None,
        queue: Optional[LochosQueue] = None,
        checkpointer: Optional[GitMilestoneCheckpointer] = None,
    ):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.spec = rig_spec or create_starter_rig()
        self.queue = queue or LochosQueue(self.workspace_root / ".metis" / "lochos.sqlite")
        self.checkpointer = checkpointer or GitMilestoneCheckpointer(self.workspace_root)
        self.seats: Dict[str, SeatRuntime] = {}
        self._subagent_managers: Dict[str, Any] = {}
        self._event_listeners: List[Callable[[Dict[str, Any]], Any]] = []
        self._init_seats()

    def add_event_listener(self, listener: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a callback or async coroutine to receive Lochos squad events."""
        if listener not in self._event_listeners:
            self._event_listeners.append(listener)

    def remove_event_listener(self, listener: Callable[[Dict[str, Any]], Any]) -> None:
        """Unregister an event listener."""
        if listener in self._event_listeners:
            self._event_listeners.remove(listener)

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcasts a Lochos squad event to all registered listeners."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        for listener in list(self._event_listeners):
            try:
                res = listener(event)
                if asyncio.iscoroutine(res):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(res)
                    except RuntimeError:
                        pass
            except Exception:
                pass

    def _init_seats(self) -> None:
        """Initializes runtimes for all seats defined in the RigSpec."""
        skills_root = self.workspace_root / ".metis" / "skills"
        for seat_spec in self.spec.list_seats():
            addr = seat_spec.full_address(self.spec.name)
            worktree_path: Optional[Path] = None

            if seat_spec.worktree_branch and self.checkpointer.is_git_repo():
                worktree_name = f"squad-{seat_spec.id}"
                try:
                    worktree_path = self.checkpointer.create_worktree_sandbox(worktree_name)
                except Exception:
                    worktree_path = self.workspace_root
            else:
                worktree_path = self.workspace_root

            runtime = SeatRuntime(
                spec=seat_spec,
                rig_name=self.spec.name,
                worktree_path=worktree_path,
            )
            runtime.project_context_pack(skills_root)
            self.seats[seat_spec.id] = runtime
            self.seats[addr] = runtime

    def get_seat(self, seat_id_or_address: str) -> Optional[SeatRuntime]:
        clean = seat_id_or_address.split("@")[0].strip()
        return self.seats.get(clean) or self.seats.get(seat_id_or_address)

    def list_seats(self) -> List[SeatRuntime]:
        seen = set()
        unique = []
        for s in self.seats.values():
            if s.spec.id not in seen:
                seen.add(s.spec.id)
                unique.append(s)
        return unique

    def dispatch_work(
        self,
        mission: str,
        slice_id: str,
        title: str,
        description: str = "",
        target_seat: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> WorkItem:
        """Enqueues and immediately claims a task for a target seat if specified."""
        target_addr = None
        if target_seat:
            seat = self.get_seat(target_seat)
            if not seat:
                raise ValueError(f"Target seat '{target_seat}' not found in squad '{self.spec.name}'")
            target_addr = seat.full_address

        item = self.queue.enqueue(
            rig_id=self.spec.name,
            mission=mission,
            slice_id=slice_id,
            title=title,
            description=description,
            payload=payload,
            assigned_to=target_addr,
        )

        result_item = item
        if target_addr:
            claimed = self.queue.claim_baton(target_addr, item.task_id)
            if claimed:
                seat = self.get_seat(target_seat)  # type: ignore
                seat.current_task = claimed
                seat.state_machine.transition(AgentSemanticState.PLANNING, reason="Task claimed by seat")
                result_item = claimed

        self.emit_event("task_dispatched", result_item.to_dict())
        return result_item

    def transfer_baton(
        self,
        task_id: str,
        from_seat: str,
        to_seat: str,
        note: Optional[str] = None,
    ) -> WorkItem:
        """Passes the work baton from one seat to another."""
        source = self.get_seat(from_seat)
        dest = self.get_seat(to_seat)
        if not source or not dest:
            raise ValueError(f"Invalid source or destination seat: {from_seat} -> {to_seat}")

        transferred = self.queue.handoff_baton(
            task_id=task_id,
            from_seat=source.full_address,
            to_seat=dest.full_address,
            note=note,
        )

        source.current_task = None
        if source.current_state in (AgentSemanticState.MUTATING, AgentSemanticState.VERIFYING):
            source.state_machine.transition(AgentSemanticState.COMPLETED, reason="Slice work completed")
        if source.current_state != AgentSemanticState.IDLE:
            source.state_machine.transition(AgentSemanticState.IDLE, reason="Baton handed off to peer")

        result_item = transferred
        # Dest claims the transferred baton
        claimed = self.queue.claim_baton(dest.full_address, task_id)
        if claimed:
            dest.current_task = claimed
            dest.state_machine.transition(AgentSemanticState.PLANNING, reason="Baton received")
            result_item = claimed

        self.emit_event("baton_transferred", {
            "task_id": task_id,
            "from_seat": source.full_address,
            "to_seat": dest.full_address,
            "task": result_item.to_dict(),
        })
        return result_item

    def generate_handoff_brief(
        self,
        from_seat: str,
        to_seat: str,
        task_id: str,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates a continuity handoff packet between two squad seats for audit and peer review.
        """
        src = self.get_seat(from_seat)
        dst = self.get_seat(to_seat)
        task = self.queue.get_task(task_id)
        if not task:
            raise ValueError(f"Task '{task_id}' not found")
        return {
            "task_id": task.task_id,
            "mission": task.mission,
            "slice_id": task.slice_id,
            "title": task.title,
            "description": task.description,
            "payload": task.payload,
            "source_seat": src.generate_restore_brief() if src else None,
            "destination_seat": dst.generate_restore_brief() if dst else None,
            "handoff_note": note or "",
            "continuity_policy": "peer_verified_handoff",
        }

    def get_attention_summary(self) -> List[Dict[str, Any]]:
        """Returns all open attention requests across all seats in the squad."""
        return self.queue.reconcile_attention()

    def get_subagent_manager(self, seat_id: str) -> Any:
        """Returns or creates a SubagentManager scoped to the seat's working root."""
        from kybernetes.engine.subagent import SubagentManager

        seat = self.get_seat(seat_id)
        if not seat:
            raise ValueError(f"Seat '{seat_id}' not found")
        target_root = seat.worktree_path or self.workspace_root
        key = str(target_root)
        if key not in self._subagent_managers:
            self._subagent_managers[key] = SubagentManager(
                workspace_root=target_root,
                checkpointer=GitMilestoneCheckpointer(target_root),
            )
        return self._subagent_managers[key]

    def spawn_subagent(
        self,
        seat_id: str,
        subagent_id: str,
        description: str,
        action_fn: Callable[[Path], Dict[str, Any]],
        isolated: bool = True,
    ) -> Any:
        """
        Spawns an isolated background subagent task delegated from a specific seat.
        Uses SubagentManager to run in a dedicated git worktree sandbox.
        """
        seat = self.get_seat(seat_id)
        if not seat:
            raise ValueError(f"Seat '{seat_id}' not found")
        mgr = self.get_subagent_manager(seat_id)
        task = mgr.fork_subagent(
            subagent_id=subagent_id,
            description=description,
            action_fn=action_fn,
            isolated=isolated,
        )
        self.emit_event("subagent_spawned", {
            "seat_id": seat.full_address,
            "subagent_id": subagent_id,
            "description": description,
            "isolated": isolated,
        })
        return task

    def merge_subagent(self, seat_id: str, subagent_id: str) -> bool:
        """Merges a completed subagent branch into the seat's branch and prunes worktree."""
        seat = self.get_seat(seat_id)
        if not seat:
            raise ValueError(f"Seat '{seat_id}' not found")
        mgr = self.get_subagent_manager(seat_id)
        merged = mgr.merge_subagent(subagent_id)
        self.emit_event("subagent_merged", {
            "seat_id": seat.full_address,
            "subagent_id": subagent_id,
            "success": merged,
        })
        return merged

    def discard_subagent(self, seat_id: str, subagent_id: str) -> bool:
        """Discards an unsuccessful subagent branch and prunes worktree."""
        seat = self.get_seat(seat_id)
        if not seat:
            raise ValueError(f"Seat '{seat_id}' not found")
        mgr = self.get_subagent_manager(seat_id)
        discarded = mgr.discard_subagent(subagent_id)
        self.emit_event("subagent_discarded", {
            "seat_id": seat.full_address,
            "subagent_id": subagent_id,
            "success": discarded,
        })
        return discarded

    def execute_seat_task(
        self,
        seat_id: str,
        action_generator: Callable[[int, List[Any]], Optional[Dict[str, Any]]],
        verifier: Optional[Callable[[], bool]] = None,
        tool_registry: Optional[Any] = None,
        enforce_approvals: bool = False,
    ) -> Any:
        """
        Executes the current claimed task for a seat using CodingAgentHarness
        inside that seat's isolated worktree path.
        """
        from kybernetes.engine.coding_agent import CodingAgentHarness
        from kybernetes.tools.registry import ToolRegistry

        seat = self.get_seat(seat_id)
        if not seat:
            raise ValueError(f"Seat '{seat_id}' not found")
        if not seat.current_task:
            raise ValueError(f"Seat '{seat_id}' has no claimed task to execute")

        seat_root = seat.worktree_path or self.workspace_root
        seat_checkpointer = (
            GitMilestoneCheckpointer(seat_root)
            if (seat.worktree_path and seat.worktree_path != self.workspace_root)
            else self.checkpointer
        )

        registry = tool_registry or ToolRegistry(
            allowed_roots=[seat_root],
            enforce_approvals=enforce_approvals,
        )

        harness = CodingAgentHarness(
            workspace_root=seat_root,
            tool_registry=registry,
            checkpointer=seat_checkpointer,
            state_machine=seat.state_machine,
        )

        res = harness.run_autonomous_loop(
            goal=seat.current_task.title,
            action_generator=action_generator,
            verifier=verifier,
        )

        if res.success:
            self.queue.complete_task(
                task_id=seat.current_task.task_id,
                seat_id=seat.full_address,
                result={"total_steps": res.total_steps, "final_message": res.final_message},
            )
            self.emit_event("task_completed", {
                "task_id": seat.current_task.task_id,
                "seat_id": seat.full_address,
                "total_steps": res.total_steps,
            })
        else:
            if seat_checkpointer.is_git_repo():
                checkpoints = seat_checkpointer.list_checkpoints()
                if checkpoints:
                    seat_checkpointer.rollback(checkpoints[0])
            self.queue.fail_task(
                task_id=seat.current_task.task_id,
                seat_id=seat.full_address,
                error=res.final_message or "Execution failed verification",
            )
            self.emit_event("task_failed", {
                "task_id": seat.current_task.task_id,
                "seat_id": seat.full_address,
                "error": res.final_message or "Execution failed verification",
            })

        seat.current_task = self.queue.get_task(seat.current_task.task_id)
        return res

    def to_status_dict(self) -> Dict[str, Any]:
        """Provides a complete snapshot for Prosopon / MSS UI."""
        return {
            "rig_name": self.spec.name,
            "summary": self.spec.summary,
            "version": self.spec.version,
            "seats": [s.to_dict() for s in self.list_seats()],
            "active_tasks": [t.to_dict() for t in self.queue.list_tasks(status="in_progress")],
            "attention_required": self.get_attention_summary(),
        }
