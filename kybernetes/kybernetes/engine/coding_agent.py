"""Metis Sovereign Coding Agent Execution Loop.
Coordinates Pi-style minimal tool contracts, Odyssey milestone checkpointing,
Dianoia backtracking, and Hermes dynamic skill learning.
"""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional

from kybernetes.contracts.enums import AgentSemanticState, TurnOutcomeStatus
from kybernetes.contracts.tool_spec import ToolCallRequest, ToolResult
from kybernetes.engine.checkpoint import GitMilestoneCheckpointer, MilestoneCheckpoint
from kybernetes.engine.state import AgentStateMachine, StateTransitionEvent
from kybernetes.tools.registry import ToolRegistry


class EarlyFlowAbort(Exception):
    """Raised when stream flow rules or Phylax guards detect token loop or policy violation mid-stream."""
    pass


class StreamFlowMonitor:
    """Monitors streaming tokens to abort infinite repetition loops and unapproved action patterns."""

    def __init__(self, max_repetition_threshold: int = 5):
        self.max_repetition_threshold = max_repetition_threshold
        self._recent_chunks: List[str] = []

    def feed(self, chunk: str) -> None:
        """Feed a stream chunk. Raises EarlyFlowAbort if repetition loop or illegal pattern detected."""
        self._recent_chunks.append(chunk)
        if len(self._recent_chunks) > 20:
            self._recent_chunks.pop(0)

        # Check for repetitive loop (identical 15+ char sequences repeating)
        text = "".join(self._recent_chunks)
        if len(text) > 45:
            tail = text[-15:]
            if text.count(tail) >= self.max_repetition_threshold:
                raise EarlyFlowAbort(f"Detected infinite repetition loop on pattern: '{tail}'. Early aborting turn.")

    def reset(self) -> None:
        self._recent_chunks.clear()


@dataclass
class CodingTurnState:
    step: int
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[ToolResult] = None
    passed_validation: bool = False
    error: Optional[str] = None
    semantic_state: AgentSemanticState = AgentSemanticState.IDLE


@dataclass
class CodingSessionResult:
    session_id: str
    success: bool
    status: TurnOutcomeStatus
    total_steps: int
    history: List[CodingTurnState] = field(default_factory=list)
    final_message: str = ""
    skill_synthesized: Optional[str] = None
    checkpoints_taken: int = 0
    backtracks_executed: int = 0
    final_state: AgentSemanticState = AgentSemanticState.IDLE
    state_transitions: int = 0


class CodingAgentHarness:
    """The central coding agent runtime embodying the synthesized harness principles."""

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        tool_registry: Optional[ToolRegistry] = None,
        checkpointer: Optional[GitMilestoneCheckpointer] = None,
        max_steps: int = 15,
        state_machine: Optional[AgentStateMachine] = None,
    ):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.tool_registry = tool_registry or ToolRegistry(allowed_roots=[self.workspace_root])
        self.checkpointer = checkpointer or GitMilestoneCheckpointer(self.workspace_root)
        self.max_steps = max_steps
        self.flow_monitor = StreamFlowMonitor()
        self.state_machine = state_machine or AgentStateMachine()

    def execute_step(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        approval_token: Optional[str] = None,
    ) -> ToolResult:
        """Execute a single vetted tool call against the workspace."""
        req = ToolCallRequest(
            tool_id=tool_name,
            arguments=arguments,
            approval_id=approval_token,
        )
        return self.tool_registry.execute(req)

    def run_autonomous_loop(
        self,
        goal: str,
        action_generator: Callable[[int, List[CodingTurnState]], Optional[Dict[str, Any]]],
        verifier: Optional[Callable[[], bool]] = None,
    ) -> CodingSessionResult:
        """
        Run the autonomous coding loop with Odyssey checkpointing and Dianoia backtracking.
        action_generator: returns {'tool_name': str, 'arguments': dict} or None when done.
        verifier: returns True if code passes syntax/tests, False if broken.
        """
        session_id = f"coding-{int(time.time())}"
        history: List[CodingTurnState] = []
        backtracks = 0

        # 1. Take initial Odyssey milestone checkpoint and enter PLANNING
        self.state_machine.transition(
            AgentSemanticState.PLANNING,
            reason=f"Started autonomous loop for: {goal}",
            metadata={"goal": goal, "session_id": session_id},
        )
        initial_cp = self.checkpointer.create_checkpoint(f"Initial baseline for: {goal}")
        active_checkpoint = initial_cp

        for step in range(1, self.max_steps + 1):
            action = action_generator(step, history)
            if not action:
                # Agent indicates task completion
                break

            tool_name = action.get("tool_name", "")
            args = action.get("arguments", {})
            approval_token = action.get("approval_token")

            # If mutating tool, create milestone checkpoint and transition state
            is_mutating = tool_name in ("edit_file_block", "write_file", "sandboxed_bash")
            if is_mutating:
                active_checkpoint = self.checkpointer.create_checkpoint(f"Pre-{tool_name} step {step}")
                # If awaiting approval is requested in action:
                if action.get("requires_approval") and not approval_token:
                    self.state_machine.transition(
                        AgentSemanticState.AWAITING_APPROVAL,
                        reason=f"Phylax gating required for mutating tool '{tool_name}'",
                        metadata={"step": step, "tool_name": tool_name, "arguments": args},
                    )
                self.state_machine.transition(
                    AgentSemanticState.MUTATING,
                    reason=f"Executing mutating tool '{tool_name}'",
                    metadata={"step": step, "tool_name": tool_name},
                )

            result = self.execute_step(tool_name, args, approval_token=approval_token)

            turn_state = CodingTurnState(
                step=step,
                tool_name=tool_name,
                arguments=args,
                result=result,
                passed_validation=result.success,
                error=result.error,
                semantic_state=self.state_machine.current_state,
            )

            # Check verification if verifier hook provided
            if result.success and verifier:
                self.state_machine.transition(
                    AgentSemanticState.VERIFYING,
                    reason=f"Running verification gate after step {step}",
                    metadata={"step": step, "tool_name": tool_name},
                )
                test_passed = verifier()
                if not test_passed:
                    # Odyssey Rollback & Dianoia Backtrack
                    self.checkpointer.rollback(active_checkpoint)
                    turn_state.passed_validation = False
                    turn_state.error = "Verification gate failed (tests/linters returned non-zero). Working tree rolled back."
                    backtracks += 1
                    self.state_machine.transition(
                        AgentSemanticState.PLANNING,
                        reason=f"Rollback to checkpoint after verification failure at step {step}",
                        metadata={"step": step, "rolled_back_to": active_checkpoint.id},
                    )
                else:
                    self.state_machine.transition(
                        AgentSemanticState.PLANNING,
                        reason=f"Step {step} verified; continuing planning",
                        metadata={"step": step},
                    )

            history.append(turn_state)

        # Final verification
        all_passed = (len(history) > 0 and history[-1].passed_validation and (verifier() if verifier else True))
        status = TurnOutcomeStatus.VERIFIED if all_passed else TurnOutcomeStatus.FAILED

        final_state = AgentSemanticState.COMPLETED if all_passed else AgentSemanticState.FAILED
        self.state_machine.transition(
            final_state,
            reason="Autonomous coding loop completed" if all_passed else "Loop ended with open verification failures",
            metadata={"success": all_passed, "total_steps": len(history)},
        )

        return CodingSessionResult(
            session_id=session_id,
            success=all_passed,
            status=status,
            total_steps=len(history),
            history=history,
            final_message="Task completed and verified." if all_passed else "Task completed with open verification issues.",
            checkpoints_taken=len(self.checkpointer.list_checkpoints()),
            backtracks_executed=backtracks,
            final_state=final_state,
            state_transitions=len(self.state_machine.history),
        )
