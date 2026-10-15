"""Tests for Agent Semantic State Machine (Herdr-inspired FSM) and CodingAgentHarness integration."""

from pathlib import Path
import pytest

from kybernetes.contracts.enums import AgentSemanticState, TurnOutcomeStatus
from kybernetes.engine.checkpoint import GitMilestoneCheckpointer
from kybernetes.engine.coding_agent import CodingAgentHarness
from kybernetes.engine.state import AgentStateMachine, InvalidStateTransitionError, StateTransitionEvent
from kybernetes.tools.registry import ToolRegistry


def test_state_machine_initial_state():
    fsm = AgentStateMachine()
    assert fsm.current_state == AgentSemanticState.IDLE
    assert not fsm.is_blocked()
    assert not fsm.is_terminal()
    assert len(fsm.history) == 0


def test_valid_lifecycle_transitions():
    fsm = AgentStateMachine()

    # IDLE -> PLANNING
    ev1 = fsm.transition(AgentSemanticState.PLANNING, reason="Analyzing request")
    assert fsm.current_state == AgentSemanticState.PLANNING
    assert ev1.from_state == AgentSemanticState.IDLE
    assert ev1.to_state == AgentSemanticState.PLANNING

    # PLANNING -> AWAITING_APPROVAL
    ev2 = fsm.transition(AgentSemanticState.AWAITING_APPROVAL, reason="Waiting on Phylax token")
    assert fsm.current_state == AgentSemanticState.AWAITING_APPROVAL
    assert fsm.is_blocked()

    # AWAITING_APPROVAL -> MUTATING
    ev3 = fsm.transition(AgentSemanticState.MUTATING, reason="Phylax token verified")
    assert fsm.current_state == AgentSemanticState.MUTATING
    assert not fsm.is_blocked()

    # MUTATING -> VERIFYING
    ev4 = fsm.transition(AgentSemanticState.VERIFYING, reason="Running syntax and tests")
    assert fsm.current_state == AgentSemanticState.VERIFYING

    # VERIFYING -> COMPLETED
    ev5 = fsm.transition(AgentSemanticState.COMPLETED, reason="All checks passed")
    assert fsm.current_state == AgentSemanticState.COMPLETED
    assert fsm.is_terminal()
    assert len(fsm.history) == 5


def test_illegal_state_transition_raises():
    fsm = AgentStateMachine()
    # Cannot jump directly from IDLE to VERIFYING
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        fsm.transition(AgentSemanticState.VERIFYING)
    assert "Illegal state transition from idle to verifying" in str(exc_info.value)


def test_approval_notification_listener_triggered():
    fsm = AgentStateMachine()
    approval_events = []

    def on_blocked(ev: StateTransitionEvent):
        approval_events.append(ev)

    fsm.on_awaiting_approval(on_blocked)

    # Transition to PLANNING then AWAITING_APPROVAL
    fsm.transition(AgentSemanticState.PLANNING)
    assert len(approval_events) == 0

    fsm.transition(AgentSemanticState.AWAITING_APPROVAL, reason="Requires user confirmation")
    assert len(approval_events) == 1
    assert approval_events[0].to_state == AgentSemanticState.AWAITING_APPROVAL
    assert approval_events[0].reason == "Requires user confirmation"


def test_coding_agent_harness_records_semantic_states(tmp_path: Path):
    registry = ToolRegistry(allowed_roots=[tmp_path])
    checkpointer = GitMilestoneCheckpointer(tmp_path)
    harness = CodingAgentHarness(
        workspace_root=tmp_path,
        tool_registry=registry,
        checkpointer=checkpointer,
    )

    test_file = tmp_path / "app.py"
    test_file.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    def action_generator(step, history):
        if step == 1:
            return {
                "tool_name": "read_file_range",
                "arguments": {"path": str(test_file), "start_line": 1, "end_line": 10},
            }
        return None

    result = harness.run_autonomous_loop(
        goal="Read app.py",
        action_generator=action_generator,
        verifier=lambda: True,
    )

    assert result.success is True
    assert result.final_state == AgentSemanticState.COMPLETED
    assert result.state_transitions >= 2
    assert harness.state_machine.current_state == AgentSemanticState.COMPLETED
