"""Agent Semantic State Finite State Machine (inspired by Herdr agent multiplexer).
Tracks live agent lifecycle states, enforces valid transitions, and broadcasts
state change notifications (especially when blocked awaiting Phylax approvals).
"""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Set

from kybernetes.contracts.enums import AgentSemanticState


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal state transition is attempted in the agent FSM."""
    pass


@dataclass
class StateTransitionEvent:
    """Event emitted whenever the agent transitions between semantic states."""
    from_state: AgentSemanticState
    to_state: AgentSemanticState
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class AgentStateMachine:
    """
    State machine managing semantic states of a coding agent session.
    Guarantees observable, deterministic state tracking across Metis interfaces.
    """

    ALLOWED_TRANSITIONS: Dict[AgentSemanticState, Set[AgentSemanticState]] = {
        AgentSemanticState.IDLE: {
            AgentSemanticState.PLANNING,
            AgentSemanticState.AWAITING_APPROVAL,
        },
        AgentSemanticState.PLANNING: {
            AgentSemanticState.MUTATING,
            AgentSemanticState.AWAITING_APPROVAL,
            AgentSemanticState.VERIFYING,
            AgentSemanticState.COMPLETED,
            AgentSemanticState.FAILED,
            AgentSemanticState.IDLE,
        },
        AgentSemanticState.MUTATING: {
            AgentSemanticState.AWAITING_APPROVAL,
            AgentSemanticState.VERIFYING,
            AgentSemanticState.PLANNING,
            AgentSemanticState.COMPLETED,
            AgentSemanticState.FAILED,
        },
        AgentSemanticState.AWAITING_APPROVAL: {
            AgentSemanticState.MUTATING,
            AgentSemanticState.PLANNING,
            AgentSemanticState.VERIFYING,
            AgentSemanticState.FAILED,
            AgentSemanticState.IDLE,
        },
        AgentSemanticState.VERIFYING: {
            AgentSemanticState.PLANNING,    # Backtracking upon gate failure
            AgentSemanticState.MUTATING,
            AgentSemanticState.COMPLETED,
            AgentSemanticState.FAILED,
        },
        AgentSemanticState.COMPLETED: {
            AgentSemanticState.IDLE,
            AgentSemanticState.PLANNING,
        },
        AgentSemanticState.FAILED: {
            AgentSemanticState.IDLE,
            AgentSemanticState.PLANNING,
        },
    }

    def __init__(self, initial_state: AgentSemanticState = AgentSemanticState.IDLE):
        self._current_state = initial_state
        self._history: List[StateTransitionEvent] = []
        self._listeners: List[Callable[[StateTransitionEvent], None]] = []
        self._approval_listeners: List[Callable[[StateTransitionEvent], None]] = []

    @property
    def current_state(self) -> AgentSemanticState:
        return self._current_state

    @property
    def history(self) -> List[StateTransitionEvent]:
        return list(self._history)

    def is_blocked(self) -> bool:
        """Returns True if the agent is blocked awaiting authorization / human approval."""
        return self._current_state == AgentSemanticState.AWAITING_APPROVAL

    def is_terminal(self) -> bool:
        """Returns True if in a finished state."""
        return self._current_state in (AgentSemanticState.COMPLETED, AgentSemanticState.FAILED)

    def add_listener(self, callback: Callable[[StateTransitionEvent], None]) -> None:
        """Register a callback for all state transition events."""
        self._listeners.append(callback)

    def on_awaiting_approval(self, callback: Callable[[StateTransitionEvent], None]) -> None:
        """Register a callback specifically triggered when entering AWAITING_APPROVAL."""
        self._approval_listeners.append(callback)

    def transition(
        self,
        target_state: AgentSemanticState,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> StateTransitionEvent:
        """
        Transition agent into target_state.
        Raises InvalidStateTransitionError if transition is disallowed and not forced.
        """
        allowed_targets = self.ALLOWED_TRANSITIONS.get(self._current_state, set())
        if not force and target_state != self._current_state and target_state not in allowed_targets:
            raise InvalidStateTransitionError(
                f"Illegal state transition from {self._current_state.value} to {target_state.value}. "
                f"Allowed target states: {[s.value for s in allowed_targets]}"
            )

        from_state = self._current_state
        self._current_state = target_state
        event = StateTransitionEvent(
            from_state=from_state,
            to_state=target_state,
            reason=reason,
            metadata=metadata or {},
        )
        self._history.append(event)

        # Notify general listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

        # Trigger approval listeners if entering AWAITING_APPROVAL
        if target_state == AgentSemanticState.AWAITING_APPROVAL:
            for alistener in self._approval_listeners:
                try:
                    alistener(event)
                except Exception:
                    pass

        return event

    def reset(self) -> None:
        """Reset state machine back to IDLE."""
        self.transition(AgentSemanticState.IDLE, reason="Reset state machine", force=True)
