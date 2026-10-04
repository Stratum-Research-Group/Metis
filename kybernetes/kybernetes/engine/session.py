"""Session management, turn history, cancellation handling, and privacy redaction."""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from kybernetes.contracts.turn import TurnEvent, TurnRequest


class SessionTurn(BaseModel):
    """Historical turn stored in session memory."""
    turn_id: str
    query: str
    response_text: str
    events: List[TurnEvent] = Field(default_factory=list)
    status: str = "completed"
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SessionState(BaseModel):
    """Full session state with turn history and cancellation control."""
    session_id: str
    turns: List[SessionTurn] = Field(default_factory=list)
    max_context_turns: int = 10
    active_turn_id: Optional[str] = None
    is_cancelled: bool = False


class SessionManager:
    """
    Manages active conversations, historical turn replay, cancellation,
    and sensitive data redaction.
    """

    REDACTION_PATTERNS = [
        re.compile(r"(?i)api[_-]?key[\s]*[=:][\s]*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?"),
        re.compile(r"(?i)bearer[\s]+([a-zA-Z0-9_\-\.]{20,})"),
        re.compile(r"(?i)password[\s]*[=:][\s]*['\"]?([^'\"\s]{6,})['\"]?"),
    ]

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
        self._active_cancellations: Dict[str, bool] = {}

    def get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def request_cancellation(self, turn_id: str) -> bool:
        """Mark an in-flight turn for cancellation."""
        self._active_cancellations[turn_id] = True
        return True

    def is_turn_cancelled(self, turn_id: str) -> bool:
        return self._active_cancellations.get(turn_id, False)

    def clear_cancellation(self, turn_id: str) -> None:
        self._active_cancellations.pop(turn_id, None)

    def record_turn_events(self, session_id: str, turn: SessionTurn) -> None:
        """Store completed turn in session memory."""
        session = self.get_or_create_session(session_id)
        session.turns.append(turn)
        if len(session.turns) > session.max_context_turns:
            session.turns.pop(0)

    def get_past_events_for_reconnect(self, session_id: str, turn_id: str) -> List[TurnEvent]:
        """Support reconnects by retrieving previously emitted events without repeating side effects."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        for t in session.turns:
            if t.turn_id == turn_id:
                return t.events
        return []

    def redact_sensitive_text(self, text: str) -> str:
        """Redact secrets and sensitive tokens from log and trace strings."""
        redacted = text
        for pattern in self.REDACTION_PATTERNS:
            redacted = pattern.sub("[REDACTED_SECRET]", redacted)
        return redacted
