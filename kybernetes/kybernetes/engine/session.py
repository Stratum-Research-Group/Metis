"""Session management, turn history, cancellation handling, and privacy redaction."""

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from kybernetes.contracts.turn import TurnEvent, TurnRequest

logger = logging.getLogger(__name__)


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
        re.compile(r"(?i)token[\s]*[=:][\s]*['\"]?([a-zA-Z0-9_\-\.]{16,})['\"]?"),
        re.compile(r"(?i)secret[\s]*[=:][\s]*['\"]?([^'\"\s]{6,})['\"]?"),
    ]

    def __init__(
        self,
        max_sessions: int = 500,
        session_ttl_seconds: float = 86400.0,
        storage_dir: Optional[Path] = None,
    ):
        import asyncio
        self.max_sessions = max_sessions
        self.session_ttl_seconds = session_ttl_seconds
        self.storage_dir = Path(storage_dir).resolve() if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._sessions: Dict[str, SessionState] = {}
        self._session_access: Dict[str, float] = {}
        self._active_turns: set[str] = set()
        self._active_cancellations: Dict[str, bool] = {}
        self._cancellation_events: Dict[str, asyncio.Event] = {}
        # In-flight real-time event buffer and subscriber queues for reconnects
        self._turn_buffers: Dict[str, List[TurnEvent]] = {}
        self._turn_subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._turn_completed_at: Dict[str, float] = {}
        self.max_buffered_turns: int = 100
        self.turn_buffer_ttl_seconds: float = 3600.0

    def _prune_turn_buffers(self) -> None:
        """Prune completed turn event buffers to prevent unbounded memory growth (Finding B2)."""
        now = time.time()
        # 1. Prune expired completed turns
        expired = [
            tid for tid, finished_at in self._turn_completed_at.items()
            if (now - finished_at) > self.turn_buffer_ttl_seconds and tid not in self._active_turns
        ]
        for tid in expired:
            self._turn_buffers.pop(tid, None)
            self._turn_completed_at.pop(tid, None)
            self._cancellation_events.pop(tid, None)
            self._active_cancellations.pop(tid, None)

        # 2. Prune excess completed turns if count exceeds max_buffered_turns
        completed_tids = [tid for tid in self._turn_completed_at if tid not in self._active_turns]
        if len(completed_tids) > self.max_buffered_turns:
            completed_tids.sort(key=lambda tid: self._turn_completed_at.get(tid, 0.0))
            excess = len(completed_tids) - self.max_buffered_turns
            for tid in completed_tids[:excess]:
                self._turn_buffers.pop(tid, None)
                self._turn_completed_at.pop(tid, None)
                self._cancellation_events.pop(tid, None)
                self._active_cancellations.pop(tid, None)

    def _prune_sessions(self) -> None:
        """Evict expired sessions or excess sessions exceeding max_sessions by LRU."""
        now = time.time()
        expired_ids = [
            sid for sid, last_acc in self._session_access.items()
            if now - last_acc > self.session_ttl_seconds
        ]
        for sid in expired_ids:
            self._sessions.pop(sid, None)
            self._session_access.pop(sid, None)

        if len(self._sessions) > self.max_sessions:
            sorted_by_lru = sorted(self._session_access.items(), key=lambda kv: kv[1])
            excess = len(self._sessions) - self.max_sessions
            for sid, _ in sorted_by_lru[:excess]:
                self._sessions.pop(sid, None)
                self._session_access.pop(sid, None)

        self._prune_turn_buffers()

    def _validate_session_id(self, session_id: str) -> None:
        """Validate session_id against path traversal attempts."""
        if not session_id or not isinstance(session_id, str):
            raise ValueError(f"Invalid session_id: {session_id!r}")
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise ValueError(f"Path traversal detected in session_id: {session_id}")

    def _get_session_path(self, session_id: str) -> Optional[Path]:
        """Resolve safe session persistence path inside storage_dir."""
        if not self.storage_dir:
            return None
        self._validate_session_id(session_id)
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)
        if not safe_id:
            raise ValueError(f"Invalid empty session_id after sanitization: {session_id}")
        session_file = (self.storage_dir / f"{safe_id}.jsonl").resolve()
        try:
            session_file.relative_to(self.storage_dir)
        except ValueError:
            raise ValueError(f"Path traversal detected in session_id: {session_id}")
        return session_file

    def get_or_create_session(self, session_id: str) -> SessionState:
        self._validate_session_id(session_id)
        now = time.time()
        if session_id not in self._sessions:
            # Check if persisted on disk
            loaded = self._load_session_from_disk(session_id)
            if loaded:
                self._sessions[session_id] = loaded
            else:
                self._sessions[session_id] = SessionState(session_id=session_id)
            self._session_access[session_id] = now
            self._prune_sessions()
        else:
            self._session_access[session_id] = now
        return self._sessions[session_id]

    def get_cancellation_event(self, turn_id: str) -> "asyncio.Event":
        import asyncio
        if turn_id not in self._cancellation_events:
            self._cancellation_events[turn_id] = asyncio.Event()
        return self._cancellation_events[turn_id]

    def register_active_turn(self, turn_id: str) -> None:
        """Register a turn as currently active/in-flight."""
        self._active_turns.add(turn_id)
        if turn_id not in self._turn_buffers:
            self._turn_buffers[turn_id] = []
        if turn_id not in self._turn_subscribers:
            self._turn_subscribers[turn_id] = []

    def unregister_active_turn(self, turn_id: str) -> None:
        """Unregister a turn upon completion or termination and notify subscribers."""
        self._active_turns.discard(turn_id)
        self.clear_cancellation(turn_id)
        self._turn_completed_at[turn_id] = time.time()
        # Notify subscribers of terminal close by sending None
        subs = self._turn_subscribers.pop(turn_id, [])
        for q in subs:
            q.put_nowait(None)
        self._prune_turn_buffers()

    def is_turn_active(self, turn_id: str) -> bool:
        return turn_id in self._active_turns

    def append_turn_event(self, turn_id: str, event: TurnEvent) -> None:
        """Append an event to the in-flight buffer and broadcast to live subscribers."""
        if turn_id not in self._turn_buffers:
            self._turn_buffers[turn_id] = []
        self._turn_buffers[turn_id].append(event)

        # Broadcast to active subscriber queues
        subs = self._turn_subscribers.get(turn_id, [])
        for q in subs:
            q.put_nowait(event)

    def get_turn_events_after(self, turn_id: str, after_seq: int = -1) -> List[TurnEvent]:
        """Retrieve buffered events for a turn with seq_num > after_seq."""
        buf = self._turn_buffers.get(turn_id, [])
        if after_seq < 0:
            return list(buf)
        return [e for e in buf if e.seq_num > after_seq]

    async def subscribe_turn_events(self, turn_id: str, after_seq: int = -1):
        """
        Async generator yielding historical buffered events after `after_seq`,
        and then streaming subsequent real-time events until turn ends.
        Registers queue prior to snapshot to close the event gap (Finding B2).
        """
        import asyncio

        # If turn is inactive and no buffer exists, nothing to stream
        if not self.is_turn_active(turn_id) and turn_id not in self._turn_buffers:
            return

        # 1. Register live subscriber queue FIRST to ensure no event is missed during snapshot
        q: asyncio.Queue = asyncio.Queue()
        if turn_id not in self._turn_subscribers:
            self._turn_subscribers[turn_id] = []
        self._turn_subscribers[turn_id].append(q)

        # 2. Snapshot historical buffered events AFTER queue is registered
        past = self.get_turn_events_after(turn_id, after_seq)
        last_yielded_seq = after_seq

        try:
            for ev in past:
                yield ev
                if ev.seq_num > last_yielded_seq:
                    last_yielded_seq = ev.seq_num

            # 3. Stream from live queue, skipping any events already yielded from history
            while True:
                if not self.is_turn_active(turn_id) and q.empty():
                    break
                ev = await q.get()
                if ev is None:
                    # Terminal signal
                    break
                if ev.seq_num > last_yielded_seq:
                    yield ev
                    last_yielded_seq = ev.seq_num
        finally:
            if turn_id in self._turn_subscribers and q in self._turn_subscribers[turn_id]:
                self._turn_subscribers[turn_id].remove(q)

    def request_cancellation(self, turn_id: str) -> bool:
        """Mark an in-flight turn for cancellation and notify waiters."""
        self._active_cancellations[turn_id] = True
        if turn_id in self._cancellation_events:
            self._cancellation_events[turn_id].set()
        return True

    def is_turn_cancelled(self, turn_id: str) -> bool:
        return self._active_cancellations.get(turn_id, False)

    def clear_cancellation(self, turn_id: str) -> None:
        self._active_cancellations.pop(turn_id, None)
        self._cancellation_events.pop(turn_id, None)

    def _redact_value(self, key: Optional[str], val: Any) -> Any:
        """Recursively redact sensitive text and key values."""
        if isinstance(val, str):
            if key and any(k in key.lower() for k in ("password", "api_key", "secret", "token", "auth", "credential")):
                return "[REDACTED_SECRET]"
            return self.redact_sensitive_text(val)
        elif isinstance(val, dict):
            return {k: self._redact_value(str(k), v) for k, v in val.items()}
        elif isinstance(val, list):
            return [self._redact_value(None, item) for item in val]
        return val

    def redact_turn(self, turn: SessionTurn) -> SessionTurn:
        """Sanitize turn query, response_text, and event payloads by removing sensitive tokens."""
        redacted_query = self.redact_sensitive_text(turn.query)
        redacted_response = self.redact_sensitive_text(turn.response_text)
        redacted_events = []
        for ev in turn.events:
            redacted_payload = self._redact_value(None, ev.payload) if ev.payload else {}
            redacted_events.append(ev.model_copy(update={"payload": redacted_payload}))
        return turn.model_copy(
            update={
                "query": redacted_query,
                "response_text": redacted_response,
                "events": redacted_events,
            }
        )

    def record_turn_events(self, session_id: str, turn: SessionTurn) -> None:
        """Store completed turn in session memory and write to disk if enabled."""
        self._validate_session_id(session_id)
        redacted_turn = self.redact_turn(turn)
        session = self.get_or_create_session(session_id)
        session.turns.append(redacted_turn)
        if len(session.turns) > session.max_context_turns:
            session.turns.pop(0)

        # Store in turn buffers in case of post-completion queries
        self._turn_buffers[turn.turn_id] = list(redacted_turn.events)
        self._persist_turn_to_disk(session_id, redacted_turn)

    def get_past_events_for_reconnect(self, session_id: str, turn_id: str) -> List[TurnEvent]:
        """Support reconnects by retrieving previously emitted events without repeating side effects."""
        # Check turn buffer first
        if turn_id in self._turn_buffers:
            return list(self._turn_buffers[turn_id])

        session = self._sessions.get(session_id)
        if not session:
            session = self._load_session_from_disk(session_id)
        if not session:
            return []
        for t in session.turns:
            if t.turn_id == turn_id:
                return t.events
        return []

    def _persist_turn_to_disk(self, session_id: str, turn: SessionTurn) -> None:
        """Append-only serialization of turn record to JSONL."""
        if not self.storage_dir:
            return
        session_file = self._get_session_path(session_id)
        if not session_file:
            return
        redacted_turn = self.redact_turn(turn)
        try:
            with open(session_file, "a", encoding="utf-8") as f:
                f.write(redacted_turn.model_dump_json() + "\n")
        except Exception as e:
            logger.error("Failed to persist turn %s for session %s to disk: %s", turn.turn_id, session_id, e)
            raise

    def _load_session_from_disk(self, session_id: str) -> Optional[SessionState]:
        """Load session turns from disk JSONL file if available."""
        if not self.storage_dir:
            return None
        session_file = self._get_session_path(session_id)
        if not session_file or not session_file.exists():
            return None

        state = SessionState(session_id=session_id)
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            state.turns.append(SessionTurn.model_validate_json(line))
                        except Exception as e:
                            logger.warning(
                                "Skipping corrupted turn at line %d in session file %s: %s",
                                line_num, session_file, e
                            )
            return state
        except Exception as e:
            logger.error("Failed to load session file %s: %s", session_file, e)
            return None

    def redact_sensitive_text(self, text: str) -> str:
        """Redact secrets and sensitive tokens from log and trace strings."""
        if not text:
            return text
        redacted = text
        for pattern in self.REDACTION_PATTERNS:
            redacted = pattern.sub("[REDACTED_SECRET]", redacted)
        return redacted
