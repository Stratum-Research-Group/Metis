"""Persistent session storage and lifecycle management."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from prosopon.contracts.session import ChatMessage, SessionRecord


class SessionStore:
    """
    Manages local storage of named conversational sessions.
    Employs atomic file replacement to ensure durability and crash consistency.
    """
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir / "sessions"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, name: Optional[str] = None) -> SessionRecord:
        """Create a new conversational session."""
        session_id = str(uuid.uuid4())
        title = name or f"Session {datetime.now(timezone.utc).strftime('%b %d, %H:%M')}"
        now_str = datetime.now(timezone.utc).isoformat()

        session = SessionRecord(
            session_id=session_id,
            name=title,
            created_at_utc=now_str,
            updated_at_utc=now_str,
            messages=[],
        )
        self.save_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Load session record by ID."""
        file_path = self.storage_dir / f"{session_id}.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return SessionRecord.model_validate(data)
        except Exception:
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all available sessions ordered by last updated."""
        sessions = []
        for p in self.storage_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "name": data.get("name", "Untitled"),
                    "updated_at_utc": data.get("updated_at_utc", ""),
                    "message_count": len(data.get("messages", [])),
                })
            except Exception:
                continue

        sessions.sort(key=lambda s: s.get("updated_at_utc", ""), reverse=True)
        return sessions

    def save_session(self, session: SessionRecord) -> None:
        """Atomically persist session record."""
        session.updated_at_utc = datetime.now(timezone.utc).isoformat()
        file_path = self.storage_dir / f"{session.session_id}.json"
        tmp_path = file_path.with_suffix(".tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(session.model_dump_json(indent=2))
        tmp_path.replace(file_path)

    def append_message(self, session_id: str, message: ChatMessage) -> Optional[SessionRecord]:
        """Append a message to an existing session."""
        session = self.get_session(session_id)
        if not session:
            return None
        session.messages.append(message)
        self.save_session(session)
        return session

    def delete_session(self, session_id: str) -> bool:
        """Permanently delete session history."""
        file_path = self.storage_dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
