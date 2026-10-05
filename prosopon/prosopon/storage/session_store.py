"""Persistent session storage and lifecycle management."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from prosopon.contracts.session import ChatMessage, SessionRecord

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class SessionStore:
    """Manages local storage of named conversational sessions.
    
    Employs atomic file replacement with unique temporary files and strict path
    traversal validation to ensure durability and security.
    """

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = (storage_dir / "sessions").resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # Ensure directory has restricted permissions
        try:
            os.chmod(self.storage_dir, 0o700)
        except OSError:
            pass

    def _resolve_session_path(self, session_id: str) -> Path:
        """Validate and resolve session ID to a safe file path within storage_dir."""
        if not session_id or not _SAFE_ID_RE.match(session_id):
            raise ValueError(f"Invalid session ID format: '{session_id}'")
        resolved = (self.storage_dir / f"{session_id}.json").resolve()
        try:
            resolved.relative_to(self.storage_dir)
        except ValueError as exc:
            raise ValueError(f"Path traversal detected in session ID: '{session_id}'") from exc
        return resolved

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
        try:
            file_path = self._resolve_session_path(session_id)
        except ValueError:
            return None

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
        """Atomically persist session record using a unique temporary file."""
        session.updated_at_utc = datetime.now(timezone.utc).isoformat()
        file_path = self._resolve_session_path(session.session_id)

        # Write to an unpredictable unique temporary file in storage_dir to avoid collision
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(self.storage_dir),
            prefix=f"{session.session_id}_",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            os.chmod(tmp_path, 0o600)
            payload = session.model_dump_json(indent=2).encode("utf-8")
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, file_path)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

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
        try:
            file_path = self._resolve_session_path(session_id)
        except ValueError:
            return False

        if file_path.exists():
            file_path.unlink()
            return True
        return False
