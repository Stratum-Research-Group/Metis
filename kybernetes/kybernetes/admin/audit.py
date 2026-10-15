"""Append-only in-memory admin audit log for sensitive operations and control actions."""

from datetime import datetime, timezone
import hashlib
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    id: str = Field(..., description="Unique event identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
    actor: str = Field(..., description="Redacted actor identifier or token fingerprint")
    action: str = Field(..., description="Action performed (e.g., read_config, probe_trigger, apply_config)")
    target: str = Field(..., description="Target module or resource URI")
    status: str = Field(..., description="Outcome: allowed | denied | completed | failed")
    detail: Optional[str] = Field(default=None, description="Additional context or error message")
    client_ip: Optional[str] = Field(default=None, description="Client IP address")


class AdminAuditLogger:
    """
    Append-only bounded in-memory audit log for Kybernetes admin operations.
    Keeps the last N entries (default 1000) in FIFO order.
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._entries: List[AuditEntry] = []
        self._counter: int = 0

    def log(
        self,
        actor_token: Optional[str],
        action: str,
        target: str,
        status: str,
        detail: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> AuditEntry:
        self._counter += 1
        
        # Redact token: produce secure SHA-256 fingerprint prefix
        if not actor_token:
            actor = "anonymous"
        else:
            token_clean = actor_token.strip()
            if token_clean.lower().startswith("bearer "):
                token_clean = token_clean[7:].strip()
            fingerprint = hashlib.sha256(token_clean.encode("utf-8")).hexdigest()[:12]
            actor = f"token:{fingerprint}"

        if isinstance(detail, (dict, list)):
            import json
            detail = json.dumps(detail)
        elif detail is not None:
            detail = str(detail)

        entry = AuditEntry(
            id=f"audit-{self._counter:06d}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor=actor,
            action=action,
            target=target,
            status=status,
            detail=detail,
            client_ip=client_ip,
        )

        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

        return entry

    def get_entries(self, limit: int = 100, offset: int = 0) -> List[AuditEntry]:
        """Return reversed chronological list of audit entries."""
        reversed_list = list(reversed(self._entries))
        return reversed_list[offset : offset + limit]

    def count(self) -> int:
        return len(self._entries)


admin_audit_logger = AdminAuditLogger()
