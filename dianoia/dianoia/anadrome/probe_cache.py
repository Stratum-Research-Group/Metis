"""In-process SQLite cache for intermediate answer probes in Anadrome.

Prevents redundant LLM synthesis and verification calls at intermediate decision states
keyed by (state_id_hash, model_id, prompt_hash).
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class AnswerProbeCache:
    """Persistent SQLite cache for intermediate answer probe generations."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".dianoia" / "probe_cache.db")
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS probe_cache (
                cache_key TEXT PRIMARY KEY,
                state_id_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                draft_answer TEXT NOT NULL,
                verified INTEGER NOT NULL,
                conflict_detail TEXT,
                created_at_utc TEXT NOT NULL
            );
            """)

    @staticmethod
    def make_key(state_id_hash: str, model_id: str, prompt_hash: str) -> str:
        raw = f"{state_id_hash}:{model_id}:{prompt_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self,
        state_id_hash: str,
        model_id: str,
        prompt_hash: str,
    ) -> Optional[Dict[str, Any]]:
        key = self.make_key(state_id_hash, model_id, prompt_hash)
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT draft_answer, verified, conflict_detail FROM probe_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row:
                return {
                    "draft_answer": row["draft_answer"],
                    "verified": bool(row["verified"]),
                    "conflict_detail": row["conflict_detail"],
                }
            return None

    def put(
        self,
        state_id_hash: str,
        model_id: str,
        prompt_hash: str,
        draft_answer: str,
        verified: bool,
        conflict_detail: Optional[str] = None,
    ) -> None:
        key = self.make_key(state_id_hash, model_id, prompt_hash)
        now_str = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO probe_cache (
                    cache_key, state_id_hash, model_id, prompt_hash,
                    draft_answer, verified, conflict_detail, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    state_id_hash,
                    model_id,
                    prompt_hash,
                    draft_answer,
                    1 if verified else 0,
                    conflict_detail,
                    now_str,
                ),
            )
