"""Lochos Transactional SQLite Work Queue and Baton Transfer Engine.
Ensures durable task ownership, crash survival, and cross-seat attention reconciliation.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class WorkItem:
    """Represents a durable unit of owned work tracked in the Lochos SQLite queue."""
    task_id: str
    rig_id: str
    mission: str
    slice_id: str
    title: str
    description: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # queued, in_progress, attention_required, completed, failed
    baton_owner: Optional[str] = None  # e.g. coder@squad
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rig_id": self.rig_id,
            "mission": self.mission,
            "slice_id": self.slice_id,
            "title": self.title,
            "description": self.description,
            "payload": self.payload,
            "status": self.status,
            "baton_owner": self.baton_owner,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
        }


class LochosQueue:
    """
    Durable SQLite work queue with explicit baton semantics.
    Tasks survive restarts; no task is dropped if Kybernetes crashes.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            metis_dir = Path.cwd() / ".metis"
            metis_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = metis_dir / "lochos.sqlite"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS work_items (
                    task_id TEXT PRIMARY KEY,
                    rig_id TEXT NOT NULL,
                    mission TEXT NOT NULL,
                    slice_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    payload_json TEXT,
                    status TEXT NOT NULL,
                    baton_owner TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    result_json TEXT,
                    error TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_work_items_status ON work_items(status);
                CREATE INDEX IF NOT EXISTS idx_work_items_owner ON work_items(baton_owner);

                CREATE TABLE IF NOT EXISTS baton_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    from_seat TEXT,
                    to_seat TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    note TEXT
                );

                CREATE TABLE IF NOT EXISTS attention_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    seat_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    details_json TEXT,
                    created_at REAL NOT NULL,
                    resolved_at REAL,
                    resolution TEXT
                );
            """)

    def _row_to_item(self, row: sqlite3.Row) -> WorkItem:
        return WorkItem(
            task_id=row["task_id"],
            rig_id=row["rig_id"],
            mission=row["mission"],
            slice_id=row["slice_id"],
            title=row["title"],
            description=row["description"] or "",
            payload=json.loads(row["payload_json"]) if row["payload_json"] else {},
            status=row["status"],
            baton_owner=row["baton_owner"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
        )

    def enqueue(
        self,
        rig_id: str,
        mission: str,
        slice_id: str,
        title: str,
        description: str = "",
        payload: Optional[Dict[str, Any]] = None,
        assigned_to: Optional[str] = None,
    ) -> WorkItem:
        """Enqueue a new task into the SQLite queue."""
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        now = time.time()
        status = "queued"
        payload_data = payload or {}

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO work_items (
                    task_id, rig_id, mission, slice_id, title, description,
                    payload_json, status, baton_owner, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    rig_id,
                    mission,
                    slice_id,
                    title,
                    description,
                    json.dumps(payload_data),
                    status,
                    assigned_to,
                    now,
                    now,
                ),
            )
            if assigned_to:
                conn.execute(
                    """
                    INSERT INTO baton_history (task_id, from_seat, to_seat, timestamp, note)
                    VALUES (?, NULL, ?, ?, 'Initial task assignment')
                    """,
                    (task_id, assigned_to, now),
                )

        return self.get_task(task_id)  # type: ignore

    def get_task(self, task_id: str) -> Optional[WorkItem]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM work_items WHERE task_id = ?", (task_id,)).fetchone()
            return self._row_to_item(row) if row else None

    def list_tasks(
        self,
        status: Optional[str] = None,
        seat_id: Optional[str] = None,
        rig_id: Optional[str] = None,
    ) -> List[WorkItem]:
        query = "SELECT * FROM work_items WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if seat_id:
            query += " AND baton_owner = ?"
            params.append(seat_id)
        if rig_id:
            query += " AND rig_id = ?"
            params.append(rig_id)

        query += " ORDER BY created_at ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_item(r) for r in rows]

    def claim_baton(self, seat_id: str, task_id: Optional[str] = None) -> Optional[WorkItem]:
        """
        Atomically claims a task for a seat.
        If task_id is specified, claims that exact task if it is queued or assigned to seat_id.
        Otherwise claims the oldest queued task assigned to seat_id or unassigned.
        """
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if task_id:
                row = cursor.execute(
                    """
                    SELECT * FROM work_items
                    WHERE task_id = ? AND status = 'queued'
                    AND (baton_owner IS NULL OR baton_owner = ?)
                    """,
                    (task_id, seat_id),
                ).fetchone()
            else:
                row = cursor.execute(
                    """
                    SELECT * FROM work_items
                    WHERE status = 'queued'
                    AND (baton_owner IS NULL OR baton_owner = ?)
                    ORDER BY created_at ASC LIMIT 1
                    """,
                    (seat_id,),
                ).fetchone()

            if not row:
                return None

            claimed_id = row["task_id"]
            prev_owner = row["baton_owner"]

            cursor.execute(
                """
                UPDATE work_items
                SET status = 'in_progress', baton_owner = ?, updated_at = ?
                WHERE task_id = ? AND status = 'queued'
                """,
                (seat_id, now, claimed_id),
            )
            if cursor.rowcount == 0:
                # Lost race to another worker
                return None

            cursor.execute(
                """
                INSERT INTO baton_history (task_id, from_seat, to_seat, timestamp, note)
                VALUES (?, ?, ?, ?, 'Baton claimed')
                """,
                (claimed_id, prev_owner, seat_id, now),
            )

        return self.get_task(claimed_id)

    def handoff_baton(
        self,
        task_id: str,
        from_seat: str,
        to_seat: str,
        note: Optional[str] = None,
    ) -> WorkItem:
        """
        Atomically transfers the baton from one seat to another.
        The task status is set to 'queued' so the receiving seat can pick it up.
        """
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                "SELECT * FROM work_items WHERE task_id = ?", (task_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Task '{task_id}' does not exist")
            if row["baton_owner"] != from_seat:
                raise PermissionError(
                    f"Seat '{from_seat}' does not hold the baton for task '{task_id}' (held by '{row['baton_owner']}')"
                )

            cursor.execute(
                """
                UPDATE work_items
                SET baton_owner = ?, status = 'queued', updated_at = ?
                WHERE task_id = ?
                """,
                (to_seat, now, task_id),
            )
            cursor.execute(
                """
                INSERT INTO baton_history (task_id, from_seat, to_seat, timestamp, note)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task_id, from_seat, to_seat, now, note or "Baton handoff"),
            )

        return self.get_task(task_id)  # type: ignore

    def raise_attention(
        self,
        task_id: str,
        seat_id: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> WorkItem:
        """
        Signals that a seat is blocked and requires operator/human attention
        (e.g. Phylax approval required, compilation error beyond budget, verification conflict).
        """
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE work_items
                SET status = 'attention_required', updated_at = ?
                WHERE task_id = ?
                """,
                (now, task_id),
            )
            cursor.execute(
                """
                INSERT INTO attention_events (
                    task_id, seat_id, reason, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (task_id, seat_id, reason, json.dumps(details or {}), now),
            )

        return self.get_task(task_id)  # type: ignore

    def resolve_attention(
        self,
        task_id: str,
        seat_id: str,
        resolution: str = "approved",
        return_to_status: str = "in_progress",
    ) -> WorkItem:
        """Resolves an open attention block for a task."""
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE work_items
                SET status = ?, updated_at = ?
                WHERE task_id = ? AND status = 'attention_required'
                """,
                (return_to_status, now, task_id),
            )
            cursor.execute(
                """
                UPDATE attention_events
                SET resolved_at = ?, resolution = ?
                WHERE task_id = ? AND seat_id = ? AND resolved_at IS NULL
                """,
                (now, resolution, task_id, seat_id),
            )

        return self.get_task(task_id)  # type: ignore

    def complete_task(
        self,
        task_id: str,
        seat_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> WorkItem:
        """Marks a task as completed with optional result output."""
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                "SELECT * FROM work_items WHERE task_id = ?", (task_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Task '{task_id}' does not exist")
            if row["baton_owner"] != seat_id:
                raise PermissionError(
                    f"Seat '{seat_id}' is not the baton owner for task '{task_id}'"
                )
            if row["status"] == "attention_required":
                raise PermissionError(
                    f"Task '{task_id}' is blocked on attention_required and must be resolved before completion"
                )
            cursor.execute(
                """
                UPDATE work_items
                SET status = 'completed', result_json = ?, updated_at = ?
                WHERE task_id = ? AND baton_owner = ?
                """,
                (json.dumps(result or {}), now, task_id, seat_id),
            )

        return self.get_task(task_id)  # type: ignore

    def fail_task(
        self,
        task_id: str,
        seat_id: str,
        error: str,
    ) -> WorkItem:
        """Marks a task as failed with an error description."""
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE work_items
                SET status = 'failed', error = ?, updated_at = ?
                WHERE task_id = ? AND baton_owner = ?
                """,
                (error, now, task_id, seat_id),
            )

        return self.get_task(task_id)  # type: ignore

    def get_seat_current_work(self, seat_id: str) -> Optional[WorkItem]:
        """Retrieves the active work item currently owned and being executed by a seat."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM work_items
                WHERE baton_owner = ? AND status IN ('in_progress', 'attention_required')
                ORDER BY updated_at DESC LIMIT 1
                """,
                (seat_id,),
            ).fetchone()
            return self._row_to_item(row) if row else None

    def reconcile_attention(self) -> List[Dict[str, Any]]:
        """
        Cross-seat attention surface reconciler.
        Returns all open blocks requiring human action across the squad.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT a.id, a.task_id, a.seat_id, a.reason, a.details_json, a.created_at,
                       w.title, w.mission, w.slice_id
                FROM attention_events a
                JOIN work_items w ON a.task_id = w.task_id
                WHERE a.resolved_at IS NULL
                ORDER BY a.created_at ASC
                """
            ).fetchall()
            return [
                {
                    "event_id": r["id"],
                    "task_id": r["task_id"],
                    "seat_id": r["seat_id"],
                    "reason": r["reason"],
                    "details": json.loads(r["details_json"]) if r["details_json"] else {},
                    "created_at": r["created_at"],
                    "task_title": r["title"],
                    "mission": r["mission"],
                    "slice_id": r["slice_id"],
                }
                for r in rows
            ]

    def recover_after_crash(self) -> List[WorkItem]:
        """
        Crash-recovery protocol:
        Scans for tasks that were 'in_progress' when the daemon crashed or restarted.
        Transitions them to 'attention_required' with an unclean_shutdown note,
        guaranteeing that no baton or work is silently lost.
        """
        now = time.time()
        recovered: List[WorkItem] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute(
                "SELECT * FROM work_items WHERE status = 'in_progress'"
            ).fetchall()

            for row in rows:
                task_id = row["task_id"]
                seat_id = row["baton_owner"] or "unknown"
                cursor.execute(
                    """
                    UPDATE work_items
                    SET status = 'attention_required', updated_at = ?
                    WHERE task_id = ?
                    """,
                    (now, task_id),
                )
                cursor.execute(
                    """
                    INSERT INTO attention_events (
                        task_id, seat_id, reason, details_json, created_at
                    ) VALUES (?, ?, 'unclean_shutdown_detected', '{"resumed_after_restart": true}', ?)
                    """,
                    (task_id, seat_id, now),
                )

            # Re-fetch updated rows within this transaction
            if rows:
                updated_rows = cursor.execute(
                    "SELECT * FROM work_items WHERE status = 'attention_required'"
                ).fetchall()
                recovered = [self._row_to_item(r) for r in updated_rows]

        return recovered
