"""SQLite storage layer for Pinax Tool Catalogue."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, Iterator, List, Optional

from kybernetes.pinax.models import (
    EntryKind,
    InstallState,
    RiskTier,
    ToolCapability,
    ToolDoc,
    ToolKind,
    ToolRecord,
    ToolRiskOverride,
    ToolSignature,
)


class PinaxDatabase:
    """
    Durable SQLite repository for the Pinax tool catalogue.
    Enforces foreign key integrity and supports FTS5 full-text search with LIKE fallback.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.fts5_enabled = False
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 30000;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Main tools table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    id TEXT PRIMARY KEY,
                    host_id TEXT NOT NULL DEFAULT 'localhost',
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    entry_kind TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    invocation_path TEXT,
                    resolved_path TEXT,
                    env_identity TEXT,
                    interpreter TEXT,
                    sha256 TEXT,
                    version TEXT,
                    package TEXT DEFAULT 'unknown',
                    package_manager TEXT DEFAULT 'unknown',
                    install_state TEXT NOT NULL,
                    risk_tier TEXT NOT NULL,
                    discovered_by TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    last_verified REAL NOT NULL,
                    notes TEXT
                );
            """)

            # 2. Tool documentation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_docs (
                    tool_id TEXT NOT NULL,
                    doc_kind TEXT NOT NULL,
                    locator TEXT NOT NULL,
                    version_scope TEXT,
                    sha256 TEXT,
                    size_bytes INTEGER DEFAULT 0,
                    fetched_at REAL NOT NULL,
                    PRIMARY KEY (tool_id, doc_kind, locator),
                    FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE
                );
            """)

            # 3. Tool capabilities
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_capabilities (
                    tool_id TEXT NOT NULL,
                    capability TEXT NOT NULL,
                    detail TEXT DEFAULT '',
                    PRIMARY KEY (tool_id, capability),
                    FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE
                );
            """)

            # 4. Tool risk overrides
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_risk_overrides (
                    tool_id TEXT NOT NULL,
                    pattern_kind TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    note TEXT DEFAULT '',
                    PRIMARY KEY (tool_id, pattern_kind, pattern),
                    FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE
                );
            """)

            # 5. Tool signatures (structured tools)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_signatures (
                    tool_id TEXT NOT NULL,
                    call_name TEXT NOT NULL,
                    arg_schema_json TEXT NOT NULL,
                    curated_by TEXT NOT NULL,
                    curated_at REAL NOT NULL,
                    PRIMARY KEY (tool_id, call_name),
                    FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE
                );
            """)

            # 6. Scan audit trail
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_runs (
                    id TEXT PRIMARY KEY,
                    host_id TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    finished_at REAL NOT NULL,
                    scanner_version TEXT NOT NULL,
                    added INTEGER NOT NULL DEFAULT 0,
                    removed INTEGER NOT NULL DEFAULT 0,
                    changed INTEGER NOT NULL DEFAULT 0
                );
            """)

            # Indexes for fast lookup
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_category ON tools(category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_tier ON tools(risk_tier);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_state ON tools(install_state);")

            # Check FTS5 support
            try:
                cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS tools_fts USING fts5(id, name, summary, category, notes);")
                self.fts5_enabled = True
            except sqlite3.OperationalError:
                self.fts5_enabled = False

    def upsert_tool(self, tool: ToolRecord) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tools (
                    id, host_id, name, kind, entry_kind, summary, category,
                    invocation_path, resolved_path, env_identity, interpreter,
                    sha256, version, package, package_manager, install_state,
                    risk_tier, discovered_by, first_seen, last_verified, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    host_id=excluded.host_id,
                    name=excluded.name,
                    kind=excluded.kind,
                    entry_kind=excluded.entry_kind,
                    summary=excluded.summary,
                    category=excluded.category,
                    invocation_path=excluded.invocation_path,
                    resolved_path=excluded.resolved_path,
                    env_identity=excluded.env_identity,
                    interpreter=excluded.interpreter,
                    sha256=excluded.sha256,
                    version=excluded.version,
                    package=excluded.package,
                    package_manager=excluded.package_manager,
                    install_state=excluded.install_state,
                    risk_tier=excluded.risk_tier,
                    discovered_by=excluded.discovered_by,
                    last_verified=excluded.last_verified,
                    notes=excluded.notes;
            """, (
                tool.id, tool.host_id, tool.name,
                tool.kind.value if isinstance(tool.kind, ToolKind) else str(tool.kind),
                tool.entry_kind.value if isinstance(tool.entry_kind, EntryKind) else str(tool.entry_kind),
                tool.summary, tool.category, tool.invocation_path, tool.resolved_path,
                tool.env_identity, tool.interpreter, tool.sha256, tool.version,
                tool.package, tool.package_manager,
                tool.install_state.value if isinstance(tool.install_state, InstallState) else str(tool.install_state),
                tool.risk_tier.value if isinstance(tool.risk_tier, RiskTier) else str(tool.risk_tier),
                tool.discovered_by, tool.first_seen, tool.last_verified, tool.notes,
            ))

            if self.fts5_enabled:
                cursor.execute("DELETE FROM tools_fts WHERE id = ?;", (tool.id,))
                cursor.execute(
                    "INSERT INTO tools_fts (id, name, summary, category, notes) VALUES (?, ?, ?, ?, ?);",
                    (tool.id, tool.name, tool.summary, tool.category, tool.notes or ""),
                )

    def batch_upsert_tools(self, tools: List[ToolRecord]) -> None:
        if not tools:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for tool in tools:
                cursor.execute("""
                    INSERT INTO tools (
                        id, host_id, name, kind, entry_kind, summary, category,
                        invocation_path, resolved_path, env_identity, interpreter,
                        sha256, version, package, package_manager, install_state,
                        risk_tier, discovered_by, first_seen, last_verified, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        host_id=excluded.host_id,
                        name=excluded.name,
                        kind=excluded.kind,
                        entry_kind=excluded.entry_kind,
                        summary=excluded.summary,
                        category=excluded.category,
                        invocation_path=excluded.invocation_path,
                        resolved_path=excluded.resolved_path,
                        env_identity=excluded.env_identity,
                        interpreter=excluded.interpreter,
                        sha256=excluded.sha256,
                        version=excluded.version,
                        package=excluded.package,
                        package_manager=excluded.package_manager,
                        install_state=excluded.install_state,
                        risk_tier=excluded.risk_tier,
                        discovered_by=excluded.discovered_by,
                        last_verified=excluded.last_verified,
                        notes=excluded.notes;
                """, (
                    tool.id, tool.host_id, tool.name,
                    tool.kind.value if isinstance(tool.kind, ToolKind) else str(tool.kind),
                    tool.entry_kind.value if isinstance(tool.entry_kind, EntryKind) else str(tool.entry_kind),
                    tool.summary, tool.category, tool.invocation_path, tool.resolved_path,
                    tool.env_identity, tool.interpreter, tool.sha256, tool.version,
                    tool.package, tool.package_manager,
                    tool.install_state.value if isinstance(tool.install_state, InstallState) else str(tool.install_state),
                    tool.risk_tier.value if isinstance(tool.risk_tier, RiskTier) else str(tool.risk_tier),
                    tool.discovered_by, tool.first_seen, tool.last_verified, tool.notes,
                ))
                if self.fts5_enabled:
                    cursor.execute("DELETE FROM tools_fts WHERE id = ?;", (tool.id,))
                    cursor.execute(
                        "INSERT INTO tools_fts (id, name, summary, category, notes) VALUES (?, ?, ?, ?, ?);",
                        (tool.id, tool.name, tool.summary, tool.category, tool.notes or ""),
                    )

    def get_tool(self, tool_id_or_name: str) -> Optional[ToolRecord]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM tools WHERE id = ? OR name = ? LIMIT 1;",
                (tool_id_or_name, tool_id_or_name),
            ).fetchone()
            if not row:
                return None
            return self._row_to_tool(row)

    def list_tools(
        self,
        category: Optional[str] = None,
        risk_tier: Optional[str] = None,
        install_state: Optional[str] = None,
        entry_kind: Optional[str] = None,
        limit: int = 100,
    ) -> List[ToolRecord]:
        query = "SELECT * FROM tools WHERE 1=1"
        params: List[Any] = []
        if category:
            query += " AND category = ?"
            params.append(category)
        if risk_tier:
            query += " AND risk_tier = ?"
            params.append(risk_tier)
        if install_state:
            query += " AND install_state = ?"
            params.append(install_state)
        if entry_kind:
            query += " AND entry_kind = ?"
            params.append(entry_kind)
        query += " ORDER BY name ASC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_tool(r) for r in rows]

    def search_tools(self, query: str, limit: int = 20) -> List[ToolRecord]:
        query_clean = query.strip()
        if not query_clean:
            return self.list_tools(limit=limit)

        with self._get_connection() as conn:
            if self.fts5_enabled:
                try:
                    # Sanitize FTS5 query
                    fts_query = " ".join([f'"{part}"*' for part in query_clean.split() if part.isalnum()])
                    if fts_query:
                        rows = conn.execute("""
                            SELECT t.* FROM tools t
                            JOIN tools_fts f ON t.id = f.id
                            WHERE tools_fts MATCH ?
                            ORDER BY rank LIMIT ?;
                        """, (fts_query, limit)).fetchall()
                        if rows:
                            return [self._row_to_tool(r) for r in rows]
                except sqlite3.OperationalError:
                    pass

            # Fallback to LIKE matching across name, summary, category, notes
            like_pat = f"%{query_clean}%"
            rows = conn.execute("""
                SELECT * FROM tools
                WHERE name LIKE ? OR summary LIKE ? OR category LIKE ? OR notes LIKE ?
                ORDER BY CASE WHEN name = ? THEN 0 WHEN name LIKE ? THEN 1 ELSE 2 END, name ASC
                LIMIT ?;
            """, (like_pat, like_pat, like_pat, like_pat, query_clean, f"{query_clean}%", limit)).fetchall()
            return [self._row_to_tool(r) for r in rows]

    def add_doc(self, doc: ToolDoc) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO tool_docs (tool_id, doc_kind, locator, version_scope, sha256, size_bytes, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tool_id, doc_kind, locator) DO UPDATE SET
                    version_scope=excluded.version_scope,
                    sha256=excluded.sha256,
                    size_bytes=excluded.size_bytes,
                    fetched_at=excluded.fetched_at;
            """, (
                doc.tool_id, doc.doc_kind, doc.locator,
                doc.version_scope, doc.sha256, doc.size_bytes, doc.fetched_at,
            ))

    def get_docs(self, tool_id: str) -> List[ToolDoc]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM tool_docs WHERE tool_id = ? ORDER BY doc_kind ASC;", (tool_id,)).fetchall()
            return [
                ToolDoc(
                    tool_id=r["tool_id"],
                    doc_kind=r["doc_kind"],
                    locator=r["locator"],
                    version_scope=r["version_scope"],
                    sha256=r["sha256"],
                    size_bytes=r["size_bytes"],
                    fetched_at=r["fetched_at"],
                )
                for r in rows
            ]

    def add_capability(self, cap: ToolCapability) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO tool_capabilities (tool_id, capability, detail)
                VALUES (?, ?, ?)
                ON CONFLICT(tool_id, capability) DO UPDATE SET detail=excluded.detail;
            """, (cap.tool_id, cap.capability, cap.detail))

    def get_capabilities(self, tool_id: str) -> List[ToolCapability]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM tool_capabilities WHERE tool_id = ?;", (tool_id,)).fetchall()
            return [ToolCapability(tool_id=r["tool_id"], capability=r["capability"], detail=r["detail"]) for r in rows]

    def add_risk_override(self, override: ToolRiskOverride) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO tool_risk_overrides (tool_id, pattern_kind, pattern, tier, note)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tool_id, pattern_kind, pattern) DO UPDATE SET
                    tier=excluded.tier,
                    note=excluded.note;
            """, (
                override.tool_id, override.pattern_kind, override.pattern,
                override.tier.value if isinstance(override.tier, RiskTier) else str(override.tier),
                override.note,
            ))

    def get_risk_overrides(self, tool_id: str) -> List[ToolRiskOverride]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM tool_risk_overrides WHERE tool_id = ?;", (tool_id,)).fetchall()
            return [
                ToolRiskOverride(
                    tool_id=r["tool_id"],
                    pattern_kind=r["pattern_kind"],
                    pattern=r["pattern"],
                    tier=RiskTier(r["tier"]),
                    note=r["note"],
                )
                for r in rows
            ]

    def add_signature(self, sig: ToolSignature) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO tool_signatures (tool_id, call_name, arg_schema_json, curated_by, curated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tool_id, call_name) DO UPDATE SET
                    arg_schema_json=excluded.arg_schema_json,
                    curated_by=excluded.curated_by,
                    curated_at=excluded.curated_at;
            """, (sig.tool_id, sig.call_name, sig.arg_schema_json, sig.curated_by, sig.curated_at))

    def get_signatures(self, tool_id: str) -> List[ToolSignature]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM tool_signatures WHERE tool_id = ?;", (tool_id,)).fetchall()
            return [
                ToolSignature(
                    tool_id=r["tool_id"],
                    call_name=r["call_name"],
                    arg_schema_json=r["arg_schema_json"],
                    curated_by=r["curated_by"],
                    curated_at=r["curated_at"],
                )
                for r in rows
            ]

    def record_scan_run(self, run: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO scan_runs (id, host_id, started_at, finished_at, scanner_version, added, removed, changed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                run["id"], run.get("host_id", "localhost"),
                run["started_at"], run["finished_at"], run.get("scanner_version", "0.1.0"),
                run.get("added", 0), run.get("removed", 0), run.get("changed", 0),
            ))

    def _row_to_tool(self, r: sqlite3.Row) -> ToolRecord:
        return ToolRecord(
            id=r["id"],
            host_id=r["host_id"],
            name=r["name"],
            kind=ToolKind(r["kind"]),
            entry_kind=EntryKind(r["entry_kind"]),
            summary=r["summary"],
            category=r["category"],
            invocation_path=r["invocation_path"],
            resolved_path=r["resolved_path"],
            env_identity=r["env_identity"],
            interpreter=r["interpreter"],
            sha256=r["sha256"],
            version=r["version"],
            package=r["package"],
            package_manager=r["package_manager"],
            install_state=InstallState(r["install_state"]),
            risk_tier=RiskTier(r["risk_tier"]),
            discovered_by=r["discovered_by"],
            first_seen=r["first_seen"],
            last_verified=r["last_verified"],
            notes=r["notes"],
        )
