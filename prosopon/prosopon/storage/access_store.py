"""SQLite-backed Access Store for devices, passkeys, sessions, pairing links, routes, and chained audit."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prosopon.auth.scopes import get_scopes_for_preset
from prosopon.contracts.access import (
    AccessAuditRecord,
    DeviceRecord,
    DeviceSessionRecord,
    PairingLinkRecord,
    PasskeyRecord,
    RouteRecord,
)

GENESIS_HASH = "0" * 64


class AccessStore:
    """
    Durable SQLite store for Metis device identities, WebAuthn credentials,
    active sessions, one-time pairing links, and SHA-256 hash-chained audit ledger.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=15.0,
            )
            self._conn.row_factory = sqlite3.Row
            with self._conn:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA foreign_keys=ON;")
            # Ensure 0600 file permissions
            try:
                os.chmod(self.db_path, 0o600)
            except OSError:
                pass
        return self._conn

    def _init_db(self) -> None:
        """Create tables and ensure local host device exists."""
        conn = self._get_connection()
        with self._lock, conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                paired_via TEXT NOT NULL,
                preset TEXT NOT NULL,
                scopes TEXT NOT NULL,
                expires_at_utc TEXT,
                revoked_at_utc TEXT,
                revoked_reason TEXT,
                pinned_ip_range TEXT
            );

            CREATE TABLE IF NOT EXISTS passkeys (
                credential_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
                public_key_cbor TEXT NOT NULL,
                sign_count INTEGER NOT NULL DEFAULT 0,
                aaguid TEXT,
                created_at_utc TEXT NOT NULL,
                last_used_at_utc TEXT NOT NULL,
                transports TEXT
            );

            CREATE TABLE IF NOT EXISTS device_sessions (
                session_id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL,
                method TEXT NOT NULL,
                issued_at_utc TEXT NOT NULL,
                expires_at_utc TEXT NOT NULL,
                revoked_at_utc TEXT,
                route_id TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_token_hash ON device_sessions(token_hash);
            CREATE INDEX IF NOT EXISTS idx_sessions_device_id ON device_sessions(device_id);

            CREATE TABLE IF NOT EXISTS pairing_links (
                id TEXT PRIMARY KEY,
                credential_hash TEXT NOT NULL,
                label TEXT NOT NULL,
                preset TEXT NOT NULL,
                scopes TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                expires_at_utc TEXT NOT NULL,
                consumed_at_utc TEXT,
                consumed_by_device_id TEXT,
                revoked_at_utc TEXT,
                created_by_device_id TEXT
            );

            CREATE TABLE IF NOT EXISTS routes (
                route_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                address TEXT NOT NULL,
                state TEXT NOT NULL,
                learned INTEGER NOT NULL DEFAULT 0,
                priority INTEGER NOT NULL DEFAULT 0,
                last_verified_at_utc TEXT,
                tls TEXT NOT NULL DEFAULT 'none'
            );

            CREATE TABLE IF NOT EXISTS access_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                actor_device_id TEXT,
                action TEXT NOT NULL,
                target_device_id TEXT,
                ip TEXT,
                route_id TEXT,
                detail_json TEXT NOT NULL,
                prev_row_hash TEXT NOT NULL,
                row_hash TEXT NOT NULL
            );
            """)

            # Ensure host-owner device exists
            row = conn.execute("SELECT device_id FROM devices WHERE device_id = 'host_owner'").fetchone()
            if not row:
                now_str = datetime.now(timezone.utc).isoformat()
                owner_scopes = json.dumps(get_scopes_for_preset("owner"))
                conn.execute(
                    """
                    INSERT INTO devices (
                        device_id, label, created_at_utc, paired_via, preset, scopes
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("host_owner", "This computer (host)", now_str, "local", "owner", owner_scopes),
                )
                self._log_audit_locked(
                    conn,
                    action="host_owner_initialized",
                    actor_device_id="host_owner",
                    target_device_id="host_owner",
                    detail={"message": "Default host owner device initialized on startup"},
                )

    # -------------------------------------------------------------------------
    # Audit Logging with Cryptographic Hash Chaining
    # -------------------------------------------------------------------------

    def _compute_audit_hash(
        self,
        prev_row_hash: str,
        ts_utc: str,
        actor_device_id: Optional[str],
        action: str,
        target_device_id: Optional[str],
        ip: Optional[str],
        route_id: Optional[str],
        detail_json: str,
    ) -> str:
        payload = (
            f"{prev_row_hash}:{ts_utc}:{actor_device_id or ''}:{action}:"
            f"{target_device_id or ''}:{ip or ''}:{route_id or ''}:{detail_json}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _log_audit_locked(
        self,
        conn: sqlite3.Connection,
        action: str,
        actor_device_id: Optional[str] = None,
        target_device_id: Optional[str] = None,
        ip: Optional[str] = None,
        route_id: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> str:
        ts_utc = datetime.now(timezone.utc).isoformat()
        detail_json = json.dumps(detail or {}, sort_keys=True)

        last_row = conn.execute(
            "SELECT row_hash FROM access_audit ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = last_row["row_hash"] if last_row else GENESIS_HASH

        row_hash = self._compute_audit_hash(
            prev_row_hash=prev_hash,
            ts_utc=ts_utc,
            actor_device_id=actor_device_id,
            action=action,
            target_device_id=target_device_id,
            ip=ip,
            route_id=route_id,
            detail_json=detail_json,
        )

        conn.execute(
            """
            INSERT INTO access_audit (
                ts_utc, actor_device_id, action, target_device_id, ip, route_id,
                detail_json, prev_row_hash, row_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts_utc,
                actor_device_id,
                action,
                target_device_id,
                ip,
                route_id,
                detail_json,
                prev_hash,
                row_hash,
            ),
        )
        return row_hash

    def log_audit(
        self,
        action: str,
        actor_device_id: Optional[str] = None,
        target_device_id: Optional[str] = None,
        ip: Optional[str] = None,
        route_id: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> str:
        conn = self._get_connection()
        with self._lock, conn:
            return self._log_audit_locked(
                conn,
                action=action,
                actor_device_id=actor_device_id,
                target_device_id=target_device_id,
                ip=ip,
                route_id=route_id,
                detail=detail,
            )

    def verify_audit_chain(self) -> Tuple[bool, int]:
        """
        Verify the integrity of the cryptographic hash chain.
        Returns (is_valid, record_count).
        """
        conn = self._get_connection()
        with self._lock:
            rows = conn.execute("SELECT * FROM access_audit ORDER BY id ASC").fetchall()
            expected_prev = GENESIS_HASH
            for row in rows:
                if row["prev_row_hash"] != expected_prev:
                    return False, row["id"]
                calc_hash = self._compute_audit_hash(
                    prev_row_hash=expected_prev,
                    ts_utc=row["ts_utc"],
                    actor_device_id=row["actor_device_id"],
                    action=row["action"],
                    target_device_id=row["target_device_id"],
                    ip=row["ip"],
                    route_id=row["route_id"],
                    detail_json=row["detail_json"],
                )
                if row["row_hash"] != calc_hash:
                    return False, row["id"]
                expected_prev = row["row_hash"]
            return True, len(rows)

    def list_audit_logs(self, limit: int = 100) -> List[AccessAuditRecord]:
        conn = self._get_connection()
        with self._lock:
            rows = conn.execute(
                "SELECT * FROM access_audit ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                AccessAuditRecord(
                    id=row["id"],
                    ts_utc=row["ts_utc"],
                    actor_device_id=row["actor_device_id"],
                    action=row["action"],
                    target_device_id=row["target_device_id"],
                    ip=row["ip"],
                    route_id=row["route_id"],
                    detail_json=row["detail_json"],
                    prev_row_hash=row["prev_row_hash"],
                    row_hash=row["row_hash"],
                )
                for row in rows
            ]

    # -------------------------------------------------------------------------
    # Devices Management
    # -------------------------------------------------------------------------

    def create_device(
        self,
        device_id: str,
        label: str,
        paired_via: str,
        preset: str = "companion",
        scopes: Optional[List[str]] = None,
        expires_at_utc: Optional[str] = None,
        pinned_ip_range: Optional[str] = None,
        actor_device_id: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> DeviceRecord:
        conn = self._get_connection()
        now_str = datetime.now(timezone.utc).isoformat()
        resolved_scopes = scopes if scopes is not None else get_scopes_for_preset(preset)
        scopes_json = json.dumps(resolved_scopes)

        with self._lock, conn:
            conn.execute(
                """
                INSERT INTO devices (
                    device_id, label, created_at_utc, paired_via, preset,
                    scopes, expires_at_utc, pinned_ip_range
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    device_id,
                    label,
                    now_str,
                    paired_via,
                    preset,
                    scopes_json,
                    expires_at_utc,
                    pinned_ip_range,
                ),
            )
            self._log_audit_locked(
                conn,
                action="device_created",
                actor_device_id=actor_device_id,
                target_device_id=device_id,
                ip=ip,
                detail={"label": label, "paired_via": paired_via, "preset": preset},
            )

        return DeviceRecord(
            device_id=device_id,
            label=label,
            created_at_utc=now_str,
            paired_via=paired_via,
            preset=preset,
            scopes=resolved_scopes,
            expires_at_utc=expires_at_utc,
            pinned_ip_range=pinned_ip_range,
            passkey_count=0,
            is_active=True,
        )

    def get_device(self, device_id: str) -> Optional[DeviceRecord]:
        conn = self._get_connection()
        with self._lock:
            row = conn.execute(
                "SELECT * FROM devices WHERE device_id = ?", (device_id,)
            ).fetchone()
            if not row:
                return None
            passkey_count = conn.execute(
                "SELECT COUNT(*) FROM passkeys WHERE device_id = ?", (device_id,)
            ).fetchone()[0]

            scopes = json.loads(row["scopes"])
            now_iso = datetime.now(timezone.utc).isoformat()
            is_revoked = bool(row["revoked_at_utc"])
            is_expired = bool(row["expires_at_utc"] and row["expires_at_utc"] <= now_iso)
            is_active = not is_revoked and not is_expired

            return DeviceRecord(
                device_id=row["device_id"],
                label=row["label"],
                created_at_utc=row["created_at_utc"],
                paired_via=row["paired_via"],
                preset=row["preset"],
                scopes=scopes,
                expires_at_utc=row["expires_at_utc"],
                revoked_at_utc=row["revoked_at_utc"],
                revoked_reason=row["revoked_reason"],
                pinned_ip_range=row["pinned_ip_range"],
                passkey_count=passkey_count,
                is_active=is_active,
            )

    def list_devices(self, include_revoked: bool = True) -> List[DeviceRecord]:
        conn = self._get_connection()
        with self._lock:
            query = "SELECT * FROM devices ORDER BY created_at_utc DESC"
            rows = conn.execute(query).fetchall()
            devices = []
            now_iso = datetime.now(timezone.utc).isoformat()
            for row in rows:
                is_revoked = bool(row["revoked_at_utc"])
                is_expired = bool(row["expires_at_utc"] and row["expires_at_utc"] <= now_iso)
                if not include_revoked and (is_revoked or is_expired):
                    continue

                passkey_count = conn.execute(
                    "SELECT COUNT(*) FROM passkeys WHERE device_id = ?", (row["device_id"],)
                ).fetchone()[0]

                devices.append(
                    DeviceRecord(
                        device_id=row["device_id"],
                        label=row["label"],
                        created_at_utc=row["created_at_utc"],
                        paired_via=row["paired_via"],
                        preset=row["preset"],
                        scopes=json.loads(row["scopes"]),
                        expires_at_utc=row["expires_at_utc"],
                        revoked_at_utc=row["revoked_at_utc"],
                        revoked_reason=row["revoked_reason"],
                        pinned_ip_range=row["pinned_ip_range"],
                        passkey_count=passkey_count,
                        is_active=(not is_revoked and not is_expired),
                    )
                )
            return devices

    def revoke_device(
        self,
        device_id: str,
        reason: str = "owner",
        actor_device_id: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> bool:
        if device_id == "host_owner":
            # Cannot revoke host_owner
            return False

        conn = self._get_connection()
        now_str = datetime.now(timezone.utc).isoformat()
        with self._lock, conn:
            cur = conn.execute(
                """
                UPDATE devices
                SET revoked_at_utc = ?, revoked_reason = ?
                WHERE device_id = ? AND revoked_at_utc IS NULL
                """,
                (now_str, reason, device_id),
            )
            if cur.rowcount == 0:
                return False

            # Invalidate all active sessions for this device
            conn.execute(
                """
                UPDATE device_sessions
                SET revoked_at_utc = ?
                WHERE device_id = ? AND revoked_at_utc IS NULL
                """,
                (now_str, device_id),
            )

            self._log_audit_locked(
                conn,
                action="device_revoked",
                actor_device_id=actor_device_id,
                target_device_id=device_id,
                ip=ip,
                detail={"reason": reason},
            )
            return True

    def lockdown_all(
        self,
        reason: str = "panic",
        actor_device_id: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        Emergency lockdown: immediately revoke all remote devices and active sessions.
        Retains only host_owner. Returns (revoked_devices_count, revoked_sessions_count).
        """
        conn = self._get_connection()
        now_str = datetime.now(timezone.utc).isoformat()
        with self._lock, conn:
            dev_cur = conn.execute(
                """
                UPDATE devices
                SET revoked_at_utc = ?, revoked_reason = ?
                WHERE device_id != 'host_owner' AND revoked_at_utc IS NULL
                """,
                (now_str, reason),
            )
            revoked_devs = dev_cur.rowcount

            sess_cur = conn.execute(
                """
                UPDATE device_sessions
                SET revoked_at_utc = ?
                WHERE device_id != 'host_owner' AND revoked_at_utc IS NULL
                """,
                (now_str,),
            )
            revoked_sessions = sess_cur.rowcount

            # Also revoke all active pairing links
            conn.execute(
                """
                UPDATE pairing_links
                SET revoked_at_utc = ?
                WHERE revoked_at_utc IS NULL AND consumed_at_utc IS NULL
                """,
                (now_str,),
            )

            self._log_audit_locked(
                conn,
                action="panic_lockdown",
                actor_device_id=actor_device_id,
                detail={
                    "reason": reason,
                    "revoked_devices": revoked_devs,
                    "revoked_sessions": revoked_sessions,
                },
                ip=ip,
            )
            return revoked_devs, revoked_sessions

    # -------------------------------------------------------------------------
    # WebAuthn Passkeys Management
    # -------------------------------------------------------------------------

    def register_passkey(
        self,
        credential_id: str,
        device_id: str,
        public_key_cbor: str,
        sign_count: int = 0,
        aaguid: Optional[str] = None,
        transports: Optional[List[str]] = None,
        actor_device_id: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> PasskeyRecord:
        conn = self._get_connection()
        now_str = datetime.now(timezone.utc).isoformat()
        transports_json = json.dumps(transports or [])

        with self._lock, conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO passkeys (
                    credential_id, device_id, public_key_cbor, sign_count,
                    aaguid, created_at_utc, last_used_at_utc, transports
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    credential_id,
                    device_id,
                    public_key_cbor,
                    sign_count,
                    aaguid,
                    now_str,
                    now_str,
                    transports_json,
                ),
            )
            self._log_audit_locked(
                conn,
                action="passkey_registered",
                actor_device_id=actor_device_id or device_id,
                target_device_id=device_id,
                ip=ip,
                detail={"credential_id": credential_id[:16] + "..."},
            )

        return PasskeyRecord(
            credential_id=credential_id,
            device_id=device_id,
            public_key_cbor=public_key_cbor,
            sign_count=sign_count,
            aaguid=aaguid,
            created_at_utc=now_str,
            last_used_at_utc=now_str,
            transports=transports or [],
        )

    def get_passkey(self, credential_id: str) -> Optional[PasskeyRecord]:
        conn = self._get_connection()
        with self._lock:
            row = conn.execute(
                "SELECT * FROM passkeys WHERE credential_id = ?", (credential_id,)
            ).fetchone()
            if not row:
                return None
            return PasskeyRecord(
                credential_id=row["credential_id"],
                device_id=row["device_id"],
                public_key_cbor=row["public_key_cbor"],
                sign_count=row["sign_count"],
                aaguid=row["aaguid"],
                created_at_utc=row["created_at_utc"],
                last_used_at_utc=row["last_used_at_utc"],
                transports=json.loads(row["transports"] or "[]"),
            )

    def get_passkeys_for_device(self, device_id: str) -> List[PasskeyRecord]:
        conn = self._get_connection()
        with self._lock:
            rows = conn.execute(
                "SELECT * FROM passkeys WHERE device_id = ? ORDER BY created_at_utc DESC",
                (device_id,),
            ).fetchall()
            return [
                PasskeyRecord(
                    credential_id=row["credential_id"],
                    device_id=row["device_id"],
                    public_key_cbor=row["public_key_cbor"],
                    sign_count=row["sign_count"],
                    aaguid=row["aaguid"],
                    created_at_utc=row["created_at_utc"],
                    last_used_at_utc=row["last_used_at_utc"],
                    transports=json.loads(row["transports"] or "[]"),
                )
                for row in rows
            ]

    def update_passkey_sign_count(
        self,
        credential_id: str,
        new_sign_count: int,
        ip: Optional[str] = None,
    ) -> bool:
        conn = self._get_connection()
        now_str = datetime.now(timezone.utc).isoformat()
        with self._lock, conn:
            cur = conn.execute(
                """
                UPDATE passkeys
                SET sign_count = ?, last_used_at_utc = ?
                WHERE credential_id = ?
                """,
                (new_sign_count, now_str, credential_id),
            )
            return cur.rowcount > 0

    # -------------------------------------------------------------------------
    # Device Sessions
    # -------------------------------------------------------------------------

    def create_session(
        self,
        device_id: str,
        method: str = "bearer",
        validity_hours: int = 72,
        route_id: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> Tuple[str, DeviceSessionRecord]:
        """
        Mint a new session token for an active device.
        Returns (raw_bearer_token, session_record). Raw token is only returned once.
        """
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        session_id = secrets.token_hex(16)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=validity_hours)

        now_str = now.isoformat()
        expires_str = expires_at.isoformat()

        conn = self._get_connection()
        with self._lock, conn:
            conn.execute(
                """
                INSERT INTO device_sessions (
                    session_id, device_id, token_hash, method, issued_at_utc,
                    expires_at_utc, route_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    device_id,
                    token_hash,
                    method,
                    now_str,
                    expires_str,
                    route_id,
                ),
            )
            self._log_audit_locked(
                conn,
                action="session_created",
                actor_device_id=device_id,
                target_device_id=device_id,
                ip=ip,
                route_id=route_id,
                detail={"session_id": session_id, "method": method},
            )

        record = DeviceSessionRecord(
            session_id=session_id,
            device_id=device_id,
            token_hash=token_hash,
            method=method,
            issued_at_utc=now_str,
            expires_at_utc=expires_str,
            route_id=route_id,
        )
        return raw_token, record

    def validate_session_token(
        self, raw_token: str
    ) -> Optional[Tuple[DeviceRecord, DeviceSessionRecord]]:
        """Validate raw bearer token against active unrevoked session and device."""
        if not raw_token:
            return None
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

        conn = self._get_connection()
        with self._lock:
            row = conn.execute(
                """
                SELECT s.*, d.label, d.created_at_utc as d_created, d.paired_via,
                       d.preset, d.scopes, d.expires_at_utc as d_expires,
                       d.revoked_at_utc as d_revoked, d.revoked_reason as d_reason,
                       d.pinned_ip_range
                FROM device_sessions s
                JOIN devices d ON s.device_id = d.device_id
                WHERE s.token_hash = ? AND s.revoked_at_utc IS NULL
                """,
                (token_hash,),
            ).fetchone()

            if not row:
                return None

            now_iso = datetime.now(timezone.utc).isoformat()
            if row["expires_at_utc"] <= now_iso:
                return None
            if row["d_revoked"] or (row["d_expires"] and row["d_expires"] <= now_iso):
                return None

            passkey_count = conn.execute(
                "SELECT COUNT(*) FROM passkeys WHERE device_id = ?", (row["device_id"],)
            ).fetchone()[0]

            dev = DeviceRecord(
                device_id=row["device_id"],
                label=row["label"],
                created_at_utc=row["d_created"],
                paired_via=row["paired_via"],
                preset=row["preset"],
                scopes=json.loads(row["scopes"]),
                expires_at_utc=row["d_expires"],
                revoked_at_utc=row["d_revoked"],
                revoked_reason=row["d_reason"],
                pinned_ip_range=row["pinned_ip_range"],
                passkey_count=passkey_count,
                is_active=True,
            )
            sess = DeviceSessionRecord(
                session_id=row["session_id"],
                device_id=row["device_id"],
                token_hash=row["token_hash"],
                method=row["method"],
                issued_at_utc=row["issued_at_utc"],
                expires_at_utc=row["expires_at_utc"],
                revoked_at_utc=row["revoked_at_utc"],
                route_id=row["route_id"],
            )
            return dev, sess

    # -------------------------------------------------------------------------
    # Pairing Links
    # -------------------------------------------------------------------------

    def create_pairing_link(
        self,
        label: str = "Phone Client",
        preset: str = "companion",
        scopes: Optional[List[str]] = None,
        validity_seconds: int = 300,
        created_by_device_id: Optional[str] = "host_owner",
        ip: Optional[str] = None,
    ) -> Tuple[str, str, PairingLinkRecord]:
        """
        Create a single-use pairing link/QR code credential.
        Returns (pairing_id, pairing_secret, record).
        """
        pairing_id = f"pair_{secrets.token_hex(8)}"
        secret = secrets.token_urlsafe(32)
        cred_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=validity_seconds)
        now_str = now.isoformat()
        expires_str = expires_at.isoformat()

        resolved_scopes = scopes if scopes is not None else get_scopes_for_preset(preset)
        scopes_json = json.dumps(resolved_scopes)

        conn = self._get_connection()
        with self._lock, conn:
            conn.execute(
                """
                INSERT INTO pairing_links (
                    id, credential_hash, label, preset, scopes,
                    created_at_utc, expires_at_utc, created_by_device_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pairing_id,
                    cred_hash,
                    label,
                    preset,
                    scopes_json,
                    now_str,
                    expires_str,
                    created_by_device_id,
                ),
            )
            self._log_audit_locked(
                conn,
                action="pair_created",
                actor_device_id=created_by_device_id,
                target_device_id=None,
                ip=ip,
                detail={"pairing_id": pairing_id, "preset": preset, "label": label},
            )

        record = PairingLinkRecord(
            id=pairing_id,
            credential_hash=cred_hash,
            label=label,
            preset=preset,
            scopes=resolved_scopes,
            created_at_utc=now_str,
            expires_at_utc=expires_str,
            created_by_device_id=created_by_device_id,
        )
        return pairing_id, secret, record

    def redeem_pairing_link(
        self,
        pairing_id: str,
        secret: str,
        device_id: str,
        ip: Optional[str] = None,
    ) -> Optional[PairingLinkRecord]:
        """Redeem a single-use pairing link. Atomic verification and consumption."""
        cred_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        now_str = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        with self._lock, conn:
            row = conn.execute(
                """
                SELECT * FROM pairing_links
                WHERE id = ? AND credential_hash = ?
                  AND consumed_at_utc IS NULL AND revoked_at_utc IS NULL
                """,
                (pairing_id, cred_hash),
            ).fetchone()

            if not row:
                self._log_audit_locked(
                    conn,
                    action="pair_denied",
                    target_device_id=device_id,
                    ip=ip,
                    detail={"pairing_id": pairing_id, "reason": "invalid_or_consumed"},
                )
                return None

            if row["expires_at_utc"] <= now_str:
                self._log_audit_locked(
                    conn,
                    action="pair_denied",
                    target_device_id=device_id,
                    ip=ip,
                    detail={"pairing_id": pairing_id, "reason": "expired"},
                )
                return None

            conn.execute(
                """
                UPDATE pairing_links
                SET consumed_at_utc = ?, consumed_by_device_id = ?
                WHERE id = ?
                """,
                (now_str, device_id, pairing_id),
            )

            self._log_audit_locked(
                conn,
                action="pair_redeemed",
                actor_device_id=device_id,
                target_device_id=device_id,
                ip=ip,
                detail={"pairing_id": pairing_id},
            )

            return PairingLinkRecord(
                id=row["id"],
                credential_hash=row["credential_hash"],
                label=row["label"],
                preset=row["preset"],
                scopes=json.loads(row["scopes"]),
                created_at_utc=row["created_at_utc"],
                expires_at_utc=row["expires_at_utc"],
                consumed_at_utc=now_str,
                consumed_by_device_id=device_id,
                created_by_device_id=row["created_by_device_id"],
            )

    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------

    def register_route(
        self,
        route_id: str,
        kind: str,
        address: str,
        state: str = "active",
        learned: bool = False,
        priority: int = 0,
        tls: str = "none",
    ) -> RouteRecord:
        now_str = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        with self._lock, conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO routes (
                    route_id, kind, address, state, learned, priority,
                    last_verified_at_utc, tls
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    route_id,
                    kind,
                    address,
                    state,
                    1 if learned else 0,
                    priority,
                    now_str,
                    tls,
                ),
            )
        return RouteRecord(
            route_id=route_id,
            kind=kind,
            address=address,
            state=state,
            learned=learned,
            priority=priority,
            last_verified_at_utc=now_str,
            tls=tls,
        )

    def list_routes(self) -> List[RouteRecord]:
        conn = self._get_connection()
        with self._lock:
            rows = conn.execute("SELECT * FROM routes ORDER BY priority ASC").fetchall()
            return [
                RouteRecord(
                    route_id=row["route_id"],
                    kind=row["kind"],
                    address=row["address"],
                    state=row["state"],
                    learned=bool(row["learned"]),
                    priority=row["priority"],
                    last_verified_at_utc=row["last_verified_at_utc"],
                    tls=row["tls"],
                )
                for row in rows
            ]
