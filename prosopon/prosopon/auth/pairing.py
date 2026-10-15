"""Single-owner device pairing and token authentication manager."""

import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field


class PairedDevice(BaseModel):
    """Metadata for an authorized client device."""
    token_hash: str = Field(..., description="SHA-256 hash of cryptographically secure bearer token")
    token: Optional[str] = Field(default=None, description="Optional legacy raw token")
    role: str = Field(default="chat", description="Token scope/role: 'chat' or 'admin'")
    device_name: str = Field(default="Phone Client", description="Reported or friendly client name")
    paired_at_utc: str = Field(..., description="Timestamp when pairing succeeded")
    expires_at_utc: str = Field(..., description="Timestamp when token expires")
    ip_address: Optional[str] = Field(default=None, description="IP address observed during pairing")


class PairingManager:
    """
    Manages single-owner PIN pairing, active bearer tokens, and revocation.
    Prevents unauthorized LAN access to chat history and Phylax approvals.
    Coordinates across processes via owner-only (0600) files and token hashes.
    """
    def __init__(self, data_dir: Path, expiry_hours: int = 72):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_file = self.data_dir / "paired_tokens.json"
        self.pin_file = self.data_dir / "pairing_pin.json"
        self.expiry_hours = expiry_hours
        
        # In-memory ephemeral pairing code state
        self._current_pin: Optional[str] = None
        self._pin_expires_at: float = 0.0
        self._failed_attempts: int = 0
        self._lockout_until: float = 0.0
        self._tokens_file_sig: Optional[Tuple[int, int, int]] = None

        # Load persisted tokens
        self._paired_tokens: Dict[str, PairedDevice] = self._load_tokens()

    @staticmethod
    def _hash_token(token: str) -> str:
        """Compute SHA-256 hash of a bearer token."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def generate_pairing_pin(self, validity_seconds: int = 300, role: str = "admin") -> str:
        """Generate a fresh 6-digit pairing PIN valid for 5 minutes and coordinate across processes."""
        pin = f"{secrets.randbelow(1000000):06d}"
        now = time.time()
        expires_at = now + validity_seconds

        # Preserve active lockout if still running
        active_lockout = self._lockout_until if now < self._lockout_until else 0.0
        record = self._load_pin_record()
        if record and now < record.get("lockout_until", 0.0):
            active_lockout = max(active_lockout, record.get("lockout_until", 0.0))
        
        self._current_pin = pin
        self._pin_expires_at = expires_at
        self._failed_attempts = 0
        self._lockout_until = active_lockout

        new_record = {
            "pin": pin,
            "expires_at": expires_at,
            "failed_attempts": 0,
            "lockout_until": active_lockout,
            "role": role,
        }
        self._save_pin_record(new_record)
        return pin

    def verify_pin_and_pair(
        self,
        pin: str,
        device_name: str = "Mobile Browser",
        ip_address: Optional[str] = None,
        role: str = "chat",
    ) -> Optional[str]:
        """
        Validate entered PIN and return a newly minted bearer token if successful.
        Coordinates with persisted pin state so CLI-issued PINs work on running servers.
        """
        now = time.time()
        record = self._load_pin_record()
        
        lockout = record.get("lockout_until", 0.0) if record else self._lockout_until
        if now < lockout:
            return None  # Rate-limited due to brute-force attempts

        expected_pin = record.get("pin") if record else self._current_pin
        expires_at = record.get("expires_at", 0.0) if record else self._pin_expires_at

        if not expected_pin or now > expires_at:
            if record and now > expires_at:
                self._save_pin_record(None)
            return None  # Expired or uninitialized

        if secrets.compare_digest(str(expected_pin).strip(), str(pin).strip()):
            # Correct PIN -> generate token
            token = secrets.token_urlsafe(32)
            tok_hash = self._hash_token(token)
            now_dt = datetime.now(timezone.utc)
            exp_dt = now_dt + timedelta(hours=self.expiry_hours)

            assigned_role = record.get("role", role) if record else role
            paired = PairedDevice(
                token_hash=tok_hash,
                role=assigned_role,
                device_name=device_name,
                paired_at_utc=now_dt.isoformat(),
                expires_at_utc=exp_dt.isoformat(),
                ip_address=ip_address,
            )
            self._paired_tokens = self._load_tokens()
            self._paired_tokens[tok_hash] = paired
            self._save_tokens()

            # Clean up used PIN
            self._current_pin = None
            self._pin_expires_at = 0.0
            self._failed_attempts = 0
            self._save_pin_record(None)
            return token

        # Failed attempt -> record and potentially trigger lockout
        self._failed_attempts += 1
        fails = (record.get("failed_attempts", 0) if record else 0) + 1
        if fails >= 5:
            lockout_time = now + 900.0  # 15-minute brute-force lockout
            self._lockout_until = lockout_time
            if record:
                record["lockout_until"] = lockout_time
                record["failed_attempts"] = fails
                self._save_pin_record(record)
        else:
            if record:
                record["failed_attempts"] = fails
                self._save_pin_record(record)
        return None

    def generate_loopback_token(self, device_name: str = "Local Browser", role: str = "chat") -> str:
        """Generate a token for local loopback sessions without PIN entry."""
        token = secrets.token_urlsafe(32)
        tok_hash = self._hash_token(token)
        now_dt = datetime.now(timezone.utc)
        exp_dt = now_dt + timedelta(hours=self.expiry_hours)

        paired = PairedDevice(
            token_hash=tok_hash,
            role=role,
            device_name=device_name,
            paired_at_utc=now_dt.isoformat(),
            expires_at_utc=exp_dt.isoformat(),
            ip_address="127.0.0.1",
        )
        self._paired_tokens = self._load_tokens()
        self._paired_tokens[tok_hash] = paired
        self._save_tokens()
        return token

    def generate_admin_token(self, device_name: str = "CLI Admin") -> str:
        """Generate an admin-privileged bearer token (used via terminal)."""
        return self.generate_loopback_token(device_name=device_name, role="admin")

    def validate_token(self, token: Optional[str], required_role: Optional[str] = None) -> bool:
        """Verify whether token exists, has not expired, and satisfies required_role if specified."""
        if not token:
            return False

        # Live sync: reload tokens if disk state changed or removed
        self._sync_disk_state()

        tok_hash = self._hash_token(token)
        paired = self._paired_tokens.get(tok_hash) or self._paired_tokens.get(token)
        if not paired:
            return False

        if required_role == "admin" and paired.role != "admin":
            return False

        key_to_use = tok_hash if tok_hash in self._paired_tokens else token
        try:
            exp = datetime.fromisoformat(paired.expires_at_utc)
            if datetime.now(timezone.utc) > exp:
                del self._paired_tokens[key_to_use]
                self._save_tokens()
                return False
            return True
        except Exception:
            return False

    def revoke_token(self, token: str) -> bool:
        """Revoke a specific device token and write immediately to disk."""
        self._sync_disk_state()
        tok_hash = self._hash_token(token)
        
        target_key = None
        if tok_hash in self._paired_tokens:
            target_key = tok_hash
        elif token in self._paired_tokens:
            target_key = token

        if target_key:
            del self._paired_tokens[target_key]
            self._save_tokens()
            return True
        return False

    def revoke_all(self) -> int:
        """Revoke all active device tokens and invalidate any pending PIN."""
        self._sync_disk_state()
        count = len(self._paired_tokens)
        self._paired_tokens.clear()
        self._save_tokens()

        self._current_pin = None
        self._pin_expires_at = 0.0
        self._failed_attempts = 0
        self._save_pin_record(None)
        return count

    def get_paired_devices(self) -> Dict[str, PairedDevice]:
        self._sync_disk_state()
        return dict(self._paired_tokens)

    def _sync_disk_state(self) -> None:
        """Check file signature (mtime_ns, inode, size) and reload if modified."""
        if self.tokens_file.exists():
            try:
                stat = self.tokens_file.stat()
                current_sig = (stat.st_mtime_ns, stat.st_ino, stat.st_size)
                if current_sig != self._tokens_file_sig:
                    self._paired_tokens = self._load_tokens()
            except Exception:
                pass
        else:
            if self._paired_tokens or self._tokens_file_sig is not None:
                self._paired_tokens = {}
                self._tokens_file_sig = None

    def _load_pin_record(self) -> Optional[Dict[str, Any]]:
        if not self.pin_file.exists():
            return None
        try:
            with open(self.pin_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_pin_record(self, record: Optional[Dict[str, Any]]) -> None:
        if record is None:
            if self.pin_file.exists():
                try:
                    self.pin_file.unlink()
                except Exception:
                    pass
            return
        tmp = self.pin_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        try:
            os.chmod(tmp, 0o600)
        except Exception:
            pass
        tmp.replace(self.pin_file)
        try:
            os.chmod(self.pin_file, 0o600)
        except Exception:
            pass

    def _load_tokens(self) -> Dict[str, PairedDevice]:
        if not self.tokens_file.exists():
            self._tokens_file_sig = None
            return {}
        try:
            stat = self.tokens_file.stat()
            self._tokens_file_sig = (stat.st_mtime_ns, stat.st_ino, stat.st_size)
            with open(self.tokens_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = {}
            for k, p in data.items():
                if "token_hash" not in p and "token" in p:
                    p["token_hash"] = self._hash_token(p["token"])
                if "role" not in p:
                    p["role"] = "chat"
                result[k] = PairedDevice.model_validate(p)
            return result
        except Exception:
            return {}

    def _save_tokens(self) -> None:
        data = {tok: p.model_dump() for tok, p in self._paired_tokens.items()}
        tmp = self.tokens_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        try:
            os.chmod(tmp, 0o600)
        except Exception:
            pass
        tmp.replace(self.tokens_file)
        try:
            os.chmod(self.tokens_file, 0o600)
            stat = self.tokens_file.stat()
            self._tokens_file_sig = (stat.st_mtime_ns, stat.st_ino, stat.st_size)
        except Exception:
            pass
