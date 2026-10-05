"""Single-owner device pairing and token authentication manager."""

import json
import os
import secrets
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Set
from pydantic import BaseModel, Field


class PairedDevice(BaseModel):
    """Metadata for an authorized client device."""
    token: str = Field(..., description="Cryptographically secure bearer token")
    device_name: str = Field(default="Phone Client", description="Reported or friendly client name")
    paired_at_utc: str = Field(..., description="Timestamp when pairing succeeded")
    expires_at_utc: str = Field(..., description="Timestamp when token expires")
    ip_address: Optional[str] = Field(default=None, description="IP address observed during pairing")


class PairingManager:
    """
    Manages single-owner PIN pairing, active bearer tokens, and revocation.
    Prevents unauthorized LAN access to chat history and Phylax approvals.
    Coordinates across processes via owner-only (0600) files.
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
        self._tokens_file_mtime: float = 0.0

        # Load persisted tokens
        self._paired_tokens: Dict[str, PairedDevice] = self._load_tokens()

    def generate_pairing_pin(self, validity_seconds: int = 300) -> str:
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
        }
        self._save_pin_record(new_record)
        return pin

    def verify_pin_and_pair(self, pin: str, device_name: str = "Mobile Browser", ip_address: Optional[str] = None) -> Optional[str]:
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
            now_dt = datetime.now(timezone.utc)
            exp_dt = now_dt + timedelta(hours=self.expiry_hours)

            paired = PairedDevice(
                token=token,
                device_name=device_name,
                paired_at_utc=now_dt.isoformat(),
                expires_at_utc=exp_dt.isoformat(),
                ip_address=ip_address,
            )
            # Sync existing tokens from disk first
            self._paired_tokens = self._load_tokens()
            self._paired_tokens[token] = paired
            self._save_tokens()

            # Invalidate PIN after successful one-time use
            self._current_pin = None
            self._pin_expires_at = 0.0
            self._failed_attempts = 0
            self._save_pin_record(None)
            return token
        else:
            failed = (record.get("failed_attempts", 0) if record else self._failed_attempts) + 1
            new_lockout = (now + 60.0) if failed >= 5 else 0.0
            if record:
                record["failed_attempts"] = failed
                record["lockout_until"] = new_lockout
                self._save_pin_record(record)
            self._failed_attempts = failed
            self._lockout_until = new_lockout
            return None

    def generate_loopback_token(self, device_name: str = "Local Browser") -> str:
        """Generate and register a token for local loopback sessions without PIN entry."""
        token = secrets.token_urlsafe(32)
        now_dt = datetime.now(timezone.utc)
        exp_dt = now_dt + timedelta(hours=self.expiry_hours)

        paired = PairedDevice(
            token=token,
            device_name=device_name,
            paired_at_utc=now_dt.isoformat(),
            expires_at_utc=exp_dt.isoformat(),
            ip_address="127.0.0.1",
        )
        self._paired_tokens = self._load_tokens()
        self._paired_tokens[token] = paired
        self._save_tokens()
        return token

    def validate_token(self, token: Optional[str]) -> bool:
        """Verify whether token exists and has not expired, reloading live state from disk if changed."""
        if not token:
            return False

        # Live sync: reload tokens if disk state changed or removed
        if self.tokens_file.exists():
            try:
                mtime = self.tokens_file.stat().st_mtime
                if mtime > self._tokens_file_mtime:
                    self._paired_tokens = self._load_tokens()
            except Exception:
                pass
        else:
            if self._paired_tokens:
                self._paired_tokens = {}
                self._tokens_file_mtime = 0.0

        if token not in self._paired_tokens:
            return False

        paired = self._paired_tokens[token]
        try:
            exp = datetime.fromisoformat(paired.expires_at_utc)
            if datetime.now(timezone.utc) > exp:
                del self._paired_tokens[token]
                self._save_tokens()
                return False
            return True
        except Exception:
            return False

    def revoke_token(self, token: str) -> bool:
        """Revoke a specific device token and write immediately to disk."""
        self._paired_tokens = self._load_tokens()
        if token in self._paired_tokens:
            del self._paired_tokens[token]
            self._save_tokens()
            return True
        return False

    def revoke_all(self) -> int:
        """Revoke all active device tokens and invalidate any pending PIN."""
        self._paired_tokens = self._load_tokens()
        count = len(self._paired_tokens)
        self._paired_tokens.clear()
        self._save_tokens()

        self._current_pin = None
        self._pin_expires_at = 0.0
        self._failed_attempts = 0
        self._save_pin_record(None)
        return count

    def get_paired_devices(self) -> Dict[str, PairedDevice]:
        if self.tokens_file.exists():
            try:
                mtime = self.tokens_file.stat().st_mtime
                if mtime > self._tokens_file_mtime:
                    self._paired_tokens = self._load_tokens()
            except Exception:
                pass
        return dict(self._paired_tokens)

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
            self._tokens_file_mtime = 0.0
            return {}
        try:
            stat = self.tokens_file.stat()
            self._tokens_file_mtime = stat.st_mtime
            with open(self.tokens_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {tok: PairedDevice.model_validate(p) for tok, p in data.items()}
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
            self._tokens_file_mtime = self.tokens_file.stat().st_mtime
        except Exception:
            pass
