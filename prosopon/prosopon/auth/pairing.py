"""Single-owner device pairing and token authentication manager."""

import json
import secrets
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Set
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
    """
    def __init__(self, data_dir: Path, expiry_hours: int = 72):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tokens_file = self.data_dir / "paired_tokens.json"
        self.expiry_hours = expiry_hours
        
        # In-memory ephemeral pairing code (e.g. 6-digit PIN)
        self._current_pin: Optional[str] = None
        self._pin_expires_at: float = 0.0
        self._failed_attempts: int = 0
        self._lockout_until: float = 0.0

        # Load persisted tokens
        self._paired_tokens: Dict[str, PairedDevice] = self._load_tokens()

    def generate_pairing_pin(self, validity_seconds: int = 300) -> str:
        """Generate a fresh 6-digit pairing PIN valid for 5 minutes."""
        self._current_pin = f"{secrets.randbelow(1000000):06d}"
        self._pin_expires_at = time.time() + validity_seconds
        self._failed_attempts = 0
        return self._current_pin

    def verify_pin_and_pair(self, pin: str, device_name: str = "Mobile Browser", ip_address: Optional[str] = None) -> Optional[str]:
        """
        Validate entered PIN and return a newly minted bearer token if successful.
        """
        now = time.time()
        if now < self._lockout_until:
            return None  # Rate-limited due to brute-force attempts

        if not self._current_pin or now > self._pin_expires_at:
            return None  # Expired or uninitialized

        if secrets.compare_digest(self._current_pin.strip(), pin.strip()):
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
            self._paired_tokens[token] = paired
            self._save_tokens()

            # Invalidate PIN after successful one-time use
            self._current_pin = None
            self._pin_expires_at = 0.0
            return token
        else:
            self._failed_attempts += 1
            if self._failed_attempts >= 5:
                self._lockout_until = now + 60.0  # Lock out for 60 seconds
            return None

    def validate_token(self, token: Optional[str]) -> bool:
        """Verify whether token exists and has not expired."""
        if not token or token not in self._paired_tokens:
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
        """Revoke a specific device token."""
        if token in self._paired_tokens:
            del self._paired_tokens[token]
            self._save_tokens()
            return True
        return False

    def revoke_all(self) -> int:
        """Revoke all active device tokens."""
        count = len(self._paired_tokens)
        self._paired_tokens.clear()
        self._save_tokens()
        return count

    def get_paired_devices(self) -> Dict[str, PairedDevice]:
        return dict(self._paired_tokens)

    def _load_tokens(self) -> Dict[str, PairedDevice]:
        if not self.tokens_file.exists():
            return {}
        try:
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
        tmp.replace(self.tokens_file)
