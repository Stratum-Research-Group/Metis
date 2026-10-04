"""Unit tests for device pairing and token authentication."""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

from prosopon.auth.pairing import PairingManager


def test_pairing_pin_generation_and_verification(tmp_path):
    mgr = PairingManager(tmp_path)
    pin = mgr.generate_pairing_pin(validity_seconds=300)
    assert len(pin) == 6
    assert pin.isdigit()

    # Wrong PIN fails
    bad_token = mgr.verify_pin_and_pair("000000" if pin != "000000" else "111111")
    assert bad_token is None

    # Correct PIN succeeds and returns token
    token = mgr.verify_pin_and_pair(pin)
    assert token is not None
    assert len(token) > 20

    # PIN cannot be reused
    second_attempt = mgr.verify_pin_and_pair(pin)
    assert second_attempt is None

    # Token is valid
    assert mgr.validate_token(token) is True


def test_token_revocation(tmp_path):
    mgr = PairingManager(tmp_path)
    pin = mgr.generate_pairing_pin()
    token = mgr.verify_pin_and_pair(pin)
    assert mgr.validate_token(token) is True

    # Revoke token
    revoked = mgr.revoke_token(token)
    assert revoked is True
    assert mgr.validate_token(token) is False


def test_brute_force_lockout(tmp_path):
    mgr = PairingManager(tmp_path)
    mgr.generate_pairing_pin()

    for _ in range(5):
        mgr.verify_pin_and_pair("999999")

    # 6th attempt is blocked by lockout
    pin = mgr.generate_pairing_pin()
    assert mgr.verify_pin_and_pair(pin) is None


def test_tls_generation(tmp_path):
    from prosopon.auth.tls import generate_self_signed_cert
    cert_path, key_path = generate_self_signed_cert(tmp_path)
    assert cert_path.exists()
    assert key_path.exists()
    assert cert_path.read_text().startswith("-----BEGIN CERTIFICATE-----")
    assert key_path.read_text().startswith("-----BEGIN RSA PRIVATE KEY-----")

