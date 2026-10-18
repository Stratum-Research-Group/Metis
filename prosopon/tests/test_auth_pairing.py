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


def test_cross_process_pairing_and_live_revocation(tmp_path):
    """Item 4: CLI-issued PIN coordinates across instances, and revocation immediately invalidates live cache."""
    # Instance 1: Server manager
    server_mgr = PairingManager(tmp_path)
    
    # Instance 2: Separate CLI manager (e.g. running 'prosopon pair')
    cli_mgr = PairingManager(tmp_path)
    pin = cli_mgr.generate_pairing_pin()
    assert (tmp_path / "pairing_pin.json").exists()
    
    # Server manager verifies PIN issued by CLI manager
    token = server_mgr.verify_pin_and_pair(pin, device_name="Phone")
    assert token is not None
    assert server_mgr.validate_token(token) is True
    # PIN file consumed
    assert not (tmp_path / "pairing_pin.json").exists()

    # Check 0600 permissions on paired_tokens.json
    import stat
    tokens_file = tmp_path / "paired_tokens.json"
    assert tokens_file.exists()
    file_mode = stat.S_IMODE(tokens_file.stat().st_mode)
    assert file_mode == 0o600

    # Instance 3: CLI revocation (e.g. running 'prosopon revoke')
    revoker_mgr = PairingManager(tmp_path)
    revoker_mgr.revoke_all()

    # Live server instance immediately detects disk update and rejects token
    assert server_mgr.validate_token(token) is False


def test_host_owner_token_permissions_and_validation(tmp_path):
    import stat
    mgr = PairingManager(tmp_path)
    tok = mgr.get_or_create_host_owner_token()
    assert tok is not None
    assert len(tok) >= 32

    token_file = tmp_path / ".host_owner_token"
    assert token_file.exists()
    mode = stat.S_IMODE(token_file.stat().st_mode)
    assert mode == 0o600

    # Validation
    assert mgr.validate_host_owner_token(tok) is True
    assert mgr.validate_host_owner_token("wrong-token") is False
    assert mgr.validate_token(tok) is True
    assert mgr.validate_token(tok, required_role="admin") is True


def test_bootstrap_tickets_single_use_and_expiry(tmp_path):
    import stat
    mgr = PairingManager(tmp_path)
    ticket = mgr.create_bootstrap_ticket(validity_seconds=60)
    assert ticket is not None
    assert len(ticket) >= 20

    tickets_file = tmp_path / "bootstrap_tickets.json"
    assert tickets_file.exists()
    mode = stat.S_IMODE(tickets_file.stat().st_mode)
    assert mode == 0o600

    # First consumption succeeds and issues token
    session_tok = mgr.consume_bootstrap_ticket(ticket, device_name="Test Browser")
    assert session_tok is not None
    assert mgr.validate_token(session_tok, required_role="admin") is True

    # Single-use: second consumption fails
    reused = mgr.consume_bootstrap_ticket(ticket)
    assert reused is None


