"""Local FIDO2 WebAuthn Relying Party (RP) Manager for Metis Pyle."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from webauthn import (
    base64url_to_bytes,
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import bytes_to_base64url
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialCreationOptions,
    PublicKeyCredentialDescriptor,
    PublicKeyCredentialRequestOptions,
    UserVerificationRequirement,
)

logger = logging.getLogger("prosopon.auth.webauthn")


class WebAuthnManager:
    """
    Local WebAuthn Relying Party (RP) implementing W3C Web Authentication ceremonies.
    Enables hardware-bound Passkey enrollment and passwordless biometric authentication
    for mobile and desktop companion devices.
    """

    def __init__(
        self,
        rp_id: str = "localhost",
        rp_name: str = "Metis Sovereign Studio",
        allowed_origins: Optional[List[str]] = None,
        challenge_ttl_seconds: int = 120,
    ) -> None:
        self.rp_id = rp_id
        self.rp_name = rp_name
        self.challenge_ttl = challenge_ttl_seconds
        self.allowed_origins = allowed_origins or [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "https://localhost:8000",
            "https://127.0.0.1:8000",
        ]
        # In-memory challenge store: challenge_key -> (challenge_bytes, expires_at_timestamp, metadata)
        self._challenges: Dict[str, Tuple[bytes, float, Dict[str, Any]]] = {}

    def _cleanup_expired_challenges(self) -> None:
        now = time.time()
        expired = [k for k, (_, exp, _) in self._challenges.items() if exp <= now]
        for k in expired:
            self._challenges.pop(k, None)

    def record_challenge(
        self, challenge_key: str, challenge_bytes: bytes, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self._cleanup_expired_challenges()
        self._challenges[challenge_key] = (
            challenge_bytes,
            time.time() + self.challenge_ttl,
            metadata or {},
        )

    def pop_challenge(self, challenge_key: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        self._cleanup_expired_challenges()
        entry = self._challenges.pop(challenge_key, None)
        if not entry:
            return None
        c_bytes, exp, meta = entry
        if exp < time.time():
            return None
        return c_bytes, meta

    # -------------------------------------------------------------------------
    # Registration (Pairing Ceremony)
    # -------------------------------------------------------------------------

    def create_registration_options(
        self,
        user_id: bytes,
        user_name: str,
        user_display_name: str,
        rp_id: Optional[str] = None,
        exclude_credential_ids: Optional[List[bytes]] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate PublicKeyCredentialCreationOptions for registering a new passkey.
        Returns (options_dict, challenge_b64).
        """
        active_rp_id = rp_id or self.rp_id
        exclude_descriptors = (
            [
                PublicKeyCredentialDescriptor(id=cid)
                for cid in exclude_credential_ids
            ]
            if exclude_credential_ids
            else None
        )

        options: PublicKeyCredentialCreationOptions = generate_registration_options(
            rp_id=active_rp_id,
            rp_name=self.rp_name,
            user_id=user_id,
            user_name=user_name,
            user_display_name=user_display_name,
            authenticator_selection=AuthenticatorSelectionCriteria(
                user_verification=UserVerificationRequirement.PREFERRED,
            ),
            exclude_credentials=exclude_descriptors,
        )

        options_json = options_to_json(options)
        parsed = json.loads(options_json)
        challenge_b64 = bytes_to_base64url(options.challenge)

        return parsed, challenge_b64

    def verify_registration(
        self,
        credential_payload: Union[str, Dict[str, Any]],
        expected_challenge: bytes,
        rp_id: Optional[str] = None,
        origin: Optional[str] = None,
    ) -> Tuple[str, str, int, Optional[str]]:
        """
        Verify the WebAuthn registration response.
        Returns (credential_id_b64, public_key_cbor_b64, sign_count, aaguid_b64).
        """
        active_rp_id = rp_id or self.rp_id
        origins = [origin] if origin else list(self.allowed_origins)

        verified = verify_registration_response(
            credential=credential_payload,
            expected_challenge=expected_challenge,
            expected_rp_id=active_rp_id,
            expected_origin=origins,
            require_user_presence=True,
            require_user_verification=False,
        )

        cred_id_b64 = bytes_to_base64url(verified.credential_id)
        pubkey_b64 = bytes_to_base64url(verified.credential_public_key)
        aaguid_b64 = bytes_to_base64url(verified.aaguid) if verified.aaguid else None

        return cred_id_b64, pubkey_b64, verified.sign_count, aaguid_b64

    # -------------------------------------------------------------------------
    # Authentication Ceremony
    # -------------------------------------------------------------------------

    def create_authentication_options(
        self,
        rp_id: Optional[str] = None,
        allow_credential_ids: Optional[List[bytes]] = None,
        user_verification: str = "preferred",
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate PublicKeyCredentialRequestOptions for authenticating an enrolled device.
        Returns (options_dict, challenge_b64).
        """
        active_rp_id = rp_id or self.rp_id
        allow_descriptors = (
            [
                PublicKeyCredentialDescriptor(id=cid)
                for cid in allow_credential_ids
            ]
            if allow_credential_ids
            else None
        )

        uv_req = (
            UserVerificationRequirement.REQUIRED
            if user_verification == "required"
            else UserVerificationRequirement.PREFERRED
        )

        options: PublicKeyCredentialRequestOptions = generate_authentication_options(
            rp_id=active_rp_id,
            allow_credentials=allow_descriptors,
            user_verification=uv_req,
        )

        options_json = options_to_json(options)
        parsed = json.loads(options_json)
        challenge_b64 = bytes_to_base64url(options.challenge)

        return parsed, challenge_b64

    def verify_authentication(
        self,
        credential_payload: Union[str, Dict[str, Any]],
        expected_challenge: bytes,
        credential_public_key_b64: str,
        current_sign_count: int,
        rp_id: Optional[str] = None,
        origin: Optional[str] = None,
        require_user_verification: bool = False,
    ) -> int:
        """
        Verify the WebAuthn authentication response and assert monotonic sign_count.
        Returns new_sign_count.
        """
        active_rp_id = rp_id or self.rp_id
        origins = [origin] if origin else list(self.allowed_origins)
        pubkey_bytes = base64url_to_bytes(credential_public_key_b64)

        verified = verify_authentication_response(
            credential=credential_payload,
            expected_challenge=expected_challenge,
            expected_rp_id=active_rp_id,
            expected_origin=origins,
            credential_public_key=pubkey_bytes,
            credential_current_sign_count=current_sign_count,
            require_user_verification=require_user_verification,
        )

        return verified.new_sign_count
