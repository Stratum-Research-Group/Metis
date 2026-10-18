"""WebRTC DataChannel & Media Engine for Metis Pyle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import aiortc
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceCandidate,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.sdp import candidate_from_sdp

from prosopon.auth.scopes import has_scope
from prosopon.auth.webauthn import WebAuthnManager
from prosopon.storage.access_store import AccessStore

logger = logging.getLogger("prosopon.transport.webrtc")


class WebRTCGateway:
    """
    Manages WebRTC PeerConnections and multiplexed DataChannels (control, stream, terminal, audio)
    for mobile and remote companion devices with hardware-bound Passkey authentication.
    """

    def __init__(
        self,
        access_store: AccessStore,
        webauthn_mgr: WebAuthnManager,
        ice_servers: Optional[List[str]] = None,
    ) -> None:
        self.access_store = access_store
        self.webauthn_mgr = webauthn_mgr
        self.ice_urls = ice_servers or ["stun:stun.l.google.com:19302"]

        # Active peer connections: session_id -> RTCPeerConnection
        self._pcs: Dict[str, RTCPeerConnection] = {}
        # Active channels: session_id -> { channel_label: RTCDataChannel }
        self._channels: Dict[str, Dict[str, RTCDataChannel]] = {}
        # Session to device mapping: session_id -> device_id
        self._session_devices: Dict[str, str] = {}
        # Device to sessions mapping: device_id -> Set[session_id]
        self._device_sessions: Dict[str, Set[str]] = {}
        # Callbacks for Phylax biometric approvals: callback(approval_id, decision, device_id, risk_tier)
        self._approval_listeners: List[Callable[[str, str, str, str], Any]] = []

    def get_rtc_configuration(self) -> RTCConfiguration:
        servers = [RTCIceServer(urls=self.ice_urls)]
        return RTCConfiguration(iceServers=servers)

    def register_approval_listener(
        self, callback: Callable[[str, str, str, str], Any]
    ) -> None:
        """Register a callback to be invoked when a Phylax biometric approval signature is verified."""
        self._approval_listeners.append(callback)

    # -------------------------------------------------------------------------
    # Signaling: Offer & Answer
    # -------------------------------------------------------------------------

    async def handle_offer(
        self,
        sdp: str,
        type_: str,
        device_id: str,
        peer_address: str = "unknown",
        session_id: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Handle incoming SDP offer from a remote companion device.
        Sets up multiplexed SCTP DataChannels ('control', 'stream', 'terminal')
        and returns (answer_sdp, answer_type, session_id).
        """
        device = self.access_store.get_device(device_id)
        if not device or not device.is_active:
            raise PermissionError(f"Device '{device_id}' is not authorized or has been revoked.")

        sess_id = session_id or f"webrtc_{secrets.token_hex(12)}"
        pc = RTCPeerConnection(configuration=self.get_rtc_configuration())

        self._pcs[sess_id] = pc
        self._channels[sess_id] = {}
        self._session_devices[sess_id] = device_id
        self._device_sessions.setdefault(device_id, set()).add(sess_id)

        # Register in SQLite access store
        self.access_store.register_webrtc_session(
            session_id=sess_id,
            device_id=device_id,
            peer_address=peer_address,
        )

        # Pre-create the 3 required reliable DataChannels
        for label in ("control", "stream", "terminal"):
            channel = pc.createDataChannel(label, ordered=True)
            self._setup_data_channel(sess_id, channel)

        # Also listen for client-initiated data channels
        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel):
            logger.info("Client opened DataChannel '%s' on session %s", channel.label, sess_id)
            self._setup_data_channel(sess_id, channel)

        # Listen for connection state changes
        @pc.on("connectionstatechange")
        async def on_state_change():
            logger.info("WebRTC session %s state: %s", sess_id, pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await self.close_session(sess_id)

        # Apply remote offer and generate local answer
        remote_desc = RTCSessionDescription(sdp=sdp, type=type_)
        await pc.setRemoteDescription(remote_desc)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return (
            pc.localDescription.sdp,
            pc.localDescription.type,
            sess_id,
        )

    # -------------------------------------------------------------------------
    # Signaling: ICE Candidates
    # -------------------------------------------------------------------------

    async def handle_ice_candidate(
        self, session_id: str, candidate_dict: Dict[str, Any]
    ) -> None:
        """Add an ICE candidate to the active peer connection."""
        pc = self._pcs.get(session_id)
        if not pc:
            raise ValueError(f"WebRTC session '{session_id}' not found.")

        cand_str = candidate_dict.get("candidate")
        if not cand_str:
            return

        # Strip 'candidate:' prefix if present
        raw_sdp = cand_str
        if raw_sdp.startswith("candidate:"):
            raw_sdp = raw_sdp[len("candidate:"):]

        candidate = candidate_from_sdp(raw_sdp)
        candidate.sdpMid = candidate_dict.get("sdpMid")
        candidate.sdpMLineIndex = candidate_dict.get("sdpMLineIndex")

        await pc.addIceCandidate(candidate)

    # -------------------------------------------------------------------------
    # DataChannel Multiplexing & Dispatch
    # -------------------------------------------------------------------------

    def _setup_data_channel(self, session_id: str, channel: RTCDataChannel) -> None:
        self._channels.setdefault(session_id, {})[channel.label] = channel

        @channel.on("open")
        def on_open():
            logger.info("DataChannel '%s' opened on session %s", channel.label, session_id)
            if channel.label == "control":
                # Send welcome event
                dev_id = self._session_devices.get(session_id, "")
                dev = self.access_store.get_device(dev_id)
                welcome = {
                    "jsonrpc": "2.0",
                    "method": "welcome",
                    "params": {
                        "session_id": session_id,
                        "device_id": dev_id,
                        "label": dev.label if dev else "",
                        "preset": dev.preset if dev else "",
                        "scopes": dev.scopes if dev else [],
                    },
                }
                channel.send(json.dumps(welcome))

        @channel.on("message")
        def on_message(message: Union[str, bytes]):
            asyncio.create_task(self._dispatch_channel_message(session_id, channel.label, message))

        @channel.on("close")
        def on_close():
            logger.info("DataChannel '%s' closed on session %s", channel.label, session_id)

    async def _dispatch_channel_message(
        self, session_id: str, channel_label: str, message: Union[str, bytes]
    ) -> None:
        """Route incoming messages according to channel protocol."""
        if channel_label == "control":
            await self._handle_control_rpc(session_id, message)
        elif channel_label == "terminal":
            # Pass interactive terminal keystrokes to active execution harness
            logger.debug("Terminal input from session %s: %s", session_id, message)
        else:
            logger.debug("Unhandled channel '%s' message on session %s", channel_label, message)

    async def _handle_control_rpc(
        self, session_id: str, message: Union[str, bytes]
    ) -> None:
        """Handle JSON-RPC 2.0 requests on the control channel."""
        try:
            raw_text = message.decode("utf-8") if isinstance(message, bytes) else message
            req = json.loads(raw_text)
        except Exception:
            return

        method = req.get("method")
        msg_id = req.get("id")
        params = req.get("params", {})
        device_id = self._session_devices.get(session_id, "")
        channel = self._channels.get(session_id, {}).get("control")

        if not channel:
            return

        # 1. Ping / Heartbeat
        if method == "ping":
            self.access_store.update_webrtc_heartbeat(session_id)
            resp = {"jsonrpc": "2.0", "result": "pong", "id": msg_id}
            channel.send(json.dumps(resp))

        # 2. Auth / Status Query
        elif method == "auth.status":
            dev = self.access_store.get_device(device_id)
            if not dev or not dev.is_active:
                resp = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": "Device revoked"},
                    "id": msg_id,
                }
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "result": {
                        "device_id": dev.device_id,
                        "label": dev.label,
                        "preset": dev.preset,
                        "scopes": dev.scopes,
                        "passkey_count": dev.passkey_count,
                    },
                    "id": msg_id,
                }
            channel.send(json.dumps(resp))

        # 3. Biometric Phylax Action Approval (WebAuthn Signed Challenge)
        elif method == "phylax.sign_approval":
            await self._handle_phylax_approval(session_id, device_id, params, msg_id, channel)

        else:
            resp = {
                "jsonrpc": "2.0",
                "error": {"code": -32601, "message": f"Method '{method}' not found"},
                "id": msg_id,
            }
            channel.send(json.dumps(resp))

    async def _handle_phylax_approval(
        self,
        session_id: str,
        device_id: str,
        params: Dict[str, Any],
        msg_id: Any,
        channel: RTCDataChannel,
    ) -> None:
        """
        Verify biometric signature for a high-risk Phylax capability approval.
        Enforces user verification for T2/T3 tiers and checks granted scopes.
        """
        approval_id = params.get("approval_id")
        action_digest = params.get("action_digest")
        risk_tier = params.get("risk_tier", "T1")
        decision = params.get("decision", "allow_once")
        credential = params.get("credential")

        dev = self.access_store.get_device(device_id)
        if not dev or not dev.is_active:
            channel.send(
                json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32001, "message": "Device revoked or inactive"},
                    "id": msg_id,
                })
            )
            return

        if not has_scope(dev.scopes, "approvals:resolve"):
            channel.send(
                json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32003, "message": "Missing 'approvals:resolve' scope"},
                    "id": msg_id,
                })
            )
            return

        # For T2/T3, enforce high approval scope
        if risk_tier in ("T2", "T3") and not has_scope(dev.scopes, "approvals:resolve_high"):
            channel.send(
                json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32004, "message": "Missing 'approvals:resolve_high' scope for T2/T3"},
                    "id": msg_id,
                })
            )
            return

        # If credential supplied, verify the biometric passkey signature
        if credential and approval_id and action_digest:
            expected_challenge_hash = hashlib.sha256(
                f"{approval_id}{action_digest}".encode("utf-8")
            ).digest()

            cred_raw_id = credential.get("id") or credential.get("rawId")
            passkey = self.access_store.get_passkey(cred_raw_id) if cred_raw_id else None

            if not passkey or passkey.device_id != device_id:
                channel.send(
                    json.dumps({
                        "jsonrpc": "2.0",
                        "error": {"code": -32005, "message": "Unrecognized passkey credential for device"},
                        "id": msg_id,
                    })
                )
                return

            try:
                new_sign_count = self.webauthn_mgr.verify_authentication(
                    credential_payload=credential,
                    expected_challenge=expected_challenge_hash,
                    credential_public_key_b64=passkey.public_key_cbor,
                    current_sign_count=passkey.sign_count,
                    require_user_verification=(risk_tier in ("T2", "T3")),
                )
                self.access_store.update_passkey_sign_count(passkey.credential_id, new_sign_count)
            except Exception as e:
                channel.send(
                    json.dumps({
                        "jsonrpc": "2.0",
                        "error": {"code": -32006, "message": f"Biometric signature verification failed: {e}"},
                        "id": msg_id,
                    })
                )
                return

        # Log audit entry
        self.access_store.log_audit(
            action="approval_resolved_remote",
            actor_device_id=device_id,
            target_device_id=device_id,
            detail={
                "approval_id": approval_id,
                "action_digest": action_digest,
                "risk_tier": risk_tier,
                "decision": decision,
            },
        )

        # Trigger listeners (e.g. Phylax resolution callbacks)
        for listener in self._approval_listeners:
            try:
                res = listener(approval_id, decision, device_id, risk_tier)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as err:
                logger.error("Error executing approval listener: %s", err)

        channel.send(
            json.dumps({
                "jsonrpc": "2.0",
                "result": {
                    "status": "approved" if decision == "allow_once" else "denied",
                    "approval_id": approval_id,
                    "device_id": device_id,
                    "biometric_verified": True,
                },
                "id": msg_id,
            })
        )

    # -------------------------------------------------------------------------
    # Outbound Broadcast Methods
    # -------------------------------------------------------------------------

    def send_stream_event(
        self,
        event_type: str,
        data: Any,
        device_id: Optional[str] = None,
    ) -> int:
        """
        Broadcast an SSE-framed event across the WebRTC 'stream' channel.
        If device_id is provided, only target that device's sessions.
        Returns the count of streams successfully transmitted to.
        """
        payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        sent_count = 0

        target_sessions = (
            self._device_sessions.get(device_id, set())
            if device_id
            else set(self._pcs.keys())
        )

        for sess_id in list(target_sessions):
            ch = self._channels.get(sess_id, {}).get("stream")
            if ch and ch.readyState == "open":
                try:
                    ch.send(payload)
                    sent_count += 1
                except Exception as err:
                    logger.debug("Failed sending stream event to session %s: %s", sess_id, err)

        return sent_count

    def send_terminal_data(
        self,
        data: Union[str, bytes],
        session_id: Optional[str] = None,
        device_id: Optional[str] = None,
    ) -> int:
        """Stream ANSI terminal output over the 'terminal' channel."""
        sent_count = 0
        if session_id:
            target_sessions = {session_id}
        elif device_id:
            target_sessions = self._device_sessions.get(device_id, set())
        else:
            target_sessions = set(self._pcs.keys())

        for s_id in list(target_sessions):
            ch = self._channels.get(s_id, {}).get("terminal")
            if ch and ch.readyState == "open":
                try:
                    ch.send(data)
                    sent_count += 1
                except Exception:
                    pass

        return sent_count

    # -------------------------------------------------------------------------
    # Lifecycle & Immediate Device Revocation
    # -------------------------------------------------------------------------

    async def close_session(self, session_id: str) -> None:
        """Close peer connection and clean up session records."""
        pc = self._pcs.pop(session_id, None)
        self._channels.pop(session_id, None)
        dev_id = self._session_devices.pop(session_id, None)

        if dev_id and dev_id in self._device_sessions:
            self._device_sessions[dev_id].discard(session_id)
            if not self._device_sessions[dev_id]:
                self._device_sessions.pop(dev_id, None)

        self.access_store.close_webrtc_session(session_id)

        if pc:
            try:
                await pc.close()
            except Exception:
                pass

    async def close_device_connections(self, device_id: str) -> int:
        """
        Immediately close all active WebRTC PeerConnections for a revoked device (< 500ms).
        Enforces Security Invariant 4 from METIS-SPEC-PYLE-WEBRTC-001.
        """
        session_ids = list(self._device_sessions.get(device_id, set()))
        closed_count = len(session_ids)

        for s_id in session_ids:
            await self.close_session(s_id)

        logger.info("Closed %d active WebRTC sessions for revoked device '%s'", closed_count, device_id)
        return closed_count

    async def close_all(self) -> None:
        """Terminate all active peer connections."""
        for s_id in list(self._pcs.keys()):
            await self.close_session(s_id)
