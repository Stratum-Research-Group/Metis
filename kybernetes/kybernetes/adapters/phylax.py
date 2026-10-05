"""Phylax capability gateway adapter and authorization broker."""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from kybernetes.contracts.enums import CheckpointAction


class PhylaxAdapter:
    """
    Adapter interfacing with the Phylax Capability Gateway.
    Brokers permission requests, one-use approval digests, and budget policy enforcement.
    """

    def __init__(self, shim: Any = None):
        self.shim = shim
        self.is_simulated = (shim is None)
        self._approvals: Dict[str, Dict[str, Any]] = {}
        self._approval_events: Dict[str, asyncio.Event] = {}

    @property
    def _pending_approvals(self) -> Dict[str, Dict[str, Any]]:
        # Backwards compatibility property
        return self._approvals

    def _is_expired(self, req: Dict[str, Any]) -> bool:
        expiry_ts = req.get("expiry_timestamp")
        if expiry_ts is None:
            expiry_str = req.get("expiry_utc")
            if not expiry_str:
                return True
            try:
                expiry_dt = datetime.fromisoformat(expiry_str)
                if expiry_dt.tzinfo is None:
                    expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                expiry_ts = expiry_dt.timestamp()
                req["expiry_timestamp"] = expiry_ts
            except Exception:
                return True
        return datetime.now(timezone.utc).timestamp() >= expiry_ts

    def is_action_permitted(
        self,
        capability: str,
        target: str,
        arguments: Dict[str, Any],
        custom_expiry_utc: Optional[str] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check whether an action is pre-authorized or requires an interactive user approval.
        Returns: (is_allowed_immediately, approval_request_dict_if_needed)
        """
        # Read-only actions and safe calculations do not require interactive prompt
        if capability in ("compute", "read_file") and not any(kw in target.lower() for kw in ["secret", "passwd", "shadow", "key"]):
            return True, None

        # Actions involving file write, mutation, or outbound send require interactive approval
        approval_id = f"app-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        if custom_expiry_utc is not None:
            expiry_utc = custom_expiry_utc
            try:
                expiry_dt = datetime.fromisoformat(custom_expiry_utc)
                if expiry_dt.tzinfo is None:
                    expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
                expiry_ts = expiry_dt.timestamp()
            except Exception:
                expiry_ts = now.timestamp()
        else:
            expiry_dt = now + timedelta(minutes=5)
            expiry_utc = expiry_dt.isoformat()
            expiry_ts = expiry_dt.timestamp()

        req = {
            "approval_id": approval_id,
            "action": capability.upper(),
            "target": target,
            "argument_summary": arguments,
            "policy_rule": f"PhylaxPolicy: Action '{capability}' requires explicit single-use user authorization.",
            "expiry_utc": expiry_utc,
            "expiry_timestamp": expiry_ts,
            "status": "pending",
            "decision": None,
            "consumed": False,
        }
        self._approvals[approval_id] = req
        self._approval_events[approval_id] = asyncio.Event()

        return False, req

    async def wait_for_approval(self, approval_id: str, timeout_seconds: float = 30.0) -> str:
        """
        Wait for user approval decision ('allow_once' or 'deny') and atomically consume it.
        Enforces one-use consumption (replays are denied) and rejects expired approvals.
        """
        if approval_id not in self._approvals or approval_id not in self._approval_events:
            return "deny"

        req = self._approvals[approval_id]

        if self._is_expired(req):
            req["status"] = "expired"
            req["decision"] = "deny"
            return "deny"

        event = self._approval_events[approval_id]
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            req["status"] = "expired"
            req["decision"] = "deny"
            return "deny"

        # Check expiry after wait
        if self._is_expired(req):
            req["status"] = "expired"
            req["decision"] = "deny"
            return "deny"

        # Check atomic consumption: strictly single-use
        if req.get("consumed", False):
            return "deny"

        # Atomically consume the approval
        req["consumed"] = True
        decision = req.get("decision", "deny")
        req["status"] = "consumed"
        return decision

    def resolve_approval(self, approval_id: str, decision: str) -> bool:
        """
        Resolve a pending approval from UI / API.
        Rejects double resolution, expired approvals, and uninitialized requests.
        """
        if approval_id not in self._approvals or approval_id not in self._approval_events:
            return False

        req = self._approvals[approval_id]

        # Check expiry at resolution
        if self._is_expired(req):
            req["status"] = "expired"
            req["decision"] = "deny"
            self._approval_events[approval_id].set()
            return False

        # Reject double resolution if already terminal
        if req.get("status") != "pending":
            return False

        req["status"] = "approved" if decision == "allow_once" else "denied"
        req["decision"] = decision
        self._approval_events[approval_id].set()
        return True

    def verify_action_execution(
        self,
        approval_id: str,
        capability: str,
        target: str,
        arguments: Dict[str, Any],
    ) -> bool:
        """
        Verify that an approval matches the exact action, target, and arguments being executed.
        Prevents capability tampering between approval and execution.
        """
        if approval_id not in self._approvals:
            return False
        req = self._approvals[approval_id]
        if req.get("action") != capability.upper():
            return False
        if req.get("target") != target:
            return False
        if req.get("argument_summary") != arguments:
            return False
        return True

    def enforce_budget_guardrail(
        self,
        task_id: str,
        checkpoint_action: CheckpointAction,
        reason: str,
    ) -> bool:
        """
        Send bounded budget enforcement decision to Phylax.
        Phylax receives ONLY the bounded decision and reason, without private task content.
        Returns True if execution may proceed, False if blocked.
        """
        if checkpoint_action == CheckpointAction.BLOCK:
            return False
        return True
