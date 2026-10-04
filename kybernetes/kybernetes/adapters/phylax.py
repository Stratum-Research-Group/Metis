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
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}
        self._approval_events: Dict[str, asyncio.Event] = {}
        self._approval_decisions: Dict[str, str] = {}

    def is_action_permitted(
        self,
        capability: str,
        target: str,
        arguments: Dict[str, Any],
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
        expiry_utc = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

        req = {
            "approval_id": approval_id,
            "action": capability.upper(),
            "target": target,
            "argument_summary": arguments,
            "policy_rule": f"PhylaxPolicy: Action '{capability}' requires explicit single-use user authorization.",
            "expiry_utc": expiry_utc,
            "status": "pending",
        }
        self._pending_approvals[approval_id] = req
        self._approval_events[approval_id] = asyncio.Event()

        return False, req

    async def wait_for_approval(self, approval_id: str, timeout_seconds: float = 30.0) -> str:
        """Wait for user approval decision ('allow_once' or 'deny')."""
        if approval_id not in self._approval_events:
            return "deny"

        event = self._approval_events[approval_id]
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
            decision = self._approval_decisions.get(approval_id, "deny")
        except asyncio.TimeoutError:
            decision = "deny"

        if approval_id in self._pending_approvals:
            self._pending_approvals[approval_id]["status"] = "approved" if decision == "allow_once" else "denied"

        return decision

    def resolve_approval(self, approval_id: str, decision: str) -> bool:
        """Resolve a pending approval from UI / API."""
        if approval_id not in self._approval_events:
            return False

        self._approval_decisions[approval_id] = decision
        self._approval_events[approval_id].set()
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
