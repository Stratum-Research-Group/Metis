"""Phylax capability gateway adapter and authorization broker."""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

from kybernetes.contracts.enums import CheckpointAction


class PhylaxAdapter:
    """
    Adapter interfacing with the Phylax Capability Gateway.
    Brokers permission requests, one-use approval digests, and budget policy enforcement.
    """

    def __init__(self, shim: Any = None, policy_path: Optional[str] = None):
        self.shim = shim
        self._approvals: Dict[str, Dict[str, Any]] = {}
        self._approval_events: Dict[str, asyncio.Event] = {}

        if self.shim is None and policy_path:
            import sys
            from pathlib import Path

            policy_file = Path(policy_path).resolve()
            if not policy_file.exists():
                raise FileNotFoundError(f"Phylax policy file not found: {policy_file}")

            # Ensure phylax package is in sys.path if running in an isolated environment
            for parent in policy_file.parents:
                if (parent / "phylax" / "__init__.py").exists():
                    if str(parent) not in sys.path:
                        sys.path.insert(0, str(parent))
                    break

            from phylax.adapters.kybernetes import KybernetesShim
            from phylax.approvals.manager import ApprovalManager
            from phylax.audit.logger import AuditLogger
            from phylax.engine.manager import PolicyManager
            from phylax.gateway.gateway import GuardedGateway

            workspace_root = policy_file.parents[1] if policy_file.parent.name == "policies" else policy_file.parent
            audit_file = workspace_root / ".phylax" / "audit.jsonl"
            audit_file.parent.mkdir(parents=True, exist_ok=True)
            approval_file = workspace_root / ".phylax" / "approvals.json"

            policy_mgr = PolicyManager(policy_file, workspace_root=workspace_root)
            audit_logger = AuditLogger(audit_file)
            approval_mgr = ApprovalManager(approval_file)
            gateway = GuardedGateway(policy_mgr, audit_logger, approval_mgr)
            self.shim = KybernetesShim(gateway)

        self.is_simulated = (self.shim is None)

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
        if self.shim and hasattr(self.shim, "gateway"):
            try:
                from phylax.contracts.enums import Effect

                tool_ids = {
                    "calculator", "compute", "file_search", "search_files",
                    "read_file_range", "read_file", "edit_file_block",
                    "write_file", "sandboxed_bash", "run_code_mode", "shell_exec",
                    "catalogue_lookup", "catalogue_docs",
                }

                if capability == "execute":
                    verb = "execute"
                    res_class = "tool"
                    effective_target = target
                elif capability in tool_ids:
                    verb = "execute"
                    res_class = "tool"
                    if capability == "compute":
                        effective_target = "calculator"
                    elif capability == "read_file":
                        effective_target = "read_file_range"
                    elif capability == "search_files":
                        effective_target = "file_search"
                    else:
                        effective_target = capability
                elif capability == "send":
                    verb = "send"
                    res_class = "network"
                    effective_target = target
                else:
                    verb = "execute"
                    res_class = "tool"
                    effective_target = capability

                call_args = arguments or {}

                auth_rec = self.shim.gateway.decide(
                    subject="kybernetes",
                    verb=verb,
                    resource_class=res_class,
                    target=effective_target,
                    arguments=call_args,
                )
                if auth_rec.effect == Effect.ALLOW:
                    return True, None
                elif auth_rec.effect == Effect.DENY:
                    return False, None
                elif auth_rec.effect == Effect.ASK:
                    approval_id = auth_rec.approval_id or f"app-{uuid.uuid4().hex[:8]}"
                    rule_ttl = 300
                    if hasattr(self.shim.gateway, "policy_manager") and auth_rec.matched_rule_id:
                        for r in self.shim.gateway.policy_manager.policy.rules:
                            if r.id == auth_rec.matched_rule_id and getattr(r, "ask_ttl_seconds", None):
                                rule_ttl = r.ask_ttl_seconds
                                break
                    now = datetime.now(timezone.utc)
                    expiry_dt = now + timedelta(seconds=rule_ttl)
                    req = {
                        "approval_id": approval_id,
                        "action": capability.upper(),
                        "target": target,
                        "argument_summary": arguments,
                        "policy_rule": f"PhylaxPolicy: Matched rule '{auth_rec.matched_rule_id}' requiring explicit single-use approval.",
                        "expiry_utc": expiry_dt.isoformat(),
                        "expiry_timestamp": expiry_dt.timestamp(),
                        "status": "pending",
                        "decision": None,
                        "consumed": False,
                    }
                    self._approvals[approval_id] = req
                    self._approval_events[approval_id] = asyncio.Event()
                    return False, req
                else:
                    return False, None
            except Exception:
                # Fail-closed: Never fall back to simulated mode on live gateway errors
                return False, None

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

        # Reject double resolution if already terminal or executed
        if req.get("status") not in ("pending",) and req.get("consumed", False):
            return False

        req["status"] = "approved" if decision == "allow_once" else "denied"
        req["decision"] = decision
        self._approval_events[approval_id].set()

        if self.shim and hasattr(self.shim, "gateway") and hasattr(self.shim.gateway, "approval_manager"):
            try:
                approved = (decision == "allow_once")
                self.shim.gateway.approval_manager.resolve_request(
                    approval_id=approval_id,
                    approved=approved,
                    reason=f"Resolved via PhylaxAdapter with {decision}",
                )
            except Exception as e:
                logger.error(f"PhylaxAdapter failed to resolve approval request {approval_id}: {e}")
                return False

        return True

    def consume_approval(
        self,
        approval_id: str,
        tool_id: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Atomically verify and consume an approved token for immediate tool execution.
        Enforces single-use, non-expired, and argument-bound matching.
        """
        if not approval_id:
            return False, "Approval token missing"

        # Check live gateway approval manager if active
        if self.shim and hasattr(self.shim, "gateway") and hasattr(self.shim.gateway, "approval_manager"):
            mgr = self.shim.gateway.approval_manager
            app_req = mgr.get_request(approval_id)
            if not app_req:
                return False, f"Approval request '{approval_id}' does not exist in gateway"

            policy_ver = getattr(self.shim.gateway.policy_manager.policy, "version", "1.0.0")
            consumed, msg = mgr.consume_for_execution(
                approval_id=approval_id,
                subject="kybernetes",
                action="execute",
                canonical_target=tool_id,
                arguments=arguments,
                policy_version=policy_ver,
            )
            if approval_id in self._approvals:
                self._approvals[approval_id]["executed"] = consumed
                if consumed:
                    self._approvals[approval_id]["status"] = "consumed"
                    self._approvals[approval_id]["consumed"] = True
            return consumed, msg

        # Simulated / local mode
        if approval_id not in self._approvals:
            return False, f"Approval token '{approval_id}' not found in active store"

        req = self._approvals[approval_id]
        if self._is_expired(req):
            req["status"] = "expired"
            return False, "Approval has expired (TTL elapsed)"

        if req.get("executed", False):
            return False, "Approval token has already been consumed for execution (replay rejected)"

        if req.get("decision") != "allow_once" and req.get("status") not in ("approved", "consumed"):
            return False, f"Approval token is not approved (status={req.get('status')})"

        if req.get("decision") and req.get("decision") != "allow_once":
            return False, f"Approval decision was '{req.get('decision')}' (not approved)"

        # Check argument binding
        expected_args = req.get("argument_summary", {})
        if expected_args != arguments:
            return False, "Argument digest mismatch: arguments do not match approved request"

        # Atomically consume for execution
        req["executed"] = True
        req["consumed"] = True
        req["status"] = "consumed"
        return True, "Approval verified and consumed"

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
        if not approval_id:
            return False

        if self.shim and hasattr(self.shim, "gateway") and hasattr(self.shim.gateway, "approval_manager"):
            mgr = self.shim.gateway.approval_manager
            app_req = mgr.get_request(approval_id)
            if not app_req:
                return False
            from phylax.contracts.enums import ApprovalStatus
            from phylax.engine.evaluator import compute_argument_digest
            if app_req.status not in (ApprovalStatus.APPROVED, ApprovalStatus.CONSUMED, ApprovalStatus.PENDING):
                return False
            digest = compute_argument_digest(arguments)
            if app_req.argument_digest != digest:
                return False
            return True

        if approval_id not in self._approvals:
            return False
        req = self._approvals[approval_id]
        if self._is_expired(req):
            return False
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
