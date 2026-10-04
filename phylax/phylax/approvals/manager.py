"""Approval lifecycle manager.

Handles generation, persistent storage, expiration checks, owner resolution,
and single-use atomic consumption of ASK requests.
"""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import secrets
import threading
from typing import Any

from phylax.contracts.enums import ApprovalStatus
from phylax.contracts.approval import ApprovalRequest
from phylax.contracts.decision import AuthorizationRecord
from phylax.engine.evaluator import compute_argument_digest


class ApprovalManager:
    """Manages durable approval requests for human-in-the-loop decisions."""

    def __init__(self, storage_path: Path | str | None = None):
        if storage_path:
            self.storage_path = Path(storage_path)
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.storage_path = None
        
        self._lock = threading.Lock()
        self._requests: dict[str, ApprovalRequest] = {}
        if self.storage_path and self.storage_path.exists():
            self._load_storage()

    def _load_storage(self) -> None:
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for item in data:
                req = ApprovalRequest.model_validate(item)
                self._requests[req.approval_id] = req
        except Exception:
            # If storage corrupt or empty, start fresh
            pass

    def _save_storage(self) -> None:
        if not self.storage_path:
            return
        # Atomic write to temp file then rename
        tmp_file = self.storage_path.with_suffix(".tmp")
        data = [r.model_dump() for r in self._requests.values()]
        tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_file.replace(self.storage_path)

    def create_request(
        self,
        auth_record: AuthorizationRecord,
        action: str,
        target: str,
        arguments: dict[str, Any] | None,
        ttl_seconds: int = 300,
        turn_id: str | None = None,
        trace_id: str | None = None,
    ) -> ApprovalRequest:
        """Create a new pending approval request bound to arguments and target."""
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(seconds=ttl_seconds)
        arg_digest = compute_argument_digest(arguments)
        
        # Build brief argument summary
        args = arguments or {}
        summary_parts = [f"{k}={v!r}" for k, v in list(args.items())[:3]]
        arg_summary = ", ".join(summary_parts) if summary_parts else "no arguments"

        req = ApprovalRequest(
            approval_id=auth_record.approval_id or f"appr-{secrets.token_hex(6)}",
            turn_id=turn_id or auth_record.trace_id,
            trace_id=trace_id or auth_record.trace_id,
            subject=auth_record.subject,
            action=action,
            target=target,
            argument_summary=arg_summary,
            argument_digest=arg_digest,
            raw_arguments=args,
            policy_rule=auth_record.matched_rule_id or "ask_rule",
            policy_version=auth_record.policy_version,
            created_at_utc=now.isoformat(),
            expiry_utc=expiry.isoformat(),
            status=ApprovalStatus.PENDING,
            nonce=secrets.token_hex(16),
            consumed_at_utc=None,
        )

        with self._lock:
            self._requests[req.approval_id] = req
            self._save_storage()

        return req

    def get_request(self, approval_id: str) -> ApprovalRequest | None:
        """Fetch request, automatically updating expired status if TTL elapsed."""
        with self._lock:
            req = self._requests.get(approval_id)
            if not req:
                return None

            now = datetime.now(timezone.utc)
            expiry = datetime.fromisoformat(req.expiry_utc)
            if req.status == ApprovalStatus.PENDING and now > expiry:
                req.status = ApprovalStatus.EXPIRED
                req.resolution_reason = "Approval timed out (TTL expired)"
                self._save_storage()

            return req

    def resolve_request(
        self,
        approval_id: str,
        approved: bool,
        reason: str | None = None,
        resolved_by: str = "owner",
    ) -> ApprovalRequest:
        """Resolve a pending request to APPROVED or REJECTED."""
        with self._lock:
            req = self._requests.get(approval_id)
            if not req:
                raise KeyError(f"Approval request '{approval_id}' not found")

            now = datetime.now(timezone.utc)
            expiry = datetime.fromisoformat(req.expiry_utc)
            if now > expiry:
                req.status = ApprovalStatus.EXPIRED
                req.resolution_reason = "Approval timed out before resolution"
                self._save_storage()
                raise TimeoutError(f"Approval request '{approval_id}' has expired")

            if req.status != ApprovalStatus.PENDING:
                raise ValueError(
                    f"Cannot resolve request '{approval_id}' with status '{req.status.value}'"
                )

            req.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            req.resolution_reason = reason or (f"Approved by {resolved_by}" if approved else f"Rejected by {resolved_by}")
            self._save_storage()
            return req

    def consume_for_execution(
        self,
        approval_id: str,
        subject: str,
        action: str,
        canonical_target: str,
        arguments: dict[str, Any] | None,
        policy_version: str,
    ) -> tuple[bool, str]:
        """Atomically verify and consume an approved request for immediate execution.
        
        Returns (success: bool, message: str).
        """
        with self._lock:
            req = self._requests.get(approval_id)
            if not req:
                return False, f"Approval request '{approval_id}' does not exist"

            # 1. Check expiration
            now = datetime.now(timezone.utc)
            expiry = datetime.fromisoformat(req.expiry_utc)
            if now > expiry:
                req.status = ApprovalStatus.EXPIRED
                req.resolution_reason = "Expired before execution"
                self._save_storage()
                return False, "Approval has expired (TTL elapsed)"

            # 2. Check status
            if req.status == ApprovalStatus.CONSUMED:
                return False, "Approval token has already been consumed (replay rejected)"
            if req.status == ApprovalStatus.REJECTED:
                return False, f"Approval was rejected: {req.resolution_reason}"
            if req.status == ApprovalStatus.PENDING:
                return False, "Approval is still pending owner confirmation"
            if req.status != ApprovalStatus.APPROVED:
                return False, f"Invalid approval status '{req.status.value}'"

            # 3. Check subject binding
            if req.subject != subject:
                return False, f"Subject mismatch: request was for '{req.subject}', but caller is '{subject}'"

            # 4. Check action / verb binding
            if req.action != action:
                return False, f"Action mismatch: approved '{req.action}', but attempted '{action}'"

            # 5. Check target binding
            if req.target != canonical_target:
                return False, f"Target mismatch: approved '{req.target}', but attempted '{canonical_target}'"

            # 6. Check argument digest binding (Anti-Argument-Swap)
            current_digest = compute_argument_digest(arguments)
            if current_digest != req.argument_digest:
                return False, (
                    f"Argument digest mismatch! Approved digest was '{req.argument_digest[:10]}...', "
                    f"but arguments provided have digest '{current_digest[:10]}...'"
                )

            # 7. Check policy version
            if req.policy_version != policy_version:
                return False, (
                    f"Policy version changed from '{req.policy_version}' to '{policy_version}' "
                    "since approval was granted"
                )

            # Atomic consumption
            req.status = ApprovalStatus.CONSUMED
            req.consumed_at_utc = now.isoformat()
            self._save_storage()
            return True, "Approval verified and consumed"

    def list_pending(self) -> list[ApprovalRequest]:
        """List all currently pending and unexpired approvals."""
        with self._lock:
            now = datetime.now(timezone.utc)
            pending: list[ApprovalRequest] = []
            for req in self._requests.values():
                expiry = datetime.fromisoformat(req.expiry_utc)
                if req.status == ApprovalStatus.PENDING:
                    if now > expiry:
                        req.status = ApprovalStatus.EXPIRED
                    else:
                        pending.append(req)
            return pending
