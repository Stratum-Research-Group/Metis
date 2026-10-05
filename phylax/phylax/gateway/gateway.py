"""Guarded capability gateway providing point-of-action authorization and execution."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
import uuid

from phylax.approvals.manager import ApprovalManager
from phylax.audit.logger import AuditLogger, PhylaxAuditFailureError
from phylax.contracts.enums import Effect, ResourceClass, Verb
from phylax.contracts.decision import AuthorizationRecord, ExecutionReceipt
from phylax.engine.manager import PolicyManager


class GuardedGateway:
    """Central gateway mediating capability authorization and execution."""

    def __init__(
        self,
        policy_manager: PolicyManager,
        audit_logger: AuditLogger,
        approval_manager: ApprovalManager | None = None,
    ):
        self.policy_manager = policy_manager
        self.audit_logger = audit_logger
        self.approval_manager = approval_manager or ApprovalManager()

    def decide(
        self,
        subject: str,
        verb: Verb | str,
        resource_class: ResourceClass | str,
        target: str,
        arguments: dict[str, Any] | None = None,
        trace_id: str | None = None,
        turn_id: str | None = None,
    ) -> AuthorizationRecord:
        """Evaluate policy, record pending approval if ASK, and log decision fail-closed."""
        evaluator = self.policy_manager.evaluator
        auth_rec = evaluator.decide(
            subject=subject,
            verb=verb,
            resource_class=resource_class,
            target=target,
            arguments=arguments,
            trace_id=trace_id,
        )

        # If ASK, create pending approval request in approval manager
        if auth_rec.effect == Effect.ASK:
            # Find matched rule's TTL if available
            rule_ttl = 300
            if auth_rec.matched_rule_id:
                for r in self.policy_manager.policy.rules:
                    if r.id == auth_rec.matched_rule_id:
                        rule_ttl = r.ask_ttl_seconds
                        break

            appr_req = self.approval_manager.create_request(
                auth_record=auth_rec,
                action=auth_rec.verb.value,
                target=auth_rec.canonical_resource,
                arguments=arguments,
                ttl_seconds=rule_ttl,
                turn_id=turn_id,
                trace_id=trace_id,
            )
            # Ensure approval_id is set
            auth_rec = auth_rec.model_copy(update={"approval_id": appr_req.approval_id})

        # Fail-closed audit logging
        try:
            self.audit_logger.log_event(
                auth_record=auth_rec,
                receipt=None,
                context=arguments,
            )
        except PhylaxAuditFailureError:
            # Return fail-closed denial
            return AuthorizationRecord(
                decision_id=f"dec-failclosed-{uuid.uuid4().hex[:8]}",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                subject=auth_rec.subject,
                verb=auth_rec.verb,
                resource_class=auth_rec.resource_class,
                canonical_resource=auth_rec.canonical_resource,
                effect=Effect.DENY,
                matched_rule_id="audit_failure_fail_closed",
                policy_version=auth_rec.policy_version,
                reason="Denied due to audit logging failure (Fail-Closed)",
                trace_id=trace_id,
                approval_id=None,
                argument_digest=auth_rec.argument_digest,
            )

        return auth_rec

    def authorize_and_execute(
        self,
        subject: str,
        verb: Verb | str,
        resource_class: ResourceClass | str,
        target: str,
        action_fn: Callable[..., Any],
        arguments: dict[str, Any] | None = None,
        approval_id: str | None = None,
        trace_id: str | None = None,
        turn_id: str | None = None,
    ) -> tuple[Any, ExecutionReceipt]:
        """Authorize at point-of-action and execute immediately upon approval.
        
        If approval_id is provided, verifies and consumes approval atomically.
        """
        args = arguments or {}
        norm_verb = Verb(verb) if isinstance(verb, str) else verb
        norm_rc = ResourceClass(resource_class) if isinstance(resource_class, str) else resource_class
        policy_ver = self.policy_manager.policy.version

        if approval_id:
            # Pre-approved path: verify and consume approval token atomically
            consumed, msg = self.approval_manager.consume_for_execution(
                approval_id=approval_id,
                subject=subject,
                action=norm_verb.value,
                canonical_target=target,
                arguments=args,
                policy_version=policy_ver,
            )
            if not consumed:
                auth_rec = AuthorizationRecord(
                    decision_id=f"dec-{uuid.uuid4().hex[:12]}",
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    subject=subject,
                    verb=norm_verb,
                    resource_class=norm_rc,
                    canonical_resource=target,
                    effect=Effect.DENY,
                    matched_rule_id="invalid_approval",
                    policy_version=policy_ver,
                    reason=f"Approval rejected: {msg}",
                    trace_id=trace_id,
                    argument_digest=None,
                )
                self.audit_logger.log_event(auth_rec, None, args)
                raise PermissionError(f"Phylax authorization denied: {msg}")

            # Approval consumed successfully, construct authorization record
            auth_rec = AuthorizationRecord(
                decision_id=f"dec-{uuid.uuid4().hex[:12]}",
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                subject=subject,
                verb=norm_verb,
                resource_class=norm_rc,
                canonical_resource=target,
                effect=Effect.ALLOW,
                matched_rule_id=f"consumed_{approval_id}",
                policy_version=policy_ver,
                reason="Execution authorized via consumed owner approval",
                trace_id=trace_id,
                approval_id=approval_id,
            )
            # Fail-closed: durably log authorized execution intent before executing side effect
            self.audit_logger.log_event(auth_rec, None, args)
        else:
            # Standard authorization check
            auth_rec = self.decide(
                subject=subject,
                verb=norm_verb,
                resource_class=norm_rc,
                target=target,
                arguments=args,
                trace_id=trace_id,
                turn_id=turn_id,
            )
            if auth_rec.effect == Effect.DENY:
                raise PermissionError(
                    f"Phylax denied action '{norm_verb.value}' on '{target}' for subject '{subject}': {auth_rec.reason}"
                )
            if auth_rec.effect == Effect.ASK:
                raise PermissionError(
                    f"Phylax requires owner approval for action '{norm_verb.value}' on '{target}' "
                    f"(approval_id={auth_rec.approval_id})"
                )

        # Point-of-action execution
        receipt_id = f"rcpt-{uuid.uuid4().hex[:12]}"
        now_utc = datetime.now(timezone.utc).isoformat()
        try:
            result = action_fn(**args)
            
            # Compute execution digest if possible
            exec_digest = None
            bytes_processed = None
            if isinstance(result, (bytes, bytearray)):
                exec_digest = hashlib.sha256(result).hexdigest()
                bytes_processed = len(result)
            elif isinstance(result, str):
                exec_digest = hashlib.sha256(result.encode("utf-8")).hexdigest()
                bytes_processed = len(result.encode("utf-8"))

            receipt = ExecutionReceipt(
                receipt_id=receipt_id,
                decision_id=auth_rec.decision_id,
                timestamp_utc=now_utc,
                success=True,
                error_message=None,
                bytes_processed=bytes_processed,
                execution_digest=exec_digest,
            )
            # Log receipt to audit
            try:
                self.audit_logger.log_event(auth_rec, receipt, args)
            except PhylaxAuditFailureError as audit_err:
                raise PhylaxAuditFailureError(
                    f"Action '{norm_verb.value}' on '{target}' was EXECUTED successfully, but post-execution "
                    f"receipt audit logging failed (executed/audit-unacknowledged). DO NOT RETRY side effect: {audit_err}"
                ) from audit_err
            return result, receipt

        except PhylaxAuditFailureError:
            raise
        except Exception as exc:
            receipt = ExecutionReceipt(
                receipt_id=receipt_id,
                decision_id=auth_rec.decision_id,
                timestamp_utc=now_utc,
                success=False,
                error_message=str(exc),
                bytes_processed=None,
                execution_digest=None,
            )
            try:
                self.audit_logger.log_event(auth_rec, receipt, args)
            except PhylaxAuditFailureError as audit_err:
                raise PhylaxAuditFailureError(
                    f"Action '{norm_verb.value}' on '{target}' FAILED during execution ({exc}), and post-execution "
                    f"failure audit logging also failed: {audit_err}"
                ) from audit_err
            raise
