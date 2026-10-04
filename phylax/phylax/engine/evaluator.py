"""Policy evaluation engine implementing rule matching and deny-precedence."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

from phylax.contracts.enums import Effect, ResourceClass, Verb
from phylax.contracts.decision import AuthorizationRecord
from phylax.contracts.policy import DeclarativeCondition, PolicyDocument, Rule
from phylax.engine.canonical import canonicalize_path, matches_path_pattern
from phylax.engine.context import verify_caller_authorization


def compute_argument_digest(arguments: dict[str, Any] | None) -> str:
    """Compute deterministic SHA-256 digest of argument dictionary."""
    if not arguments:
        payload = b"{}"
    else:
        try:
            norm_json = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
            payload = norm_json.encode("utf-8")
        except Exception:
            payload = str(sorted(arguments.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PolicyEvaluator:
    """Evaluates access requests against an active PolicyDocument."""

    def __init__(self, policy: PolicyDocument, workspace_root: Path | None = None):
        self.policy = policy
        self.workspace_root = (workspace_root or Path.cwd()).resolve()

    def decide(
        self,
        subject: str,
        verb: Verb | str,
        resource_class: ResourceClass | str,
        target: str,
        arguments: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> AuthorizationRecord:
        """Evaluate authorization request and return immutable AuthorizationRecord."""
        # 1. Verify caller authorization
        verified_subject = verify_caller_authorization(subject)

        # 2. Normalize enums
        norm_verb = Verb(verb) if isinstance(verb, str) else verb
        norm_rc = ResourceClass(resource_class) if isinstance(resource_class, str) else resource_class

        # 3. Canonicalize target
        if norm_rc == ResourceClass.FILE:
            canon_target = str(canonicalize_path(target, base_dir=self.workspace_root))
        else:
            canon_target = str(target).strip()

        # 4. Find all matching rules
        matching_rules: list[Rule] = []
        for rule in self.policy.rules:
            if self._rule_matches(rule, verified_subject, norm_verb, norm_rc, canon_target, arguments):
                matching_rules.append(rule)

        arg_digest = compute_argument_digest(arguments)
        now_utc = datetime.now(timezone.utc).isoformat()
        decision_id = f"dec-{uuid.uuid4().hex[:12]}"

        if not matching_rules:
            # Explicit default deny
            return AuthorizationRecord(
                decision_id=decision_id,
                timestamp_utc=now_utc,
                subject=verified_subject,
                verb=norm_verb,
                resource_class=norm_rc,
                canonical_resource=canon_target,
                effect=Effect.DENY,
                matched_rule_id=None,
                policy_version=self.policy.version,
                reason="Default deny: No matching rule found",
                trace_id=trace_id,
                argument_digest=arg_digest,
            )

        # 5. Resolve conflicts: Deny-precedence
        # Sort by priority (lowest int = highest priority)
        # If priorities are equal, DENY takes precedence over ALLOW and ASK
        def sort_key(r: Rule) -> tuple[int, int]:
            # Priority first, then effect severity: DENY (0) > ASK (1) > ALLOW (2)
            effect_rank = 0 if r.effect == Effect.DENY else (1 if r.effect == Effect.ASK else 2)
            return (r.priority, effect_rank)

        sorted_rules = sorted(matching_rules, key=sort_key)
        winning_rule = sorted_rules[0]

        # Check if an explicit DENY at the same priority level wins
        approval_id = None
        if winning_rule.effect == Effect.ASK:
            approval_id = f"appr-{uuid.uuid4().hex[:12]}"

        reason = (
            f"Matched rule '{winning_rule.id}' (priority {winning_rule.priority}): "
            f"{winning_rule.description or winning_rule.effect.value}"
        )

        return AuthorizationRecord(
            decision_id=decision_id,
            timestamp_utc=now_utc,
            subject=verified_subject,
            verb=norm_verb,
            resource_class=norm_rc,
            canonical_resource=canon_target,
            effect=winning_rule.effect,
            matched_rule_id=winning_rule.id,
            policy_version=self.policy.version,
            reason=reason,
            trace_id=trace_id,
            approval_id=approval_id,
            argument_digest=arg_digest,
        )

    def _rule_matches(
        self,
        rule: Rule,
        subject: str,
        verb: Verb,
        resource_class: ResourceClass,
        canon_target: str,
        arguments: dict[str, Any] | None,
    ) -> bool:
        # Check subject
        if rule.subject != "*" and rule.subject != "all" and rule.subject != subject:
            return False

        # Check verb
        if rule.verb != Verb.ALL and rule.verb != verb:
            return False

        # Check resource class
        if rule.resource.resource_class != ResourceClass.ALL and rule.resource.resource_class != resource_class:
            return False

        # Check resource specifics
        res = rule.resource
        if resource_class == ResourceClass.FILE:
            target_path = Path(canon_target)
            root_path = Path(res.path_root) if res.path_root else None
            if not matches_path_pattern(target_path, root_path, res.path_glob, base_dir=self.workspace_root):
                return False

        elif resource_class == ResourceClass.TOOL:
            if res.tool_id and res.tool_id != "*" and res.tool_id != canon_target:
                return False

        elif resource_class == ResourceClass.NETWORK:
            if res.destination and res.destination != "*":
                # Check destination pattern match
                if res.destination not in canon_target and canon_target not in res.destination:
                    return False

        elif resource_class == ResourceClass.SYSTEM:
            if res.command and res.command != "*" and res.command != canon_target:
                return False

        # Check condition if present
        if rule.condition:
            if not self._check_condition(rule.condition, canon_target, arguments):
                return False

        return True

    def _check_condition(
        self,
        cond: DeclarativeCondition,
        canon_target: str,
        arguments: dict[str, Any] | None,
    ) -> bool:
        args = arguments or {}

        if cond.path_must_exist is not None:
            p = Path(canon_target)
            if p.exists() != cond.path_must_exist:
                return False

        if cond.allowed_extensions is not None:
            p = Path(canon_target)
            if p.suffix.lower() not in [ext.lower() for ext in cond.allowed_extensions]:
                return False

        if cond.max_bytes is not None:
            # Check file size or argument content size
            p = Path(canon_target)
            if p.exists() and p.is_file():
                if p.stat().st_size > cond.max_bytes:
                    return False
            if "content" in args:
                content_len = len(str(args["content"]).encode("utf-8"))
                if content_len > cond.max_bytes:
                    return False

        if cond.argument_patterns:
            for k, expected in cond.argument_patterns.items():
                if k not in args or str(args[k]) != expected:
                    return False

        return True
