"""Static analysis linter for Phylax YAML policies."""

from enum import Enum
from pathlib import Path
from typing import Any
import yaml
from pydantic import ValidationError

from phylax.contracts.enums import Effect, ResourceClass, Verb
from phylax.contracts.policy import PolicyDocument, Rule


class LintSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class LintIssue:
    def __init__(
        self,
        severity: LintSeverity,
        message: str,
        rule_id: str | None = None,
        field: str | None = None,
    ):
        self.severity = severity
        self.message = message
        self.rule_id = rule_id
        self.field = field

    def __repr__(self) -> str:
        loc = f" [Rule: {self.rule_id}]" if self.rule_id else ""
        fld = f" (Field: {self.field})" if self.field else ""
        return f"[{self.severity.value}]{loc}{fld} {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "rule_id": self.rule_id,
            "field": self.field,
            "message": self.message,
        }


class PolicyLinter:
    """Lints policy YAML for syntax, structural errors, shadowed rules, and security flaws."""

    def __init__(self, workspace_root: Path | None = None):
        self.workspace_root = workspace_root or Path.cwd()

    def lint_yaml_string(self, yaml_content: str) -> tuple[PolicyDocument | None, list[LintIssue]]:
        issues: list[LintIssue] = []

        # 1. Parse YAML syntax
        try:
            raw_data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            issues.append(
                LintIssue(
                    severity=LintSeverity.ERROR,
                    message=f"YAML syntax error: {exc}",
                )
            )
            return None, issues

        if not isinstance(raw_data, dict):
            issues.append(
                LintIssue(
                    severity=LintSeverity.ERROR,
                    message="Policy document must be a YAML mapping/dictionary",
                )
            )
            return None, issues

        # 2. Validate against Pydantic schema
        try:
            policy = PolicyDocument.model_validate(raw_data)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                issues.append(
                    LintIssue(
                        severity=LintSeverity.ERROR,
                        message=f"{err['msg']}",
                        field=loc,
                    )
                )
            return None, issues

        # 3. Perform semantic checks on rules
        self._check_semantic_rules(policy, issues)

        return policy, issues

    def lint_file(self, path: Path | str) -> tuple[PolicyDocument | None, list[LintIssue]]:
        file_path = Path(path)
        if not file_path.exists():
            return None, [
                LintIssue(
                    severity=LintSeverity.ERROR,
                    message=f"Policy file does not exist: {file_path}",
                )
            ]
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return None, [
                LintIssue(
                    severity=LintSeverity.ERROR,
                    message=f"Failed to read policy file: {exc}",
                )
            ]
        return self.lint_yaml_string(content)

    def _check_semantic_rules(self, policy: PolicyDocument, issues: list[LintIssue]) -> None:
        rules = policy.rules

        # Check path traversal and relative roots
        for rule in rules:
            res = rule.resource
            if res.resource_class == ResourceClass.FILE:
                if not res.path_root and not res.path_glob:
                    issues.append(
                        LintIssue(
                            severity=LintSeverity.ERROR,
                            rule_id=rule.id,
                            field="resource",
                            message="File resource must specify at least path_root or path_glob",
                        )
                    )
                if res.path_root:
                    if ".." in res.path_root:
                        issues.append(
                            LintIssue(
                                severity=LintSeverity.ERROR,
                                rule_id=rule.id,
                                field="resource.path_root",
                                message=f"Path traversal sequence '..' forbidden in path_root: '{res.path_root}'",
                            )
                        )
            elif res.resource_class == ResourceClass.TOOL:
                if not res.tool_id:
                    issues.append(
                        LintIssue(
                            severity=LintSeverity.ERROR,
                            rule_id=rule.id,
                            field="resource.tool_id",
                            message="Tool resource must specify tool_id",
                        )
                    )
            elif res.resource_class == ResourceClass.NETWORK:
                if not res.destination:
                    issues.append(
                        LintIssue(
                            severity=LintSeverity.ERROR,
                            rule_id=rule.id,
                            field="resource.destination",
                            message="Network resource must specify destination pattern",
                        )
                    )

        # Check rule shadowing / unreachability
        for i in range(len(rules)):
            rule_a = rules[i]
            for j in range(i + 1, len(rules)):
                rule_b = rules[j]
                if self._rule_subsumes(rule_a, rule_b):
                    # rule_a completely covers rule_b, so rule_b is unreachable
                    if j == len(rules) - 1 and rule_b.id.startswith("default_deny"):
                        # If a rule shadows the default deny, that's a critical safety error
                        issues.append(
                            LintIssue(
                                severity=LintSeverity.ERROR,
                                rule_id=rule_a.id,
                                message=f"Rule '{rule_a.id}' shadows the final default-deny rule, creating an open policy!",
                            )
                        )
                    else:
                        issues.append(
                            LintIssue(
                                severity=LintSeverity.WARNING,
                                rule_id=rule_b.id,
                                message=(
                                    f"Rule '{rule_b.id}' is shadowed and unreachable because "
                                    f"earlier rule '{rule_a.id}' subsumes its subject, verb, and resource."
                                ),
                            )
                        )

        # Conflict check: broad allow placed ahead of specific deny
        # Deny must win over conflicting broad allow!
        for i in range(len(rules)):
            rule_a = rules[i]
            if rule_a.effect == Effect.ALLOW:
                for j in range(i + 1, len(rules)):
                    rule_b = rules[j]
                    if rule_b.effect == Effect.DENY and self._rule_overlaps(rule_a, rule_b):
                        issues.append(
                            LintIssue(
                                severity=LintSeverity.WARNING,
                                rule_id=rule_a.id,
                                message=(
                                    f"Broad allow rule '{rule_a.id}' appears before narrower deny rule '{rule_b.id}'. "
                                    "Phylax evaluator enforces deny-precedence, but ordering should place specific deny rules first."
                                ),
                            )
                        )

    def _rule_subsumes(self, a: Rule, b: Rule) -> bool:
        """Returns True if rule A strictly subsumes rule B (making B unreachable)."""
        # Subject matching
        if a.subject != "*" and a.subject != b.subject:
            return False

        # Verb matching
        if a.verb != b.verb:
            return False

        # Resource class
        if a.resource.resource_class != b.resource.resource_class:
            return False

        # Conditions: if A has a condition that B lacks, A does not subsume B
        if a.condition is not None and b.condition is None:
            return False

        # Resource specifics
        rc = a.resource.resource_class
        if rc == ResourceClass.FILE:
            # If A specifies path_root = '*' or '/' or same root and broad glob
            a_root = a.resource.path_root
            b_root = b.resource.path_root
            a_glob = a.resource.path_glob or "*"
            b_glob = b.resource.path_glob or "*"

            if a_root == "*" or a_root == "":
                return True
            if a_root and b_root and a_root == b_root:
                if a_glob == "*" or a_glob == "**/*":
                    return True
            return False

        elif rc == ResourceClass.TOOL:
            if a.resource.tool_id == "*" or a.resource.tool_id == b.resource.tool_id:
                return True
            return False

        elif rc == ResourceClass.NETWORK:
            if a.resource.destination == "*" or a.resource.destination == b.resource.destination:
                return True
            return False

        return False

    def _rule_overlaps(self, a: Rule, b: Rule) -> bool:
        """Returns True if rule A and rule B share subjects, verbs, and overlapping resources."""
        # Check subject
        subject_match = (a.subject == "*" or b.subject == "*" or a.subject == b.subject)
        if not subject_match:
            return False

        # Check verb
        if a.verb != b.verb:
            return False

        # Check resource class
        if a.resource.resource_class != b.resource.resource_class:
            return False

        rc = a.resource.resource_class
        if rc == ResourceClass.FILE:
            # Overlapping path roots or globs
            a_root = a.resource.path_root or ""
            b_root = b.resource.path_root or ""
            if a_root == "*" or b_root == "*" or a_root == b_root:
                return True
            if a_root and b_root:
                try:
                    p_a = Path(a_root)
                    p_b = Path(b_root)
                    if p_b.is_relative_to(p_a) or p_a.is_relative_to(p_b):
                        return True
                except Exception:
                    pass
        elif rc == ResourceClass.TOOL:
            return a.resource.tool_id == "*" or b.resource.tool_id == "*" or a.resource.tool_id == b.resource.tool_id
        elif rc == ResourceClass.NETWORK:
            return a.resource.destination == "*" or b.resource.destination == "*" or a.resource.destination == b.resource.destination

        return False
