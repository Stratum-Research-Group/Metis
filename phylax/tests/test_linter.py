"""Tests for static analysis policy linter."""

from phylax.linter.linter import LintSeverity, PolicyLinter


def test_linter_detects_shadowed_rules():
    yaml_content = """
version: "1.0.0"
owner: "alice"
rules:
  - id: "broad_rule"
    subject: "*"
    verb: "read"
    resource:
      class: "file"
      path_root: "*"
    effect: "allow"
    priority: 10

  - id: "shadowed_rule"
    subject: "kybernetes"
    verb: "read"
    resource:
      class: "file"
      path_root: "docs"
    effect: "deny"
    priority: 10

  - id: "default_deny"
    subject: "*"
    verb: "*"
    resource:
      class: "*"
      path_root: "*"
    effect: "deny"
    priority: 1000
"""
    linter = PolicyLinter()
    policy, issues = linter.lint_yaml_string(yaml_content)

    shadowed = [i for i in issues if "shadowed" in i.message.lower()]
    assert len(shadowed) > 0
    assert shadowed[0].rule_id == "shadowed_rule"


def test_linter_detects_path_traversal_in_root():
    yaml_content = """
version: "1.0.0"
owner: "alice"
rules:
  - id: "traversal_rule"
    subject: "kybernetes"
    verb: "read"
    resource:
      class: "file"
      path_root: "/app/docs/../../etc"
    effect: "allow"
    priority: 10

  - id: "default_deny"
    subject: "*"
    verb: "*"
    resource:
      class: "*"
      path_root: "*"
    effect: "deny"
    priority: 1000
"""
    linter = PolicyLinter()
    policy, issues = linter.lint_yaml_string(yaml_content)

    errors = [i for i in issues if i.severity == LintSeverity.ERROR]
    assert any(".." in i.message for i in errors)


def test_linter_detects_unknown_verbs():
    yaml_content = """
version: "1.0.0"
owner: "alice"
rules:
  - id: "invalid_verb_rule"
    subject: "kybernetes"
    verb: "destroy_everything"
    resource:
      class: "file"
      path_root: "/tmp"
    effect: "allow"

  - id: "default_deny"
    subject: "*"
    verb: "*"
    resource:
      class: "*"
      path_root: "*"
    effect: "deny"
"""
    linter = PolicyLinter()
    policy, issues = linter.lint_yaml_string(yaml_content)
    assert any(i.severity == LintSeverity.ERROR for i in issues)
