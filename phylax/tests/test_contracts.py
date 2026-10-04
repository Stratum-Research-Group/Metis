"""Tests for typed policy schema contracts and models."""

import pytest
from pydantic import ValidationError

from phylax.contracts.enums import Effect, ResourceClass, Verb
from phylax.contracts.policy import PolicyDocument, ResourceSpec, Rule


def test_valid_policy_document():
    doc = PolicyDocument(
        version="1.0.0",
        owner="alice",
        rules=[
            Rule(
                id="rule_1",
                subject="kybernetes",
                verb=Verb.READ,
                resource=ResourceSpec(resource_class=ResourceClass.FILE, path_root="/tmp"),
                effect=Effect.ALLOW,
                priority=10,
            ),
            Rule(
                id="default_deny",
                subject="*",
                verb=Verb.ALL,
                resource=ResourceSpec(resource_class=ResourceClass.ALL, path_root="*"),
                effect=Effect.DENY,
                priority=1000,
            ),
        ],
    )
    assert doc.version == "1.0.0"
    assert len(doc.rules) == 2


def test_reject_duplicate_rule_id():
    with pytest.raises(ValidationError, match="Duplicate rule ID"):
        PolicyDocument(
            version="1.0.0",
            owner="alice",
            rules=[
                Rule(
                    id="rule_dup",
                    subject="kybernetes",
                    verb=Verb.READ,
                    resource=ResourceSpec(resource_class=ResourceClass.FILE, path_root="/tmp"),
                    effect=Effect.ALLOW,
                ),
                Rule(
                    id="rule_dup",
                    subject="stoma",
                    verb=Verb.WRITE,
                    resource=ResourceSpec(resource_class=ResourceClass.FILE, path_root="/tmp"),
                    effect=Effect.ALLOW,
                ),
                Rule(
                    id="default_deny",
                    subject="*",
                    verb=Verb.ALL,
                    resource=ResourceSpec(resource_class=ResourceClass.ALL, path_root="*"),
                    effect=Effect.DENY,
                ),
            ],
        )


def test_reject_missing_final_default_deny():
    with pytest.raises(ValidationError, match="Policy must end with an explicit default-deny rule"):
        PolicyDocument(
            version="1.0.0",
            owner="alice",
            rules=[
                Rule(
                    id="rule_1",
                    subject="kybernetes",
                    verb=Verb.READ,
                    resource=ResourceSpec(resource_class=ResourceClass.FILE, path_root="/tmp"),
                    effect=Effect.ALLOW,
                )
            ],
        )


def test_reject_unknown_keys():
    with pytest.raises(ValidationError):
        PolicyDocument.model_validate(
            {
                "version": "1.0.0",
                "owner": "alice",
                "unauthorized_field": "dangerous_injection",
                "rules": [
                    {
                        "id": "default_deny",
                        "subject": "*",
                        "verb": "*",
                        "resource": {"class": "*", "path_root": "*"},
                        "effect": "deny",
                    }
                ],
            }
        )
