"""Tests for policy evaluation and deny-precedence."""

from pathlib import Path
import pytest

from phylax.contracts.enums import Effect, ResourceClass, Verb
from phylax.contracts.policy import DeclarativeCondition, PolicyDocument, ResourceSpec, Rule
from phylax.engine.context import trusted_caller
from phylax.engine.evaluator import PolicyEvaluator


@pytest.fixture
def sample_policy(tmp_path: Path) -> PolicyDocument:
    docs_dir = str(tmp_path / "docs")
    private_dir = str(tmp_path / "docs" / "private")

    return PolicyDocument(
        version="1.0.0",
        owner="test-owner",
        rules=[
            # Specific Deny
            Rule(
                id="deny_private_subdir",
                subject="kybernetes",
                verb=Verb.READ,
                resource=ResourceSpec(resource_class=ResourceClass.FILE, path_root=private_dir),
                effect=Effect.DENY,
                priority=1,
            ),
            # Broad Allow
            Rule(
                id="allow_broad_docs",
                subject="kybernetes",
                verb=Verb.READ,
                resource=ResourceSpec(resource_class=ResourceClass.FILE, path_root=docs_dir),
                effect=Effect.ALLOW,
                priority=10,
            ),
            # Condition-based Allow
            Rule(
                id="allow_json_only",
                subject="stoma",
                verb=Verb.WRITE,
                resource=ResourceSpec(resource_class=ResourceClass.FILE, path_root=docs_dir),
                condition=DeclarativeCondition(allowed_extensions=[".json"], max_bytes=1000),
                effect=Effect.ALLOW,
                priority=10,
            ),
            # Universal Default Deny
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


def test_deny_precedence_over_broad_allow(tmp_path: Path, sample_policy: PolicyDocument):
    evaluator = PolicyEvaluator(sample_policy, workspace_root=tmp_path)

    docs_file = tmp_path / "docs" / "readme.txt"
    private_file = tmp_path / "docs" / "private" / "keys.pem"

    # Reading general doc is allowed
    rec1 = evaluator.decide("kybernetes", Verb.READ, ResourceClass.FILE, str(docs_file))
    assert rec1.effect == Effect.ALLOW
    assert rec1.matched_rule_id == "allow_broad_docs"

    # Reading file in private subdir must be DENIED even though it's inside docs
    rec2 = evaluator.decide("kybernetes", Verb.READ, ResourceClass.FILE, str(private_file))
    assert rec2.effect == Effect.DENY
    assert rec2.matched_rule_id == "deny_private_subdir"


def test_default_deny_for_unlisted_action(tmp_path: Path, sample_policy: PolicyDocument):
    evaluator = PolicyEvaluator(sample_policy, workspace_root=tmp_path)

    # stoma attempting to execute a system command
    rec = evaluator.decide("stoma", Verb.EXECUTE, ResourceClass.SYSTEM, "/bin/rm -rf /")
    assert rec.effect == Effect.DENY


def test_declarative_condition_evaluation(tmp_path: Path, sample_policy: PolicyDocument):
    evaluator = PolicyEvaluator(sample_policy, workspace_root=tmp_path)

    json_file = str(tmp_path / "docs" / "data.json")
    txt_file = str(tmp_path / "docs" / "data.txt")

    # Allowed: .json extension with small payload
    rec_ok = evaluator.decide(
        "stoma", Verb.WRITE, ResourceClass.FILE, json_file, arguments={"content": "{}"}
    )
    assert rec_ok.effect == Effect.ALLOW

    # Denied: .txt extension violates allowed_extensions condition
    rec_bad_ext = evaluator.decide(
        "stoma", Verb.WRITE, ResourceClass.FILE, txt_file, arguments={"content": "hello"}
    )
    assert rec_bad_ext.effect == Effect.DENY

    # Denied: payload exceeds max_bytes
    huge_content = "x" * 2000
    rec_too_big = evaluator.decide(
        "stoma", Verb.WRITE, ResourceClass.FILE, json_file, arguments={"content": huge_content}
    )
    assert rec_too_big.effect == Effect.DENY


def test_trusted_caller_spoofing_prevention(tmp_path: Path, sample_policy: PolicyDocument):
    evaluator = PolicyEvaluator(sample_policy, workspace_root=tmp_path)

    # Set trusted caller context to stoma
    with trusted_caller("stoma"):
        # If caller tries to claim they are kybernetes, PermissionError must be raised
        with pytest.raises(PermissionError, match="Caller context violation"):
            evaluator.decide("kybernetes", Verb.READ, ResourceClass.FILE, "/tmp/doc.txt")


def test_network_destination_matching_not_substring(tmp_path: Path):
    """Verify network destination matching avoids substring traps (Finding R4)."""
    policy = PolicyDocument(
        version="1.0.0",
        owner="test-owner",
        rules=[
            Rule(
                id="allow_api_com",
                subject="worker",
                verb=Verb.SEND,
                resource=ResourceSpec(resource_class=ResourceClass.NETWORK, destination="api.com"),
                effect=Effect.ALLOW,
                priority=10,
            ),
            Rule(
                id="allow_subdomains",
                subject="worker",
                verb=Verb.SEND,
                resource=ResourceSpec(resource_class=ResourceClass.NETWORK, destination="*.trusted.org"),
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
    evaluator = PolicyEvaluator(policy, workspace_root=tmp_path)

    # 1. Exact match allowed
    assert evaluator.decide("worker", Verb.SEND, ResourceClass.NETWORK, "api.com").effect == Effect.ALLOW

    # 2. Substring bypasses must be DENIED
    assert evaluator.decide("worker", Verb.SEND, ResourceClass.NETWORK, "api.com.evil.com").effect == Effect.DENY
    assert evaluator.decide("worker", Verb.SEND, ResourceClass.NETWORK, "evil-api.com").effect == Effect.DENY
    assert evaluator.decide("worker", Verb.SEND, ResourceClass.NETWORK, "api.co").effect == Effect.DENY

    # 3. Wildcard matching
    assert evaluator.decide("worker", Verb.SEND, ResourceClass.NETWORK, "app.trusted.org").effect == Effect.ALLOW
    assert evaluator.decide("worker", Verb.SEND, ResourceClass.NETWORK, "trusted.org").effect == Effect.ALLOW
    assert evaluator.decide("worker", Verb.SEND, ResourceClass.NETWORK, "evil-trusted.org").effect == Effect.DENY

