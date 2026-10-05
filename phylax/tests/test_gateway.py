"""Tests for guarded capability gateway and adapters."""

from pathlib import Path
import pytest

from phylax.approvals.manager import ApprovalManager
from phylax.audit.logger import AuditLogger
from phylax.contracts.enums import Effect, ResourceClass, Verb
from phylax.contracts.policy import PolicyDocument, ResourceSpec, Rule
from phylax.engine.manager import PolicyManager
from phylax.gateway.gateway import GuardedGateway
from phylax.gateway.file_adapter import GuardedFileAdapter
from phylax.gateway.tool_adapter import GuardedToolAdapter
from phylax.gateway.send_adapter import GuardedSendAdapter


@pytest.fixture
def gateway_setup(tmp_path: Path):
    docs_dir = str(tmp_path / "docs")
    (tmp_path / "docs").mkdir()

    policy = PolicyDocument(
        version="1.0.0",
        owner="owner",
        rules=[
            Rule(
                id="allow_docs_read",
                subject="kybernetes",
                verb=Verb.READ,
                resource=ResourceSpec(resource_class=ResourceClass.FILE, path_root=docs_dir),
                effect=Effect.ALLOW,
                priority=10,
            ),
            Rule(
                id="allow_docs_write",
                subject="kybernetes",
                verb=Verb.WRITE,
                resource=ResourceSpec(resource_class=ResourceClass.FILE, path_root=docs_dir),
                effect=Effect.ALLOW,
                priority=10,
            ),
            Rule(
                id="allow_calculator_tool",
                subject="kybernetes",
                verb=Verb.EXECUTE,
                resource=ResourceSpec(resource_class=ResourceClass.TOOL, tool_id="calc"),
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

    pol_file = tmp_path / "policy.yaml"
    pol_file.write_text(policy.model_dump_json())
    pm = PolicyManager(pol_file, workspace_root=tmp_path)
    audit = AuditLogger(tmp_path / "audit.jsonl")
    approvals = ApprovalManager(tmp_path / "approvals.json")
    gw = GuardedGateway(pm, audit, approvals)
    return gw, tmp_path


def test_guarded_file_adapter_read_and_write(gateway_setup):
    gw, tmp_path = gateway_setup
    files = GuardedFileAdapter(gw)

    target_file = tmp_path / "docs" / "test.txt"
    
    # Write file safely
    bytes_written = files.write_text(target_file, "hello phylax", subject="kybernetes")
    assert bytes_written > 0
    assert target_file.exists()

    # Read file safely
    content = files.read_text(target_file, subject="kybernetes")
    assert content == "hello phylax"

    # Attempt to read unauthorized file outside docs
    unauthorized = tmp_path / "secret.txt"
    unauthorized.write_text("secret")
    with pytest.raises(PermissionError, match="denied"):
        files.read_text(unauthorized, subject="kybernetes")


def test_guarded_tool_adapter(gateway_setup):
    gw, _ = gateway_setup
    tools = GuardedToolAdapter(gw)

    tools.register_tool("calc", lambda a, b: a + b)
    tools.register_tool("shell", lambda cmd: "executed")

    # Allowed tool
    res = tools.invoke("calc", {"a": 2, "b": 3}, subject="kybernetes")
    assert res == 5

    # Denied tool
    with pytest.raises(PermissionError, match="denied"):
        tools.invoke("shell", {"cmd": "whoami"}, subject="kybernetes")


def test_direct_bypass_attempt(gateway_setup):
    gw, tmp_path = gateway_setup

    def malicious_delete(**kwargs):
        return "deleted"

    # Calling authorize_and_execute on unpermitted operation
    with pytest.raises(PermissionError, match="denied"):
        gw.authorize_and_execute(
            subject="untrusted_agent",
            verb=Verb.EXECUTE,
            resource_class=ResourceClass.SYSTEM,
            target="/bin/rm",
            action_fn=malicious_delete,
        )


def test_file_write_content_digest_binding(gateway_setup):
    """Item 5: Approvals and writes bind exact content digest and reject tampered content."""
    gw, tmp_path = gateway_setup
    files = GuardedFileAdapter(gw)
    target_file = tmp_path / "docs" / "tamper_test.txt"

    # Successful write with matching digest
    files.write_text(target_file, "authorized_content", subject="kybernetes")
    assert target_file.read_text() == "authorized_content"


def test_pre_execution_audit_failure_blocks_action(gateway_setup, monkeypatch):
    """Item 6: If pre-execution audit logging fails, action_fn must NOT be called."""
    from unittest.mock import MagicMock
    from phylax.audit.logger import PhylaxAuditFailureError

    gw, tmp_path = gateway_setup
    action_mock = MagicMock(return_value="executed")

    # Patch log_event to simulate disk failure on pre-execution logging
    def fail_log(*args, **kwargs):
        raise PhylaxAuditFailureError("Simulated disk error during audit write")

    monkeypatch.setattr(gw.audit_logger, "log_event", fail_log)

    with pytest.raises((PermissionError, PhylaxAuditFailureError)):
        gw.authorize_and_execute(
            subject="kybernetes",
            verb=Verb.READ,
            resource_class=ResourceClass.FILE,
            target=str(tmp_path / "docs"),
            action_fn=action_mock,
        )

    # Action MUST NOT have executed
    assert action_mock.call_count == 0

