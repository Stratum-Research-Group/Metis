"""Integration tests for Kybernetes orchestrator shim."""

from pathlib import Path
import pytest

from phylax.adapters.kybernetes import KybernetesShim
from phylax.approvals.manager import ApprovalManager
from phylax.audit.logger import AuditLogger
from phylax.contracts.enums import Effect, ResourceClass, Verb
from phylax.contracts.policy import PolicyDocument, ResourceSpec, Rule
from phylax.engine.manager import PolicyManager
from phylax.gateway.gateway import GuardedGateway


@pytest.fixture
def kybernetes_env(tmp_path: Path):
    docs_dir = str(tmp_path / "docs")
    (tmp_path / "docs").mkdir()

    policy = PolicyDocument(
        version="1.0.0",
        owner="owner",
        rules=[
            Rule(
                id="allow_read_docs",
                subject="kybernetes",
                verb=Verb.READ,
                resource=ResourceSpec(resource_class=ResourceClass.FILE, path_root=docs_dir),
                effect=Effect.ALLOW,
                priority=10,
            ),
            Rule(
                id="allow_write_docs",
                subject="kybernetes",
                verb=Verb.WRITE,
                resource=ResourceSpec(resource_class=ResourceClass.FILE, path_root=docs_dir),
                effect=Effect.ALLOW,
                priority=10,
            ),
            Rule(
                id="ask_send",
                subject="kybernetes",
                verb=Verb.SEND,
                resource=ResourceSpec(resource_class=ResourceClass.NETWORK, destination="*"),
                effect=Effect.ASK,
                priority=20,
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
    shim = KybernetesShim(gw)
    return shim, tmp_path, approvals


def test_kybernetes_read_write_file(kybernetes_env):
    shim, tmp_path, _ = kybernetes_env
    doc = tmp_path / "docs" / "plan.md"

    # Write
    shim.write_file(doc, "# Kybernetes Plan\n1. Search\n2. Act")
    assert doc.exists()

    # Read
    content = shim.read_file(doc, trace_id="trace-kyb-1")
    assert "Kybernetes Plan" in content


def test_kybernetes_send_approval_flow(kybernetes_env):
    shim, _, approvals = kybernetes_env
    sent = []

    def mock_sender(**args):
        sent.append(args)
        return "ok"

    # Direct send requires approval
    with pytest.raises(PermissionError, match="requires owner approval"):
        shim.send("https://api.external/hook", {"data": 123}, mock_sender, trace_id="trace-send-1")

    # Locate and approve request
    pending = approvals.list_pending()
    assert len(pending) == 1
    req = pending[0]
    approvals.resolve_request(req.approval_id, approved=True)

    # Re-execute with approval_id
    res = shim.send("https://api.external/hook", {"data": 123}, mock_sender, approval_id=req.approval_id)
    assert res == "ok"
    assert len(sent) == 1
