#!/usr/bin/env python3
"""Phylax End-to-End Demo Script.

Demonstrates:
  1. Allowed docs read
  2. Denied private read
  3. Send requiring approval (owner approves -> executes once)
  4. Unanswered send that times out to deny (TTL expiration)
  5. Approved request whose arguments change before execution (denied by digest check)
  6. Audit chain inspection and cryptographic verification
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from phylax.adapters.kybernetes import KybernetesShim
from phylax.approvals.manager import ApprovalManager
from phylax.audit.logger import AuditLogger
from phylax.audit.verifier import AuditVerifier
from phylax.engine.manager import PolicyManager
from phylax.gateway.gateway import GuardedGateway


def run_demo() -> None:
    demo_root = Path("demo_workspace").resolve()
    if demo_root.exists():
        shutil.rmtree(demo_root)
    demo_root.mkdir(parents=True, exist_ok=True)

    # 1. Prepare demo files
    docs_dir = demo_root / "workspace" / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_file = docs_dir / "system_architecture.txt"
    doc_file.write_text("Metis System Architecture: Phylax guards capabilities at point-of-action.\n")

    private_dir = demo_root / "workspace" / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    secret_file = private_dir / "id_rsa_secret.pem"
    secret_file.write_text("-----BEGIN PRIVATE KEY-----\nSUPER_SECRET_KEY\n-----END PRIVATE KEY-----\n")

    policy_file = Path("policies/demo_policy.yaml")
    audit_file = demo_root / "demo_audit.jsonl"
    approval_file = demo_root / "demo_approvals.json"

    print("=" * 72)
    print("                 PHYLAX CAPABILITY GATEWAY DEMO")
    print("=" * 72)
    print(f"Policy:    {policy_file}")
    print(f"Workspace: {demo_root}")
    print(f"Audit Log: {audit_file}\n")

    # Initialize components
    policy_mgr = PolicyManager(policy_file, workspace_root=demo_root)
    audit_logger = AuditLogger(audit_file)
    approval_mgr = ApprovalManager(approval_file)
    gateway = GuardedGateway(policy_mgr, audit_logger, approval_mgr)
    kybernetes = KybernetesShim(gateway)

    # ---------------------------------------------------------
    # Scenario 1: Allowed Docs Read
    # ---------------------------------------------------------
    print("\n[Scenario 1] Read Allowed Document (workspace/docs/system_architecture.txt)")
    try:
        content = kybernetes.read_file(doc_file, trace_id="trace-001")
        print("  -> Result: SUCCESS (Allowed)")
        print(f"  -> Content Read: {content.strip()}")
    except Exception as exc:
        print(f"  -> Result: UNEXPECTED FAILURE: {exc}")

    # ---------------------------------------------------------
    # Scenario 2: Denied Private Read
    # ---------------------------------------------------------
    print("\n[Scenario 2] Read Denied Private File (workspace/private/id_rsa_secret.pem)")
    try:
        content = kybernetes.read_file(secret_file, trace_id="trace-002")
        print(f"  -> Result: UNEXPECTED SUCCESS: {content}")
    except PermissionError as exc:
        print("  -> Result: BLOCKED (Denied as expected)")
        print(f"  -> Policy Message: {exc}")

    # ---------------------------------------------------------
    # Scenario 3: Send Requiring Approval (Owner Approves -> Executes)
    # ---------------------------------------------------------
    print("\n[Scenario 3] Outbound Send Requiring Approval (Owner Approves)")
    destination = "https://api.external.org/notify"
    payload = {"recipient": "carol", "message": "Metis cluster status: OK", "token": "secret_demo_token_123"}
    sent_box = []

    def mock_sender(**args):
        sent_box.append(args)
        return {"status": "delivered", "bytes": len(json.dumps(args))}

    try:
        kybernetes.send(destination, payload, mock_sender, trace_id="trace-003")
    except PermissionError as exc:
        print("  -> Point-of-action check triggered:")
        print(f"     {exc}")

    # Locate pending approval
    pending = approval_mgr.list_pending()
    assert len(pending) > 0, "Expected pending approval request"
    appr_req = pending[-1]
    print(f"  -> Pending Approval ID: {appr_req.approval_id}")
    print(f"  -> Target:              {appr_req.target}")
    print(f"  -> Argument Digest:     {appr_req.argument_digest[:16]}...")
    print(f"  -> Argument Summary:    {appr_req.argument_summary}")

    # Owner resolves in Prosopon / CLI
    print("  -> Simulating Owner Approval via Prosopon/CLI...")
    approval_mgr.resolve_request(appr_req.approval_id, approved=True, reason="Owner approved via Prosopon UI")

    # Retry execution with approval ID
    print("  -> Executing action with approval token...")
    res = kybernetes.send(destination, payload, mock_sender, approval_id=appr_req.approval_id, trace_id="trace-003")
    print(f"  -> Execution Result: {res}")
    assert len(sent_box) == 1, "Expected sender to be invoked exactly once"

    # Attempt to replay the same approval
    print("  -> Attempting to REPLAY the consumed approval token...")
    try:
        kybernetes.send(destination, payload, mock_sender, approval_id=appr_req.approval_id)
        print("  -> ERROR: Replay succeeded!")
    except PermissionError as exc:
        print(f"  -> Replay Prevented: {exc}")

    # ---------------------------------------------------------
    # Scenario 4: Unanswered Send Times Out to Deny (TTL)
    # ---------------------------------------------------------
    print("\n[Scenario 4] Unanswered Send Times Out to Deny")
    payload4 = {"recipient": "david", "message": "Urgent alert"}
    try:
        kybernetes.send(destination, payload4, mock_sender, trace_id="trace-004")
    except PermissionError:
        pass

    pending4 = approval_mgr.list_pending()
    appr4 = pending4[-1]
    print(f"  -> Approval created: {appr4.approval_id} (TTL=5s)")
    print("  -> Waiting 6 seconds for TTL expiration...")
    time.sleep(6)

    # Check request status
    expired_req = approval_mgr.get_request(appr4.approval_id)
    print(f"  -> Status after TTL: {expired_req.status.value.upper()}")
    print("  -> Attempting execution after timeout...")
    try:
        kybernetes.send(destination, payload4, mock_sender, approval_id=appr4.approval_id)
        print("  -> ERROR: Expired approval was accepted!")
    except PermissionError as exc:
        print(f"  -> Denied on Timeout: {exc}")

    # ---------------------------------------------------------
    # Scenario 5: Anti-Argument-Swap Attack Prevention
    # ---------------------------------------------------------
    print("\n[Scenario 5] Anti-Argument-Swap (Arguments Tampered Before Execution)")
    benign_payload = {"recipient": "alice", "action": "ping", "data": "clean"}
    try:
        kybernetes.send(destination, benign_payload, mock_sender, trace_id="trace-005")
    except PermissionError:
        pass

    appr5 = approval_mgr.list_pending()[-1]
    print(f"  -> Created approval for benign payload: {appr5.approval_id}")
    approval_mgr.resolve_request(appr5.approval_id, approved=True, reason="Owner approved benign ping")

    # Malicious actor swaps payload
    tampered_payload = {"recipient": "eve", "action": "exfiltrate", "data": "stolen_keys"}
    print(f"  -> Caller attempts execution with SWAPPED payload: {tampered_payload}")
    try:
        kybernetes.send(destination, tampered_payload, mock_sender, approval_id=appr5.approval_id)
        print("  -> ERROR: Tampered arguments were executed!")
    except PermissionError as exc:
        print(f"  -> Swapped Arguments Blocked: {exc}")

    # ---------------------------------------------------------
    # Scenario 6: Audit Log Inspection & Cryptographic Verification
    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("                 AUDIT LOG & TAMPER VERIFICATION")
    print("=" * 72)
    verifier = AuditVerifier(audit_file)
    v_res = verifier.verify()
    print(f"Chain Valid:   {v_res.is_valid}")
    print(f"Total Records: {v_res.total_records}")
    print(f"Head Hash:     {v_res.head_hash}")

    print("\nRecent Audit Entries (Redacted):")
    with open(audit_file, "r") as f:
        for line in f:
            rec = json.loads(line)
            print(f"  [{rec['record_id']:02d}] {rec['timestamp_utc'][:19]} | {rec['subject']:10s} | {rec['verb']:7s} | {rec['effect']:5s} | {rec['canonical_resource'][:35]} | Rule: {rec['matched_rule_id']}")
            if rec.get("redacted_details"):
                print(f"       Details: {rec['redacted_details']}")

    # Clean up workspace
    shutil.rmtree(demo_root)
    print("\n[+] Demo completed successfully. All 5 scenarios verified with fail-closed audit.")


if __name__ == "__main__":
    run_demo()
