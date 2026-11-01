"""Realistic local mock implementation of Kybernetes orchestrator for development and testing."""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional

from prosopon.contracts.kybernetes import Citation, TurnEvent
from prosopon.contracts.phylax import ApprovalRequest


class MockKybernetesClient:
    """
    Simulates Kybernetes orchestrator turn streaming, deliberation, citations, and Phylax approvals.
    """
    def __init__(self):
        self._pending_approvals: Dict[str, ApprovalRequest] = {}
        self._approval_events: Dict[str, asyncio.Event] = {}
        self._approval_decisions: Dict[str, str] = {}

    async def health_check(self) -> Dict[str, Any]:
        return {"status": "ok", "mode": "local_mock", "version": "1.0.0"}

    async def get_admin_overview(self, admin_token: Optional[str] = None) -> Dict[str, Any]:
        return {
            "service": "kybernetes",
            "version": "0.1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git": {"commit": "mock-dev", "dirty": False},
            "stack_compatibility": "unchecked",
            "budget_storage_mode": "in-memory",
            "orchestrator_mode": "mock",
            "simulation_banner": True,
            "mock_banner": True,
            "components": {
                "kybernetes": {"state": "mock", "detail": "Mock orchestrator client active", "freshness": "fresh", "checked_at": datetime.now(timezone.utc).isoformat(), "evidence": "In-memory mock", "is_simulated": True},
                "hestia": {"state": "mock", "detail": "Simulated local model pool (llama-3.1-8b-instruct)", "freshness": "fresh", "checked_at": datetime.now(timezone.utc).isoformat(), "evidence": "Simulated inference", "is_simulated": True},
                "lthcm": {"state": "simulated", "detail": "In-memory mock corpus", "freshness": "fresh", "checked_at": datetime.now(timezone.utc).isoformat(), "evidence": "Mock adapter", "is_simulated": True},
                "dianoia": {"state": "simulated", "detail": "Mock thought-graph deliberation", "freshness": "fresh", "checked_at": datetime.now(timezone.utc).isoformat(), "evidence": "Mock deliberation", "is_simulated": True},
                "phylax": {"state": "simulated", "detail": "Mock approval manager", "freshness": "fresh", "checked_at": datetime.now(timezone.utc).isoformat(), "evidence": "Mock approvals", "is_simulated": True},
                "krisis": {"state": "simulated", "detail": "Mock verification gates", "freshness": "fresh", "checked_at": datetime.now(timezone.utc).isoformat(), "evidence": "Mock verifier", "is_simulated": True},
            },
        }

    async def get_admin_modules(self, admin_token: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            {"id": "kybernetes", "display_name": "Kybernetes Orchestrator", "section": "Overview", "kind": "service", "config_format": "pydantic"},
            {"id": "hestia", "display_name": "Hestia Inference Server", "section": "Models & Resources", "kind": "service", "config_format": "yaml"},
            {"id": "prosopon", "display_name": "Prosopon Frontend & BFF", "section": "Overview", "kind": "service", "config_format": "pydantic"},
            {"id": "lthcm", "display_name": "LTHCM Vector Memory", "section": "Memory & Ingestion", "kind": "library", "config_format": "none"},
            {"id": "dianoia", "display_name": "Dianoia Deliberation", "section": "Reasoning", "kind": "library", "config_format": "none"},
            {"id": "krisis", "display_name": "Krisis Verifier", "section": "Reasoning", "kind": "library", "config_format": "json"},
            {"id": "phylax", "display_name": "Phylax Guardrails", "section": "Guardrails", "kind": "library", "config_format": "yaml"},
            {"id": "dokime", "display_name": "Dokime Benchmarking", "section": "Evaluation", "kind": "batch_tool", "config_format": "toml"},
            {"id": "stoma", "display_name": "Stoma Ingestion", "section": "Memory & Ingestion", "kind": "batch_tool", "config_format": "yaml"},
            {"id": "paideia", "display_name": "Paideia Trainer", "section": "Training", "kind": "batch_tool", "config_format": "none"},
            {"id": "kleroterion", "display_name": "Kleroterion Consensus", "section": "Reasoning", "kind": "library", "config_format": "none"},
        ]

    async def get_admin_module_config(self, module_id: str, admin_token: Optional[str] = None) -> Dict[str, Any]:
        return {
            "module_id": module_id,
            "saved": {"mode": "mock"},
            "effective": {"mode": "mock"},
            "sources": {"mode": "mock_default"},
            "drift": False,
        }

    async def validate_admin_module_config(
        self, module_id: str, proposed: Dict[str, Any], admin_token: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "module_id": module_id,
            "valid": True,
            "errors": [],
            "warnings": ["Mock validator invoked"],
            "restart_required": True,
        }

    async def get_admin_sessions(self, limit: int = 50, offset: int = 0, admin_token: Optional[str] = None) -> Dict[str, Any]:
        return {"total": 0, "limit": limit, "offset": offset, "storage": "in-memory mock", "sessions": []}

    async def get_admin_traces(self, limit: int = 50, offset: int = 0, admin_token: Optional[str] = None) -> Dict[str, Any]:
        return {"total": 0, "limit": limit, "offset": offset, "storage": "in-memory mock", "traces": []}

    async def get_admin_audit_log(self, limit: int = 50, offset: int = 0, admin_token: Optional[str] = None) -> Dict[str, Any]:
        return {"total": 0, "limit": limit, "offset": offset, "entries": []}

    async def run_admin_probe(self, component_id: str, admin_token: Optional[str] = None) -> Dict[str, Any]:
        return {
            "state": "mock",
            "detail": f"Mock functional probe for {component_id} passed",
            "freshness": "fresh",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": 5,
            "evidence": "Mock probe simulator",
            "is_simulated": True,
        }

    async def submit_turn_stream(
        self,
        session_id: str,
        user_text: str,
        turn_id: Optional[str] = None,
    ) -> AsyncIterator[TurnEvent]:
        tid = turn_id or str(uuid.uuid4())
        seq = 0

        # 1. Event: started
        yield TurnEvent(
            turn_id=tid,
            seq_num=seq,
            event_type="started",
            payload={"session_id": session_id, "query": user_text},
            status="draft",
        )
        seq += 1
        await asyncio.sleep(0.05)

        # 2. Event: plan_selected
        q_lower = user_text.lower()
        if "compare" in q_lower or "both" in q_lower or " or " in q_lower:
            plan = "flat_rerank"
            reason = "Direct comparison detected; bypassing graph walk."
        else:
            plan = "dianoia_walk"
            reason = "Multi-hop dependency detected; initiating bounded deliberation walk."

        yield TurnEvent(
            turn_id=tid,
            seq_num=seq,
            event_type="plan_selected",
            payload={"route": plan, "reason": reason, "components": ["LTHCM", "Dianoia", "Hestia"]},
            status="draft",
        )
        seq += 1
        await asyncio.sleep(0.05)

        # 3. Check for side-effect / approval trigger (e.g. "write file", "save", "delete", "export")
        if any(kw in q_lower for kw in ["write file", "save", "delete", "export report"]):
            app_id = f"app-{uuid.uuid4().hex[:8]}"
            exp = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

            app_req = ApprovalRequest(
                approval_id=app_id,
                turn_id=tid,
                action="WRITE_FILE",
                target="/tmp/metis_report.txt",
                argument_summary={"path": "/tmp/metis_report.txt", "bytes": 1024, "operation": "overwrite"},
                policy_rule="PhylaxRule: Require user consent for local filesystem modification",
                expiry_utc=exp,
                status="pending",
            )
            self._pending_approvals[app_id] = app_req
            event_sync = asyncio.Event()
            self._approval_events[app_id] = event_sync

            yield TurnEvent(
                turn_id=tid,
                seq_num=seq,
                event_type="ask_approval",
                payload={"approval": app_req.model_dump()},
                status="draft",
            )
            seq += 1

            # Wait for user decision
            try:
                await asyncio.wait_for(event_sync.wait(), timeout=30.0)
                decision = self._approval_decisions.get(app_id, "deny")
            except asyncio.TimeoutError:
                decision = "deny"

            yield TurnEvent(
                turn_id=tid,
                seq_num=seq,
                event_type="approval_resolved",
                payload={"approval_id": app_id, "decision": decision},
                status="draft",
            )
            seq += 1

            if decision == "deny":
                msg = "Action was denied by user. Operation safely cancelled without side-effects."
                for word in msg.split():
                    yield TurnEvent(
                        turn_id=tid,
                        seq_num=seq,
                        event_type="delta",
                        payload={"text": word + " "},
                        status="draft",
                    )
                    seq += 1
                    await asyncio.sleep(0.02)
                yield TurnEvent(
                    turn_id=tid,
                    seq_num=seq,
                    event_type="completed",
                    payload={"route_summary": {"route": plan, "action_status": "denied"}},
                    status="verified",
                )
                return

        # 4. Normal query evidence
        citation1 = Citation(
            citation_id="[1]",
            passage_id="pass_7890",
            source_id="doc_metis_architecture",
            source_title="Metis Stack Specification",
            text="Prosopon is the single-owner local interface layer communicating strictly with Kybernetes.",
            score=0.94,
        )
        yield TurnEvent(
            turn_id=tid,
            seq_num=seq,
            event_type="evidence",
            payload={"citations": [citation1.model_dump()]},
            status="draft",
        )
        seq += 1
        await asyncio.sleep(0.05)

        # 5. Delta tokens
        response_text = (
            f"Based on the local knowledge base [1], Prosopon provides authenticated text and voice deliberation "
            f"while enforcing strict Phylax approvals. The query '{user_text}' was routed via {plan}."
        )
        for token in response_text.split():
            yield TurnEvent(
                turn_id=tid,
                seq_num=seq,
                event_type="delta",
                payload={"text": token + " "},
                status="draft",
            )
            seq += 1
            await asyncio.sleep(0.015)

        # 6. Completed event with verified outcome
        yield TurnEvent(
            turn_id=tid,
            seq_num=seq,
            event_type="completed",
            payload={
                "route_summary": {
                    "route": plan,
                    "components_used": ["LTHCM", "Dianoia", "Hestia"],
                    "latency_ms": 142.5,
                    "verifier_status": "verified",
                }
            },
            status="verified",
        )

    async def resolve_approval(self, approval_id: str, decision: str) -> Dict[str, Any]:
        self._approval_decisions[approval_id] = decision
        if approval_id in self._pending_approvals:
            self._pending_approvals[approval_id].status = "approved" if decision == "allow_once" else "denied"
        if approval_id in self._approval_events:
            self._approval_events[approval_id].set()
        return {"status": "ok", "approval_id": approval_id, "decision": decision}

    async def cancel_turn(self, turn_id: str) -> bool:
        return True

    async def apply_admin_module_config(
        self,
        module_id: str,
        base_hash: str,
        patch: Dict[str, Any],
        confirm_token: Optional[str] = None,
        admin_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "id": f"cs-mock-{module_id}-0001",
            "module": module_id,
            "base_hash": base_hash,
            "patch": patch,
            "requested_by": "mock-admin",
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "status": "restart_pending",
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "prev_snapshot_id": f"snap_mock_{module_id}_0001",
            "files_affected": [f"{module_id}.yaml"],
            "restart_required": True,
        }

    async def get_admin_module_history(
        self, module_id: str, admin_token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return [
            {
                "id": f"cs-mock-{module_id}-0001",
                "module": module_id,
                "base_hash": "mock-base-hash",
                "patch": {"mode": "mock"},
                "requested_by": "mock-admin",
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "status": "restart_pending",
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "prev_snapshot_id": f"snap_mock_{module_id}_0001",
                "files_affected": [f"{module_id}.yaml"],
                "restart_required": True,
            }
        ]

    async def rollback_admin_module_config(
        self, module_id: str, changeset_id: str, admin_token: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            "id": f"cs-rollback-mock-{module_id}-0002",
            "module": module_id,
            "base_hash": "mock-hash-prev",
            "patch": {"mode": "mock-restored"},
            "requested_by": "mock-admin",
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "status": "rolled_back_on_disk",
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "prev_snapshot_id": f"snap_mock_rb_{module_id}_0002",
            "files_affected": [f"{module_id}.yaml"],
            "restart_required": True,
        }

    async def load_model(
        self, model: str, confirm: bool = True, admin_token: Optional[str] = None
    ) -> Dict[str, Any]:
        return {"status": "loaded", "model": model, "mode": "mock"}

    async def unload_model(
        self, model: str, confirm: bool = True, admin_token: Optional[str] = None
    ) -> Dict[str, Any]:
        return {"status": "unloaded", "model": model, "mode": "mock"}

    async def get_dokime_runs(self, limit: int = 20, admin_token: Optional[str] = None) -> Dict[str, Any]:
        return {
            "total": 1,
            "limit": limit,
            "tool": "dokime",
            "runs": [
                {
                    "run_id": "run_mock_dokime_001",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "models": ["mock-llama"],
                    "tasks": ["arc_easy_25"],
                    "is_simulated": True,
                    "status": "COMPLETED",
                    "summary": {
                        "mock-llama/arc_easy_25": {
                            "mean": 0.85,
                            "ci_lower": 0.72,
                            "ci_upper": 0.94,
                            "count": 25,
                        }
                    },
                }
            ],
        }

    async def run_dokime_batch(
        self,
        confirm_token: str,
        task_id: Optional[str] = "arc_easy_25",
        model_id: Optional[str] = "llama-3.1-8b-instruct",
        smoke_test: bool = True,
        admin_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "success",
            "tool": "dokime",
            "run": {
                "run_id": "run_mock_dokime_002",
                "total_queries": 25,
                "is_simulated": True,
                "status": "COMPLETED",
            },
        }

    async def get_paideia_runs(self, limit: int = 20, admin_token: Optional[str] = None) -> Dict[str, Any]:
        return {
            "total": 1,
            "limit": limit,
            "tool": "paideia",
            "runs": [
                {
                    "run_id": "mock-paideia-run-001",
                    "task_name": "qlora_router",
                    "status": "SIMULATED",
                    "is_simulated": True,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": 1.25,
                    "adapter_digest": "mock-adapter-digest-sha256",
                }
            ],
        }

    async def run_paideia_batch(
        self,
        confirm_token: str,
        task: str = "router",
        allow_simulated: bool = True,
        admin_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "success",
            "tool": "paideia",
            "run": {
                "run_id": "mock-paideia-run-002",
                "task_name": f"qlora_{task}",
                "status": "SIMULATED",
                "is_simulated": True,
                "duration_seconds": 0.85,
                "adapter_digest": "mock-new-adapter-sha256",
            },
        }

    async def get_lochos_squad(self) -> Dict[str, Any]:
        return {
            "rig_name": "metis-sovereign-squad",
            "version": "0.2",
            "summary": "Mock sovereign engineering squad",
            "seats": [
                {
                    "id": "coder",
                    "label": "Autonomous Coding Agent",
                    "role": "Primary code implementation",
                    "model": "qwen2.5-coder-32b",
                    "state": "idle",
                    "worktree_branch": "squad/coder",
                    "skills": ["python", "refactoring"],
                    "current_task": None,
                },
                {
                    "id": "reviewer",
                    "label": "Independent Review Agent",
                    "role": "Zero-trust diff auditing",
                    "model": "qwen2.5-coder-32b",
                    "state": "idle",
                    "worktree_branch": "squad/reviewer",
                    "skills": ["security", "pytest"],
                    "current_task": None,
                },
            ],
            "active_tasks": [],
            "attention_required": [],
        }

    async def get_lochos_queue(self, status: Optional[str] = None, seat_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return []

    async def dispatch_lochos_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": "mock-task-123",
            "status": "in_progress",
            "title": payload.get("title", "Mock Task"),
            "assigned_to": payload.get("target_seat", "coder"),
        }

    async def transfer_lochos_baton(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": payload.get("task_id", "mock-task"),
            "status": "transferred",
            "baton_owner": payload.get("to_seat", "reviewer"),
            "note": payload.get("note", ""),
        }

    async def resolve_lochos_attention(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": payload.get("task_id", "mock-task"),
            "status": "in_progress",
            "resolved": True,
        }

    async def run_lochos_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": "mock-pipeline-task-001",
            "success": True,
            "stage": "completed",
            "baton_owner": payload.get("reviewer_seat", "reviewer"),
            "milestones": ["baseline_checkpoint", "coding_complete", "verification_passed"],
            "rolled_back": False,
            "verification_passed": True,
            "details": {"mission": payload.get("mission")},
        }

    async def complete_lochos_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": payload.get("task_id", "mock-task"),
            "status": "completed",
            "result": payload.get("result", {"status": "ok"}),
        }

    # --- Pinax Tool Catalogue Mock Methods ---
    async def get_pinax_tools(
        self,
        query: str = "",
        category: Optional[str] = None,
        risk_tier: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        mock_tools = [
            {
                "id": "tool:rg",
                "name": "rg",
                "summary": "Recursively searches directories for a regex pattern",
                "category": "text",
                "risk_tier": "T0",
                "install_state": "installed",
                "kind": "binary",
                "has_structured_signature": False,
            },
            {
                "id": "tool:ls",
                "name": "ls",
                "summary": "List directory contents",
                "category": "filesystem",
                "risk_tier": "T0",
                "install_state": "installed",
                "kind": "binary",
                "has_structured_signature": False,
            },
            {
                "id": "tool:git",
                "name": "git",
                "summary": "Fast, scalable, distributed revision control system",
                "category": "vcs",
                "risk_tier": "T0",
                "install_state": "installed",
                "kind": "binary",
                "has_structured_signature": False,
            },
            {
                "id": "tool:curl",
                "name": "curl",
                "summary": "Command line tool for transferring data with URL syntax",
                "category": "network",
                "risk_tier": "T2",
                "install_state": "installed",
                "kind": "binary",
                "has_structured_signature": False,
            },
            {
                "id": "tool:rm",
                "name": "rm",
                "summary": "Remove files or directories",
                "category": "filesystem",
                "risk_tier": "T3",
                "install_state": "installed",
                "kind": "binary",
                "has_structured_signature": False,
            },
        ]
        res = mock_tools
        if query:
            q = query.lower()
            res = [t for t in res if q in t["name"].lower() or q in t["summary"].lower()]
        if category:
            res = [t for t in res if t["category"] == category]
        if risk_tier:
            res = [t for t in res if t["risk_tier"] == risk_tier]
        return res[:limit]

    async def get_pinax_tool_detail(self, tool_id: str) -> Dict[str, Any]:
        name = tool_id.replace("tool:", "")
        return {
            "id": f"tool:{name}",
            "name": name,
            "host_id": "localhost",
            "kind": "binary",
            "entry_kind": "installed",
            "summary": f"Detailed specification for {name}",
            "category": "filesystem" if name in ("ls", "rm") else ("network" if name == "curl" else "text"),
            "risk_tier": "T3" if name == "rm" else ("T2" if name == "curl" else "T0"),
            "install_state": "installed",
            "invocation_path": f"/usr/bin/{name}",
            "resolved_path": f"/usr/bin/{name}",
            "sha256": "abcdef0123456789" * 4,
            "discovered_by": "curated",
            "capabilities": [{"capability": "file-read", "risk_weight": 0.1}],
            "risk_overrides": [
                {"pattern_kind": "flag", "pattern": "-T", "tier": "T3", "note": "File upload egress"}
            ] if name == "curl" else [],
            "signatures": [],
            "docs": [],
        }

    async def get_pinax_tool_docs(self, tool_id: str, max_bytes: int = 4096) -> Dict[str, Any]:
        name = tool_id.replace("tool:", "")
        return {
            "tool": name,
            "found": True,
            "doc_kind": "man_page",
            "locator": f"/usr/share/man/man1/{name}.1.gz",
            "content": f"{name.upper()}(1) User Commands\n\nNAME\n       {name} - command utility\n\nDESCRIPTION\n       Mock documentation slice for {name}.",
            "truncated": False,
            "size_bytes": 128,
        }

    async def trigger_pinax_scan(self, path_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "id": "mock-scan-01",
            "host_id": "localhost",
            "started_at": 1700000000.0,
            "finished_at": 1700000000.15,
            "elapsed_seconds": 0.15,
            "scanner_version": "0.1.0",
            "added": 12,
            "removed": 0,
            "changed": 35,
            "installed_updated": 35,
            "unvetted_candidates": 12,
            "total_discovered": 47,
        }

    async def evaluate_pinax_invocation(self, tool_name: str, args: List[str]) -> Dict[str, Any]:
        tier = "T0"
        req_approval = False
        reason = "Base catalogue tier"
        matched = []
        if tool_name == "curl" and any(a in ("-T", "--upload-file", "-d") for a in args):
            tier = "T3"
            req_approval = True
            reason = "Override: File upload egress"
            matched.append({"pattern": "-T", "tier": "T3"})
        elif tool_name == "rm":
            tier = "T3"
            req_approval = True
            reason = "Base destructive tool"
        return {
            "tool": tool_name,
            "args": args,
            "tier": tier,
            "requires_approval": req_approval,
            "reason": reason,
            "matched_overrides": matched,
        }


