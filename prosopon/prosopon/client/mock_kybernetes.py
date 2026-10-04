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
