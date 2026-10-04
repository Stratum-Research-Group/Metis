"""Typed asynchronous client for Kybernetes orchestrator API."""

import json
from typing import Any, AsyncIterator, Dict, Optional
import httpx

from prosopon.contracts.kybernetes import TurnEvent
from prosopon.contracts.phylax import ApprovalResponse


class KybernetesClient:
    """HTTP client communicating with Kybernetes orchestrator."""
    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", timeout_seconds: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

    async def health_check(self) -> Dict[str, Any]:
        """Verify upstream Kybernetes server readiness."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    async def submit_turn_stream(
        self,
        session_id: str,
        user_text: str,
        turn_id: Optional[str] = None,
    ) -> AsyncIterator[TurnEvent]:
        """
        Submit a conversational turn and stream Server-Sent Events (SSE).
        """
        payload = {
            "session_id": session_id,
            "turn_id": turn_id,
            "query": user_text,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/turn",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if line.startswith("data:"):
                        raw_data = line[len("data:"):].strip()
                        if raw_data:
                            event_dict = json.loads(raw_data)
                            yield TurnEvent.model_validate(event_dict)

    async def resolve_approval(self, approval_id: str, decision: str) -> Dict[str, Any]:
        """Submit Phylax approval decision (allow_once or deny)."""
        response_model = ApprovalResponse(approval_id=approval_id, decision=decision)  # type: ignore
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.base_url}/approvals/{approval_id}/resolve",
                json=response_model.model_dump(),
            )
            resp.raise_for_status()
            return resp.json()

    async def cancel_turn(self, turn_id: str) -> bool:
        """Cancel an in-flight turn."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{self.base_url}/chat/turns/{turn_id}/cancel")
                return resp.status_code == 200
        except Exception:
            return False
