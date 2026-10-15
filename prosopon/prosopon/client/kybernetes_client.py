"""Typed asynchronous client for Kybernetes orchestrator API."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

from prosopon.contracts.kybernetes import TurnEvent
from prosopon.contracts.phylax import ApprovalResponse


import os


class KybernetesClient:
    """HTTP client communicating with Kybernetes orchestrator."""
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081/v1",
        service_token: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ):
        url = (base_url or os.environ.get("KYBERNETES_URL", "http://127.0.0.1:8081/v1")).rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        self.base_url = url
        self.service_token = service_token or os.environ.get("KYBERNETES_SERVICE_TOKEN")
        self.timeout = timeout_seconds

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers = dict(extra or {})
        if self.service_token:
            headers["Authorization"] = f"Bearer {self.service_token}"
        return headers

    async def health_check(self) -> Dict[str, Any]:
        """Verify upstream Kybernetes server readiness."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/health", headers=self._headers())
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
        headers = self._headers({"Accept": "text/event-stream"})
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/turn",
                json=payload,
                headers=headers,
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
        headers = self._headers()
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.base_url}/approvals/{approval_id}/resolve",
                json=response_model.model_dump(),
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def cancel_turn(self, turn_id: str) -> bool:
        """Cancel an in-flight turn."""
        try:
            headers = self._headers()
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{self.base_url}/chat/turns/{turn_id}/cancel", headers=headers)
                return resp.status_code == 200
        except Exception:
            return False

    async def get_admin_overview(self, admin_token: Optional[str] = None) -> Dict[str, Any]:
        """Fetch centralized stack overview from Kybernetes admin API."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/admin/overview", headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def get_admin_modules(self, admin_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch module descriptors."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/admin/modules", headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def get_admin_module_config(self, module_id: str, admin_token: Optional[str] = None) -> Dict[str, Any]:
        """Fetch module config view."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/admin/modules/{module_id}/config", headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def validate_admin_module_config(
        self, module_id: str, proposed: Dict[str, Any], admin_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate proposed module configuration against upstream module validator."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{self.base_url}/admin/modules/{module_id}/config/validate",
                json={"proposed": proposed},
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def get_admin_sessions(self, limit: int = 50, offset: int = 0, admin_token: Optional[str] = None) -> Dict[str, Any]:
        """Fetch orchestrator sessions."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/admin/sessions?limit={limit}&offset={offset}", headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def get_admin_traces(self, limit: int = 50, offset: int = 0, admin_token: Optional[str] = None) -> Dict[str, Any]:
        """Fetch execution traces."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/admin/traces?limit={limit}&offset={offset}", headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def get_admin_audit_log(self, limit: int = 50, offset: int = 0, admin_token: Optional[str] = None) -> Dict[str, Any]:
        """Fetch admin audit log."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.base_url}/admin/audit-log?limit={limit}&offset={offset}", headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def run_admin_probe(self, component_id: str, admin_token: Optional[str] = None) -> Dict[str, Any]:
        """Trigger functional probe in Kybernetes."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.base_url}/admin/probes/run",
                json={"component_id": component_id},
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def apply_admin_module_config(
        self,
        module_id: str,
        base_hash: str,
        patch: Dict[str, Any],
        confirm_token: Optional[str] = None,
        admin_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply proposed module configuration via upstream control plane."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.base_url}/admin/modules/{module_id}/config/apply",
                json={"base_hash": base_hash, "patch": patch, "confirm_token": confirm_token},
                headers=headers,
            )
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise HTTPException(status_code=resp.status_code, detail=detail)
            return resp.json()

    async def get_admin_module_history(
        self, module_id: str, admin_token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch change history for a given module."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{self.base_url}/admin/modules/{module_id}/history",
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def rollback_admin_module_config(
        self, module_id: str, changeset_id: str, admin_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rollback configuration to snapshot preceding a specific ChangeSet."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.base_url}/admin/modules/{module_id}/config/rollback/{changeset_id}",
                headers=headers,
            )
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise HTTPException(status_code=resp.status_code, detail=detail)
            return resp.json()

    async def load_model(
        self, model: str, confirm: bool = True, admin_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Trigger audited model load via Kybernetes."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/admin/models/load",
                json={"model": model, "confirm": confirm},
                headers=headers,
            )
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise HTTPException(status_code=resp.status_code, detail=detail)
            return resp.json()

    async def unload_model(
        self, model: str, confirm: bool = True, admin_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Trigger audited model unload via Kybernetes."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/admin/models/unload",
                json={"model": model, "confirm": confirm},
                headers=headers,
            )
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise HTTPException(status_code=resp.status_code, detail=detail)
            return resp.json()

    async def get_dokime_runs(self, limit: int = 20, admin_token: Optional[str] = None) -> Dict[str, Any]:
        """Fetch historical Dokime evaluation runs."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/admin/batch/dokime/runs?limit={limit}", headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def run_dokime_batch(
        self,
        confirm_token: str,
        task_id: Optional[str] = "arc_easy_25",
        model_id: Optional[str] = "llama-3.1-8b-instruct",
        smoke_test: bool = True,
        admin_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger an audited Dokime benchmark evaluation run."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/admin/batch/dokime/run",
                json={
                    "confirm_token": confirm_token,
                    "task_id": task_id,
                    "model_id": model_id,
                    "smoke_test": smoke_test,
                },
                headers=headers,
            )
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise HTTPException(status_code=resp.status_code, detail=detail)
            return resp.json()

    async def get_paideia_runs(self, limit: int = 20, admin_token: Optional[str] = None) -> Dict[str, Any]:
        """Fetch historical Paideia fine-tuning and evidence runs."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/admin/batch/paideia/runs?limit={limit}", headers=headers)
            resp.raise_for_status()
            return resp.json()

    async def run_paideia_batch(
        self,
        confirm_token: str,
        task: str = "router",
        allow_simulated: bool = True,
        admin_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger an audited Paideia fine-tuning run."""
        token = admin_token or os.environ.get("KYBERNETES_ADMIN_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/admin/batch/paideia/run",
                json={
                    "confirm_token": confirm_token,
                    "task": task,
                    "allow_simulated": allow_simulated,
                },
                headers=headers,
            )
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise HTTPException(status_code=resp.status_code, detail=detail)
            return resp.json()
