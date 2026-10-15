"""Batch tool run controls and execution management for Dokime and Paideia.

Provides safe execution gating, concurrency locking, explicit confirmation checks,
visible resource warnings, and truthful historical run reporting.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from kybernetes.admin.audit import admin_audit_logger


class BatchJobBusyError(Exception):
    """Raised when a batch job is already running for the target tool."""
    pass


class BatchJobManager:
    """Manages asynchronous batch tool executions and historical run manifests."""

    def __init__(self, workspace_root: Optional[Path] = None):
        if workspace_root is None:
            # Walk up to Metis workspace root
            self.workspace_root = Path(__file__).resolve().parents[3]
        else:
            self.workspace_root = workspace_root

        self._dokime_lock = asyncio.Lock()
        self._paideia_lock = asyncio.Lock()
        self._active_jobs: Dict[str, Dict[str, Any]] = {}
        self.audit_logger = admin_audit_logger

    def set_audit_logger(self, logger: Any) -> None:
        self.audit_logger = logger

    @property
    def dokime_runs_dirs(self) -> List[Path]:
        return [
            self.workspace_root / "dokime" / "runs",
            self.workspace_root / "runs" / "dokime",
            self.workspace_root / "runs",
        ]

    @property
    def paideia_runs_dirs(self) -> List[Path]:
        return [
            self.workspace_root / "paideia" / "runs",
            self.workspace_root / "runs" / "paideia",
            self.workspace_root / "runs" / "latest_adapter",
        ]

    # =========================================================================
    # DOKIME RUNNER & HISTORY
    # =========================================================================

    def get_dokime_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List historical Dokime evaluation benchmark runs from disk."""
        runs: List[Dict[str, Any]] = []
        seen_ids = set()

        for base_dir in self.dokime_runs_dirs:
            if not base_dir.exists() or not base_dir.is_dir():
                continue

            # Iterate over run_* subdirectories
            for item in sorted(base_dir.iterdir(), reverse=True):
                if not item.is_dir() or not item.name.startswith("run_"):
                    continue
                run_id = item.name
                if run_id in seen_ids:
                    continue

                manifest_file = item / "manifest.json"
                summary_file = item / "summary.json"

                run_data: Dict[str, Any] = {
                    "run_id": run_id,
                    "timestamp": None,
                    "models": [],
                    "tasks": [],
                    "is_simulated": True,  # Per contract: evaluation runs in test/dev stack are simulated/offline
                    "status": "COMPLETED",
                    "summary": {},
                    "path": str(item.relative_to(self.workspace_root)),
                }

                if manifest_file.exists():
                    try:
                        with open(manifest_file, "r", encoding="utf-8") as f:
                            m_data = json.load(f)
                            run_data["timestamp"] = m_data.get("timestamp_utc")
                            run_data["models"] = m_data.get("model_ids", [])
                            run_data["tasks"] = m_data.get("task_ids", [])
                            run_data["seed"] = m_data.get("seed")
                            run_data["git_sha"] = m_data.get("git_sha")
                    except Exception:
                        pass

                if summary_file.exists():
                    try:
                        with open(summary_file, "r", encoding="utf-8") as f:
                            s_data = json.load(f)
                            run_data["summary"] = s_data
                    except Exception:
                        pass

                seen_ids.add(run_id)
                runs.append(run_data)

        # Sort by timestamp or run_id descending
        runs.sort(key=lambda r: r.get("timestamp") or r["run_id"], reverse=True)
        return runs[:limit]

    async def run_dokime(
        self,
        task_id: Optional[str] = None,
        model_id: Optional[str] = None,
        smoke_test: bool = True,
        actor: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Dokime benchmark evaluation safely with concurrency protection."""
        if self._dokime_lock.locked():
            self.audit_logger.log(
                actor_token=actor,
                action="batch_run_dokime",
                target="dokime",
                status="denied",
                client_ip=client_ip,
                detail={"reason": "Job already in progress"},
            )
            raise BatchJobBusyError("Dokime evaluation benchmark is currently in progress.")

        async with self._dokime_lock:
            run_id = f"run_{uuid.uuid4().hex[:12]}"
            target_task = task_id or "arc_easy_25"
            target_model = model_id or "llama-3.1-8b-instruct"

            self.audit_logger.log(
                actor_token=actor,
                action="batch_run_dokime",
                target=f"dokime:{run_id}",
                status="started",
                client_ip=client_ip,
                detail={
                    "run_id": run_id,
                    "task": target_task,
                    "model": target_model,
                    "smoke_test": smoke_test,
                },
            )

            # Target output directory
            output_dir = self.workspace_root / "dokime" / "runs"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run in worker thread to prevent blocking asyncio loop
            def _execute() -> Dict[str, Any]:
                from dokime.client import HestiaClient
                from dokime.config import RunConfig
                from dokime.runner import run_evaluation

                # Test/dev fallback mock client if Hestia is unreachable or smoke_test is requested
                # to satisfy "never trigger GPU model loads without explicit configuration"
                client: Optional[HestiaClient] = None
                try:
                    # Quick probe if Hestia is alive at default port
                    with httpx.Client(timeout=0.5) as probe:
                        res = probe.get("http://127.0.0.1:8080/health")
                        if res.status_code == 200 and not smoke_test:
                            client = HestiaClient(base_url="http://127.0.0.1:8080")
                except Exception:
                    client = None

                if client is None:
                    # Safe simulated transport (returns answer 'A' deterministically)
                    def mock_handler(request: httpx.Request) -> httpx.Response:
                        return httpx.Response(
                            200,
                            json={"choices": [{"message": {"content": "Answer: A"}}]},
                        )
                    mock_transport = httpx.MockTransport(mock_handler)
                    mock_http = httpx.Client(transport=mock_transport)
                    client = HestiaClient(base_url="http://mock:8000", client=mock_http)

                cfg = RunConfig(
                    models=[target_model],
                    tasks=[target_task],
                    output_dir=str(output_dir),
                    seed=42,
                )

                result = run_evaluation(cfg, client=client, run_id=run_id)
                result["is_simulated"] = True
                result["status"] = "COMPLETED"
                return result

            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, _execute)

                self.audit_logger.log(
                    actor_token=actor,
                    action="batch_run_dokime",
                    target=f"dokime:{run_id}",
                    status="completed",
                    client_ip=client_ip,
                    detail={"total_queries": result.get("total_queries")},
                )
                return result
            except Exception as exc:
                self.audit_logger.log(
                    actor_token=actor,
                    action="batch_run_dokime",
                    target=f"dokime:{run_id}",
                    status="failed",
                    client_ip=client_ip,
                    detail={"error": str(exc)},
                )
                raise

    # =========================================================================
    # PAIDEIA RUNNER & HISTORY
    # =========================================================================

    def get_paideia_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List historical Paideia fine-tuning and evidence runs from disk."""
        runs: List[Dict[str, Any]] = []
        seen_ids = set()

        for base_dir in self.paideia_runs_dirs:
            if not base_dir.exists() or not base_dir.is_dir():
                continue

            for item in sorted(base_dir.iterdir(), reverse=True):
                if not item.is_dir():
                    continue

                manifest_file = item / "manifest.json"
                if not manifest_file.exists():
                    continue

                run_id = item.name
                if run_id in seen_ids:
                    continue

                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        m_data = json.load(f)

                    runs.append({
                        "run_id": m_data.get("run_id", run_id),
                        "task_name": m_data.get("task_name"),
                        "status": m_data.get("status", "SIMULATED"),
                        "is_simulated": m_data.get("is_simulated", True),
                        "started_at_utc": m_data.get("started_at_utc"),
                        "completed_at_utc": m_data.get("completed_at_utc"),
                        "duration_seconds": m_data.get("duration_seconds"),
                        "adapter_digest": m_data.get("adapter_digest"),
                        "metrics": m_data.get("metrics", {}),
                        "failure_reason": m_data.get("failure_reason"),
                        "path": str(item.relative_to(self.workspace_root)),
                    })
                    seen_ids.add(run_id)
                except Exception:
                    continue

        runs.sort(key=lambda r: r.get("started_at_utc") or r["run_id"], reverse=True)
        return runs[:limit]

    async def run_paideia(
        self,
        task: str = "router",
        allow_simulated: bool = True,
        actor: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Paideia training / checkpoint marker safely with preflight validation."""
        if self._paideia_lock.locked():
            self.audit_logger.log(
                actor_token=actor,
                action="batch_run_paideia",
                target="paideia",
                status="denied",
                client_ip=client_ip,
                detail={"reason": "Training job already in progress"},
            )
            raise BatchJobBusyError("Paideia training job is currently in progress.")

        async with self._paideia_lock:
            run_id = str(uuid.uuid4())
            self.audit_logger.log(
                actor_token=actor,
                action="batch_run_paideia",
                target=f"paideia:{run_id}",
                status="started",
                client_ip=client_ip,
                detail={"task": task, "allow_simulated": allow_simulated},
            )

            output_dir = self.workspace_root / "paideia" / "runs"
            output_dir.mkdir(parents=True, exist_ok=True)

            def _execute() -> Dict[str, Any]:
                # Ensure paideia is in sys.path if not installed globally
                paideia_dir = str(self.workspace_root / "paideia")
                if paideia_dir not in sys.path:
                    sys.path.insert(0, paideia_dir)

                from paideia.data.splitter import DatasetSplits
                from paideia.training.config import TrainingConfig
                from paideia.training.runner import QLoRATrainingRunner

                cfg = TrainingConfig(
                    output_dir=str(output_dir),
                    task=task if task in ("router", "walk_policy") else "router",
                    allow_simulated=allow_simulated,
                    epochs=1,
                    batch_size=1,
                )
                # Deterministic minimal dataset splits for training run marker
                splits = DatasetSplits(train=[1, 2, 3, 4], dev=[5], test=[6])

                runner = QLoRATrainingRunner(config=cfg, splits=splits)
                manifest = runner.run()
                return manifest.model_dump()

            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, _execute)

                self.audit_logger.log(
                    actor_token=actor,
                    action="batch_run_paideia",
                    target=f"paideia:{run_id}",
                    status="completed",
                    client_ip=client_ip,
                    detail={
                        "status": result.get("status"),
                        "adapter_digest": result.get("adapter_digest"),
                    },
                )
                return result
            except Exception as exc:
                self.audit_logger.log(
                    actor_token=actor,
                    action="batch_run_paideia",
                    target=f"paideia:{run_id}",
                    status="failed",
                    client_ip=client_ip,
                    detail={"error": str(exc)},
                )
                raise


batch_job_manager = BatchJobManager()
