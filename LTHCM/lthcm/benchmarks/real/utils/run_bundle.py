"""
Replayable Run Bundle & Artifact Persistence
===========================================
Captures exact experimental provenance, environment locks, dataset checksums,
per-query records, and ranked TREC run outputs for reproducible benchmarking.
"""

import os
import sys
import json
import time
import hashlib
import platform
import subprocess
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch


def get_git_info() -> Dict[str, Any]:
    """Retrieve current Git commit SHA and dirty working tree status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = len(status) > 0
    except Exception:
        dirty = False

    return {"commit": commit, "dirty": dirty}


def get_system_hardware_info() -> Dict[str, Any]:
    """Capture hardware and runtime platform specifications."""
    info = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    try:
        import psutil
        info["total_ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        pass
    return info


def compute_content_sha256(content: Union[str, bytes]) -> str:
    """Compute SHA-256 hash of string or byte buffer."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class RunBundleBuilder:
    """
    Constructs an immutable, auditable experiment bundle for a benchmark suite.
    """

    def __init__(
        self,
        benchmark_name: str,
        dataset_name: str,
        split: str,
        run_id: str,
        seed: int = 42,
        output_dir: str = "runs",
    ):
        self.benchmark_name = benchmark_name
        self.dataset_name = dataset_name
        self.split = split
        self.run_id = run_id
        self.seed = seed
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.metadata = {
            "run_id": run_id,
            "benchmark_name": benchmark_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git": get_git_info(),
            "hardware": get_system_hardware_info(),
            "seed": seed,
        }
        self.dataset_info = {
            "name": dataset_name,
            "split": split,
            "revision": "latest",
            "checksum": None,
            "total_queries": 0,
            "query_ids": [],
            "corpus_size": 0,
        }
        self.models_info: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
        self.timings: Dict[str, Any] = {}
        self.per_query_records: List[Dict[str, Any]] = []
        self.per_arm_metrics: Dict[str, Any] = {}
        self.significance_tests: Dict[str, Any] = {}
        self.extra_artifacts: Dict[str, str] = {}

    def set_dataset_provenance(
        self,
        query_ids: List[str],
        corpus_size: int,
        revision: Optional[str] = None,
        checksum: Optional[str] = None,
    ):
        self.dataset_info["total_queries"] = len(query_ids)
        self.dataset_info["query_ids"] = list(query_ids)
        self.dataset_info["corpus_size"] = corpus_size
        if revision:
            self.dataset_info["revision"] = revision
        if checksum:
            self.dataset_info["checksum"] = checksum

    def set_models(self, encoder: str, reranker: Optional[str] = None):
        self.models_info["encoder"] = encoder
        if reranker:
            self.models_info["reranker"] = reranker

    def add_query_record(
        self,
        query_id: str,
        question: str,
        gold_targets: List[str],
        ranked_predictions: Dict[str, List[str]],
        metrics_by_arm: Dict[str, Dict[str, float]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        rec = {
            "query_id": str(query_id),
            "question": str(question),
            "gold_targets": list(gold_targets),
            "ranked_predictions": ranked_predictions,
            "metrics": metrics_by_arm,
            "metadata": metadata or {},
        }
        self.per_query_records.append(rec)

    def write_trec_run(self, arm_name: str, runs: Dict[str, List[str]], scores: Optional[Dict[str, List[float]]] = None) -> str:
        """
        Write standard TREC format run file:
        <query_id> Q0 <doc_id> <rank> <score> <run_tag>
        """
        fname = f"{self.benchmark_name}_{arm_name}.run"
        fpath = os.path.join(self.output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            for qid, docs in runs.items():
                q_scores = scores.get(qid, [1.0 / (r + 1) for r in range(len(docs))]) if scores else [1.0 / (r + 1) for r in range(len(docs))]
                for rank, (doc_id, score) in enumerate(zip(docs, q_scores), start=1):
                    f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {arm_name}\n")
        self.extra_artifacts[f"trec_run_{arm_name}"] = fname
        return fpath

    def finalize(
        self,
        summary_results: Dict[str, Any],
        summary_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save summary JSON, per-query records JSONL, and manifest with checksums.
        """
        # Attach provenance fields into summary
        summary_results["run_id"] = self.run_id
        summary_results["is_synthetic"] = summary_results.get("is_synthetic", False)
        summary_results["provenance"] = {
            "created_at": self.metadata["created_at"],
            "git": self.metadata["git"],
            "hardware": self.metadata["hardware"],
            "dataset": self.dataset_info,
            "models": self.models_info,
            "seed": self.seed,
        }

        # 1. Write summary JSON
        s_name = summary_filename or f"{self.benchmark_name}_summary.json"
        s_path = os.path.join(self.output_dir, s_name)
        with open(s_path, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2)

        # 2. Write per-query records JSONL
        pq_name = f"{self.benchmark_name}_per_query.jsonl"
        pq_path = os.path.join(self.output_dir, pq_name)
        with open(pq_path, "w", encoding="utf-8") as f:
            for rec in self.per_query_records:
                f.write(json.dumps(rec) + "\n")

        # 3. Write manifest
        manifest = {
            "run_id": self.run_id,
            "benchmark": self.benchmark_name,
            "files": {
                s_name: compute_file_sha256(s_path),
                pq_name: compute_file_sha256(pq_path),
            },
        }
        for label, fname in self.extra_artifacts.items():
            f_full = os.path.join(self.output_dir, fname)
            if os.path.isfile(f_full):
                manifest["files"][fname] = compute_file_sha256(f_full)

        m_path = os.path.join(self.output_dir, f"{self.benchmark_name}_manifest.json")
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return summary_results
