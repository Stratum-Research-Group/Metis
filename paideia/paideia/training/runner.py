"""Atomic QLoRA / SFT training runner with OOM handling and manifest provenance."""

import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from paideia.contracts.manifest_schema import RunManifest
from paideia.data.splitter import DatasetSplits
from paideia.training.config import TrainingConfig
from paideia.training.hardware import (
    acquire_gpu_lock,
    inspect_hardware,
    release_gpu_lock,
)


class TrainingPreflightError(Exception):
    """Raised when hardware, VRAM, or configuration preflight check fails."""
    pass


def get_git_info() -> Dict[str, Any]:
    """Retrieve git HEAD commit and dirty status if inside a git repo."""
    commit = None
    dirty = False
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = len(status_out) > 0
    except Exception:
        pass
    return {"git_commit": commit, "dirty_tree": dirty}


def compute_dir_sha256(directory: Path, exclude_files: Optional[Any] = None) -> str:
    """
    Compute deterministic collective sha256 checksum of files in a directory.
    - Stable sorted traversal of directory trees.
    - Encodes relative paths into hash to prevent layout collisions.
    - Excludes registry metadata files (e.g. hestia_registry_entry.json).
    - Fails closed on unreadable files instead of silently skipping.
    """
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory for digest computation does not exist or is not a directory: {directory}")

    hasher = hashlib.sha256()
    exclude = set(exclude_files) if exclude_files else set()

    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for f in sorted(files):
            if f in exclude:
                continue
            fp = Path(root) / f
            rel_path = fp.relative_to(directory).as_posix()
            hasher.update(rel_path.encode("utf-8"))
            try:
                with open(fp, "rb") as fh:
                    while chunk := fh.read(65536):
                        hasher.update(chunk)
            except Exception as e:
                raise OSError(f"Failed to read file for digest calculation: {fp} ({e})") from e

    return hasher.hexdigest()



class QLoRATrainingRunner:
    """
    Manages local fine-tuning execution, safety gates, and manifest generation.
    """
    def __init__(self, config: TrainingConfig, splits: DatasetSplits):
        self.config = config
        self.splits = splits
        self.run_id = str(uuid.uuid4())
        self.output_dir = Path(config.output_dir) / self.run_id
        self.manifest_path = self.output_dir / "manifest.json"

    def preflight_check(self, required_free_vram_mb: float = 12000.0) -> Dict[str, Any]:
        """
        Verify VRAM admission, lock coordination, and backend availability.
        Refuses execution if preconditions are violated.
        """
        hardware = inspect_hardware()

        # If on GPU, check VRAM
        if hardware.cuda_available:
            if hardware.free_vram_mb < required_free_vram_mb:
                raise TrainingPreflightError(
                    f"Insufficient free VRAM: {hardware.free_vram_mb:.0f} MB available, "
                    f"minimum {required_free_vram_mb:.0f} MB required. "
                    "Ensure Hestia serving model is unloaded before starting training."
                )
            
            # Acquire exclusive lock
            if not acquire_gpu_lock():
                raise TrainingPreflightError(
                    "Could not acquire GPU lock (/tmp/metis_gpu.lock). "
                    "Another training process or Hestia operation is holding the GPU."
                )

        return hardware.model_dump()

    def run(self) -> RunManifest:
        """
        Execute training with atomic manifest lifecycle and safe exception handling.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        git_info = get_git_info()
        hardware_info = inspect_hardware()

        manifest = RunManifest(
            run_id=self.run_id,
            task_name=f"qlora_{self.config.task}",
            status="RUNNING",
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            git_commit=git_info["git_commit"],
            dirty_tree=git_info["dirty_tree"],
            hardware_line=hardware_info.hardware_line,
            package_versions={
                "python": sys.version.split()[0],
                "paideia": "0.1.0",
            },
            dataset_checksum=self.splits.train_checksum,
            split_checksums={
                "train": self.splits.train_checksum,
                "dev": self.splits.dev_checksum,
                "test": self.splits.test_checksum,
            },
            split_counts={
                "train": len(self.splits.train),
                "dev": len(self.splits.dev),
                "test": len(self.splits.test),
            },
            seed=self.config.seed,
            hyperparameters=self.config.model_dump(),
        )

        # Write atomic initial manifest
        self._write_manifest(manifest)
        t0 = time.perf_counter()

        try:
            # 1. Run Preflight
            self.preflight_check()

            # 2. Check if PyTorch / GPU training packages are installed
            try:
                import torch
            except (ImportError, ModuleNotFoundError):
                torch = None

            # In v1 / smoke profile: simulate or invoke backend
            # Real training execution:
            # If backend is unsloth and supported:
            #   use unsloth FastLanguageModel
            # else:
            #   use peft + bitsandbytes fallback
            # We record checkpointing
            checkpoint_dir = self.output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint state marker
            with open(checkpoint_dir / "adapter_config.json", "w") as f:
                json.dump({
                    "base_model": self.config.base_model_name,
                    "rank": self.config.rank,
                    "lora_alpha": self.config.lora_alpha,
                    "target_modules": self.config.target_modules,
                    "task": self.config.task,
                }, f, indent=2)

            t_elapsed = time.perf_counter() - t0
            manifest.status = "SIMULATED"
            manifest.is_simulated = True
            manifest.completed_at_utc = datetime.now(timezone.utc).isoformat()
            manifest.duration_seconds = t_elapsed
            manifest.adapter_digest = compute_dir_sha256(checkpoint_dir, exclude_files={"manifest.json", "hestia_registry_entry.json"})
            manifest.metrics = {
                "train_samples": len(self.splits.train),
                "epochs_completed": 0,
                "simulated": True,
                "note": "Marker-only checkpoint written without optimizer steps or weight updates.",
                "peak_vram_mb": hardware_info.total_vram_mb - hardware_info.free_vram_mb,
            }

        except Exception as exc:
            t_elapsed = time.perf_counter() - t0
            manifest.status = "FAILED"
            manifest.completed_at_utc = datetime.now(timezone.utc).isoformat()
            manifest.duration_seconds = t_elapsed
            manifest.failure_reason = f"{type(exc).__name__}: {str(exc)}"
        finally:
            release_gpu_lock()
            self._write_manifest(manifest)

        return manifest

    def _write_manifest(self, manifest: RunManifest) -> None:
        """Write manifest atomically via temporary file."""
        tmp_path = self.manifest_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        tmp_path.replace(self.manifest_path)
