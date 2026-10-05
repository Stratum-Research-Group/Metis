"""Hardware discovery, VRAM measurement, and GPU admission coordination."""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, Field


class HardwareProfile(BaseModel):
    """Runtime hardware environment status."""
    cuda_available: bool = False
    device_name: str = "CPU"
    compute_capability: Tuple[int, int] = (0, 0)
    total_vram_mb: float = 0.0
    free_vram_mb: float = 0.0
    unsloth_supported: bool = False
    hardware_line: str = "CPU-only"


DEFAULT_LOCK_PATH = Path("/tmp/metis_gpu.lock")


def inspect_hardware() -> HardwareProfile:
    """Inspect local hardware and detect GPU capabilities."""
    profile = HardwareProfile()

    try:
        import torch
        if torch.cuda.is_available():
            profile.cuda_available = True
            dev_idx = 0
            profile.device_name = torch.cuda.get_device_name(dev_idx)
            profile.compute_capability = torch.cuda.get_device_capability(dev_idx)
            
            # Query memory
            mem_info = torch.cuda.mem_get_info(dev_idx)
            profile.free_vram_mb = mem_info[0] / (1024 * 1024)
            profile.total_vram_mb = mem_info[1] / (1024 * 1024)
            
            # Unsloth officially requires compute capability >= 7.5 (Turing/Ampere/Ada/Blackwell)
            # V100 is Volta (7.0) which lacks bfloat16 and modern Triton attention kernels
            major, minor = profile.compute_capability
            profile.unsloth_supported = (major > 7) or (major == 7 and minor >= 5)
            
            profile.hardware_line = (
                f"{profile.device_name} (CC {major}.{minor}) | "
                f"Total VRAM: {profile.total_vram_mb:.0f} MB | Free: {profile.free_vram_mb:.0f} MB"
            )
            return profile
    except Exception:
        pass

    # Fallback to nvidia-smi if torch is not installed or CPU-only torch
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if smi_out:
            parts = [p.strip() for p in smi_out.splitlines()[0].split(",")]
            profile.cuda_available = True
            profile.device_name = parts[0]
            profile.total_vram_mb = float(parts[1])
            profile.free_vram_mb = float(parts[2])
            profile.hardware_line = f"{profile.device_name} | Total: {profile.total_vram_mb:.0f} MB | Free: {profile.free_vram_mb:.0f} MB"
    except Exception:
        profile.hardware_line = "Local CPU System (No GPU detected or nvidia-smi unavailable)"

    return profile


import fcntl
import json
import time

_HELD_LEASES: Dict[str, int] = {}


def acquire_gpu_lock(lock_file: Path = DEFAULT_LOCK_PATH, device: str = "rtx-5060-ti") -> bool:
    """
    Acquire exclusive GPU lock to coordinate training with Hestia serving layer.
    Uses atomic OS-level fcntl.flock to eliminate check-then-write race conditions.
    Returns True if acquired, False if already held by another process.
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return False

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        os.close(fd)
        return False

    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        payload = json.dumps({
            "pid": os.getpid(),
            "owner": "paideia_training",
            "device": device,
            "acquired_at": time.time(),
        })
        os.write(fd, payload.encode("utf-8"))
        _HELD_LEASES[str(lock_file)] = fd
        return True
    except OSError:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
        return False


def release_gpu_lock(lock_file: Path = DEFAULT_LOCK_PATH) -> None:
    """Release GPU lock upon training completion or failure."""
    fd = _HELD_LEASES.pop(str(lock_file), None)
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        if lock_file.exists():
            lock_file.unlink()
    except OSError:
        pass

