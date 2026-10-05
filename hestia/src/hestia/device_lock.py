"""Atomic OS-level GPU device lease and mutual exclusion protocol."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def get_default_lock_dir() -> Path:
    uid = os.getuid()
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir and Path(runtime_dir).is_dir():
        d = Path(runtime_dir) / "metis"
    elif Path(f"/run/user/{uid}").is_dir():
        d = Path(f"/run/user/{uid}") / "metis"
    else:
        d = Path(f"/tmp/metis_{uid}")
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def get_device_lock_path(device: str = "rtx-5060-ti", lock_dir: Optional[Path] = None) -> Path:
    base = lock_dir or get_default_lock_dir()
    clean_device = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in device)
    return base / f"metis_gpu_{clean_device}.lock"


@dataclass
class GpuDeviceLease:
    device: str
    owner: str
    fd: int
    lock_path: Path
    default_fd: Optional[int] = None
    default_lock_path: Optional[Path] = None

    def release(self) -> None:
        _ACTIVE_LEASES.pop(self.device, None)
        for fd in (self.fd, self.default_fd):
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass
        # Permanent lock files — never unlink to avoid races between contenders


_ACTIVE_LEASES: dict[str, GpuDeviceLease] = {}


def acquire_device_lease(
    device: str = "rtx-5060-ti",
    owner: str = "hestia",
    lock_dir: Optional[Path] = None,
) -> Optional[GpuDeviceLease]:
    """Atomically acquire an exclusive OS flock lease on a GPU device.
    
    Locks both the per-device lock file and the shared default lock file
    to guarantee mutual exclusion between serving and training.
    """
    if device in _ACTIVE_LEASES:
        # Reject duplicate in-process lease attempts
        return None

    resolved_lock_dir = lock_dir or get_default_lock_dir()
    lock_path = get_device_lock_path(device, resolved_lock_dir)
    default_lock_path = resolved_lock_dir / "metis_gpu.lock"

    open_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)

    # Open and lock per-device file
    try:
        fd = os.open(str(lock_path), open_flags, 0o600)
    except OSError as exc:
        logger.warning("Failed to open device lock file %s: %s", lock_path, exc)
        return None

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        os.close(fd)
        return None

    # Open and lock shared default lock file
    default_fd: Optional[int] = None
    if default_lock_path != lock_path:
        try:
            default_fd = os.open(str(default_lock_path), open_flags, 0o600)
            fcntl.flock(default_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            # Roll back per-device lock if shared lock fails
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)
            if default_fd is not None:
                os.close(default_fd)
            return None

    meta = {
        "pid": os.getpid(),
        "owner": owner,
        "device": device,
        "acquired_at": time.time(),
    }
    encoded = json.dumps(meta).encode("utf-8")

    for fdesc in (fd, default_fd):
        if fdesc is not None:
            try:
                os.ftruncate(fdesc, 0)
                os.lseek(fdesc, 0, os.SEEK_SET)
                os.write(fdesc, encoded)
            except OSError:
                pass

    lease = GpuDeviceLease(
        device=device,
        owner=owner,
        fd=fd,
        lock_path=lock_path,
        default_fd=default_fd,
        default_lock_path=default_lock_path if default_lock_path != lock_path else None,
    )
    _ACTIVE_LEASES[device] = lease
    return lease
