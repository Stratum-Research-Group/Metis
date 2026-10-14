"""VRAM monitoring and budget enforcement."""

from __future__ import annotations

import logging
import shutil
import warnings

logger = logging.getLogger(__name__)


def get_free_vram_mb(gpu_index: int = 0) -> int:
    """Query live free VRAM in MB for the given GPU via pynvml.

    Falls back to nvidia-smi subprocess if pynvml is not available.
    Returns 0 if no GPU is detected (for testing on CPU-only machines).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = mem_info.free // (1024 * 1024)
        pynvml.nvmlShutdown()
        return int(free_mb)
    except Exception:
        logger.debug("pynvml failed, trying nvidia-smi fallback")

    return _nvidia_smi_fallback(gpu_index)


def _nvidia_smi_fallback(gpu_index: int) -> int:
    """Use nvidia-smi CLI to query free VRAM."""
    import subprocess

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        logger.warning("No nvidia-smi found — assuming 0 MB free VRAM (CPU-only mode?)")
        return 0

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                f"--id={gpu_index}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass

    logger.warning("nvidia-smi query failed — assuming 0 MB free VRAM")
    return 0


def get_total_vram_mb(gpu_index: int = 0) -> int:
    """Query total VRAM in MB for the given GPU."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mb = mem_info.total // (1024 * 1024)
        pynvml.nvmlShutdown()
        return int(total_mb)
    except Exception:
        logger.debug("pynvml failed, trying nvidia-smi fallback for total VRAM")

    return _nvidia_smi_total_fallback(gpu_index)


def _nvidia_smi_total_fallback(gpu_index: int) -> int:
    """Use nvidia-smi CLI to query total VRAM."""
    import subprocess

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        logger.warning("No nvidia-smi found — cannot query total VRAM")
        return 0

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
                f"--id={gpu_index}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError):
        pass

    logger.warning("nvidia-smi query failed — cannot query total VRAM")
    return 0


def get_gpu_name(gpu_index: int = 0) -> str:
    """Query the GPU name string."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        return name if isinstance(name, str) else name.decode()
    except Exception:
        return "unknown"


def get_gpu_count() -> int:
    """Query total count of available NVIDIA GPUs.

    Falls back to nvidia-smi CLI if pynvml is unavailable.
    Returns 0 if no GPU or driver is detected.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            import pynvml

        pynvml.nvmlInit()
        count = int(pynvml.nvmlDeviceGetCount())
        pynvml.nvmlShutdown()
        return count
    except Exception:
        logger.debug("pynvml failed, trying nvidia-smi fallback for GPU count")

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return 0

    import subprocess

    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
            if lines:
                return int(lines[0])
    except (subprocess.TimeoutExpired, ValueError):
        pass

    return 0

