"""
Deterministic Execution Utilities
=================================
Provides centralized seed setting and deterministic configuration across Python random,
NumPy, PyTorch (CPU + CUDA), and cuDNN.

IMPORTANT NOTICE ON PYTHONHASHSEED:
Setting `os.environ["PYTHONHASHSEED"]` at Python runtime only affects child processes.
Python's internal dictionary and set hashing seed is established once at interpreter startup.
For true bitwise Python hash determinism, `PYTHONHASHSEED=<seed>` must be set in the parent
shell/environment BEFORE launching Python (e.g., `PYTHONHASHSEED=42 python ...`).
The `ensure_deterministic_env()` helper can verify this or re-exec the process if desired.
"""

import os
import sys
import random
import warnings
from typing import Optional
import numpy as np
import torch


def ensure_deterministic_env(
    seed: int = 42,
    hard_fail: bool = True,
    reexec: bool = False,
) -> None:
    """
    Checks if PYTHONHASHSEED was configured before interpreter startup.
    If hard_fail=True, raises RuntimeError if PYTHONHASHSEED is not set or mismatched,
    directing the user to relaunch with the proper environment variable.
    """
    # Allow test runners (e.g. pytest) to run smoke tests without hard-failing
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST") or "PYTEST_VERSION" in os.environ:
        return

    val = os.environ.get("PYTHONHASHSEED")
    if val != str(seed):
        if reexec:
            os.environ["PYTHONHASHSEED"] = str(seed)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.execv(sys.executable, [sys.executable] + sys.argv)
        elif hard_fail:
            cmd_args = " ".join(sys.argv)
            raise RuntimeError(
                f"CRITICAL DETERMINISM ERROR: PYTHONHASHSEED is currently '{val}', expected '{seed}'.\n"
                f"Python's internal dictionary and set hashing seed is fixed once at interpreter startup\n"
                f"and cannot be configured at runtime.\n"
                f"Please relaunch the command with PYTHONHASHSEED={seed} in the parent shell:\n"
                f"    PYTHONHASHSEED={seed} python {cmd_args}\n"
            )
        else:
            warnings.warn(
                f"PYTHONHASHSEED is currently '{val}', not '{seed}'. "
                f"Set PYTHONHASHSEED={seed} in the parent shell prior to launching Python "
                f"for bitwise reproducible dictionary/set iteration orders.",
                RuntimeWarning,
                stacklevel=2,
            )


def set_all_seeds(
    seed: int = 42,
    deterministic_algorithms: bool = True,
    warn_only: bool = False,
) -> None:
    """
    Sets random seeds across all libraries and enforces deterministic algorithms.

    - Verifies PYTHONHASHSEED environment configuration
    - Sets random.seed and np.random.seed
    - Sets torch.manual_seed across CPU and CUDA
    - Configures torch.backends.cudnn.deterministic = True and benchmark = False
    - Sets CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic cuBLAS kernels
    - Calls torch.use_deterministic_algorithms(True) where supported
    """
    # Note: Setting os.environ['PYTHONHASHSEED'] at runtime does not reseed the running
    # interpreter. Call ensure_deterministic_env(seed) before startup.
    current_hash_seed = os.environ.get("PYTHONHASHSEED")
    if current_hash_seed != str(seed) and not os.environ.get("PYTEST_CURRENT_TEST"):
        warnings.warn(
            f"PYTHONHASHSEED is '{current_hash_seed}', not '{seed}'. "
            f"Set PYTHONHASHSEED={seed} before Python starts to ensure identical hash orders.",
            RuntimeWarning,
            stacklevel=2,
        )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if deterministic_algorithms:
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except Exception as e:
            warnings.warn(f"Could not enable torch.use_deterministic_algorithms: {e}", RuntimeWarning)


def set_single_threaded_blas() -> None:
    """
    Forces single-threaded BLAS execution across OpenMP, MKL, and PyTorch
    to eliminate multi-threaded floating-point non-associative reduction drift.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

