"""System diagnostics and readiness doctor check."""

import os
import sys
from pathlib import Path
from typing import Any, Dict
from tabulate import tabulate

from paideia.training.hardware import inspect_hardware


def run_doctor_check() -> Dict[str, Any]:
    """Execute comprehensive environment, hardware, and schema diagnostics."""
    report = {
        "python_version": sys.version.split()[0],
        "paideia_version": "0.1.0",
        "checks": [],
    }

    # 1. Python check
    py_ok = sys.version_info >= (3, 10)
    report["checks"].append({
        "component": "Python >= 3.10",
        "status": "PASS" if py_ok else "FAIL",
        "details": sys.version.split()[0],
    })

    # 2. Hardware / GPU check
    hw = inspect_hardware()
    gpu_status = "PASS" if hw.cuda_available else "WARN (CPU-only)"
    report["checks"].append({
        "component": "Hardware / CUDA",
        "status": gpu_status,
        "details": hw.hardware_line,
    })

    # 3. VRAM gate
    vram_status = "PASS" if hw.free_vram_mb >= 12000 else ("WARN" if hw.cuda_available else "N/A")
    report["checks"].append({
        "component": "VRAM Headroom (>=12GB free)",
        "status": vram_status,
        "details": f"{hw.free_vram_mb:.0f} MB free" if hw.cuda_available else "No GPU memory",
    })

    # 4. PyTorch check
    try:
        import torch
        torch_ok = True
        torch_ver = torch.__version__
    except ImportError:
        torch_ok = False
        torch_ver = "Not installed (install '.[gpu]' profile for training)"
    report["checks"].append({
        "component": "PyTorch",
        "status": "PASS" if torch_ok else "INFO",
        "details": torch_ver,
    })

    # 5. Core dependencies check
    for pkg in ["pydantic", "sklearn", "scipy", "numpy"]:
        try:
            __import__(pkg)
            report["checks"].append({
                "component": f"Dependency: {pkg}",
                "status": "PASS",
                "details": "Installed and imported successfully",
            })
        except ImportError:
            report["checks"].append({
                "component": f"Dependency: {pkg}",
                "status": "FAIL",
                "details": "Missing required core package",
            })

    # 6. Data directories & permissions
    runs_dir = Path("./runs")
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        test_file = runs_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        write_ok = True
    except Exception:
        write_ok = False

    report["checks"].append({
        "component": "Filesystem Write Access (./runs)",
        "status": "PASS" if write_ok else "FAIL",
        "details": str(runs_dir.resolve()),
    })

    return report


def print_doctor_report() -> None:
    """Print human-readable doctor diagnostics table."""
    report = run_doctor_check()
    headers = ["Diagnostic Component", "Status", "Details"]
    rows = [[c["component"], c["status"], c["details"]] for c in report["checks"]]
    print(f"\n=== Paideia Doctor Readiness Report (v{report['paideia_version']}) ===")
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print()
