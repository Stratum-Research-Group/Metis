"""System diagnostics and readiness doctor check for Prosopon."""

import shutil
import sys
from pathlib import Path
from typing import Any, Dict
from tabulate import tabulate

from prosopon.config import ProsoponConfig


def run_doctor_check(config: ProsoponConfig) -> Dict[str, Any]:
    """Execute comprehensive environment, voice tool, and network diagnostics."""
    report = {
        "prosopon_version": "0.1.0",
        "python_version": sys.version.split()[0],
        "checks": [],
    }

    # 1. Python check
    py_ok = sys.version_info >= (3, 10)
    report["checks"].append({
        "component": "Python >= 3.10",
        "status": "PASS" if py_ok else "FAIL",
        "details": sys.version.split()[0],
    })

    # 2. ffmpeg check
    ffmpeg_path = shutil.which("ffmpeg")
    report["checks"].append({
        "component": "Audio Transcoder (ffmpeg)",
        "status": "PASS" if ffmpeg_path else "FAIL",
        "details": ffmpeg_path or "Missing! Required for browser WebM to WAV conversion",
    })

    # 3. whisper.cpp check
    whisper_path = shutil.which(config.whisper_binary) or shutil.which("whisper-cli") or shutil.which("whisper")
    report["checks"].append({
        "component": "STT Engine (whisper.cpp)",
        "status": "PASS" if whisper_path else "WARN (Mock Fallback)",
        "details": whisper_path or f"'{config.whisper_binary}' not on PATH; using built-in mock",
    })

    # 4. piper TTS check
    piper_path = shutil.which(config.piper_binary)
    report["checks"].append({
        "component": "TTS Engine (piper)",
        "status": "PASS" if piper_path else "WARN (Mock Fallback)",
        "details": piper_path or f"'{config.piper_binary}' not on PATH; using built-in mock",
    })

    # 5. Network & TLS mode
    if config.is_lan():
        tls_status = "PASS" if config.enable_tls else "WARN (Insecure LAN)"
        details = "TLS Enabled (Secure Context active for mobile mic)" if config.enable_tls else "Plain HTTP: Mobile browser will block mic on LAN"
    else:
        tls_status = "PASS"
        details = "Loopback (127.0.0.1) active (Secure Context exempt)"

    report["checks"].append({
        "component": "Network & Secure Context",
        "status": tls_status,
        "details": details,
    })

    # 6. Session directory permissions
    try:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        test_file = config.data_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        write_ok = True
    except Exception:
        write_ok = False

    report["checks"].append({
        "component": "Session Storage Permissions",
        "status": "PASS" if write_ok else "FAIL",
        "details": str(config.data_dir.resolve()),
    })

    return report


def print_doctor_report(config: ProsoponConfig) -> None:
    """Print human-readable doctor diagnostics table."""
    report = run_doctor_check(config)
    headers = ["Component", "Status", "Details"]
    rows = [[c["component"], c["status"], c["details"]] for c in report["checks"]]
    print(f"\n=== Prosopon Doctor Readiness Report (v{report['prosopon_version']}) ===")
    print(tabulate(rows, headers=headers, tablefmt="github"))
    print()
