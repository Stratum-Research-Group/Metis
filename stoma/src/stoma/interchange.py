"""
Stoma <-> Dianoia Index Interchange Verification.
Exercises pinned Dianoia's actual CLI loader via controlled subprocess invocation.
Captures the real Dianoia edge-gate refusal (cli.py:206-219) and reports
accurately that direct CLI walk is unsupported due to zero graph edges in Stoma v1.
"""

from __future__ import annotations

import logging
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from lthcm.engine.pipeline import LTHCMEngine

logger = logging.getLogger(__name__)


def verify_dianoia_interchange(index_path: Path, allow_trusted_pickle: bool = False) -> Dict[str, Any]:
    """
    Test direct interchange against Dianoia's actual CLI load and query path.
    1. Check serialization format (pickle) and engine type (LTHCMEngine).
    2. Check model dimensions (d_model=384, num_facets=1).
    3. Invoke Dianoia's actual CLI in a controlled subprocess and capture the real edge gate refusal.
    """
    path = Path(index_path).resolve()
    if not path.is_file():
        return {
            "compatible": False,
            "status": "error",
            "message": f"Index file not found: {path}",
        }

    if not allow_trusted_pickle:
        return {
            "compatible": False,
            "status": "untrusted_pickle_rejected",
            "message": "Untrusted pickle deserialization refused without explicit authorization.",
        }

    if path.stat().st_size > 2 * 1024 * 1024 * 1024:
        return {
            "compatible": False,
            "status": "size_exceeded",
            "message": f"Index file '{path}' exceeds 2GB maximum limit.",
        }

    # Step 1: Serialization format check
    try:
        with open(path, "rb") as f:
            engine = pickle.load(f)
    except Exception as e:
        return {
            "compatible": False,
            "serialization_match": False,
            "status": "incompatible_format",
            "message": f"Failed to deserialize index with pickle: {e}",
        }

    if not isinstance(engine, LTHCMEngine):
        return {
            "compatible": False,
            "serialization_match": True,
            "is_lthcm_engine": False,
            "status": "type_mismatch",
            "message": f"Loaded object is not LTHCMEngine (got {type(engine).__name__})",
        }

    d_model = getattr(engine, "d_model", None)
    num_facets = getattr(engine, "num_facets", None)
    dimension_match = (d_model == 384 and num_facets == 1)

    edge_count = 0
    if hasattr(engine, "wormhole") and hasattr(engine.wormhole, "graph"):
        if hasattr(engine.wormhole.graph, "number_of_edges"):
            edge_count = engine.wormhole.graph.number_of_edges()
        elif hasattr(engine.wormhole.graph, "edges"):
            edge_count = len(engine.wormhole.graph.edges)

    # Step 2: Actual invocation of Dianoia's CLI loader
    cmd = [sys.executable, "-m", "dianoia.cli", "--index", str(path), "interchange_verification_ping"]
    status_override: Optional[str] = None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        cli_returncode = proc.returncode
        cli_stdout = proc.stdout
        cli_stderr = proc.stderr
    except subprocess.TimeoutExpired:
        cli_returncode = -1
        cli_stdout = ""
        cli_stderr = "Dianoia CLI execution timed out after 20 seconds."
        status_override = "dianoia_cli_timeout"
    except Exception as e:
        cli_returncode = -1
        cli_stdout = ""
        cli_stderr = f"Subprocess execution failed: {e}"

    edge_gate_triggered = (
        cli_returncode == 1
        and "contains 0 graph edges" in (cli_stderr + cli_stdout)
    )

    if edge_gate_triggered:
        return {
            "compatible": False,
            "direct_cli_walk_supported": False,
            "serialization_match": True,
            "is_lthcm_engine": True,
            "d_model": d_model,
            "num_facets": num_facets,
            "dimension_match": dimension_match,
            "edge_count": edge_count,
            "cli_return_code": cli_returncode,
            "cli_stderr": cli_stderr.strip(),
            "edge_gate_triggered": True,
            "status": "direct_cli_walk_unsupported_zero_edges",
            "reason": (
                "Dianoia's _load_index strictly mandates edge_count > 0 (cli.py:213-219). "
                "Per Stoma spec, wormhole edge mining is explicitly out of scope for Stoma v1. "
                "The serialized index is binary- and dimension-compatible with LTHCM, "
                "but direct Dianoia CLI deliberative walk is unsupported because Stoma v1 builds 0 graph edges."
            ),
        }

    if cli_returncode != 0:
        return {
            "compatible": False,
            "direct_cli_walk_supported": False,
            "serialization_match": True,
            "is_lthcm_engine": True,
            "d_model": d_model,
            "num_facets": num_facets,
            "dimension_match": dimension_match,
            "edge_count": edge_count,
            "cli_return_code": cli_returncode,
            "cli_stderr": cli_stderr.strip(),
            "edge_gate_triggered": False,
            "status": status_override or "dianoia_cli_execution_failed",
            "message": f"Dianoia CLI execution failed with return code {cli_returncode}: {cli_stderr.strip()}",
        }

    return {
        "compatible": True,
        "direct_cli_walk_supported": True,
        "serialization_match": True,
        "is_lthcm_engine": True,
        "d_model": d_model,
        "num_facets": num_facets,
        "dimension_match": dimension_match,
        "edge_count": edge_count,
        "cli_return_code": cli_returncode,
        "edge_gate_triggered": False,
        "status": "compatible",
        "message": "Index passes all Dianoia load requirements and CLI executed successfully.",
    }
