"""
Walk logging for Dianoia.

Every walk appends JSONL records to the log file. These logs provide the
supervisory and negative training signal for a learned walk policy.

Conflict & Re-Walk Two-Record Pattern:
When a walk encounters answer-time verification contradictions:
1. The first record documents the initial walk trajectory, with
   `verification.conflicts_found=True`, `verification.rewalked=False`, and
   the detailed contradiction obstruction report (`conflict_detail`).
2. The second record documents the re-walk resolution attempt, with
   `verification.rewalked=True`, its own verification report and conflict details,
   and `initial_path` preserving the initial trajectory. If the re-walk failed to
   execute, `verification.rewalk_failed=True` is recorded.
Both records share the same `walk_id` (a UUIDv4 string) so downstream log
consumers can cleanly pair and join the negative signal and resolution.

For normal walks without contradictions, exactly one record is emitted with
its own unique `walk_id`.

Note: append is not concurrent-safe; fine for single-shot v1.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


# Schema version for forward compatibility. A future learner can
# distinguish v1 logs from later schema revisions.
SCHEMA_VERSION = 1

DEFAULT_LOG_PATH = os.path.expanduser("~/.dianoia/walks.jsonl")


def build_walk_record(
    question: str,
    seeded_nodes: List[str],
    path: List[Dict[str, Any]],
    verification: Dict[str, Any],
    final_answer: str,
    max_steps: int,
    steps_used: int,
    model: str,
    fallback_used: bool,
    initial_path: Optional[List[Dict[str, Any]]] = None,
    walk_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a single JSONL-serialisable walk record conforming to the v1 schema.

    Args:
        question: The original user question.
        seeded_nodes: List of node IDs from the initial retrieval seed.
        path: List of step dicts, each containing:
            - node: str (node ID)
            - action: "EXPAND"|"BACKTRACK"|"ANSWER"
            - raw_output: str (raw model output for this step)
            - reason: str (parsed reason)
            - parse_failed: bool
        verification: Dict containing:
            - conflicts_found: bool
            - rewalked: bool
            - report: str
            - conflict_detail: str | None
        final_answer: The final synthesized answer.
        max_steps: Maximum step budget.
        steps_used: Actual steps consumed.
        model: Name/path of the model used.
        fallback_used: Whether the fallback model was activated.
        initial_path: Optional initial walk path if this record represents a re-walk.
        walk_id: Optional unique identifier for this walk session. When a conflict
            triggers a re-walk, both records share the same walk_id.

    Returns:
        A dict ready for JSON serialisation.
    """
    if walk_id is None:
        walk_id = str(uuid.uuid4())

    record = {
        "schema_version": SCHEMA_VERSION,
        "walk_id": walk_id,
        "question": question,
        "seeded_nodes": seeded_nodes,
        "path": path,
        "verification": verification,
        "final_answer": final_answer,
        "max_steps": max_steps,
        "steps_used": steps_used,
        "model": model,
        "fallback_used": fallback_used,
    }
    if initial_path is not None:
        record["initial_path"] = initial_path
    return record


def append_walk_log(
    record: Dict[str, Any],
    log_path: Optional[str] = None,
) -> str:
    """
    Append a walk record as one JSONL line to the log file.

    Creates the parent directory if absent.

    Args:
        record: Walk record dict (from build_walk_record).
        log_path: File path to append to. Defaults to ~/.dianoia/walks.jsonl.

    Returns:
        The resolved path that was written to.
    """
    resolved = log_path or DEFAULT_LOG_PATH
    parent = Path(resolved).parent
    parent.mkdir(parents=True, exist_ok=True)

    with open(resolved, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return resolved
