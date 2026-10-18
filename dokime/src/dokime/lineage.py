"""Experiment Tree and Lineage Provenance Tracker for Dokime (Phase 3).

Prevents ungrounded evaluation regressions by:
- Explicitly tracking parent_run_id, branch_point, and parameter diffs across runs.
- Recording git_sha and git_dirty flags to guarantee evaluation reproducibility.
- Building complete DAG experiment trees from runs directory.
- Auditing candidate runs against their exact parent lineage for statistical regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_parameter_diff(
    parent_manifest: dict[str, Any],
    child_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Compute structural parameter and configuration diff between parent and child runs."""
    diff: dict[str, Any] = {}

    # 1. Hyperparameters
    for param in ("seed", "temperature", "max_tokens"):
        parent_val = parent_manifest.get(param)
        child_val = child_manifest.get(param)
        if parent_val != child_val:
            diff[f"param_{param}"] = {"parent": parent_val, "child": child_val}

    # 2. Models
    parent_models = set(parent_manifest.get("model_ids", []))
    child_models = set(child_manifest.get("model_ids", []))
    if parent_models != child_models:
        diff["models"] = {
            "added": sorted(list(child_models - parent_models)),
            "removed": sorted(list(parent_models - child_models)),
        }

    # 3. Tasks
    parent_tasks = set(parent_manifest.get("task_ids", []))
    child_tasks = set(child_manifest.get("task_ids", []))
    if parent_tasks != child_tasks:
        diff["tasks"] = {
            "added": sorted(list(child_tasks - parent_tasks)),
            "removed": sorted(list(parent_tasks - child_tasks)),
        }

    # 4. Prompt template hash
    p_tmpl = parent_manifest.get("prompt_template_hash")
    c_tmpl = child_manifest.get("prompt_template_hash")
    if p_tmpl != c_tmpl:
        diff["prompt_template_hash"] = {"parent": p_tmpl, "child": c_tmpl}

    # 5. Dataset checksums / BOMs
    p_checks = parent_manifest.get("checksums", {})
    c_checks = child_manifest.get("checksums", {})
    dataset_diff: dict[str, Any] = {}
    all_dataset_keys = set(k for k in p_checks if k.startswith("dataset_")) | set(
        k for k in c_checks if k.startswith("dataset_")
    )
    for dk in all_dataset_keys:
        pv = p_checks.get(dk)
        cv = c_checks.get(dk)
        if pv != cv:
            dataset_diff[dk] = {"parent": pv, "child": cv}
    if dataset_diff:
        diff["dataset_checksums"] = dataset_diff

    # 6. Git commit
    p_git = parent_manifest.get("git_sha")
    c_git = child_manifest.get("git_sha")
    if p_git != c_git:
        diff["git_sha"] = {"parent": p_git, "child": c_git}

    return diff


@dataclass
class ExperimentNode:
    """Node in an evaluation lineage experiment tree."""

    run_id: str
    parent_run_id: Optional[str]
    branch_point: Optional[str]
    git_sha: Optional[str]
    git_dirty: Optional[bool]
    timestamp_utc: str
    model_ids: List[str]
    task_ids: List[str]
    parameter_diff: Optional[Dict[str, Any]] = None
    children: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "branch_point": self.branch_point,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "timestamp_utc": self.timestamp_utc,
            "model_ids": self.model_ids,
            "task_ids": self.task_ids,
            "parameter_diff": self.parameter_diff,
            "children": self.children,
        }


def load_manifest(manifest_path: Path) -> Optional[dict[str, Any]]:
    """Safe loader for a manifest JSON file."""
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_experiment_tree(runs_dir: Path | str = "runs") -> dict[str, ExperimentNode]:
    """Scans all run directories and constructs a connected DAG of ExperimentNodes."""
    base = Path(runs_dir).resolve()
    if not base.is_dir():
        return {}

    manifests: dict[str, dict[str, Any]] = {}
    for candidate_dir in base.iterdir():
        if candidate_dir.is_dir():
            m_path = candidate_dir / "manifest.json"
            m = load_manifest(m_path)
            if m and "run_id" in m:
                manifests[m["run_id"]] = m

    # Instantiate nodes
    nodes: dict[str, ExperimentNode] = {}
    for run_id, m in manifests.items():
        parent_id = m.get("parent_run_id")
        nodes[run_id] = ExperimentNode(
            run_id=run_id,
            parent_run_id=parent_id,
            branch_point=m.get("branch_point"),
            git_sha=m.get("git_sha"),
            git_dirty=m.get("git_dirty"),
            timestamp_utc=m.get("timestamp_utc", ""),
            model_ids=m.get("model_ids", []),
            task_ids=m.get("task_ids", []),
            parameter_diff=m.get("parameter_diff"),
        )

    # Link children and populate parameter diff if not already saved
    for run_id, node in nodes.items():
        if node.parent_run_id and node.parent_run_id in nodes:
            nodes[node.parent_run_id].children.append(run_id)
            if not node.parameter_diff:
                parent_m = manifests[node.parent_run_id]
                child_m = manifests[run_id]
                node.parameter_diff = compute_parameter_diff(parent_m, child_m)

    return nodes


def get_lineage_path(run_id: str, runs_dir: Path | str = "runs") -> list[str]:
    """Returns the ancestor path list from root to the given run_id."""
    tree = build_experiment_tree(runs_dir)
    path: list[str] = [run_id]
    current = run_id
    visited: set[str] = {run_id}

    while current in tree and tree[current].parent_run_id:
        parent = tree[current].parent_run_id
        if not parent or parent in visited:
            break
        visited.add(parent)
        path.append(parent)
        current = parent

    return list(reversed(path))


def evaluate_lineage_regressions(
    candidate_run_id: str,
    runs_dir: Path | str = "runs",
) -> dict[str, Any]:
    """Compares candidate run against its declared parent_run_id to detect statistically significant regressions."""
    from dokime.compare import run_comparison

    base = Path(runs_dir).resolve()
    cand_manifest_path = base / candidate_run_id / "manifest.json"
    cand_manifest = load_manifest(cand_manifest_path)
    if not cand_manifest:
        raise FileNotFoundError(f"Candidate run manifest not found: {cand_manifest_path}")

    parent_id = cand_manifest.get("parent_run_id")
    if not parent_id:
        return {
            "status": "UNPARENTED",
            "message": f"Run '{candidate_run_id}' has no declared parent_run_id (root experiment).",
            "regressions": [],
            "comparisons": {},
        }

    parent_manifest_path = base / parent_id / "manifest.json"
    parent_manifest = load_manifest(parent_manifest_path)
    if not parent_manifest:
        raise FileNotFoundError(f"Parent run manifest not found: {parent_manifest_path}")

    common_models = sorted(list(set(cand_manifest.get("model_ids", [])) & set(parent_manifest.get("model_ids", []))))
    common_tasks = sorted(list(set(cand_manifest.get("task_ids", [])) & set(parent_manifest.get("task_ids", []))))

    regressions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}

    for model_id in common_models:
        for task_id in common_tasks:
            comp_key = f"{model_id}/{task_id}"
            try:
                report = run_comparison(
                    run_a_ref=parent_id,
                    run_b_ref=candidate_run_id,
                    model=model_id,
                    task=task_id,
                    base_dir=str(base),
                    write_report=False,
                )
                sig = report.get("significance", {})
                verdict = sig.get("verdict")
                delta = sig.get("delta")
                ci_upper = sig.get("ci_upper")
                comparisons[comp_key] = {
                    "delta": delta,
                    "ci_lower": sig.get("ci_lower"),
                    "ci_upper": ci_upper,
                    "p_value": sig.get("p_value"),
                    "is_significant": sig.get("is_significant"),
                    "verdict": verdict,
                }

                # A regression occurs when verdict is REGRESSION or ci_upper < 0
                if verdict == "REGRESSION" or (ci_upper is not None and ci_upper < 0):
                    regressions.append({
                        "model_id": model_id,
                        "task_id": task_id,
                        "drop_mean": delta,
                        "ci_upper": ci_upper,
                        "p_value": sig.get("p_value"),
                    })
            except Exception as e:
                err_msg = str(e)
                comparisons[comp_key] = {"error": err_msg}
                errors.append({"model_id": model_id, "task_id": task_id, "error": err_msg})

    if regressions:
        status = "REGRESSION"
    elif errors:
        status = "ERROR"
    else:
        status = "PASSED"

    return {
        "candidate_run_id": candidate_run_id,
        "parent_run_id": parent_id,
        "status": status,
        "regressions_detected": len(regressions),
        "errors_detected": len(errors),
        "regressions": regressions,
        "errors": errors,
        "comparisons": comparisons,
    }
