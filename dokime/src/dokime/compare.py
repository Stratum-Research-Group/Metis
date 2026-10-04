"""Regression compare between two evaluation runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from dokime.manifest import sha256_file
from dokime.stats import paired_bootstrap_test, per_query_diff_analysis, single_run_bootstrap


def is_valid_finite_score(score_val: Any) -> bool:
    """Check if a score value is non-null and a finite float."""
    if score_val is None:
        return False
    try:
        val = float(score_val)
        return math.isfinite(val)
    except (ValueError, TypeError):
        return False


def make_compare_report_filename(
    baseline_run_id: str,
    model_id: str,
    task_id: str | None = None,
) -> str:
    """Generate a collision-free, filesystem-safe filename for a scoped comparison report."""
    scope_key = f"{baseline_run_id}::{model_id}::{task_id or ''}"
    scope_digest = hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:12]

    clean_base = re.sub(r"[^a-zA-Z0-9_-]", "_", baseline_run_id)
    clean_base = re.sub(r"_+", "_", clean_base).strip("_") or "base"
    clean_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_id)
    clean_model = re.sub(r"_+", "_", clean_model).strip("_") or "model"

    if task_id:
        clean_task = re.sub(r"[^a-zA-Z0-9_-]", "_", task_id)
        clean_task = re.sub(r"_+", "_", clean_task).strip("_") or "task"
        return f"compare_{clean_base}_{clean_model}_{clean_task}_{scope_digest}.json"
    return f"compare_{clean_base}_{clean_model}_{scope_digest}.json"


def make_cross_model_compare_report_filename(
    model_a: str,
    model_b: str,
    task_id: str,
) -> str:
    """Generate a collision-free, filesystem-safe filename for a cross-model comparison report.

    Includes filesystem-safe encodings of both model IDs and the task ID, plus a 12-char
    SHA-256 digest of the exact scope tuple to prevent collisions across similar sanitized names.
    """
    clean_a = re.sub(r"[^a-zA-Z0-9_-]", "_", model_a)
    clean_a = re.sub(r"_+", "_", clean_a).strip("_") or "model_a"
    clean_b = re.sub(r"[^a-zA-Z0-9_-]", "_", model_b)
    clean_b = re.sub(r"_+", "_", clean_b).strip("_") or "model_b"
    clean_task = re.sub(r"[^a-zA-Z0-9_-]", "_", task_id)
    clean_task = re.sub(r"_+", "_", clean_task).strip("_") or "task"

    scope_key = f"{model_a}::{model_b}::{task_id}"
    scope_digest = hashlib.sha256(scope_key.encode("utf-8")).hexdigest()[:12]

    return f"compare_cross_{clean_a}_vs_{clean_b}_{clean_task}_{scope_digest}.json"


class CompareError(Exception):
    """Raised when runs cannot be compared due to ambiguous models, missing data, or dataset incompatibility."""
    pass


def _resolve_run_files(run_ref: str, base_dir: str = "runs") -> tuple[str, Path, Path]:
    """Resolve a run_id or file path to (run_id, results_path, manifest_path)."""
    p = Path(run_ref)
    if p.is_file():
        run_id = p.parent.name if p.name == "results.jsonl" else p.stem
        results_path = p
        manifest_path = p.parent / "manifest.json"
        return run_id, results_path, manifest_path

    if p.is_dir() and (p / "results.jsonl").is_file():
        run_id = p.name
        results_path = p / "results.jsonl"
        manifest_path = p / "manifest.json"
        return run_id, results_path, manifest_path

    # Check base_dir / run_ref
    run_dir = Path(base_dir) / run_ref
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "manifest.json"

    if not results_path.is_file():
        raise FileNotFoundError(f"Cannot find results.jsonl for run reference: '{run_ref}'")

    return run_ref, results_path, manifest_path


def load_results_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all records from a results.jsonl file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_manifest(path: Path, run_id: str) -> dict[str, Any]:
    """Load manifest.json, raising CompareError if missing or invalid."""
    if not path.is_file():
        raise CompareError(
            f"Missing required manifest provenance for run '{run_id}' at {path}. "
            "Cannot verify dataset compatibility."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise CompareError(f"Failed to read manifest for run '{run_id}': {e}") from e


def _verify_results_integrity(run_id: str, results_path: Path, manifest: dict[str, Any]) -> None:
    """Verify raw bytes of results.jsonl against manifest checksums.results_jsonl SHA-256.

    A missing, malformed, or mismatched results checksum causes a clear CompareError.
    """
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict):
        raise CompareError(
            f"Run '{run_id}' manifest is missing or has invalid 'checksums' field."
        )
    expected_hash = checksums.get("results_jsonl")
    if not expected_hash or not isinstance(expected_hash, str):
        raise CompareError(
            f"Run '{run_id}' manifest is missing required 'results_jsonl' checksum."
        )
    if not results_path.is_file():
        raise CompareError(
            f"Results file not found for run '{run_id}' at '{results_path}'."
        )
    actual_hash = sha256_file(results_path)
    if actual_hash.lower() != expected_hash.lower():
        raise CompareError(
            f"Results checksum mismatch for run '{run_id}': "
            f"manifest expects {expected_hash}, but found {actual_hash}. "
            "File may be corrupted, truncated, or tampered."
        )


def run_cross_model_comparison(
    run_ref: str,
    model_a: str,
    model_b: str,
    task: str,
    base_dir: str = "runs",
    write_report: bool = True,
) -> dict[str, Any]:
    """Compare two model IDs within a single matrix run for the specified task.

    Safeguards:
    - Verifies raw bytes of results.jsonl against manifest checksum.
    - Requires distinct model IDs.
    - Requires results for both models and that task in the run; refuses if either is absent.
    - Refuses if the two models' query_id sets (including logged failure rows) differ.
    - Refuses if recorded dataset checksums for that task differ or are missing from the manifest.
    - Refuses if prompt hashes per query or run seeds mismatch before pairing.
    - Refuses if expected answers per query mismatch.
    - Excludes failure/null-result rows from paired statistics and reports each model's failure count separately.
    - On remaining complete pairs, computes paired bootstrap (10,000 resamples, seed 42), per-query diffs,
      and single-run bootstrap estimates for both models.
    - Writes the report to a distinct collision-resistant filesystem-safe filename under runs/{run_id}/.
    """
    if not task:
        raise CompareError("A specific --task <task_id> is required for cross-model compare.")

    if model_a == model_b:
        raise CompareError(
            f"Cross-model comparison requires two distinct model IDs, but got '{model_a}' for both."
        )

    run_id, results_path, manifest_path = _resolve_run_files(run_ref, base_dir)
    manifest = load_manifest(manifest_path, run_id)
    canonical_run_id = manifest.get("run_id") or run_id
    _verify_results_integrity(run_id, results_path, manifest)

    # 1. Dataset checksum safeguard
    checksums = manifest.get("checksums", {})
    if not isinstance(checksums, dict):
        raise CompareError(f"Run '{run_id}' manifest has invalid 'checksums' field.")
    dataset_key = f"dataset_{task}"
    if dataset_key not in checksums:
        raise CompareError(
            f"Missing required dataset checksum for task '{task}' in run '{run_id}' manifest."
        )
    dataset_checksum = checksums[dataset_key]

    manifest_ds_a = checksums.get(f"dataset_{task}_{model_a}")
    manifest_ds_b = checksums.get(f"dataset_{task}_{model_b}")
    if manifest_ds_a and manifest_ds_b and manifest_ds_a != manifest_ds_b:
        raise CompareError(
            f"Dataset checksum mismatch in manifest for task '{task}': "
            f"model '{model_a}' has {manifest_ds_a}, model '{model_b}' has {manifest_ds_b}."
        )

    # 2. Run seed safeguard
    manifest_seed = manifest.get("seed")
    if manifest_seed is None:
        raise CompareError(f"Run '{run_id}' manifest is missing required 'seed' field.")

    # 3. Load records and filter for task
    records = load_results_jsonl(results_path)
    task_records = [r for r in records if r.get("task_id") == task]
    if not task_records:
        raise CompareError(f"No records found for task '{task}' in run '{run_id}'.")

    # 4. Filter records for model_a and model_b
    records_a = [r for r in task_records if r.get("model_id") == model_a]
    records_b = [r for r in task_records if r.get("model_id") == model_b]

    if not records_a and not records_b:
        raise CompareError(
            f"Neither model '{model_a}' nor model '{model_b}' has results for task '{task}' in run '{run_id}'."
        )
    if not records_a:
        raise CompareError(
            f"Model '{model_a}' has no results for task '{task}' in run '{run_id}'. "
            f"Available models for task '{task}': {sorted(set(r.get('model_id') for r in task_records if r.get('model_id')))}."
        )
    if not records_b:
        raise CompareError(
            f"Model '{model_b}' has no results for task '{task}' in run '{run_id}'. "
            f"Available models for task '{task}': {sorted(set(r.get('model_id') for r in task_records if r.get('model_id')))}."
        )

    # 5. Check duplicate query records per model
    rec_map_a: dict[str, dict[str, Any]] = {}
    for r in records_a:
        qid = r["query_id"]
        if qid in rec_map_a:
            raise CompareError(
                f"Duplicate query record detected in run '{run_id}' for model '{model_a}', "
                f"task '{task}', query '{qid}'. Ambiguous results cannot be compared."
            )
        rec_map_a[qid] = r

    rec_map_b: dict[str, dict[str, Any]] = {}
    for r in records_b:
        qid = r["query_id"]
        if qid in rec_map_b:
            raise CompareError(
                f"Duplicate query record detected in run '{run_id}' for model '{model_b}', "
                f"task '{task}', query '{qid}'. Ambiguous results cannot be compared."
            )
        rec_map_b[qid] = r

    # 6. Check query_id sets equality (including logged failure rows)
    queries_a = set(rec_map_a.keys())
    queries_b = set(rec_map_b.keys())
    if queries_a != queries_b:
        missing_in_b = sorted(queries_a - queries_b)
        missing_in_a = sorted(queries_b - queries_a)
        raise CompareError(
            f"Query ID sets differ between model '{model_a}' and model '{model_b}' for task '{task}' in run '{run_id}'. "
            f"Logged queries must match exactly (including failure rows). "
            f"Missing in '{model_b}': {missing_in_b}; missing in '{model_a}': {missing_in_a}."
        )

    # 7. Check per-record dataset checksums and seeds if present
    records_ds_a = set(r["dataset_checksum"] for r in records_a if "dataset_checksum" in r)
    records_ds_b = set(r["dataset_checksum"] for r in records_b if "dataset_checksum" in r)
    if records_ds_a and records_ds_b and records_ds_a != records_ds_b:
        raise CompareError(
            f"Recorded dataset checksum mismatch: model '{model_a}' recorded {records_ds_a}, "
            f"model '{model_b}' recorded {records_ds_b} for task '{task}'."
        )
    if records_ds_a and dataset_checksum not in records_ds_a:
        raise CompareError(
            f"Recorded dataset checksum for model '{model_a}' ({records_ds_a}) does not match manifest ({dataset_checksum})."
        )
    if records_ds_b and dataset_checksum not in records_ds_b:
        raise CompareError(
            f"Recorded dataset checksum for model '{model_b}' ({records_ds_b}) does not match manifest ({dataset_checksum})."
        )

    seeds_a = set(r["seed"] for r in records_a if "seed" in r)
    seeds_b = set(r["seed"] for r in records_b if "seed" in r)
    if seeds_a and seeds_b and seeds_a != seeds_b:
        raise CompareError(
            f"Seed mismatch between model '{model_a}' ({seeds_a}) and model '{model_b}' ({seeds_b})."
        )
    if seeds_a and manifest_seed not in seeds_a:
        raise CompareError(
            f"Recorded seed for model '{model_a}' ({seeds_a}) does not match manifest seed ({manifest_seed})."
        )
    if seeds_b and manifest_seed not in seeds_b:
        raise CompareError(
            f"Recorded seed for model '{model_b}' ({seeds_b}) does not match manifest seed ({manifest_seed})."
        )

    # 8. Check prompt hashes and expected answers per query across the full query set
    for qid in sorted(queries_a):
        ra = rec_map_a[qid]
        rb = rec_map_b[qid]

        ea = str(ra.get("expected", ""))
        eb = str(rb.get("expected", ""))
        if ea != eb:
            raise CompareError(
                f"Expected answer mismatch on query '{qid}' in task '{task}': "
                f"model '{model_a}' expected '{ea}', model '{model_b}' expected '{eb}'. "
                "Cannot pair queries with conflicting expected answers."
            )

        ha = ra.get("prompt_hash")
        hb = rb.get("prompt_hash")
        if not ha or not hb or ha != hb:
            if not ha and not hb:
                desc = f"Prompt hash missing or null on query '{qid}' in task '{task}': prompt hash is null or empty for both models"
            elif not ha:
                desc = f"Prompt hash missing or null on query '{qid}' in task '{task}': model '{model_a}' has null or empty prompt hash"
            elif not hb:
                desc = f"Prompt hash missing or null on query '{qid}' in task '{task}': model '{model_b}' has null or empty prompt hash"
            else:
                desc = f"Prompt hash mismatch on query '{qid}' in task '{task}': model '{model_a}' has '{ha}', model '{model_b}' has '{hb}'"
            raise CompareError(
                f"{desc}. Cross-model compare requires verifiable, identical rendered prompt hashes "
                f"for all queries in the full query set (including failure rows) between model '{model_a}' "
                f"and model '{model_b}'."
            )

    # 9. Count failures separately and filter complete pairs
    failures_model_a = sum(1 for r in records_a if not is_valid_finite_score(r.get("score")))
    failures_model_b = sum(1 for r in records_b if not is_valid_finite_score(r.get("score")))

    paired_qids: list[str] = [
        qid
        for qid in sorted(queries_a)
        if is_valid_finite_score(rec_map_a[qid].get("score"))
        and is_valid_finite_score(rec_map_b[qid].get("score"))
    ]

    if not paired_qids:
        raise CompareError(
            f"No valid paired queries with successful scores remain between model '{model_a}' "
            f"and model '{model_b}' for task '{task}' in run '{run_id}'. "
            f"(Model '{model_a}' failures: {failures_model_a}, Model '{model_b}' failures: {failures_model_b})."
        )

    # 10. Paired bootstrap statistics (10k resamples, seed 42)
    scores_a = [float(rec_map_a[qid]["score"]) for qid in paired_qids]
    scores_b = [float(rec_map_b[qid]["score"]) for qid in paired_qids]

    sig_result = paired_bootstrap_test(
        scores_baseline=scores_a,
        scores_candidate=scores_b,
        num_samples=10000,
        confidence_level=0.95,
        seed=42,
    )

    dict_a = {f"{task}:{qid}": float(rec_map_a[qid]["score"]) for qid in paired_qids}
    dict_b = {f"{task}:{qid}": float(rec_map_b[qid]["score"]) for qid in paired_qids}
    query_diffs = per_query_diff_analysis(dict_a, dict_b)

    stats_a = single_run_bootstrap(scores_a, seed=42)
    stats_b = single_run_bootstrap(scores_b, seed=42)

    report = {
        "comparison_type": "cross_model",
        "run_id": canonical_run_id,
        "run_baseline": canonical_run_id,
        "run_candidate": canonical_run_id,
        "model_a": model_a,
        "model_b": model_b,
        "model_id": f"{model_a}_vs_{model_b}",
        "task_id": task,
        "task_filter": task,
        "seed": manifest_seed,
        "dataset_checksum": dataset_checksum,
        "paired_success_count": len(paired_qids),
        "failures_model_a": failures_model_a,
        "failures_model_b": failures_model_b,
        "failures_baseline": failures_model_a,
        "failures_candidate": failures_model_b,
        "model_a_stats": stats_a,
        "model_b_stats": stats_b,
        "baseline_stats": stats_a,
        "candidate_stats": stats_b,
        "significance": sig_result,
        "query_diffs": query_diffs,
    }

    if write_report:
        report_dir = results_path.parent
        if report_dir.is_dir():
            report_filename = make_cross_model_compare_report_filename(
                model_a=model_a,
                model_b=model_b,
                task_id=task,
            )
            report_path = report_dir / report_filename
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            report["report_path"] = str(report_path)

    return report


def run_comparison(
    run_a_ref: str,
    run_b_ref: str | None = None,
    model: str | None = None,
    task: str | None = None,
    cross_model: tuple[str, str] | list[str] | None = None,
    base_dir: str = "runs",
    write_report: bool = True,
) -> dict[str, Any]:
    """Compare two evaluation runs, checking for regression / improvement with bootstrap CIs.

    Args:
        run_a_ref: Baseline run id or path to results.jsonl (or single matrix run for cross_model).
        run_b_ref: Candidate run id or path to results.jsonl (omitted if cross_model is set).
        model: Model identifier to scope comparison. Required if either run has multiple models.
        task: Optional task identifier to slice comparison (required if cross_model is set).
        cross_model: Optional pair of model IDs for within-run cross-model compare on run_a_ref.
        base_dir: Directory where runs are stored (default: 'runs').
        write_report: Whether to write compare report to disk.

    Returns:
        dict containing paired statistics, per-query diffs, failure counts, and verdicts.
    """
    if cross_model is not None:
        if run_b_ref is not None:
            raise CompareError("--cross-model cannot be combined with a second run ID.")
        if model is not None:
            raise CompareError("--cross-model cannot be combined with --model.")
        if not task:
            raise CompareError("--task <task_id> is required for --cross-model compare.")
        if len(cross_model) != 2:
            raise CompareError("--cross-model requires exactly two model IDs.")
        return run_cross_model_comparison(
            run_ref=run_a_ref,
            model_a=cross_model[0],
            model_b=cross_model[1],
            task=task,
            base_dir=base_dir,
            write_report=write_report,
        )

    if run_b_ref is None:
        raise CompareError(
            "Two run IDs are required for ordinary compare (or use --cross-model for single-run compare)."
        )

    run_a_id, path_a, manifest_a_path = _resolve_run_files(run_a_ref, base_dir)
    run_b_id, path_b, manifest_b_path = _resolve_run_files(run_b_ref, base_dir)

    manifest_a = load_manifest(manifest_a_path, run_a_id)
    manifest_b = load_manifest(manifest_b_path, run_b_id)

    # Round 3: Verify raw bytes of each run's results.jsonl against its own manifest checksum
    _verify_results_integrity(run_a_id, path_a, manifest_a)
    _verify_results_integrity(run_b_id, path_b, manifest_b)

    records_a = load_results_jsonl(path_a)
    records_b = load_results_jsonl(path_b)

    models_a = sorted(set(r["model_id"] for r in records_a if "model_id" in r))
    models_b = sorted(set(r["model_id"] for r in records_b if "model_id" in r))

    # Item 2: Keep --model scoping within the same model; do not add cross-model comparison
    if model is None:
        if len(models_a) > 1 or len(models_b) > 1:
            raise CompareError(
                f"Cannot compare without --model: Run A has models {models_a}, Run B has models {models_b}. "
                "Specify --model <id> to scope comparison."
            )
        if len(models_a) == 1 and len(models_b) == 1:
            if models_a[0] == models_b[0]:
                target_model = models_a[0]
            else:
                raise CompareError(
                    f"Run A evaluated model '{models_a[0]}' but Run B evaluated model '{models_b[0]}'. "
                    "Dokime requires --model scoping within the same model; cross-model comparison is not supported."
                )
        else:
            raise CompareError("No models found in one or both runs.")
    else:
        target_model = model
        if target_model not in models_a:
            raise CompareError(f"Model '{target_model}' not found in Run A ({run_a_id}). Available: {models_a}")
        if target_model not in models_b:
            raise CompareError(f"Model '{target_model}' not found in Run B ({run_b_id}). Available: {models_b}")

    # Filter records by target_model
    filtered_a = [r for r in records_a if r.get("model_id") == target_model]
    filtered_b = [r for r in records_b if r.get("model_id") == target_model]

    # Filter by task if specified
    if task is not None:
        filtered_a = [r for r in filtered_a if r.get("task_id") == task]
        filtered_b = [r for r in filtered_b if r.get("task_id") == task]
        if not filtered_a:
            raise CompareError(f"No records found for task '{task}' in Run A ({run_a_id}).")
        if not filtered_b:
            raise CompareError(f"No records found for task '{task}' in Run B ({run_b_id}).")

    # Round 3: Compute explicit unpaired-query counts for baseline-only and candidate-only records after model/task scoping
    keys_a = set((r["task_id"], r["query_id"]) for r in filtered_a)
    keys_b = set((r["task_id"], r["query_id"]) for r in filtered_b)
    unpaired_baseline = len(keys_a - keys_b)
    unpaired_candidate = len(keys_b - keys_a)

    # Item 2: Verify dataset checksums from manifests for whole-task comparability
    tasks_to_compare = sorted(set(r["task_id"] for r in filtered_a) & set(r["task_id"] for r in filtered_b))
    if not tasks_to_compare:
        raise CompareError("No overlapping tasks found between Run A and Run B.")

    checksums_a = manifest_a.get("checksums", {})
    checksums_b = manifest_b.get("checksums", {})

    for tid in tasks_to_compare:
        key = f"dataset_{tid}"
        if key not in checksums_a:
            raise CompareError(f"Missing required dataset checksum for task '{tid}' in Run A manifest.")
        if key not in checksums_b:
            raise CompareError(f"Missing required dataset checksum for task '{tid}' in Run B manifest.")
        if checksums_a[key] != checksums_b[key]:
            raise CompareError(
                f"Dataset checksum mismatch for task '{tid}': "
                f"Run A has {checksums_a[key]}, Run B has {checksums_b[key]}. "
                "Incompatible datasets cannot be compared."
            )

    # Item 1: Count failures separately; do NOT turn null scores / errors into 0.0
    failures_baseline = sum(1 for r in filtered_a if not is_valid_finite_score(r.get("score")))
    failures_candidate = sum(1 for r in filtered_b if not is_valid_finite_score(r.get("score")))

    # Build per-query mappings for valid queries
    valid_scores_a: dict[tuple[str, str], float] = {}
    expected_a: dict[tuple[str, str], str] = {}
    for r in filtered_a:
        key = (r["task_id"], r["query_id"])
        if key in expected_a:
            raise CompareError(
                f"Duplicate query record detected in run {run_a_id} for task '{key[0]}', query '{key[1]}'. "
                "Run results are ambiguous and cannot be compared."
            )
        expected_a[key] = str(r.get("expected", ""))
        if is_valid_finite_score(r.get("score")):
            # Genuinely scored 0.0 remains valid
            valid_scores_a[key] = float(r["score"])

    valid_scores_b: dict[tuple[str, str], float] = {}
    expected_b: dict[tuple[str, str], str] = {}
    for r in filtered_b:
        key = (r["task_id"], r["query_id"])
        if key in expected_b:
            raise CompareError(
                f"Duplicate query record detected in run {run_b_id} for task '{key[0]}', query '{key[1]}'. "
                "Run results are ambiguous and cannot be compared."
            )
        expected_b[key] = str(r.get("expected", ""))
        if is_valid_finite_score(r.get("score")):
            valid_scores_b[key] = float(r["score"])

    # Item 2: Verify expected answers on each overlapping query
    all_common_keys = sorted(set(expected_a.keys()) & set(expected_b.keys()))
    for key in all_common_keys:
        if expected_a[key] != expected_b[key]:
            raise CompareError(
                f"Expected answer mismatch on query '{key[1]}' in task '{key[0]}': "
                f"Run A expected '{expected_a[key]}', Run B expected '{expected_b[key]}'. "
                "Cannot pair queries with conflicting expected answers."
            )

    # Item 1: Exclude failed rows from paired statistics and per-query score differences
    # Only pair rows where BOTH runs succeeded
    paired_success_keys = sorted(set(valid_scores_a.keys()) & set(valid_scores_b.keys()))

    if not paired_success_keys:
        raise CompareError(
            f"No valid paired queries with successful scores remain between Run A ({run_a_id}) "
            f"and Run B ({run_b_id}) for model '{target_model}'. "
            f"(Run A failures: {failures_baseline}, Run B failures: {failures_candidate})."
        )

    scores_a = [valid_scores_a[k] for k in paired_success_keys]
    scores_b = [valid_scores_b[k] for k in paired_success_keys]

    # Paired bootstrap test on paired successes (10k resamples, seed 42)
    sig_result = paired_bootstrap_test(
        scores_baseline=scores_a,
        scores_candidate=scores_b,
        num_samples=10000,
        confidence_level=0.95,
        seed=42,
    )

    # Per-query diff analysis on paired successes
    dict_a = {f"{k[0]}:{k[1]}": valid_scores_a[k] for k in paired_success_keys}
    dict_b = {f"{k[0]}:{k[1]}": valid_scores_b[k] for k in paired_success_keys}
    query_diffs = per_query_diff_analysis(dict_a, dict_b)

    base_stats = single_run_bootstrap(scores_a, seed=42)
    cand_stats = single_run_bootstrap(scores_b, seed=42)

    report = {
        "run_baseline": run_a_id,
        "run_candidate": run_b_id,
        "model_id": target_model,
        "task_filter": task,
        "paired_success_count": len(paired_success_keys),
        "failures_baseline": failures_baseline,
        "failures_candidate": failures_candidate,
        "unpaired_baseline": unpaired_baseline,
        "unpaired_candidate": unpaired_candidate,
        "unpaired_baseline_count": unpaired_baseline,
        "unpaired_candidate_count": unpaired_candidate,
        "baseline_only_count": unpaired_baseline,
        "candidate_only_count": unpaired_candidate,
        "baseline_stats": base_stats,
        "candidate_stats": cand_stats,
        "significance": sig_result,
        "query_diffs": query_diffs,
    }

    if write_report:
        report_dir = Path(base_dir) / run_b_id
        if report_dir.is_dir():
            report_filename = make_compare_report_filename(
                baseline_run_id=run_a_id,
                model_id=target_model,
                task_id=task,
            )
            report_path = report_dir / report_filename
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            report["report_path"] = str(report_path)

    return report
