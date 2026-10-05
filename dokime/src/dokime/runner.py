"""Execution runner matrix evaluating models across tasks with JSONL logging."""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dokime.client import HestiaAPIError, HestiaClient
from dokime.config import RunConfig, validate_unique_ids
from dokime.manifest import create_manifest, sha256_text
from dokime.stats import single_run_bootstrap
from dokime.tasks.protocol import (
    Task,
    extract_task_prediction,
    format_task_prompt,
    materialize_dataset,
)
from dokime.tasks.registry import get_task


def run_evaluation(
    config: RunConfig,
    client: HestiaClient | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Execute evaluation matrix (models x tasks) for a single run_id.

    Args:
        config: RunConfig with models, tasks, seed, temperature, etc.
        client: HestiaClient instance (created from config/env if None).
        run_id: Optional predetermined run_id (generates UUID4 if None).

    Returns:
        dict containing run_id, output_dir, stats, manifest.
    """
    if not config.models:
        raise ValueError("No models specified for evaluation run.")
    if not config.tasks:
        raise ValueError("No tasks specified for evaluation run.")

    validate_unique_ids(config.models, "model")
    validate_unique_ids(config.tasks, "task")

    if run_id is None:
        run_id = f"run_{uuid.uuid4().hex[:12]}"

    if client is None:
        client = HestiaClient(base_url=config.url)

    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl_path = output_dir / "results.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"

    # Persist prompt template artifact if provided (Item 5)
    prompt_template_artifact_path: Path | None = None
    if config.prompt_template is not None:
        prompt_template_artifact_path = output_dir / "prompt_template.txt"
        prompt_template_artifact_path.write_text(config.prompt_template, encoding="utf-8")

    # Preload tasks and resolve dataset paths (Item 3)
    instantiated_tasks: dict[str, Task] = {}
    dataset_paths: dict[str, Path | None] = {}
    loaded_queries_by_task: dict[str, list[Any]] = {}

    for tid in config.tasks:
        task = get_task(tid)
        instantiated_tasks[tid] = task
        queries = task.load()
        loaded_queries_by_task[tid] = queries

        # Check if task provides an on-disk dataset path
        src_path = getattr(task, "dataset_path", None)
        if src_path and Path(src_path).is_file():
            dataset_paths[tid] = Path(src_path)
        else:
            # Materialize loaded queries deterministically as run artifact (Item 3)
            materialized_path = output_dir / f"dataset_{tid}.json"
            materialize_dataset(queries, materialized_path)
            dataset_paths[tid] = materialized_path

    results_count = 0
    scores_by_cell: dict[tuple[str, str], list[float]] = {}
    failures_by_cell: dict[tuple[str, str], int] = {}
    totals_by_cell: dict[tuple[str, str], int] = {}

    with open(results_jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for model_id in config.models:
            for task_id in config.tasks:
                task = instantiated_tasks[task_id]
                queries = loaded_queries_by_task[task_id]
                cell_key = (model_id, task_id)
                scores_by_cell[cell_key] = []
                failures_by_cell[cell_key] = 0
                totals_by_cell[cell_key] = 0

                for query in queries:
                    totals_by_cell[cell_key] += 1

                    prompt_hash: str | None = None
                    predicted: str | None = None
                    score: float | None = None
                    error_msg: str | None = None
                    latency_ms: float = 0.0

                    try:
                        # Format prompt using task method or safe fallback (Item 3 & Round 2 Item 2)
                        rendered_prompt = format_task_prompt(task, query, template=config.prompt_template)
                        prompt_hash = sha256_text(rendered_prompt)

                        raw_response, latency_ms = client.chat_completion(
                            model=model_id,
                            prompt=rendered_prompt,
                            seed=config.seed,
                            temperature=config.temperature,
                            max_tokens=config.max_tokens,
                        )
                        predicted = extract_task_prediction(task, query, raw_response)
                        raw_score = float(task.score(query, raw_response))
                        if not math.isfinite(raw_score):
                            error_msg = f"Non-finite score returned by task scorer: {raw_score}"
                            score = None
                            failures_by_cell[cell_key] += 1
                        else:
                            score = raw_score
                            scores_by_cell[cell_key].append(score)
                    except (HestiaAPIError, Exception) as exc:
                        # Round 2 Item 2: Formatter or client error produces null-score record with null prompt_hash if rendering failed.
                        error_msg = str(exc)
                        predicted = None
                        score = None
                        failures_by_cell[cell_key] += 1

                    result_record = {
                        "run_id": run_id,
                        "task_id": task_id,
                        "query_id": query.id,
                        "model_id": model_id,
                        "prompt_hash": prompt_hash,
                        "predicted": predicted,
                        "expected": query.expected,
                        "score": score,
                        "latency_ms": round(latency_ms, 2),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    if error_msg:
                        result_record["error"] = error_msg

                    jsonl_file.write(json.dumps(result_record) + "\n")
                    results_count += 1

    # Item 1 & 7: Compute summary and write summary.json BEFORE finalizing manifest.json
    summary_data: dict[str, Any] = {}
    for (m_id, t_id), valid_scores in scores_by_cell.items():
        cell_name = f"{m_id}/{t_id}"
        total_q = totals_by_cell[(m_id, t_id)]
        failed_q = failures_by_cell[(m_id, t_id)]
        # LTHCM statistics convention: bootstrap seed is fixed at 42, independent of model-request seed
        cell_stats = single_run_bootstrap(valid_scores, seed=42)
        summary_data[cell_name] = {
            "total_queries": total_q,
            "successful_queries": len(valid_scores),
            "failed_queries": failed_q,
            "stats": cell_stats,
        }

    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf, indent=2)

    # Resolve served model identities from Hestia client
    model_identities: dict[str, Any] = {}
    try:
        available_models = client.get_models_metadata()
        models_by_name = {m.get("name"): m for m in available_models if isinstance(m, dict) and m.get("name")}
        for mid in config.models:
            if mid in models_by_name:
                model_identities[mid] = models_by_name[mid]
            else:
                model_identities[mid] = {"name": mid, "state": "unresolved"}
    except Exception:
        for mid in config.models:
            model_identities[mid] = {"name": mid, "state": "unresolved"}

    # Item 7: Finalize manifest.json including summary.json checksum
    manifest = create_manifest(
        run_id=run_id,
        model_ids=config.models,
        task_ids=config.tasks,
        seed=config.seed,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        results_jsonl_path=results_jsonl_path,
        summary_json_path=summary_path,
        dataset_paths=dataset_paths,
        prompt_template=config.prompt_template,
        prompt_template_path=config.prompt_template_path,
        prompt_template_artifact_path=prompt_template_artifact_path,
        output_path=manifest_path,
        model_identities=model_identities,
    )

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "results_path": str(results_jsonl_path),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "total_queries": results_count,
        "stats": summary_data,
        "manifest": manifest,
    }
