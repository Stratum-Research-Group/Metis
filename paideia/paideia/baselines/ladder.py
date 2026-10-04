"""Baseline ladder evaluation runner."""

import os
import resource
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from tabulate import tabulate

from paideia.baselines.base import BasePolicy
from paideia.baselines.majority import MajorityBaseline
from paideia.baselines.regex_router import RegexRouterBaseline
from paideia.baselines.lexical import LexicalLogisticBaseline
from paideia.baselines.embedding import SmallEmbeddingBaseline
from paideia.contracts.lthcm_schema import RouteUtility


class BaselineEvaluationReport:
    """Consolidated performance comparison across the baseline ladder."""
    def __init__(self, task_name: str, results: List[Dict[str, Any]]):
        self.task_name = task_name
        self.results = results

    def to_table(self) -> str:
        headers = [
            "Model",
            "Accuracy (%)",
            "Misrouting (%)",
            "p50 Lat (ms)",
            "p95 Lat (ms)",
            "Failures",
            "Mem (MB)",
        ]
        rows = []
        for r in self.results:
            rows.append([
                r["name"],
                f"{r['accuracy'] * 100:.2f}%",
                f"{r.get('misrouting_rate', 0.0) * 100:.2f}%" if "misrouting_rate" in r else "N/A",
                f"{r['p50_latency_ms']:.3f}",
                f"{r['p95_latency_ms']:.3f}",
                r["failures"],
                f"{r['peak_memory_mb']:.1f}",
            ])
        return tabulate(rows, headers=headers, tablefmt="github")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "results": self.results,
        }


def run_baseline_ladder(
    train_items: List[Any],
    eval_items: List[Any],
    target_extractor: Callable[[Any], Optional[str]],
    utility: Optional[RouteUtility] = None,
    task_name: str = "router_classification",
) -> BaselineEvaluationReport:
    """
    Train and evaluate the baseline ladder on identical splits.
    
    Ladder:
    1. Majority Baseline
    2. Regex / Heuristic Router
    3. Lexical TF-IDF + Logistic Regression
    4. Small Sub-word Embedding Classifier
    """
    candidates: List[BasePolicy] = [
        MajorityBaseline(),
        RegexRouterBaseline(),
        LexicalLogisticBaseline(),
        SmallEmbeddingBaseline(),
    ]

    # Pre-extract gold ground truth
    gold_labels: List[Optional[str]] = [target_extractor(item) for item in eval_items]
    valid_eval_indices = [i for i, g in enumerate(gold_labels) if g is not None]
    
    results: List[Dict[str, Any]] = []

    for model in candidates:
        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # Fit
        t_fit_start = time.perf_counter()
        model.fit(train_items, target_extractor=target_extractor)
        t_fit = time.perf_counter() - t_fit_start
        
        # Predict on eval items with per-item timing
        predictions, latencies = model.predict_with_timing(eval_items)
        
        mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux ru_maxrss is in KiB
        peak_mem_mb = max(mem_after, mem_before) / 1024.0

        # Compute accuracy & misrouting
        correct = 0
        failures = 0
        total_valid = len(valid_eval_indices)

        for idx in valid_eval_indices:
            gold = gold_labels[idx]
            pred = predictions[idx]
            if pred is None:
                failures += 1
            elif pred == gold:
                correct += 1

        accuracy = correct / max(total_valid, 1)
        misrouting_rate = 1.0 - accuracy

        p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

        results.append({
            "name": model.name,
            "accuracy": accuracy,
            "misrouting_rate": misrouting_rate,
            "p50_latency_ms": p50_lat,
            "p95_latency_ms": p95_lat,
            "failures": failures,
            "fit_time_seconds": t_fit,
            "peak_memory_mb": peak_mem_mb,
            "predictions": predictions,
        })

    return BaselineEvaluationReport(task_name=task_name, results=results)
