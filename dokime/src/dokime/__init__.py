"""Dokime: evaluation harness and measurement layer for local AI stack."""

from dokime.bom import DatasetBOM, generate_dataset_bom
from dokime.lineage import (
    ExperimentNode,
    build_experiment_tree,
    compute_parameter_diff,
    evaluate_lineage_regressions,
    get_lineage_path,
)

__version__ = "0.1.0"

__all__ = [
    "DatasetBOM",
    "generate_dataset_bom",
    "ExperimentNode",
    "build_experiment_tree",
    "compute_parameter_diff",
    "evaluate_lineage_regressions",
    "get_lineage_path",
]

