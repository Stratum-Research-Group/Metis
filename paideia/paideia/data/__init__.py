"""Paideia data loading and splitting package."""

from paideia.data.router_loader import load_router_dataset, RouterDatasetAudit
from paideia.data.walk_loader import load_walk_dataset, PolicyStepSample, WalkDatasetAudit
from paideia.data.splitter import (
    DatasetSplits,
    split_grouped_dataset,
    compute_item_hash,
    compute_split_hash,
    derive_question_family_key,
)
from paideia.data.card import DatasetCard

__all__ = [
    "load_router_dataset",
    "RouterDatasetAudit",
    "load_walk_dataset",
    "PolicyStepSample",
    "WalkDatasetAudit",
    "DatasetSplits",
    "split_grouped_dataset",
    "compute_item_hash",
    "compute_split_hash",
    "derive_question_family_key",
    "DatasetCard",
]
