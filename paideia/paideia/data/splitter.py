"""Leakage-safe, group-aware dataset splitting with cryptographic verification."""

import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


def compute_item_hash(item: Any) -> str:
    """Compute sha256 checksum of an item's JSON representation."""
    if hasattr(item, "model_dump"):
        data = item.model_dump()
    elif isinstance(item, dict):
        data = item
    else:
        data = str(item)
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_split_hash(items: List[Any]) -> str:
    """Compute deterministic collective sha256 checksum of an entire split."""
    hasher = hashlib.sha256()
    for item in sorted(items, key=lambda x: getattr(x, "query_id", getattr(x, "sample_id", str(x)))):
        hasher.update(compute_item_hash(item).encode("utf-8"))
    return hasher.hexdigest()


def derive_question_family_key(question: str) -> str:
    """
    Derive grouping key to prevent near-duplicate/family leakage.
    Strips punctuation, lowercases, and takes the main content tokens.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", question.lower())
    # Keep first 5 significant tokens as question stem/family
    filtered = [t for t in tokens if t not in {"is", "the", "a", "an", "what", "which", "who", "where", "when", "does"}]
    if not filtered:
        return question.strip().lower()
    return " ".join(filtered[:4])


class DatasetSplits:
    """Immutable train/dev/test dataset splits."""
    def __init__(
        self,
        train: List[Any],
        dev: List[Any],
        test: List[Any],
        seed: int = 42,
    ):
        self.train = train
        self.dev = dev
        self.test = test
        self.seed = seed
        self.train_checksum = compute_split_hash(train)
        self.dev_checksum = compute_split_hash(dev)
        self.test_checksum = compute_split_hash(test)

    def summary(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "train_count": len(self.train),
            "dev_count": len(self.dev),
            "test_count": len(self.test),
            "train_checksum": self.train_checksum,
            "dev_checksum": self.dev_checksum,
            "test_checksum": self.test_checksum,
        }


def split_grouped_dataset(
    items: List[Any],
    group_key_fn: Callable[[Any], str],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplits:
    """
    Split items into train, dev, and test sets by groups to prevent data leakage.
    Ensures that all items sharing the same group key are assigned strictly to the same split.
    
    Args:
        items: List of items to split.
        group_key_fn: Function returning grouping key for an item.
        train_ratio: Target fraction for training set.
        dev_ratio: Target fraction for dev set.
        test_ratio: Target fraction for final frozen test set.
        seed: Random seed for deterministic reproducibility.
        
    Returns:
        DatasetSplits instance.
    """
    if abs((train_ratio + dev_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("Split ratios must sum to 1.0")

    # Group items by group key
    groups: Dict[str, List[Any]] = {}
    for item in items:
        k = group_key_fn(item)
        groups.setdefault(k, []).append(item)

    rng = np.random.default_rng(seed)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    total_items = len(items)
    if total_items >= 3:
        target_dev = max(1, int(round(total_items * dev_ratio))) if dev_ratio > 0 else 0
        target_test = max(1, int(round(total_items * test_ratio))) if test_ratio > 0 else 0
        target_train = max(1, total_items - target_dev - target_test)
    else:
        target_train = total_items
        target_dev = 0
        target_test = 0

    train_items: List[Any] = []
    dev_items: List[Any] = []
    test_items: List[Any] = []

    for gk in group_keys:
        grp = groups[gk]
        if len(train_items) < target_train:
            train_items.extend(grp)
        elif len(dev_items) < target_dev:
            dev_items.extend(grp)
        else:
            test_items.extend(grp)

    return DatasetSplits(
        train=train_items,
        dev=dev_items,
        test=test_items,
        seed=seed,
    )
