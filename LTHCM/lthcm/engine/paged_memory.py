"""Paged Memory Facet Pool and Continual Learning Retention Probe (Phase 5 / mini-AGI model).

Implements:
- 3-tier storage topology: Disk (cold checkpoints) -> RAM (warm cache) -> Active (bounded working set).
- Per-facet optimizer moments: Adam/momentum moments travel with the facet file, not fixed global slots.
- Bounded working sets with LRU eviction and atomic file persistence.
- Continual learning forgetting & retention probe (comparing massed vs interleaved update degradation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch


@dataclass
class MemoryFacet:
    """A discrete memory facet / topological cluster with weights and optimizer moments."""

    facet_id: str
    domain: str
    weights: torch.Tensor
    moments: Dict[str, torch.Tensor] = field(default_factory=dict)
    step: int = 0
    last_accessed: float = field(default_factory=time.time)
    is_dirty: bool = False
    tier: str = "active"

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "facet_id": self.facet_id,
            "domain": self.domain,
            "step": self.step,
            "last_accessed": self.last_accessed,
            "is_dirty": self.is_dirty,
            "tier": self.tier,
            "weight_shape": list(self.weights.shape),
            "weight_norm": float(torch.norm(self.weights).item()),
        }


class PagedFacetPool:
    """
    Manages memory facets across Disk, RAM, and Active execution device.
    Strictly bounds active resident facets and guarantees atomic persistence.
    """

    def __init__(
        self,
        storage_dir: Path | str,
        max_active: int = 4,
        max_ram: int = 16,
        device: str = "cpu",
    ):
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_active = max_active
        self.max_ram = max_ram
        self.device = torch.device(device)

        # In-memory tiers
        self.active_facets: Dict[str, MemoryFacet] = {}
        self.ram_facets: Dict[str, MemoryFacet] = {}

    def _facet_path(self, facet_id: str) -> Path:
        clean_id = hashlib.sha256(facet_id.encode("utf-8")).hexdigest()[:16]
        return self.storage_dir / f"facet_{clean_id}.pt"

    def register_facet(self, facet: MemoryFacet) -> None:
        """Adds a newly created facet to the pool, placing it in the active set."""
        facet.weights = facet.weights.to(self.device)
        facet.tier = "active"
        facet.last_accessed = time.time()
        facet.is_dirty = True

        if len(self.active_facets) >= self.max_active:
            self.evict_lru_active()

        self.active_facets[facet.facet_id] = facet

    def get_facet(self, facet_id: str) -> MemoryFacet:
        """
        Retrieves a facet by ID, paging it into the Active working set if needed.
        Path: Disk -> RAM -> Active.
        """
        # 1. Already in Active
        if facet_id in self.active_facets:
            f = self.active_facets[facet_id]
            f.last_accessed = time.time()
            return f

        # 2. In RAM cache
        if facet_id in self.ram_facets:
            f = self.ram_facets.pop(facet_id)
            if len(self.active_facets) >= self.max_active:
                self.evict_lru_active()
            f.weights = f.weights.to(self.device)
            for k in f.moments:
                f.moments[k] = f.moments[k].to(self.device)
            f.tier = "active"
            f.last_accessed = time.time()
            self.active_facets[facet_id] = f
            return f

        # 3. On Disk
        f = self._load_from_disk(facet_id)
        if len(self.active_facets) >= self.max_active:
            self.evict_lru_active()
        f.weights = f.weights.to(self.device)
        for k in f.moments:
            f.moments[k] = f.moments[k].to(self.device)
        f.tier = "active"
        f.last_accessed = time.time()
        self.active_facets[facet_id] = f
        return f

    def evict_lru_active(self) -> Optional[str]:
        """Evicts the least recently accessed active facet into RAM."""
        if not self.active_facets:
            return None

        # Find LRU facet
        lru_id = min(self.active_facets.keys(), key=lambda k: self.active_facets[k].last_accessed)
        facet = self.active_facets.pop(lru_id)

        # Move tensors to CPU for RAM tier
        facet.weights = facet.weights.cpu()
        for k in facet.moments:
            facet.moments[k] = facet.moments[k].cpu()
        facet.tier = "ram"

        # Check RAM capacity
        if len(self.ram_facets) >= self.max_ram:
            self.evict_lru_ram()

        self.ram_facets[lru_id] = facet
        return lru_id

    def evict_lru_ram(self) -> Optional[str]:
        """Evicts the least recently accessed RAM facet to Disk."""
        if not self.ram_facets:
            return None

        lru_id = min(self.ram_facets.keys(), key=lambda k: self.ram_facets[k].last_accessed)
        facet = self.ram_facets.pop(lru_id)
        if facet.is_dirty:
            self._save_to_disk(facet)
        facet.tier = "disk"
        return lru_id

    def flush_all(self) -> None:
        """Persists all dirty active and RAM facets to disk."""
        for facet in list(self.active_facets.values()):
            if facet.is_dirty:
                self._save_to_disk(facet)
        for facet in list(self.ram_facets.values()):
            if facet.is_dirty:
                self._save_to_disk(facet)

    def _save_to_disk(self, facet: MemoryFacet) -> Path:
        target = self._facet_path(facet.facet_id)
        tmp = target.with_name(f".{target.name}.tmp")

        payload = {
            "facet_id": facet.facet_id,
            "domain": facet.domain,
            "weights": facet.weights.cpu(),
            "moments": {k: v.cpu() for k, v in facet.moments.items()},
            "step": facet.step,
            "last_accessed": facet.last_accessed,
        }
        torch.save(payload, tmp)
        os.replace(tmp, target)
        facet.is_dirty = False
        return target

    def _load_from_disk(self, facet_id: str) -> MemoryFacet:
        path = self._facet_path(facet_id)
        if not path.is_file():
            raise FileNotFoundError(f"Facet checkpoint not found on disk: {path}")

        payload = torch.load(path, map_location="cpu", weights_only=True)
        return MemoryFacet(
            facet_id=payload["facet_id"],
            domain=payload["domain"],
            weights=payload["weights"],
            moments=payload.get("moments", {}),
            step=payload.get("step", 0),
            last_accessed=payload.get("last_accessed", time.time()),
            is_dirty=False,
            tier="disk",
        )


class RetentionProbe:
    """
    Evaluates continual learning forgetting and retention metrics
    under massed vs interleaved update exposure (adapted from mini-AGI).
    """

    @classmethod
    def evaluate_drift(
        cls,
        initial_weights: torch.Tensor,
        current_weights: torch.Tensor,
    ) -> float:
        """Computes Euclidean weight drift normalized by initial parameter norm."""
        drift = torch.norm(current_weights - initial_weights).item()
        base = max(1e-9, torch.norm(initial_weights).item())
        return drift / base

    @classmethod
    def compute_forgetting_score(
        cls,
        baseline_performance: Dict[str, float],
        post_update_performance: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Computes retention and degradation across domains.
        Positive degradation indicates catastrophic forgetting.
        """
        results: Dict[str, float] = {}
        total_drop = 0.0

        for domain, base_score in baseline_performance.items():
            post_score = post_update_performance.get(domain, 0.0)
            drop = base_score - post_score
            results[domain] = drop
            total_drop += drop

        avg_drop = total_drop / max(1, len(baseline_performance))
        return {
            "domain_drops": results,
            "mean_retention_drop": avg_drop,
            "exhibits_forgetting": avg_drop > 0.05,
        }
