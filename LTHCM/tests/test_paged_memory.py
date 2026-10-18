"""Unit tests for PagedFacetPool and RetentionProbe (Phase 5)."""

from pathlib import Path
import time
import pytest
import torch

from lthcm.engine.paged_memory import MemoryFacet, PagedFacetPool, RetentionProbe


def test_paged_facet_pool_tier_transitions(tmp_path: Path):
    storage = tmp_path / "facets"
    pool = PagedFacetPool(storage_dir=storage, max_active=2, max_ram=2)

    # 1. Create and register 2 facets (filling Active tier)
    f1 = MemoryFacet(facet_id="f1", domain="code", weights=torch.ones(10))
    f2 = MemoryFacet(facet_id="f2", domain="math", weights=torch.full((10,), 2.0))
    pool.register_facet(f1)
    pool.register_facet(f2)

    assert "f1" in pool.active_facets
    assert "f2" in pool.active_facets
    assert len(pool.ram_facets) == 0

    # 2. Register 3rd facet -> causes LRU (f1) to evict to RAM
    time.sleep(0.01)
    f3 = MemoryFacet(facet_id="f3", domain="physics", weights=torch.full((10,), 3.0))
    pool.register_facet(f3)

    assert len(pool.active_facets) == 2
    assert "f1" in pool.ram_facets
    assert pool.ram_facets["f1"].tier == "ram"

    # 3. Add 4th and 5th facets -> fills RAM and evicts oldest to Disk
    f4 = MemoryFacet(facet_id="f4", domain="history", weights=torch.full((10,), 4.0))
    pool.register_facet(f4)
    f5 = MemoryFacet(facet_id="f5", domain="law", weights=torch.full((10,), 5.0))
    pool.register_facet(f5)

    # 4. Flush all dirty to ensure on-disk persistence
    pool.flush_all()

    # 5. Page in f1 (which is now on disk)
    retrieved_f1 = pool.get_facet("f1")
    assert retrieved_f1.facet_id == "f1"
    assert torch.allclose(retrieved_f1.weights, torch.ones(10))
    assert retrieved_f1.tier == "active"
    assert "f1" in pool.active_facets


def test_isolated_optimizer_moments_persistence(tmp_path: Path):
    storage = tmp_path / "moments_storage"
    pool = PagedFacetPool(storage_dir=storage, max_active=1, max_ram=1)

    # Facet with distinct Adam moments
    m1 = torch.tensor([0.1, 0.2, 0.3, 0.4])
    m2 = torch.tensor([0.01, 0.04, 0.09, 0.16])
    facet = MemoryFacet(
        facet_id="expert_opt_1",
        domain="finance",
        weights=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        moments={"exp_avg": m1, "exp_avg_sq": m2},
        step=42,
    )
    pool.register_facet(facet)

    # Force eviction to disk by registering 2 more facets
    f_dummy1 = MemoryFacet(facet_id="dummy1", domain="d", weights=torch.zeros(4))
    f_dummy2 = MemoryFacet(facet_id="dummy2", domain="d", weights=torch.zeros(4))
    pool.register_facet(f_dummy1)
    pool.register_facet(f_dummy2)

    # Verify expert_opt_1 is now completely evicted from memory
    assert "expert_opt_1" not in pool.active_facets
    assert "expert_opt_1" not in pool.ram_facets

    # Reload from disk and verify moments are perfectly preserved
    reloaded = pool.get_facet("expert_opt_1")
    assert reloaded.step == 42
    assert "exp_avg" in reloaded.moments
    assert "exp_avg_sq" in reloaded.moments
    assert torch.allclose(reloaded.moments["exp_avg"], m1)
    assert torch.allclose(reloaded.moments["exp_avg_sq"], m2)
    assert torch.allclose(reloaded.weights, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_retention_probe_metrics():
    w_initial = torch.ones(50)
    w_drifted = torch.ones(50) * 1.05
    drift = RetentionProbe.evaluate_drift(w_initial, w_drifted)
    assert drift > 0.0

    baseline_perf = {"code": 0.90, "math": 0.85, "literature": 0.80}
    # Severe drop on code under massed updates
    post_massed_perf = {"code": 0.95, "math": 0.60, "literature": 0.55}

    res = RetentionProbe.compute_forgetting_score(baseline_perf, post_massed_perf)
    assert res["exhibits_forgetting"] is True
    assert res["domain_drops"]["math"] == pytest.approx(0.25)
    assert res["domain_drops"]["literature"] == pytest.approx(0.25)
