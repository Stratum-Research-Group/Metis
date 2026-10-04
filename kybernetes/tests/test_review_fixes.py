"""Regression tests for audit fixes in Metis review (aacb3e58)."""

import pytest
from kybernetes.adapters.hestia import HestiaAdapter
from kybernetes.adapters.dianoia import DianoiaAdapter
from kybernetes.contracts.turn import Citation
from kybernetes.cli import cli
from click.testing import CliRunner


def test_item1_kybernetes_serve_default_port():
    """Verify kybernetes serve defaults to port 8081 to avoid colliding with Hestia (8080)."""
    serve_cmd = cli.commands["serve"]
    port_param = next(p for p in serve_cmd.params if p.name == "port")
    assert port_param.default == 8081


def test_item2_hestia_adapter_default_port():
    """Verify HestiaAdapter defaults to Hestia's port 8080."""
    adapter = HestiaAdapter()
    assert adapter.base_url == "http://127.0.0.1:8080/v1"


def test_item3_hestia_adapter_default_model_name():
    """Verify HestiaAdapter defaults to registered model in hestia/models.yaml."""
    adapter = HestiaAdapter()
    assert adapter.default_model == "llama-3.1-8b-instruct"


def test_item7_dianoia_adapter_simulation_disclosure():
    """Verify DianoiaAdapter discloses simulated status in both zero-edge and multi-edge walks."""
    adapter = DianoiaAdapter()
    citations = [
        Citation(citation_id="[1]", passage_id="p1", source_id="d1", source_title="D1", text="T1", score=0.9),
        Citation(citation_id="[2]", passage_id="p2", source_id="d2", source_title="D2", text="T2", score=0.8),
    ]

    # Multi-edge walk
    res_multi = adapter.deliberate(query="test", seed_citations=citations)
    assert res_multi.is_simulated is True
    assert "simulated adapter" in res_multi.deliberation_summary

    # Zero-edge walk
    res_zero = adapter.deliberate(query="test", seed_citations=citations[:1])
    assert res_zero.is_simulated is True
    assert "simulated adapter" in res_zero.deliberation_summary
