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


def test_r1_lthcm_adapter_simulation_flag():
    """Verify LTHCMAdapter marks citations as simulated when using local demo corpus."""
    from kybernetes.adapters.lthcm import LTHCMAdapter
    adapter = LTHCMAdapter()
    assert adapter.is_simulated is True
    citations = adapter.retrieve("kybernetes metis")
    assert len(citations) > 0
    assert all(c.is_simulated is True for c in citations)


def test_r2_krisis_adapter_simulation_flag():
    """Verify KrisisAdapter marks verification report as simulated."""
    from kybernetes.adapters.krisis import KrisisAdapter
    adapter = KrisisAdapter()
    citations = [
        Citation(citation_id="[1]", passage_id="p1", source_id="d1", source_title="D1", text="T1", score=0.9)
    ]
    report = adapter.verify_response(query="test", response_text="Answer based on [1]", citations=citations)
    assert report.is_simulated is True
