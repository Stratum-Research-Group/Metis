"""
Regression tests for Round 2 review defects.
Covers:
- Defect 1: Workspace source mappings & control plane runnable tools
- Defect 2: Hestia & Prosopon CLI serve options, working directory paths, port propagation
- Defect 3: Prosopon live KybernetesClient upstream selection (--no-mock)
"""

from pathlib import Path
import pytest
from click.testing import CliRunner

from hestia.cli import cli as hestia_cli
from hestia.api.app import create_app as create_hestia_app
from kybernetes.adapters.hestia import HestiaAdapter
from prosopon.cli import cli as prosopon_cli
from prosopon.config import ProsoponConfig
from prosopon.server import create_app as create_prosopon_app
from prosopon.client.kybernetes_client import KybernetesClient
from prosopon.client.mock_kybernetes import MockKybernetesClient


def test_defect2_hestia_cli_serve_options_and_paths():
    """Verify Hestia CLI serve supports --port, --host, --config, --models and path fallbacks."""
    runner = CliRunner()
    res = runner.invoke(hestia_cli, ["serve", "--help"])
    assert res.exit_code == 0
    assert "--port" in res.output
    assert "--host" in res.output
    assert "--config" in res.output
    assert "--models" in res.output

    # Verify create_app handles root-relative and hestia-relative config paths
    app = create_hestia_app(config_path="hestia.yaml", models_path="models.yaml")
    assert Path(app.state.config_path).is_file()
    assert Path(app.state.models_path).is_file()


def test_defect2_prosopon_cli_serve_and_run_options():
    """Verify Prosopon CLI supports serve with --kybernetes-url and run alias."""
    runner = CliRunner()
    res_serve = runner.invoke(prosopon_cli, ["serve", "--help"])
    assert res_serve.exit_code == 0
    assert "--kybernetes-url" in res_serve.output
    assert "--mock" in res_serve.output
    assert "--no-mock" in res_serve.output

    res_run = runner.invoke(prosopon_cli, ["run", "--help"])
    assert res_run.exit_code == 0
    assert "--kybernetes-url" in res_run.output


def test_defect2_hestia_adapter_env_port_propagation(monkeypatch):
    """Verify Kybernetes HestiaAdapter honors custom HESTIA_BASE_URL env var."""
    monkeypatch.setenv("HESTIA_BASE_URL", "http://127.0.0.1:9095/v1")
    adapter = HestiaAdapter()
    assert adapter.base_url == "http://127.0.0.1:9095/v1"


def test_defect3_prosopon_upstream_mode_selection():
    """Verify Prosopon uses real KybernetesClient when mock=False and respects custom URL."""
    cfg_mock = ProsoponConfig(mock_kybernetes=True)
    app_mock = create_prosopon_app(cfg_mock)
    assert isinstance(app_mock.state.kybernetes, MockKybernetesClient)

    custom_url = "http://127.0.0.1:8999/v1"
    cfg_real = ProsoponConfig(mock_kybernetes=False, kybernetes_api_url=custom_url)
    app_real = create_prosopon_app(cfg_real)
    assert isinstance(app_real.state.kybernetes, KybernetesClient)
    assert app_real.state.kybernetes.base_url == custom_url


@pytest.mark.asyncio
async def test_defect4_pipeline_final_status_propagates_verifier_verdict():
    """
    Verify pipeline final_status, show_work, route_summary, and trace strictly preserve
    verifier verdicts (INSUFFICIENT_EVIDENCE, CONFLICTED, VERIFIED) even with citations.
    """
    from kybernetes.contracts.enums import TurnOutcomeStatus
    from kybernetes.contracts.turn import TurnRequest, Citation
    from kybernetes.engine.pipeline import ExecutionPipeline
    from kybernetes.budget.engine import BudgetEngine
    from kybernetes.budget.storage import BudgetDatabase
    from kybernetes.engine.session import SessionManager
    from kybernetes.routing.router import PlanRouter
    from kybernetes.tools.registry import ToolRegistry
    from kybernetes.adapters.hestia import HestiaAdapter
    from kybernetes.adapters.lthcm import LTHCMAdapter
    from kybernetes.adapters.dianoia import DianoiaAdapter
    from kybernetes.adapters.phylax import PhylaxAdapter
    from kybernetes.adapters.krisis import KrisisAdapter, KrisisVerificationReport

    class MockHestia(HestiaAdapter):
        async def stream_chat(self, prompt, **kwargs):
            yield {"delta": "According to the architectural records [1], Kybernetes orchestrates Hestia."}
            yield {"usage": {"input_tokens": 10, "output_tokens": 10}}

    class VerifierInjector(KrisisAdapter):
        def __init__(self, target_status: TurnOutcomeStatus):
            self.target_status = target_status

        def verify_response(self, query, response_text, citations, is_tool_output=False):
            return KrisisVerificationReport(
                status=self.target_status,
                score=0.0 if self.target_status != TurnOutcomeStatus.VERIFIED else 0.95,
                supported_claims=1 if self.target_status == TurnOutcomeStatus.VERIFIED else 0,
                unsupported_claims=0 if self.target_status == TurnOutcomeStatus.VERIFIED else 1,
                conflict_detected=(self.target_status == TurnOutcomeStatus.CONFLICTED),
                explanation=f"Injected test status: {self.target_status.value}",
                is_simulated=False,
                verifier_mode="injected_test",
            )

    for injected_status in [
        TurnOutcomeStatus.INSUFFICIENT_EVIDENCE,
        TurnOutcomeStatus.CONFLICTED,
        TurnOutcomeStatus.VERIFIED,
    ]:
        pipeline = ExecutionPipeline(
            router=PlanRouter(),
            budget_engine=BudgetEngine(BudgetDatabase(":memory:")),
            tool_registry=ToolRegistry(),
            hestia_adapter=MockHestia(),
            lthcm_adapter=LTHCMAdapter(),
            dianoia_adapter=DianoiaAdapter(),
            phylax_adapter=PhylaxAdapter(),
            krisis_adapter=VerifierInjector(injected_status),
            session_manager=SessionManager(),
        )

        req = TurnRequest(session_id="test-defect4", query="Explain Metis ecosystem architecture.")
        events = [e async for e in pipeline.execute_turn_stream(req)]
        comp = events[-1]

        assert comp.event_type == "completed"
        assert comp.status == injected_status.value
        assert comp.payload["route_summary"]["verifier_status"] == injected_status.value
        assert comp.payload["show_work"]["verifier_status"] == injected_status.value
        assert comp.payload["route_summary"]["verifier_mode"] == "injected_test"

        trace = pipeline.get_trace(comp.turn_id)
        assert trace is not None
        assert trace.final_status == injected_status


def test_defect5_krisis_adapter_marker_validation_and_grounding():
    """
    Verify KrisisAdapter rejects unresolved markers ([999]), malformed brackets,
    unrelated claims, contradictions, and marks simulated evidence as draft.
    """
    from kybernetes.contracts.enums import TurnOutcomeStatus
    from kybernetes.contracts.turn import Citation
    from kybernetes.adapters.krisis import KrisisAdapter

    krisis = KrisisAdapter()
    real_citation = Citation(
        citation_id="[1]",
        passage_id="pass_metis_01",
        source_id="doc_arch",
        source_title="Metis Architecture",
        text="Kybernetes serves as the orchestrator coordinating Hestia and Phylax.",
        score=0.95,
        is_simulated=False,
    )

    # 1. Unresolved marker [999]
    rep_999 = krisis.verify_response(
        query="architecture",
        response_text="The moon is cheese [999].",
        citations=[real_citation],
    )
    assert rep_999.status == TurnOutcomeStatus.INSUFFICIENT_EVIDENCE
    assert rep_999.unsupported_claims == 1
    assert "unresolved" in rep_999.explanation.lower()

    # 2. Malformed unclosed bracket
    rep_malformed = krisis.verify_response(
        query="architecture",
        response_text="Kybernetes coordinates Hestia [1.",
        citations=[real_citation],
    )
    assert rep_malformed.status == TurnOutcomeStatus.INSUFFICIENT_EVIDENCE
    assert "malformed" in rep_malformed.explanation.lower()

    # 3. Known marker attached to unrelated claim
    rep_unrelated = krisis.verify_response(
        query="architecture",
        response_text="The moon is composed of aged gouda cheese [1].",
        citations=[real_citation],
    )
    assert rep_unrelated.status == TurnOutcomeStatus.INSUFFICIENT_EVIDENCE
    assert rep_unrelated.unsupported_claims == 1

    # 4. Contradiction
    rep_conflict = krisis.verify_response(
        query="architecture",
        response_text="Kybernetes never coordinates Hestia [1].",
        citations=[real_citation],
    )
    assert rep_conflict.status == TurnOutcomeStatus.CONFLICTED
    assert rep_conflict.conflict_detected is True

    # 5. Genuinely supported claim with real citation
    rep_verified = krisis.verify_response(
        query="architecture",
        response_text="Kybernetes serves as the orchestrator coordinating Hestia [1].",
        citations=[real_citation],
    )
    assert rep_verified.status == TurnOutcomeStatus.VERIFIED
    assert rep_verified.supported_claims == 1
    assert rep_verified.unsupported_claims == 0

    # 6. Genuinely supported claim with simulated citation -> draft
    sim_citation = real_citation.model_copy(update={"is_simulated": True})
    rep_sim = krisis.verify_response(
        query="architecture",
        response_text="Kybernetes serves as the orchestrator coordinating Hestia [1].",
        citations=[sim_citation],
    )
    assert rep_sim.status == TurnOutcomeStatus.DRAFT
    assert rep_sim.is_simulated is True

