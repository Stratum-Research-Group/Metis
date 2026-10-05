"""
Regression tests for Round 2 review defects.
Covers:
- Defect 1: Workspace source mappings & control plane runnable tools
- Defect 2: Hestia & Prosopon CLI serve options, working directory paths, port propagation
- Defect 3: Prosopon live KybernetesClient upstream selection (--no-mock)
"""

import sys
from pathlib import Path
from unittest.mock import patch
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


def test_defect6_lthcm_adapter_real_index_and_provenance(tmp_path):
    """
    Verify LTHCMAdapter:
    - Fails closed on absent path, ordinary file, empty dir, incompatible index.
    - Preserves is_simulated=True in demo mode.
    - Successfully loads real snapshot with unique ingested fact and generation identity.
    """
    import json
    from kybernetes.adapters.lthcm import LTHCMAdapter

    # 1. Absent path
    absent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        LTHCMAdapter(memory_root=str(absent))

    # 2. Ordinary file
    file_path = tmp_path / "ordinary_file.txt"
    file_path.write_text("not a dir")
    with pytest.raises(NotADirectoryError):
        LTHCMAdapter(memory_root=str(file_path))

    # 3. Empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="empty"):
        LTHCMAdapter(memory_root=str(empty_dir))

    # 4. Incompatible index (no manifest.json or index.json)
    incompat_dir = tmp_path / "incompatible"
    incompat_dir.mkdir()
    (incompat_dir / "random.txt").write_text("hello")
    with pytest.raises(ValueError, match="Incompatible index"):
        LTHCMAdapter(memory_root=str(incompat_dir))

    # 5. Demo mode
    demo_adapter = LTHCMAdapter(memory_root=None)
    assert demo_adapter.is_simulated is True
    demo_cits = demo_adapter.retrieve("kybernetes orchestrates")
    assert len(demo_cits) > 0
    assert all(c.is_simulated is True for c in demo_cits)
    assert any(c.passage_id == "pass_metis_arch_01" for c in demo_cits)

    # 6. Real snapshot with unique ingested fact
    real_dir = tmp_path / "real_index"
    real_dir.mkdir()
    unique_fact_manifest = {
        "manifest_version": "2.0",
        "created_at": 1700000000.0,
        "passages": {
            "pass_unique_sovereign_42": {
                "source_id": "doc_sovereign_spec_001",
                "source_title": "Sovereign Spec Protocol",
                "text": "Sovereign autonomy requires zero reliance on external cloud APIs.",
                "keywords": ["sovereign", "autonomy", "apis", "external", "reliance"],
                "ingested_at": 1700000000.0,
            }
        }
    }
    with open(real_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(unique_fact_manifest, f)
    with open(real_dir / "current_generation.json", "w", encoding="utf-8") as f:
        json.dump({"generation": "gen-42"}, f)

    real_adapter = LTHCMAdapter(memory_root=str(real_dir))
    assert real_adapter.is_simulated is False
    assert real_adapter.generation_id == "gen-42"

    real_cits = real_adapter.retrieve("sovereign autonomy")
    assert len(real_cits) == 1
    cit = real_cits[0]
    assert cit.passage_id == "pass_unique_sovereign_42"
    assert cit.source_id == "doc_sovereign_spec_001"
    assert cit.source_title == "Sovereign Spec Protocol"
    assert cit.is_simulated is False
    assert "Sovereign autonomy requires" in cit.text


@pytest.mark.asyncio
async def test_defect10_phylax_approval_lifecycle_and_security():
    """
    Verify PhylaxAdapter:
    - Expiry enforcement (past expiry rejected at resolution & consumption).
    - Single-use consumption (second wait returns 'deny').
    - Double-resolution rejection (terminal status).
    - Action and target binding verification.
    - Timeout and explicit deny handling.
    """
    import asyncio
    from kybernetes.adapters.phylax import PhylaxAdapter

    phylax = PhylaxAdapter()

    # 1. Valid single execution
    _, req1 = phylax.is_action_permitted("write_file", "/tmp/valid.txt", {"mode": "w"})
    assert req1 is not None
    app1 = req1["approval_id"]
    assert phylax.resolve_approval(app1, "allow_once") is True
    dec1 = await phylax.wait_for_approval(app1, timeout_seconds=1.0)
    assert dec1 == "allow_once"

    # 2. Replay prevention: second wait returns 'deny'
    dec1_replay = await phylax.wait_for_approval(app1, timeout_seconds=1.0)
    assert dec1_replay == "deny"

    # 3. Double resolution: cannot resolve already consumed/terminal approval
    assert phylax.resolve_approval(app1, "allow_once") is False

    # 4. Expiry: past timestamp
    _, req_exp = phylax.is_action_permitted(
        "write_file",
        "/tmp/expired.txt",
        {"data": "secret"},
        custom_expiry_utc="2000-01-01T00:00:00Z",
    )
    app_exp = req_exp["approval_id"]
    # Resolution of expired approval fails
    assert phylax.resolve_approval(app_exp, "allow_once") is False
    # Consumption of expired approval returns 'deny'
    dec_exp = await phylax.wait_for_approval(app_exp, timeout_seconds=0.1)
    assert dec_exp == "deny"

    # 5. Changed target / arguments binding check
    _, req_bind = phylax.is_action_permitted("write_file", "/tmp/fileA.txt", {"count": 1})
    app_bind = req_bind["approval_id"]
    assert phylax.verify_action_execution(app_bind, "write_file", "/tmp/fileA.txt", {"count": 1}) is True
    assert phylax.verify_action_execution(app_bind, "write_file", "/tmp/fileB.txt", {"count": 1}) is False
    assert phylax.verify_action_execution(app_bind, "read_file", "/tmp/fileA.txt", {"count": 1}) is False
    assert phylax.verify_action_execution(app_bind, "write_file", "/tmp/fileA.txt", {"count": 999}) is False

    # 6. Timeout
    _, req_timeout = phylax.is_action_permitted("write_file", "/tmp/timeout.txt", {})
    dec_timeout = await phylax.wait_for_approval(req_timeout["approval_id"], timeout_seconds=0.02)
    assert dec_timeout == "deny"

    # 7. Explicit deny
    _, req_deny = phylax.is_action_permitted("write_file", "/tmp/deny.txt", {})
    app_deny = req_deny["approval_id"]
    assert phylax.resolve_approval(app_deny, "deny") is True
    assert await phylax.wait_for_approval(app_deny, timeout_seconds=1.0) == "deny"


def test_defect8_dianoia_producer_and_paideia_schema_version_compatibility(tmp_path):
    """
    Verify Dianoia producer output conforms to Paideia schema, supports version 1 migration,
    rejects unsupported versions, and records audit schema violations.
    """
    import json
    from pydantic import ValidationError
    from dianoia.log import build_walk_record
    from paideia.contracts.dianoia_schema import DianoiaWalkRecord
    from paideia.data.walk_loader import load_walk_dataset

    # 1. Real Dianoia producer record
    prod_dict = build_walk_record(
        question="What is Metis architecture?",
        seeded_nodes=["node_arch_01"],
        path=[
            {"node": "node_arch_01", "action": "EXPAND", "raw_output": "expand", "reason": "explore", "parse_failed": False},
            {"node": "node_arch_02", "action": "ANSWER", "raw_output": "answer", "reason": "complete", "parse_failed": False},
        ],
        verification={"conflicts_found": False, "rewalked": False, "report": "ok", "conflict_detail": None},
        final_answer="Metis is a sovereign local AI stack.",
        max_steps=10,
        steps_used=2,
        model="llama-3.1-8b",
        fallback_used=False,
    )
    rec1 = DianoiaWalkRecord.model_validate(prod_dict)
    assert rec1.schema_version == "1.0.0"

    # 2. Legacy integer 1 and string '1' schema_version migration
    legacy_int = prod_dict.copy()
    legacy_int["schema_version"] = 1
    legacy_int["walk_id"] = "w-legacy-int"
    rec_int = DianoiaWalkRecord.model_validate(legacy_int)
    assert rec_int.schema_version == "1.0.0"

    legacy_str = prod_dict.copy()
    legacy_str["schema_version"] = "1"
    legacy_str["walk_id"] = "w-legacy-str"
    rec_str = DianoiaWalkRecord.model_validate(legacy_str)
    assert rec_str.schema_version == "1.0.0"

    # 3. Unsupported schema version rejected
    unsupported = prod_dict.copy()
    unsupported["schema_version"] = "99.0.0"
    with pytest.raises(ValidationError):
        DianoiaWalkRecord.model_validate(unsupported)

    # 4. JSONL file through real loader
    jsonl_path = tmp_path / "walks_compat.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(prod_dict) + "\n")
        f.write(json.dumps(legacy_int) + "\n")
        f.write(json.dumps(unsupported) + "\n")  # Should be counted as schema violation

    samples, audit = load_walk_dataset(jsonl_path)
    assert audit.valid_records == 2
    assert audit.schema_violations == 1
    assert len(samples) == 4  # 2 steps each from 2 valid records


def test_defect9_paideia_rejects_unresolved_re_walks_from_gold_data(tmp_path):
    """
    Verify Paideia walk loader rejects re-walks with continuing conflicts from gold data,
    only accepting genuine resolved re-walks or clean trajectories.
    """
    import json
    from paideia.contracts.dianoia_schema import DianoiaWalkRecord, DianoiaVerification, DianoiaStep
    from paideia.data.walk_loader import load_walk_dataset

    # 1. Unresolved re-walk (conflicts_found=True and rewalked=True)
    unresolved_record = DianoiaWalkRecord(
        walk_id="w-unresolved",
        question="Unresolved conflict question",
        model="llama-3.1-8b",
        path=[
            DianoiaStep(node="n1", action="ANSWER", reason="done"),
        ],
        verification=DianoiaVerification(conflicts_found=True, rewalked=True, report="conflict still present"),
        final_answer="Contradictory answer",
    )
    assert unresolved_record.is_defensible_gold_trajectory() is False

    # 2. Resolved re-walk (attempt 1 had conflicts, attempt 2 resolved with conflicts_found=False)
    att1 = DianoiaWalkRecord(
        walk_id="w-resolved",
        question="Resolvable question",
        model="llama-3.1-8b",
        path=[DianoiaStep(node="n1", action="EXPAND", reason="step1")],
        verification=DianoiaVerification(conflicts_found=True, rewalked=False, report="initial conflict"),
        final_answer="Draft 1",
    )
    att2 = DianoiaWalkRecord(
        walk_id="w-resolved",
        question="Resolvable question",
        model="llama-3.1-8b",
        path=[
            DianoiaStep(node="n1", action="EXPAND", reason="step1"),
            DianoiaStep(node="n2", action="ANSWER", reason="resolved answer"),
        ],
        verification=DianoiaVerification(conflicts_found=False, rewalked=True, report="conflict resolved"),
        final_answer="True verified answer",
    )
    assert att1.is_defensible_gold_trajectory() is False
    assert att2.is_defensible_gold_trajectory() is True

    # 3. Clean attempt
    clean = DianoiaWalkRecord(
        walk_id="w-clean",
        question="Clean question",
        model="llama-3.1-8b",
        path=[DianoiaStep(node="n1", action="ANSWER", reason="clean")],
        verification=DianoiaVerification(conflicts_found=False, rewalked=False, report="ok"),
        final_answer="Clean answer",
    )
    assert clean.is_defensible_gold_trajectory() is True

    # Write to dataset file
    dataset_file = tmp_path / "walks_conflicts.jsonl"
    with open(dataset_file, "w", encoding="utf-8") as f:
        f.write(unresolved_record.model_dump_json() + "\n")
        f.write(att1.model_dump_json() + "\n")
        f.write(att2.model_dump_json() + "\n")
        f.write(clean.model_dump_json() + "\n")

    samples, audit = load_walk_dataset(dataset_file)
    assert audit.unresolved_conflicts_excluded == 1  # w-unresolved was excluded
    assert audit.walk_attempts_joined == 1  # w-resolved joined 2 attempts into 1 final
    # Samples should only contain w-resolved (2 steps) and w-clean (1 step) = 3 samples
    sample_walk_ids = {s.walk_id for s in samples}
    assert "w-unresolved" not in sample_walk_ids
    assert sample_walk_ids == {"w-resolved", "w-clean"}
    assert len(samples) == 3


def test_defect11_paideia_runner_simulated_status_and_export_acceptance_gate(tmp_path):
    """
    Verify Paideia runner marks marker-only training as SIMULATED, reports 0 epochs,
    and blocks export as a production trained adapter unless allow_simulated=True.
    """
    from paideia.training.config import TrainingConfig
    from paideia.data.splitter import DatasetSplits
    from paideia.training.runner import QLoRATrainingRunner
    from paideia.export.registry import export_hestia_adapter_registry

    config = TrainingConfig(
        output_dir=str(tmp_path / "runs"),
        epochs=3,
        batch_size=1,
    )
    splits = DatasetSplits(train=[1, 2], dev=[3], test=[4])
    runner = QLoRATrainingRunner(config=config, splits=splits)
    with patch.object(runner, "preflight_check", return_value={"hardware_line": "cpu"}):
        manifest = runner.run()

    assert manifest.status == "SIMULATED"
    assert manifest.is_simulated is True
    assert manifest.metrics["epochs_completed"] == 0
    assert manifest.metrics["simulated"] is True

    checkpoint_dir = runner.output_dir / "checkpoints"
    # Copy manifest into checkpoints or parent dir
    (checkpoint_dir / "manifest.json").write_text(manifest.model_dump_json())

    # Gate must block production export of simulated run
    with pytest.raises(ValueError, match="Acceptance gate failure"):
        export_hestia_adapter_registry(
            adapter_id="sim-adapter",
            adapter_dir=checkpoint_dir,
            base_model_name="meta-llama/Llama-3.1-8B-Instruct",
            task="question_router",
            training_data_fingerprint="dummy_fp",
            evaluation_run_id="eval-1",
            resource_footprint={"vram": 100},
            allow_simulated=False,
        )

    # Allowed when explicitly declared simulated
    entry = export_hestia_adapter_registry(
        adapter_id="sim-adapter",
        adapter_dir=checkpoint_dir,
        base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        task="question_router",
        training_data_fingerprint="dummy_fp",
        evaluation_run_id="eval-1",
        resource_footprint={"vram": 100},
        allow_simulated=True,
    )
    assert entry.adapter_id == "sim-adapter"


def test_defect12_export_digest_stability_exclusions_and_artifact_identity(tmp_path):
    """
    Verify adapter export hash scope:
    - Excludes hestia_registry_entry.json from hash calculation.
    - Preserves stable digest across repeated exports.
    - Senses file mutation and layout changes.
    - Generates distinct digests for distinct model / task identities.
    - Fails closed on unreadable files.
    """
    import os
    from paideia.export.registry import export_hestia_adapter_registry, EXCLUDED_REGISTRY_FILES
    from paideia.training.runner import compute_dir_sha256

    adapter_dir = tmp_path / "real_weights"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"initial_weight_matrix_content")
    (adapter_dir / "adapter_config.json").write_text('{"rank": 8, "alpha": 16}')

    # 1. First export
    entry1 = export_hestia_adapter_registry(
        adapter_id="lora-v1",
        adapter_dir=adapter_dir,
        base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        task="retrieval_router",
        training_data_fingerprint="data_fingerprint_001",
        evaluation_run_id="eval-001",
        resource_footprint={"vram": 1000},
    )

    # 2. Check digest matches directory checksum with exclusions
    computed_digest = compute_dir_sha256(adapter_dir, exclude_files=EXCLUDED_REGISTRY_FILES)
    assert entry1.adapter_digest == computed_digest

    # 3. Repeated export must be stable and deterministic (must not hash the previous registry file)
    entry2 = export_hestia_adapter_registry(
        adapter_id="lora-v1",
        adapter_dir=adapter_dir,
        base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        task="retrieval_router",
        training_data_fingerprint="data_fingerprint_001",
        evaluation_run_id="eval-001",
        resource_footprint={"vram": 1000},
    )
    assert entry2.adapter_digest == entry1.adapter_digest

    # 4. Mutation detection
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"mutated_weight_matrix_content")
    entry_mutated = export_hestia_adapter_registry(
        adapter_id="lora-v1",
        adapter_dir=adapter_dir,
        base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        task="retrieval_router",
        training_data_fingerprint="data_fingerprint_001",
        evaluation_run_id="eval-001",
        resource_footprint={"vram": 1000},
    )
    assert entry_mutated.adapter_digest != entry1.adapter_digest

    # 5. Distinct model and task identities change digests
    entry_diff_model = export_hestia_adapter_registry(
        adapter_id="lora-v1",
        adapter_dir=adapter_dir,
        base_model_name="qwen/Qwen2.5-7B",
        task="different_task",
        training_data_fingerprint="data_fingerprint_001",
        evaluation_run_id="eval-001",
        resource_footprint={"vram": 1000},
    )
    assert entry_diff_model.base_model_digest != entry1.base_model_digest
    assert entry_diff_model.template_digest != entry1.template_digest

    # 6. Unreadable file failure
    unreadable_file = adapter_dir / "corrupted.bin"
    unreadable_file.write_bytes(b"data")
    try:
        os.chmod(unreadable_file, 0o000)
        # On root/privileged hosts chmod may not deny read, so only assert if PermissionError/OSError occurs
        try:
            with open(unreadable_file, "rb") as test_f:
                test_f.read(1)
            can_read = True
        except PermissionError:
            can_read = False

        if not can_read:
            with pytest.raises(OSError):
                compute_dir_sha256(adapter_dir, exclude_files=EXCLUDED_REGISTRY_FILES)
    finally:
        os.chmod(unreadable_file, 0o644)
        unreadable_file.unlink()


def test_defect7_stoma_monorepo_vendored_provenance():
    """
    Defect 7: Verify monorepo vendored dependencies report honest provenance:
    monorepo git revision, content digests matching release manifest, no fabricated
    submodule checkout commits, and origin verification.
    """
    from stoma.config import verify_runtime_dependencies
    res = verify_runtime_dependencies()
    assert res["layout"] == "monorepo_vendored"
    assert res["lthcm_origin_ok"] is True
    assert res["dianoia_origin_ok"] is True
    assert res["monorepo_revision"] is not None
    assert len(res["monorepo_revision"]) == 40
    assert res["lthcm_content_digest"] is not None
    assert len(res["lthcm_content_digest"]) == 64
    assert res["dianoia_content_digest"] is not None
    assert len(res["dianoia_content_digest"]) == 64
    # Real monorepo check does NOT label constants as measured submodule checkout SHAs
    assert res["lthcm_checkout_sha"] is None
    assert res["dianoia_checkout_sha"] is None
    assert res["lthcm_gitlink_sha"] is None
    assert res["dianoia_gitlink_sha"] is None


def test_defect7_stoma_monorepo_altered_source_rejected(tmp_path):
    """
    Defect 7: Alter a sibling source in a disposable copy and verify rejection
    with ordinary real functions (no mocks).
    """
    import shutil
    import subprocess
    import types
    from unittest.mock import patch
    from stoma.config import verify_runtime_dependencies, DependencyOriginError

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    stoma_dir = repo_root / "stoma"
    lthcm_dir = repo_root / "LTHCM"
    dianoia_dir = repo_root / "dianoia"

    shutil.copytree("stoma", stoma_dir, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".venv"))
    shutil.copytree("LTHCM", lthcm_dir, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".venv", "runs"))
    shutil.copytree("dianoia", dianoia_dir, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".venv"))

    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_root, check=True, capture_output=True)

    fake_lthcm = types.SimpleNamespace(__file__=str(lthcm_dir / "lthcm" / "__init__.py"))
    fake_dianoia = types.SimpleNamespace(__file__=str(dianoia_dir / "dianoia" / "__init__.py"))

    with patch.dict(sys.modules, {"lthcm": fake_lthcm, "dianoia": fake_dianoia}):
        # 1. Clean check passes
        res = verify_runtime_dependencies(stoma_root=stoma_dir)
        assert res["layout"] == "monorepo_vendored"

        # 2. Alter LTHCM source file
        lthcm_init = lthcm_dir / "lthcm" / "__init__.py"
        original_lthcm = lthcm_init.read_bytes()
        lthcm_init.write_bytes(original_lthcm + b"\n# TAMPERED CODE\n")
        with pytest.raises(DependencyOriginError, match="Content digest mismatch for vendored LTHCM"):
            verify_runtime_dependencies(stoma_root=stoma_dir)
        lthcm_init.write_bytes(original_lthcm)

        # 3. Alter dianoia source file
        dianoia_init = dianoia_dir / "dianoia" / "__init__.py"
        original_dianoia = dianoia_init.read_bytes()
        dianoia_init.write_bytes(original_dianoia + b"\n# TAMPERED CODE\n")
        with pytest.raises(DependencyOriginError, match="Content digest mismatch for vendored dianoia"):
            verify_runtime_dependencies(stoma_root=stoma_dir)
        dianoia_init.write_bytes(original_dianoia)


def test_defect7_stoma_monorepo_missing_provenance_and_replaced_dir(tmp_path):
    """
    Defect 7: Replace component directory or delete provenance manifest,
    and verify fail-closed behavior with ordinary real functions.
    """
    import shutil
    import subprocess
    import types
    from unittest.mock import patch
    from stoma.config import verify_runtime_dependencies, DependencyOriginError

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    stoma_dir = repo_root / "stoma"
    lthcm_dir = repo_root / "LTHCM"
    dianoia_dir = repo_root / "dianoia"

    shutil.copytree("stoma", stoma_dir, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".venv"))
    shutil.copytree("LTHCM", lthcm_dir, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".venv", "runs"))
    shutil.copytree("dianoia", dianoia_dir, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".venv"))

    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_root, check=True, capture_output=True)

    fake_lthcm = types.SimpleNamespace(__file__=str(lthcm_dir / "lthcm" / "__init__.py"))
    fake_dianoia = types.SimpleNamespace(__file__=str(dianoia_dir / "dianoia" / "__init__.py"))

    with patch.dict(sys.modules, {"lthcm": fake_lthcm, "dianoia": fake_dianoia}):
        # 1. Missing provenance manifest fails closed
        manifest_file = stoma_dir / "vendored_provenance.json"
        manifest_pkg = stoma_dir / "src" / "stoma" / "vendored_provenance.json"
        manifest_backup = manifest_file.read_bytes()
        manifest_file.unlink()
        if manifest_pkg.is_file():
            manifest_pkg.unlink()

        with pytest.raises(DependencyOriginError, match="Missing vendored provenance manifest"):
            verify_runtime_dependencies(stoma_root=stoma_dir)

        manifest_file.write_bytes(manifest_backup)

        # 2. Replaced / empty LTHCM directory fails closed
        shutil.rmtree(lthcm_dir)
        lthcm_dir.mkdir()
        with pytest.raises(DependencyOriginError, match="Required file .* missing in vendored LTHCM"):
            verify_runtime_dependencies(stoma_root=stoma_dir)


def test_defect7_stoma_submodule_layout_validation(tmp_path):
    """
    Defect 7: Retain proper gitlink and checkout commit validation for actual submodule deployments.
    Fails closed on altered or unreadable gitlink/checkout commit.
    """
    import types
    from unittest.mock import patch
    from stoma.config import verify_runtime_dependencies, DependencyOriginError, LOCKED_LTHCM_COMMIT, LOCKED_DIANOIA_COMMIT

    sub_root = tmp_path / "sub_stoma"
    (sub_root / "deps" / "LTHCM").mkdir(parents=True)
    (sub_root / "deps" / "dianoia").mkdir(parents=True)

    fake_lthcm = types.SimpleNamespace(__file__=str(sub_root / "deps" / "LTHCM" / "lthcm" / "__init__.py"))
    fake_dianoia = types.SimpleNamespace(__file__=str(sub_root / "deps" / "dianoia" / "dianoia" / "__init__.py"))

    with patch.dict(sys.modules, {"lthcm": fake_lthcm, "dianoia": fake_dianoia}):
        with patch("stoma.config._read_top_level_gitlink", side_effect=[LOCKED_LTHCM_COMMIT, LOCKED_DIANOIA_COMMIT]), \
             patch("stoma.config._read_submodule_commit", side_effect=[LOCKED_LTHCM_COMMIT, LOCKED_DIANOIA_COMMIT]):
            res = verify_runtime_dependencies(stoma_root=sub_root)
            assert res["layout"] == "submodule"
            assert res["lthcm_gitlink_sha"] == LOCKED_LTHCM_COMMIT
            assert res["lthcm_checkout_sha"] == LOCKED_LTHCM_COMMIT
            assert res["dianoia_gitlink_sha"] == LOCKED_DIANOIA_COMMIT
            assert res["dianoia_checkout_sha"] == LOCKED_DIANOIA_COMMIT

        # Altered gitlink fails closed
        with patch("stoma.config._read_top_level_gitlink", return_value="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"):
            with pytest.raises(DependencyOriginError, match="Submodule commit violation: LTHCM gitlink"):
                verify_runtime_dependencies(stoma_root=sub_root)

        # Missing gitlink fails closed
        with patch("stoma.config._read_top_level_gitlink", return_value=None):
            with pytest.raises(DependencyOriginError, match="Cannot verify top-level gitlink SHA"):
                verify_runtime_dependencies(stoma_root=sub_root)

        # Altered checkout fails closed
        with patch("stoma.config._read_top_level_gitlink", side_effect=[LOCKED_LTHCM_COMMIT, LOCKED_DIANOIA_COMMIT]), \
             patch("stoma.config._read_submodule_commit", return_value="badc0ffeebadc0ffeebadc0ffeebadc0ffee1234"):
            with pytest.raises(DependencyOriginError, match="Submodule commit violation: LTHCM is at"):
                verify_runtime_dependencies(stoma_root=sub_root)


def test_defect7_stoma_origin_check_violation():
    """
    Defect 7: Ensure that if lthcm or dianoia is imported from outside expected root,
    DependencyOriginError is raised.
    """
    import types
    from unittest.mock import patch
    from stoma.config import verify_runtime_dependencies, DependencyOriginError

    mock_foreign_lthcm = types.SimpleNamespace(__file__="/usr/local/lib/python3.12/site-packages/lthcm/__init__.py")
    with patch.dict(sys.modules, {"lthcm": mock_foreign_lthcm}):
        with pytest.raises(DependencyOriginError, match="Runtime import violation: 'lthcm' imported from"):
            verify_runtime_dependencies()







