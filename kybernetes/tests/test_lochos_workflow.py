import subprocess
import tempfile
from pathlib import Path

from kybernetes.contracts.enums import AgentSemanticState
from kybernetes.lochos.queue import LochosQueue
from kybernetes.lochos.spec import create_starter_rig
from kybernetes.lochos.squad import LochosSquadManager
from kybernetes.lochos.workflow import LochosWorkflowOrchestrator, PipelineStage


def _init_test_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=str(root), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Metis Pipeline Test"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "test@metis.local"], cwd=str(root), check=True)
    (root / ".gitignore").write_text(".metis/\n", encoding="utf-8")
    (root / "base.txt").write_text("initial base content\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "base.txt"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-m", "Initial baseline commit"], cwd=str(root), check=True)


def test_orchestrator_successful_flow():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _init_test_git_repo(root)

        rig = create_starter_rig("pipeline-test-squad")
        manager = LochosSquadManager(
            rig_spec=rig,
            workspace_root=root,
        )
        orchestrator = LochosWorkflowOrchestrator(squad_manager=manager)

        coder_executed = False

        def mock_coder_runner(seat, task):
            nonlocal coder_executed
            coder_executed = True
            (root / "feature.py").write_text("def hello(): return 'world'\n", encoding="utf-8")
            return {"status": "ok", "success": True}

        verifier_executed = False

        def mock_verifier():
            nonlocal verifier_executed
            verifier_executed = True
            return (root / "feature.py").exists()

        result = orchestrator.run_pipeline(
            mission="release-0.6.0",
            slice_id="OPR.01",
            title="Implement hello world feature",
            description="Add feature.py module",
            coder_runner=mock_coder_runner,
            verifier=mock_verifier,
        )

        assert coder_executed is True
        assert verifier_executed is True
        assert result.success is True
        assert result.stage == PipelineStage.COMPLETED
        assert result.rolled_back is False
        assert result.verification_passed is True
        assert len(result.milestones) == 2
        assert "reviewer" in result.baton_owner

        # Verify task state in SQLite queue
        task = manager.queue.get_task(result.task_id)
        assert task is not None
        assert task.status == "completed"
        assert task.result["verification"] == "PASSED"
        assert result.handoff_brief is not None
        assert result.handoff_brief["task_id"] == result.task_id


def test_orchestrator_reviewer_rejection_and_rollback():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _init_test_git_repo(root)

        rig = create_starter_rig("rollback-test-squad")
        manager = LochosSquadManager(
            rig_spec=rig,
            workspace_root=root,
        )
        orchestrator = LochosWorkflowOrchestrator(squad_manager=manager)

        def mock_coder_runner(seat, task):
            (root / "base.txt").write_text("mutated invalid content\n", encoding="utf-8")
            (root / "unwanted.py").write_text("unwanted file\n", encoding="utf-8")
            return True

        # Reviewer rejects the modification
        def rejecting_verifier():
            return False

        result = orchestrator.run_pipeline(
            mission="release-0.6.0",
            slice_id="OPR.02",
            title="Defective Mutation Task",
            coder_runner=mock_coder_runner,
            verifier=rejecting_verifier,
            auto_rollback_on_failure=True,
        )

        assert result.success is False
        assert result.rolled_back is True
        assert result.stage == PipelineStage.ROLLED_BACK
        assert result.verification_passed is False

        # Verify transactional rollback restored working tree exactly
        assert (root / "base.txt").read_text(encoding="utf-8") == "initial base content\n"
        assert not (root / "unwanted.py").exists()

        # Task in queue must be failed
        task = manager.queue.get_task(result.task_id)
        assert task is not None
        assert task.status == "failed"


def test_orchestrator_rejection_with_attention_escalation():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _init_test_git_repo(root)

        rig = create_starter_rig("attention-test-squad")
        manager = LochosSquadManager(
            rig_spec=rig,
            workspace_root=root,
        )
        orchestrator = LochosWorkflowOrchestrator(squad_manager=manager)

        result = orchestrator.run_pipeline(
            mission="security-fix",
            slice_id="SEC.009",
            title="Ambiguous Security Patch",
            verifier=lambda: False,
            auto_rollback_on_failure=True,
            escalate_attention_on_failure=True,
        )

        assert result.success is False
        assert result.stage == PipelineStage.ATTENTION_REQUIRED
        assert result.rolled_back is True

        # Check that open attention item was registered in Lochos queue
        attention_summary = manager.get_attention_summary()
        assert len(attention_summary) == 1
        assert attention_summary[0]["task_id"] == result.task_id
        assert attention_summary[0]["reason"] == "reviewer_verification_failed"


def test_orchestrator_coder_failure_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _init_test_git_repo(root)

        rig = create_starter_rig("coder-fail-test-squad")
        manager = LochosSquadManager(
            rig_spec=rig,
            workspace_root=root,
        )
        orchestrator = LochosWorkflowOrchestrator(squad_manager=manager)

        def failing_coder(seat, task):
            (root / "untracked.tmp").write_text("crash\n", encoding="utf-8")
            raise RuntimeError("Out of token budget")

        result = orchestrator.run_pipeline(
            mission="refactor",
            slice_id="REF.001",
            title="Crash during implementation",
            coder_runner=failing_coder,
            auto_rollback_on_failure=True,
        )

        assert result.success is False
        assert result.stage == PipelineStage.ROLLED_BACK
        assert result.rolled_back is True
        assert "Out of token budget" in (result.error or "")
        assert not (root / "untracked.tmp").exists()


def test_orchestrator_with_test_command_verifier():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _init_test_git_repo(root)

        rig = create_starter_rig("test-cmd-squad")
        manager = LochosSquadManager(
            rig_spec=rig,
            workspace_root=root,
        )
        orchestrator = LochosWorkflowOrchestrator(squad_manager=manager)

        # 1. Successful test command
        res_ok = orchestrator.run_pipeline(
            mission="ci-check",
            slice_id="CI.01",
            title="Passing command check",
            test_command="python3 -c 'assert 1 + 1 == 2'",
        )
        assert res_ok.success is True
        assert res_ok.verification_passed is True

        # 2. Failing test command
        res_fail = orchestrator.run_pipeline(
            mission="ci-check",
            slice_id="CI.02",
            title="Failing command check",
            test_command="python3 -c 'assert 1 + 1 == 999'",
        )
        assert res_fail.success is False
        assert res_fail.verification_passed is False
        assert "AssertionError" in (res_fail.error or "")
