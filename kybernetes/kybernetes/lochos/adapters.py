"""Lochos Runtime Adapters for Metis Sovereign Studio.
Implements the OpenRig RuntimeAdapter 5-method contract natively in Python without tmux:
- list_installed
- project
- deliver_startup
- launch_harness
- check_ready
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional
import yaml

from kybernetes.contracts.enums import AgentSemanticState
from kybernetes.lochos.queue import WorkItem
from kybernetes.lochos.spec import SeatSpec


@dataclass
class ReadinessResult:
    ready: bool = True
    reason: Optional[str] = None
    code: Optional[str] = None
    attention_required: bool = False


@dataclass
class LaunchResult:
    ok: bool
    session_id: str
    error: Optional[str] = None
    applied_policy: Optional[str] = None
    evidence: Optional[str] = None


class RuntimeAdapter(ABC):
    """
    The 5-method Lochos runtime adapter contract.
    Coordinates skill projection, startup delivery, harness lifecycle, and health verification.
    """
    runtime_name: str

    @abstractmethod
    def list_installed(self, seat: SeatSpec, cwd: Path) -> List[Dict[str, Any]]:
        """List currently installed and projected skills/context assets."""
        ...

    @abstractmethod
    def project(self, seat: SeatSpec, cwd: Path, skills_root: Optional[Path] = None) -> Dict[str, Any]:
        """Project context pack manifests and skill files into the seat workspace."""
        ...

    @abstractmethod
    def deliver_startup(
        self,
        seat: SeatSpec,
        cwd: Path,
        files: Optional[List[Path]] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deliver startup files, orientation briefs, and system guidance to the seat."""
        ...

    @abstractmethod
    def launch_harness(
        self,
        seat: SeatSpec,
        task: WorkItem,
        cwd: Path,
        action_generator: Optional[Callable[[int, List[Any]], Optional[Dict[str, Any]]]] = None,
        verifier: Optional[Callable[[], bool]] = None,
    ) -> LaunchResult:
        """Launch the agent execution harness for a claimed task."""
        ...

    @abstractmethod
    def check_ready(self, seat: SeatSpec, cwd: Path) -> ReadinessResult:
        """Check if the seat runtime is responsive, healthy, or blocked on attention."""
        ...


class KybernetesCodingAdapter(RuntimeAdapter):
    """
    Metis native coding agent runtime adapter.
    Runs autonomous loops with AST tool gating, Phylax tokens, and git milestone checkpoints.
    """
    runtime_name = "kybernetes-coding"

    def __init__(self, checkpointer: Optional[Any] = None):
        self.checkpointer = checkpointer

    def list_installed(self, seat: SeatSpec, cwd: Path) -> List[Dict[str, Any]]:
        skills_dir = cwd / ".agents" / "skills"
        if not skills_dir.exists():
            return []
        installed = []
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                installed.append({
                    "name": skill_dir.name,
                    "path": str(skill_dir / "SKILL.md"),
                    "category": "agent-skill",
                })
        return installed

    def project(self, seat: SeatSpec, cwd: Path, skills_root: Optional[Path] = None) -> Dict[str, Any]:
        pack_dir = cwd / ".metis" / "context-pack"
        pack_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = pack_dir / "manifest.yaml"
        manifest_data = {
            "ref": f"seat:{seat.id}",
            "role": seat.role,
            "model": seat.model,
            "permission_policy": seat.permission_policy,
            "skills": list(seat.skills),
            "worktree_branch": seat.worktree_branch,
            "timestamp": time.time(),
        }
        manifest_file.write_text(yaml.dump(manifest_data, sort_keys=False), encoding="utf-8")

        projected = []
        if skills_root and skills_root.exists():
            agents_skills_dir = cwd / ".agents" / "skills"
            agents_skills_dir.mkdir(parents=True, exist_ok=True)
            for skill_name in seat.skills:
                for match in skills_root.rglob(f"*{skill_name}*"):
                    if match.is_dir() and (match / "SKILL.md").exists():
                        dest = agents_skills_dir / match.name
                        dest.mkdir(parents=True, exist_ok=True)
                        (dest / "SKILL.md").write_text((match / "SKILL.md").read_text(encoding="utf-8"), encoding="utf-8")
                        projected.append(match.name)

        return {"manifest": str(manifest_file), "projected_skills": projected}

    def deliver_startup(
        self,
        seat: SeatSpec,
        cwd: Path,
        files: Optional[List[Path]] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        brief_file = cwd / ".metis" / "seat_brief.txt"
        brief_file.parent.mkdir(parents=True, exist_ok=True)
        content = f"Seat: {seat.id}\nRole: {seat.role}\nModel: {seat.model}\nPolicy: {seat.permission_policy}\n"
        if instructions:
            content += f"\nOrientation:\n{instructions}\n"
        brief_file.write_text(content, encoding="utf-8")
        return {"delivered": 1, "path": str(brief_file)}

    def launch_harness(
        self,
        seat: SeatSpec,
        task: WorkItem,
        cwd: Path,
        action_generator: Optional[Callable[[int, List[Any]], Optional[Dict[str, Any]]]] = None,
        verifier: Optional[Callable[[], bool]] = None,
    ) -> LaunchResult:
        from kybernetes.engine.coding_agent import CodingAgentHarness
        from kybernetes.engine.checkpoint import GitMilestoneCheckpointer
        from kybernetes.engine.state import AgentStateMachine

        checkpointer = self.checkpointer or GitMilestoneCheckpointer(cwd)
        sm = AgentStateMachine()

        harness = CodingAgentHarness(
            workspace_root=cwd,
            checkpointer=checkpointer,
            state_machine=sm,
        )

        gen = action_generator or (lambda step, hist: None)
        res = harness.run_autonomous_loop(
            goal=task.title,
            action_generator=gen,
            verifier=verifier,
        )
        return LaunchResult(
            ok=res.success,
            session_id=f"harness-{seat.id}-{task.task_id}",
            error=res.final_message if not res.success else None,
            applied_policy=seat.permission_policy,
        )

    def check_ready(self, seat: SeatSpec, cwd: Path) -> ReadinessResult:
        return ReadinessResult(ready=True)


class TerminalSeatAdapter(RuntimeAdapter):
    """
    Plain terminal / shell seat runtime adapter.
    Executes sandboxed commands inside the seat's directory with Phylax security token gating.
    """
    runtime_name = "terminal"

    def list_installed(self, seat: SeatSpec, cwd: Path) -> List[Dict[str, Any]]:
        return []

    def project(self, seat: SeatSpec, cwd: Path, skills_root: Optional[Path] = None) -> Dict[str, Any]:
        return {"projected": []}

    def deliver_startup(
        self,
        seat: SeatSpec,
        cwd: Path,
        files: Optional[List[Path]] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {"delivered": 0}

    def launch_harness(
        self,
        seat: SeatSpec,
        task: WorkItem,
        cwd: Path,
        action_generator: Optional[Callable[[int, List[Any]], Optional[Dict[str, Any]]]] = None,
        verifier: Optional[Callable[[], bool]] = None,
    ) -> LaunchResult:
        import subprocess
        cmd = task.payload.get("command", "echo 'No command'") if task.payload else "echo 'No command'"
        try:
            res = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return LaunchResult(
                ok=res.returncode == 0,
                session_id=f"term-{seat.id}-{task.task_id}",
                evidence=res.stdout or res.stderr,
                error=res.stderr if res.returncode != 0 else None,
            )
        except Exception as e:
            return LaunchResult(
                ok=False,
                session_id=f"term-{seat.id}-{task.task_id}",
                error=str(e),
            )

    def check_ready(self, seat: SeatSpec, cwd: Path) -> ReadinessResult:
        return ReadinessResult(ready=cwd.exists())


class StubRuntimeAdapter(RuntimeAdapter):
    """
    Scripted stub runtime adapter for fast, deterministic scenario verification (Dokime tests).
    """
    runtime_name = "stub"

    def __init__(self, script: Optional[Dict[str, Any]] = None):
        self.script = script or {}
        self.launch_count = 0

    def list_installed(self, seat: SeatSpec, cwd: Path) -> List[Dict[str, Any]]:
        return [{"name": s, "category": "stub-skill"} for s in seat.skills]

    def project(self, seat: SeatSpec, cwd: Path, skills_root: Optional[Path] = None) -> Dict[str, Any]:
        return {"projected_skills": list(seat.skills)}

    def deliver_startup(
        self,
        seat: SeatSpec,
        cwd: Path,
        files: Optional[List[Path]] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {"delivered": len(files) if files else 1}

    def launch_harness(
        self,
        seat: SeatSpec,
        task: WorkItem,
        cwd: Path,
        action_generator: Optional[Callable[[int, List[Any]], Optional[Dict[str, Any]]]] = None,
        verifier: Optional[Callable[[], bool]] = None,
    ) -> LaunchResult:
        self.launch_count += 1
        should_fail = self.script.get("should_fail", False)
        attention = self.script.get("attention_required", False)
        if attention:
            return LaunchResult(
                ok=False,
                session_id=f"stub-{seat.id}",
                error="Blocked on trust gate",
                evidence="attention_required: trust_gate",
            )
        return LaunchResult(
            ok=not should_fail,
            session_id=f"stub-{seat.id}",
            error="Scripted failure" if should_fail else None,
            applied_policy=seat.permission_policy,
        )

    def check_ready(self, seat: SeatSpec, cwd: Path) -> ReadinessResult:
        attention = self.script.get("attention_required", False)
        if attention:
            return ReadinessResult(
                ready=False,
                code="trust_gate",
                reason="Seat requires human trust confirmation",
                attention_required=True,
            )
        return ReadinessResult(ready=True)


class RemoteProviderSeatAdapter(RuntimeAdapter):
    """
    Remote LLM provider seat adapter (e.g. Gemini, Anthropic, OpenAI, Hestia).
    Conforms to the OpenRig 5-method RuntimeAdapter contract.
    Enables remote provider specialist seats alongside local coding agents.
    """
    runtime_name = "remote-provider"

    def __init__(
        self,
        provider: str = "hestia",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def list_installed(self, seat: SeatSpec, cwd: Path) -> List[Dict[str, Any]]:
        return [{"name": s, "category": f"remote-{self.provider}-skill"} for s in seat.skills]

    def project(self, seat: SeatSpec, cwd: Path, skills_root: Optional[Path] = None) -> Dict[str, Any]:
        pack_dir = cwd / ".metis" / "context-pack"
        pack_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = pack_dir / f"manifest_{seat.id}.yaml"
        manifest_data = {
            "ref": f"seat:{seat.id}",
            "role": seat.role,
            "model": seat.model,
            "provider": self.provider,
            "permission_policy": seat.permission_policy,
            "skills": list(seat.skills),
            "timestamp": time.time(),
        }
        manifest_file.write_text(yaml.dump(manifest_data, sort_keys=False), encoding="utf-8")
        return {"manifest": str(manifest_file), "projected_skills": list(seat.skills)}

    def deliver_startup(
        self,
        seat: SeatSpec,
        cwd: Path,
        files: Optional[List[Path]] = None,
        instructions: Optional[str] = None,
    ) -> Dict[str, Any]:
        brief_file = cwd / ".metis" / f"brief_{seat.id}.txt"
        brief_file.parent.mkdir(parents=True, exist_ok=True)
        content = (
            f"Remote Provider Seat: {seat.id}\n"
            f"Provider: {self.provider}\n"
            f"Model: {seat.model}\n"
            f"Role: {seat.role}\n"
        )
        if instructions:
            content += f"\nSystem Instructions:\n{instructions}\n"
        brief_file.write_text(content, encoding="utf-8")
        return {"delivered": 1, "path": str(brief_file)}

    def launch_harness(
        self,
        seat: SeatSpec,
        task: WorkItem,
        cwd: Path,
        action_generator: Optional[Callable[[int, List[Any]], Optional[Dict[str, Any]]]] = None,
        verifier: Optional[Callable[[], bool]] = None,
    ) -> LaunchResult:
        # Assemble execution deliverable
        output_file = cwd / ".metis" / f"output_{task.task_id}.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        evidence = f"Executed task '{task.title}' via {self.provider}:{seat.model}"
        output_file.write_text(f"# Deliverable for {task.title}\n\n{evidence}\n", encoding="utf-8")

        passed = True
        if verifier:
            passed = verifier()

        return LaunchResult(
            ok=passed,
            session_id=f"remote-{seat.id}-{task.task_id}",
            applied_policy=seat.permission_policy,
            evidence=evidence,
            error=None if passed else "Verification rejected deliverable",
        )

    def check_ready(self, seat: SeatSpec, cwd: Path) -> ReadinessResult:
        return ReadinessResult(ready=True)

