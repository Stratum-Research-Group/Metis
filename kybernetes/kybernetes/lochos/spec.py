"""Lochos Declarative Team and RigSpec Schema for Metis Sovereign Studio.
Defines squads, pods, seats, roles, and edge channels without tmux dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


VALID_EDGE_KINDS = {"delegates_to", "can_observe", "collaborates_with", "escalates_to"}
VALID_EFFORTS = {"low", "medium", "high"}
VALID_PERMISSION_POLICIES = {"strict", "audit_only", "full_bypass"}


@dataclass
class SeatSpec:
    """Represents an addressed specialist agent in a squad pod (e.g. coder@metis-squad)."""
    id: str
    label: str
    role: str
    model: str = "qwen2.5-coder-32b"
    temperature: float = 0.2
    effort: str = "medium"
    permission_policy: str = "strict"
    cwd: Optional[str] = None
    worktree_branch: Optional[str] = None
    system_prompt: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    def full_address(self, rig_name: str) -> str:
        return f"{self.id}@{rig_name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "role": self.role,
            "model": self.model,
            "temperature": self.temperature,
            "effort": self.effort,
            "permission_policy": self.permission_policy,
            "cwd": self.cwd,
            "worktree_branch": self.worktree_branch,
            "system_prompt": self.system_prompt,
            "skills": list(self.skills),
            "env": dict(self.env),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeatSpec":
        if "id" not in data or not data["id"]:
            raise ValueError("SeatSpec must define a non-empty 'id'")
        return cls(
            id=str(data["id"]).strip(),
            label=str(data.get("label", data["id"])).strip(),
            role=str(data.get("role", "Specialist Agent")).strip(),
            model=str(data.get("model", "qwen2.5-coder-32b")).strip(),
            temperature=float(data.get("temperature", 0.2)),
            effort=str(data.get("effort", "medium")).lower(),
            permission_policy=str(data.get("permission_policy", "strict")).lower(),
            cwd=data.get("cwd"),
            worktree_branch=data.get("worktree_branch"),
            system_prompt=data.get("system_prompt"),
            skills=list(data.get("skills", [])),
            env=dict(data.get("env", {})),
        )


@dataclass
class PodSpec:
    """A logical cluster of agent seats sharing a mission boundary."""
    id: str
    label: str
    seats: List[SeatSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "seats": [s.to_dict() for s in self.seats],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PodSpec":
        if "id" not in data or not data["id"]:
            raise ValueError("PodSpec must define a non-empty 'id'")
        seats = [SeatSpec.from_dict(s) for s in data.get("seats", [])]
        return cls(
            id=str(data["id"]).strip(),
            label=str(data.get("label", data["id"])).strip(),
            seats=seats,
        )


@dataclass
class EdgeSpec:
    """Defines communication, delegation, or observation channels between seats."""
    from_seat: str
    to_seat: str
    kind: str = "delegates_to"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_seat": self.from_seat,
            "to_seat": self.to_seat,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EdgeSpec":
        kind = str(data.get("kind", "delegates_to")).strip()
        if kind not in VALID_EDGE_KINDS:
            raise ValueError(f"Invalid edge kind '{kind}'. Valid kinds: {VALID_EDGE_KINDS}")
        return cls(
            from_seat=str(data["from_seat"]).strip(),
            to_seat=str(data["to_seat"]).strip(),
            kind=kind,
        )


@dataclass
class RigSpec:
    """
    Top-level declarative specification for a Lochos multi-agent squad.
    Equivalent to OpenRig RigSpec v0.2, tailored natively for MSS.
    """
    name: str
    summary: str = ""
    version: str = "0.2"
    pods: List[PodSpec] = field(default_factory=list)
    edges: List[EdgeSpec] = field(default_factory=list)

    def list_seats(self) -> List[SeatSpec]:
        seats: List[SeatSpec] = []
        for pod in self.pods:
            seats.extend(pod.seats)
        return seats

    def get_seat(self, seat_id: str) -> Optional[SeatSpec]:
        target = seat_id.split("@")[0].strip()
        for seat in self.list_seats():
            if seat.id == target:
                return seat
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "summary": self.summary,
            "pods": [p.to_dict() for p in self.pods],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RigSpec":
        if "name" not in data or not data["name"]:
            raise ValueError("RigSpec must define a non-empty 'name'")
        pods = [PodSpec.from_dict(p) for p in data.get("pods", [])]
        edges = [EdgeSpec.from_dict(e) for e in data.get("edges", [])]
        return cls(
            name=str(data["name"]).strip(),
            summary=str(data.get("summary", "")).strip(),
            version=str(data.get("version", "0.2")),
            pods=pods,
            edges=edges,
        )

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "RigSpec":
        parsed = yaml.safe_load(yaml_content)
        if not isinstance(parsed, dict):
            raise ValueError("RigSpec YAML must define a top-level mapping")
        return cls.from_dict(parsed)

    @classmethod
    def from_file(cls, path: Path) -> "RigSpec":
        content = Path(path).read_text(encoding="utf-8")
        return cls.from_yaml(content)


def create_starter_rig(name: str = "metis-sovereign-squad") -> RigSpec:
    """Creates a default starter 2-seat rig: build (coder) and review."""
    coder_seat = SeatSpec(
        id="coder",
        label="Autonomous Coding Agent",
        role="Primary code implementation, AST navigation, and exact-anchor editing",
        model="qwen2.5-coder-32b",
        temperature=0.2,
        effort="high",
        permission_policy="strict",
        worktree_branch="squad/coder",
        skills=["python", "refactoring", "git"],
    )
    reviewer_seat = SeatSpec(
        id="reviewer",
        label="Independent Review & Verification Agent",
        role="Zero-trust diff auditing, Dokime test verification, and security checks",
        model="qwen2.5-coder-32b",
        temperature=0.0,
        effort="medium",
        permission_policy="audit_only",
        worktree_branch="squad/reviewer",
        skills=["security", "code-review", "pytest"],
    )
    core_pod = PodSpec(
        id="core-pod",
        label="Autonomous Engineering Squad",
        seats=[coder_seat, reviewer_seat],
    )
    edges = [
        EdgeSpec(from_seat="coder", to_seat="reviewer", kind="delegates_to"),
        EdgeSpec(from_seat="reviewer", to_seat="coder", kind="escalates_to"),
    ]
    return RigSpec(
        name=name,
        summary="Default Metis Sovereign Studio engineering squad with build and review seats",
        version="0.2",
        pods=[core_pod],
        edges=edges,
    )
