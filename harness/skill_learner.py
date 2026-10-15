"""Nous Hermes-inspired Dynamic Skill Generation and Retrieval Engine.
Enables Metis coding agent to self-synthesize reusable skills from successful trajectories.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional


@dataclass
class DynamicSkill:
    id: str
    name: str
    domain: str
    description: str
    instructions: str
    verification_command: Optional[str]
    created_at: float
    file_path: Path


class DynamicSkillLearner:
    """Manages creation, indexing, and just-in-time retrieval of dynamic skills."""

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.skills_dir = self.workspace_root / ".metis" / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_skill(
        self,
        name: str,
        domain: str,
        description: str,
        instructions: str,
        verification_command: Optional[str] = None,
    ) -> DynamicSkill:
        """Synthesize a new skill from a verified solution trajectory."""
        slug = re.sub(r"[^a-z0-9_-]", "_", name.lower()).strip("_")
        target_dir = self.skills_dir / domain / slug
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / "SKILL.md"

        timestamp = time.time()
        content = (
            f"---\n"
            f"name: \"{name}\"\n"
            f"domain: \"{domain}\"\n"
            f"description: \"{description}\"\n"
            f"created_at: {timestamp}\n"
            f"verification_command: \"{verification_command or ''}\"\n"
            f"---\n\n"
            f"# {name}\n\n"
            f"## Description\n{description}\n\n"
            f"## Instructions & Procedure\n{instructions}\n\n"
        )
        if verification_command:
            content += f"## Verification\nRun: `{verification_command}`\n"

        file_path.write_text(content, encoding="utf-8")

        return DynamicSkill(
            id=f"{domain}/{slug}",
            name=name,
            domain=domain,
            description=description,
            instructions=instructions,
            verification_command=verification_command,
            created_at=timestamp,
            file_path=file_path,
        )

    def load_skills(self) -> List[DynamicSkill]:
        """Load all persisted skills from .metis/skills."""
        skills: List[DynamicSkill] = []
        if not self.skills_dir.exists():
            return skills

        for skill_file in self.skills_dir.glob("*/*/SKILL.md"):
            try:
                text = skill_file.read_text(encoding="utf-8")
                # Parse frontmatter
                fm_match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
                if fm_match:
                    fm_text, body = fm_match.groups()
                    meta: Dict[str, Any] = {}
                    for line in fm_text.splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            meta[k.strip()] = v.strip().strip('"')

                    skills.append(
                        DynamicSkill(
                            id=f"{meta.get('domain', 'general')}/{skill_file.parent.name}",
                            name=meta.get("name", skill_file.parent.name),
                            domain=meta.get("domain", "general"),
                            description=meta.get("description", ""),
                            instructions=body.strip(),
                            verification_command=meta.get("verification_command") or None,
                            created_at=float(meta.get("created_at", 0.0)),
                            file_path=skill_file,
                        )
                    )
            except Exception:
                continue

        return skills

    def find_relevant_skills(self, query: str, max_results: int = 3) -> List[DynamicSkill]:
        """Simple keyword matching to retrieve relevant skills for prompt injection."""
        all_skills = self.load_skills()
        query_words = set(re.findall(r"\w+", query.lower()))

        scored_skills = []
        for s in all_skills:
            score = 0
            skill_text = f"{s.name} {s.domain} {s.description}".lower()
            for w in query_words:
                if w in skill_text:
                    score += 1
            if score > 0:
                scored_skills.append((score, s))

        scored_skills.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored_skills[:max_results]]
