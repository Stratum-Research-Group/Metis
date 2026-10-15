"""Unit tests for DynamicSkillLearner."""

from pathlib import Path
import pytest
import sys

# Ensure Metis root is on python path for importing harness.skill_learner
sys.path.insert(0, str(Path(__file__).parents[2]))

from harness.skill_learner import DynamicSkillLearner


def test_synthesize_and_retrieve_skill(tmp_path: Path):
    learner = DynamicSkillLearner(workspace_root=tmp_path)

    skill = learner.synthesize_skill(
        name="Pytest Mocking Pattern",
        domain="testing",
        description="Guidelines for mocking external HTTP clients in pytest async fixtures.",
        instructions="Use httpx.AsyncClient with MockTransport instead of patching global socket.",
        verification_command="pytest tests/test_api.py",
    )

    assert skill.name == "Pytest Mocking Pattern"
    assert skill.file_path.exists()
    assert "MockTransport" in skill.file_path.read_text(encoding="utf-8")

    # Load all skills
    all_skills = learner.load_skills()
    assert len(all_skills) == 1
    assert all_skills[0].domain == "testing"

    # Search for skill
    found = learner.find_relevant_skills("How do I mock an async HTTP client in pytest?")
    assert len(found) == 1
    assert found[0].name == "Pytest Mocking Pattern"

    # Search for unrelated query
    not_found = learner.find_relevant_skills("Configure Docker Swarm cluster")
    assert len(not_found) == 0
