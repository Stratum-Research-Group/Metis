"""Tests for question class normalization and routing."""

import json
from pathlib import Path
import pytest

from krisis.contracts.request import Capabilities
from krisis.routing.normalizer import normalize_question_class
from krisis.routing.router import ClassRouter


@pytest.fixture
def router():
    policy = json.loads(Path("krisis-policy-0.1.json").read_text(encoding="utf-8"))
    return ClassRouter(policy["class_routing"])


def test_normalization():
    assert normalize_question_class("closed_form_number") == ["closed_form_number"]
    assert normalize_question_class("unknown_foo") == ["unclassified"]
    assert normalize_question_class(None) == ["unclassified"]
    assert normalize_question_class("") == ["unclassified"]
    assert normalize_question_class(["closed_form_entity", "unknown"]) == ["closed_form_entity", "unclassified"]


def test_grounded_routing_with_and_without_logprobs(router: ClassRouter):
    # With logprobs -> grounded_v1
    prof, abstain, _ = router.route(
        "closed_form_number",
        requires_grounding=True,
        capabilities=Capabilities(verifier_active=True, logprobs=True),
    )
    assert prof == "grounded_v1"
    assert abstain is None

    # Without logprobs -> grounded_support_only
    prof, abstain, _ = router.route(
        "closed_form_number",
        requires_grounding=True,
        capabilities=Capabilities(verifier_active=True, logprobs=False),
    )
    assert prof == "grounded_support_only"
    assert abstain is None


def test_verifier_outage_abstention(router: ClassRouter):
    # Verifier unavailable on grounded request -> abstain VERIFIER_UNAVAILABLE
    prof, abstain, _ = router.route(
        "closed_form_choice",
        requires_grounding=True,
        capabilities=Capabilities(verifier_active=False, logprobs=True),
    )
    assert prof is None
    assert abstain == "VERIFIER_UNAVAILABLE"


def test_non_grounded_routing(router: ClassRouter):
    # Non-grounded -> eligibility_baseline
    prof, abstain, _ = router.route(
        "closed_form_number",
        requires_grounding=False,
        capabilities=Capabilities(verifier_active=False, logprobs=False),
    )
    assert prof == "eligibility_baseline"
    assert abstain is None


def test_multi_class_conflict(router: ClassRouter):
    # number (grounded_v1) + unclassified (grounded_support_only) when logprobs=True -> PROFILE_UNAVAILABLE
    prof, abstain, _ = router.route(
        ["closed_form_number", "unclassified"],
        requires_grounding=True,
        capabilities=Capabilities(verifier_active=True, logprobs=True),
    )
    assert prof is None
    assert abstain == "PROFILE_UNAVAILABLE"
