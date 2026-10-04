"""Class routing engine matching question classes to profiles or typed abstentions."""

from typing import Any
from krisis.contracts.policy import ClassRoutingConfig, RoutingRule
from krisis.contracts.request import Capabilities
from krisis.routing.normalizer import normalize_question_class


class ClassRouter:
    """Evaluates class routing rules according to the Krisis policy specification."""

    def __init__(self, routing_config: ClassRoutingConfig | dict[str, Any]):
        if isinstance(routing_config, ClassRoutingConfig):
            self.config = routing_config
        else:
            self.config = ClassRoutingConfig.model_validate(routing_config)

        norm_conf = self.config.input_class_normalization or {}
        self.known_classes = norm_conf.get(
            "known_classes",
            ["closed_form_number", "closed_form_entity", "closed_form_choice", "unclassified"],
        )

    def route_single(
        self,
        norm_class: str,
        requires_grounding: bool,
        capabilities: Capabilities,
    ) -> tuple[str | None, str | None]:
        """Route a single normalized question class. Returns (profile, abstain)."""
        for rule in self.config.rules:
            # Match class
            if rule.question_class != "*" and rule.question_class != norm_class:
                continue

            # Match requires_grounding
            if rule.requires_grounding is not None and rule.requires_grounding != requires_grounding:
                continue

            # Match verifier_active
            if rule.verifier_active is not None and rule.verifier_active != capabilities.verifier_active:
                continue

            # Match logprobs
            if rule.logprobs is not None and rule.logprobs != capabilities.logprobs:
                continue

            # First match wins
            if rule.abstain:
                return None, rule.abstain
            if rule.profile:
                return rule.profile, None

        return None, "NO_VALID_CANDIDATE"

    def route(
        self,
        raw_question_class: Any,
        requires_grounding: bool,
        capabilities: Capabilities,
    ) -> tuple[str | None, str | None, list[str]]:
        """Route request. Returns (profile_id, abstain_reason, normalized_classes)."""
        norm_classes = normalize_question_class(raw_question_class, self.known_classes)

        if len(norm_classes) == 1:
            profile, abstain = self.route_single(norm_classes[0], requires_grounding, capabilities)
            return profile, abstain, norm_classes

        # Multi-class routing
        profiles: set[str] = set()
        abstentions: list[str] = []

        for c in norm_classes:
            prof, abst = self.route_single(c, requires_grounding, capabilities)
            if abst:
                abstentions.append(abst)
            elif prof:
                profiles.add(prof)

        # Any terminal abstention applies (e.g. VERIFIER_UNAVAILABLE)
        if abstentions:
            return None, abstentions[0], norm_classes

        # Compatible profiles combine
        if len(profiles) == 1:
            return list(profiles)[0], None, norm_classes

        # Conflicting profiles abstain PROFILE_UNAVAILABLE
        if len(profiles) > 1:
            return None, "PROFILE_UNAVAILABLE", norm_classes

        return None, "PROFILE_UNAVAILABLE", norm_classes
