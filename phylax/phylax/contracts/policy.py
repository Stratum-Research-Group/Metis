"""Typed policy schema models for Phylax.

Enforces strict YAML schemas with no executable code and extra='forbid'
to ensure unknown keys immediately fail validation.
"""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator

from phylax.contracts.enums import Effect, ResourceClass, Verb


class DeclarativeCondition(BaseModel):
    """Declarative, safe conditions with zero Python eval or code execution."""
    model_config = ConfigDict(extra="forbid")

    path_must_exist: bool | None = None
    max_bytes: int | None = Field(default=None, ge=1)
    allowed_extensions: list[str] | None = None
    argument_patterns: dict[str, str] | None = None
    environment_tags: list[str] | None = None


class ResourceSpec(BaseModel):
    """Specification of target resource matching parameters."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    resource_class: ResourceClass = Field(alias="class")
    path_root: str | None = None
    path_glob: str | None = None
    tool_id: str | None = None
    destination: str | None = None
    command: str | None = None

    @field_validator("path_root")
    @classmethod
    def validate_path_root(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("path_root cannot be empty string")
        return v


class Rule(BaseModel):
    """Individual authorization rule."""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    description: str | None = None
    subject: str = Field(min_length=1)  # May be a specific subject or wildcard '*'
    verb: Verb  # Exactly one of read, write, execute, send
    resource: ResourceSpec
    condition: DeclarativeCondition | None = None
    effect: Effect  # allow, deny, ask
    priority: int = Field(default=100, ge=1, le=1000)  # Lower value = higher priority
    ask_ttl_seconds: int = Field(default=300, ge=5, le=86400)


class RedactionConfig(BaseModel):
    """Audit log redaction preferences."""
    model_config = ConfigDict(extra="forbid")

    redact_content: bool = True
    mask_patterns: list[str] = Field(
        default_factory=lambda: ["api_key", "token", "secret", "password", "pin", "auth"]
    )
    mask_replacement: str = "[REDACTED]"


class PolicyDocument(BaseModel):
    """Top-level Phylax policy document parsed from YAML."""
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    description: str | None = None
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)
    rules: list[Rule]

    @field_validator("rules")
    @classmethod
    def validate_rules(cls, rules: list[Rule]) -> list[Rule]:
        if not rules:
            raise ValueError("Policy must contain at least one rule")
        
        # Verify unique rule IDs
        rule_ids = set()
        for r in rules:
            if r.id in rule_ids:
                raise ValueError(f"Duplicate rule ID found: '{r.id}'")
            rule_ids.add(r.id)

        # Verify that there is an explicit final default-deny rule
        last_rule = rules[-1]
        is_default_deny = (
            last_rule.effect == Effect.DENY
            and (last_rule.subject == "*" or last_rule.subject == "all")
        )
        if not is_default_deny:
            raise ValueError(
                "Policy must end with an explicit default-deny rule for subject '*' (e.g. effect: deny)"
            )

        return rules
