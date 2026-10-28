"""Enumerations for Krisis answer selection, status, and reason codes."""

from enum import Enum


class DecisionStatus(str, Enum):
    """Top-level status of a Krisis decision."""
    SELECTED = "selected"
    ABSTAINED = "abstained"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REBUILD_REQUIRED = "rebuild_required"



class Reason(str, Enum):
    """Standardized failure or abstention reasons matching reason_enum in policy."""
    NO_AUTHORIZED_CANDIDATE = "NO_AUTHORIZED_CANDIDATE"
    NO_VALID_CANDIDATE = "NO_VALID_CANDIDATE"
    QUALITY_BELOW_THRESHOLD = "QUALITY_BELOW_THRESHOLD"
    VERIFIER_UNAVAILABLE = "VERIFIER_UNAVAILABLE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    STALE = "STALE"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    UPSTREAM_GENERATION_FAILED = "UPSTREAM_GENERATION_FAILED"
    PROFILE_UNAVAILABLE = "PROFILE_UNAVAILABLE"
    FEATURE_INVALID = "FEATURE_INVALID"
    CANCELLED = "CANCELLED"
    AMBIGUOUS_PREDICTION_SET = "AMBIGUOUS_PREDICTION_SET"
    CONFORMAL_EMPTY_SET = "CONFORMAL_EMPTY_SET"


class GateType(str, Enum):
    """Hard eligibility gates applied to candidate answers."""
    ACCESS = "access"
    PROVENANCE = "provenance"
    CONTRADICTION = "contradiction"
    CITATION_RESOLUTION = "citation_resolution"
    PER_CLAIM_SUPPORT = "per_claim_support"
    FEATURE_VALID = "feature_valid"


class ClaimVerdict(str, Enum):
    """Verdict assigned by evidence verifier to individual claims."""
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"


class ProfileId(str, Enum):
    """Known selector profiles in Krisis policy."""
    ELIGIBILITY_BASELINE = "eligibility_baseline"
    GROUNDED_SUPPORT_ONLY = "grounded_support_only"
    GROUNDED_V1 = "grounded_v1"
