"""
Kleroterion errors and exceptions.
"""


class KleroterionError(Exception):
    """Base exception for all Kleroterion errors."""


class InvalidConfigError(KleroterionError):
    """Raised when configuration validation fails."""


class InvalidInputError(KleroterionError):
    """Raised when request or input validation fails."""


class NoEligibleActionError(KleroterionError):
    """Raised when no action remains eligible after intersection (NO_ELIGIBLE_ACTION)."""


class InvalidProposalError(KleroterionError):
    """Raised when a feature proposal violates bounds or invariants."""


class ScorerError(KleroterionError):
    """Raised when host scoring fails or returns invalid/non-finite logits."""


class RevalidationError(KleroterionError):
    """Raised when host revalidation fails prior to action execution."""


class TransactionAbortedError(KleroterionError):
    """Raised when a host transaction aborts or encounters a stale snapshot."""
