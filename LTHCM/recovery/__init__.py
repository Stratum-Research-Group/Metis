"""LTHCM Damage-Trained Recovery Module.

Implements verifiable recovery over disposable shadow graphs without rewriting evidence,
as specified in LTHCM-recovery-application-plan-356cbcd7.txt.
"""

from recovery.schema import (
    CanonicalDocument,
    EligibilityDecision,
    DerivedEdge,
    ShadowNode,
    ShadowGraphState,
    DamageConfig,
    RecoveryStepRecord,
    RecoveryOutcome,
    ProhibitedWriteError,
    MissingSourceError,
    CorruptedProvenanceError,
    EligibilityViolationError,
    UnsupportedLinkError,
    NonFiniteStateError,
    RecoveryBudgetExceededError,
)

from recovery.baselines import (
    BaseRecoveryModel,
    DeterministicRebuildRerankBaseline,
    RecurrentNoDamageAblation,
    RecurrentNoRecurrenceAblation,
    RecurrentRecoveryWrapperBaseline,
    SpectralChebyshevBaseline,
    StandardGNNBaseline,
    UnchangedPipelineBaseline,
)
from recovery.stress_tests import HeldOutStressTester

__all__ = [
    "CanonicalDocument",
    "EligibilityDecision",
    "DerivedEdge",
    "ShadowNode",
    "ShadowGraphState",
    "DamageConfig",
    "RecoveryStepRecord",
    "RecoveryOutcome",
    "ProhibitedWriteError",
    "MissingSourceError",
    "CorruptedProvenanceError",
    "EligibilityViolationError",
    "UnsupportedLinkError",
    "NonFiniteStateError",
    "RecoveryBudgetExceededError",
    "BaseRecoveryModel",
    "UnchangedPipelineBaseline",
    "DeterministicRebuildRerankBaseline",
    "SpectralChebyshevBaseline",
    "StandardGNNBaseline",
    "RecurrentRecoveryWrapperBaseline",
    "RecurrentNoRecurrenceAblation",
    "RecurrentNoDamageAblation",
    "HeldOutStressTester",
]
