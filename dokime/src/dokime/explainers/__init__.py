"""Dokime Explainability Subsystem.

Provides black-box attribution and post-hoc explanation algorithms for models,
rationales, and prompt minimization without requiring activation hooks or gradients.
"""

from dokime.explainers.sis import (
    SISCollectionResult,
    SISResult,
    SufficientInputSubsets,
    explain_text_sufficiency,
)
from dokime.explainers.step_causality import (
    CausalDeliberationAuditor,
    DeliberationAuditReport,
    DeliberationStep,
    StepCausalRole,
    StepInterventionAudit,
)
from dokime.explainers.prefix_executability import (
    ActionStep,
    PrefixExecutabilityAuditor,
    PrefixExecutabilityReport,
    StepExecutabilityAudit,
)
from dokime.explainers.batch_isolation import (
    BatchIsolationAuditor,
    BatchIsolationReport,
    BatchItem,
    ItemPermutationResult,
)

__all__ = [
    "SISResult",
    "SISCollectionResult",
    "SufficientInputSubsets",
    "explain_text_sufficiency",
    "StepCausalRole",
    "DeliberationStep",
    "StepInterventionAudit",
    "DeliberationAuditReport",
    "CausalDeliberationAuditor",
    "ActionStep",
    "StepExecutabilityAudit",
    "PrefixExecutabilityReport",
    "PrefixExecutabilityAuditor",
    "BatchItem",
    "ItemPermutationResult",
    "BatchIsolationReport",
    "BatchIsolationAuditor",
]
