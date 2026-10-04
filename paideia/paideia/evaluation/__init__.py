"""Paideia evaluation and promotion package."""

from paideia.evaluation.bootstrap import (
    BootstrapResult,
    compute_paired_bootstrap_ci,
)
from paideia.evaluation.null_test import (
    NullTestResult,
    paired_permutation_test,
    mcnemar_test,
)
from paideia.evaluation.promotion import (
    OperationalGuardrails,
    PromotionVerdict,
    evaluate_promotion,
)
from paideia.evaluation.dokime_bundle import DokimeRunBundle

__all__ = [
    "BootstrapResult",
    "compute_paired_bootstrap_ci",
    "NullTestResult",
    "paired_permutation_test",
    "mcnemar_test",
    "OperationalGuardrails",
    "PromotionVerdict",
    "evaluate_promotion",
    "DokimeRunBundle",
]
