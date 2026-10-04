"""Paideia baseline policies and evaluation ladder."""

from paideia.baselines.base import BasePolicy
from paideia.baselines.majority import MajorityBaseline
from paideia.baselines.regex_router import RegexRouterBaseline
from paideia.baselines.lexical import LexicalLogisticBaseline
from paideia.baselines.embedding import SmallEmbeddingBaseline
from paideia.baselines.ladder import (
    run_baseline_ladder,
    BaselineEvaluationReport,
)

__all__ = [
    "BasePolicy",
    "MajorityBaseline",
    "RegexRouterBaseline",
    "LexicalLogisticBaseline",
    "SmallEmbeddingBaseline",
    "run_baseline_ladder",
    "BaselineEvaluationReport",
]
