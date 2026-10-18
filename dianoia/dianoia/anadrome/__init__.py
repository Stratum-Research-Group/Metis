"""Anadrome: Offline Walk-Trace Replay Policy Optimization for Dianoia."""

from dianoia.anadrome.canonical import canonical_json_dumps, compute_observation_hash
from dianoia.anadrome.contracts import (
    DecisionObservation,
    EligibleAction,
    EnvironmentIdentity,
    RecordedTransition,
    ReplayTreeRecord,
)
from dianoia.anadrome.evaluator import (
    AnadromeEvaluator,
    PairedComparisonResult,
    PolicyEpisodeScore,
    PolicyEvaluationSummary,
)
from dianoia.anadrome.logger import AnadromeLogger
from dianoia.anadrome.policies import (
    AnadromePolicy,
    BaselineReplayPolicy,
    EarlyExitAnswerPolicy,
    GreedySimilarityPolicy,
    HighEntropyStochasticPolicy,
    SheafContradictionBacktrackPolicy,
)
from dianoia.anadrome.probe_cache import AnswerProbeCache
from dianoia.anadrome.replay import AnadromeReplayEngine, EpisodeReplayResult

__all__ = [
    "canonical_json_dumps",
    "compute_observation_hash",
    "EnvironmentIdentity",
    "EligibleAction",
    "DecisionObservation",
    "RecordedTransition",
    "ReplayTreeRecord",
    "AnswerProbeCache",
    "AnadromeLogger",
    "AnadromeReplayEngine",
    "EpisodeReplayResult",
    "AnadromePolicy",
    "GreedySimilarityPolicy",
    "HighEntropyStochasticPolicy",
    "EarlyExitAnswerPolicy",
    "SheafContradictionBacktrackPolicy",
    "BaselineReplayPolicy",
    "AnadromeEvaluator",
    "PolicyEvaluationSummary",
    "PairedComparisonResult",
    "PolicyEpisodeScore",
]

