"""Kybernetes core enumerations and route/stage definitions."""

from enum import Enum


class RouteTag(str, Enum):
    """
    Observability tags describing capabilities used during a turn.
    Note: These are NOT mutually exclusive routing silos; a single turn
    can combine retrieval, deliberation, and tools.
    """
    DIRECT = "DIRECT"
    MEMORY = "MEMORY"
    DELIBERATE = "DELIBERATE"
    TOOL = "TOOL"


class StageName(str, Enum):
    """Sequential pipeline stages in an inspectable execution plan."""
    AUTHORIZE = "AUTHORIZE"
    RETRIEVE = "RETRIEVE"
    DELIBERATE = "DELIBERATE"
    TOOL = "TOOL"
    SYNTHESIZE = "SYNTHESIZE"
    VERIFY = "VERIFY"
    RESPOND = "RESPOND"


class TurnOutcomeStatus(str, Enum):
    """
    Outcome status of a conversational turn.
    Factual correctness is distinguished from raw generation status.
    """
    VERIFIED = "verified"
    CITED_UNVERIFIED = "cited_unverified"
    DENIED = "denied"
    CONFLICTED = "conflicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"
    DRAFT = "draft"
    ABSTAINED = "abstained"


class EventType(str, Enum):
    """Streaming event types emitted by Kybernetes over SSE."""
    STARTED = "started"
    PLAN_SELECTED = "plan_selected"
    DELTA = "delta"
    EVIDENCE = "evidence"
    ASK_APPROVAL = "ask_approval"
    APPROVAL_RESOLVED = "approval_resolved"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolSideEffectClass(str, Enum):
    """Side effect classification for tools."""
    READ_ONLY = "read_only"
    FILESYSTEM_MUTATION = "filesystem_mutation"
    OUTBOUND_NETWORK = "outbound_network"
    SYSTEM_EXECUTION = "system_execution"


class BudgetUnit(str, Enum):
    """Distinct measurement units for resource tracking."""
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    CACHED_TOKENS = "cached_tokens"
    TIME_SECONDS = "time_seconds"
    MONEY_USD = "money_usd"


class CheckpointAction(str, Enum):
    """Decision emitted at pre-dispatch and in-phase budget checkpoints."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
