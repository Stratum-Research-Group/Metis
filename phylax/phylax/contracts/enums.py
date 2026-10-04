"""Enumerations for Phylax policy, actions, and decisions."""

from enum import Enum


class Verb(str, Enum):
    """Allowed capability verbs in Phylax."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    SEND = "send"
    ALL = "*"


class Effect(str, Enum):
    """Authorization decision effects."""
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class ResourceClass(str, Enum):
    """Resource classes mediated by Phylax."""
    FILE = "file"
    TOOL = "tool"
    NETWORK = "network"
    SYSTEM = "system"
    ALL = "*"


class ApprovalStatus(str, Enum):
    """Lifecycle states of an ASK approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CONSUMED = "consumed"
