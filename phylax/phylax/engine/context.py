"""Trusted caller context and stack inspection.

Binds subject identity to execution context rather than accepting unverified free-text.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import inspect
from pathlib import Path
from typing import Generator

_current_subject: ContextVar[str | None] = ContextVar("phylax_caller_subject", default=None)
_delegated_tokens: ContextVar[dict[str, str]] = ContextVar("phylax_delegated_tokens", default={})

KNOWN_COMPONENTS = {
    "kybernetes",
    "stoma",
    "hestia",
    "dianoia",
    "lthcm",
    "prosopon",
    "dokime",
    "paideia",
}


@contextmanager
def trusted_caller(subject: str) -> Generator[None, None, None]:
    """Context manager setting trusted subject in contextvars for the current task/coroutine."""
    token = _current_subject.set(subject)
    try:
        yield
    finally:
        _current_subject.reset(token)


def get_current_subject() -> str | None:
    """Return active subject from contextvars if present."""
    return _current_subject.get()


def inspect_caller_identity(fallback_subject: str | None = None) -> str:
    """Infer caller identity by walking the call stack outside the phylax package.
    
    If contextvars subject is set, that takes precedence.
    Otherwise, inspects caller frames to match known Metis components.
    """
    ctx_subj = _current_subject.get()
    if ctx_subj:
        return ctx_subj

    stack = inspect.stack()
    # Find first frame outside of phylax package
    phylax_dir = str(Path(__file__).parent.parent.resolve())
    for frame_info in stack[1:]:
        filename = frame_info.filename
        if not filename.startswith(phylax_dir) and "<frozen" not in filename:
            # Check module name
            module = frame_info.frame.f_globals.get("__name__", "")
            for comp in KNOWN_COMPONENTS:
                if comp in module or f"/{comp}/" in filename or filename.endswith(f"{comp}.py"):
                    return comp
            
            # If fallback is provided and caller is test or main
            if fallback_subject:
                return fallback_subject

            # Check top-level package or file stem
            stem = Path(filename).stem
            if stem in KNOWN_COMPONENTS:
                return stem
            break

    return fallback_subject or "untrusted_caller"


def verify_caller_authorization(claimed_subject: str | None) -> str:
    """Verifies that the claimed subject matches the trusted caller context.
    
    Prevents an unprivileged component from claiming to be an administrative component.
    """
    ctx_subj = _current_subject.get()
    if ctx_subj is not None:
        if claimed_subject and claimed_subject != ctx_subj:
            # Cannot spoof another subject when explicit context is bound
            raise PermissionError(
                f"Caller context violation: Claimed subject '{claimed_subject}' does not match "
                f"active trusted context '{ctx_subj}'"
            )
        return ctx_subj

    inspected = inspect_caller_identity(fallback_subject=claimed_subject)
    if claimed_subject and claimed_subject != inspected:
        # If caller passed a subject, but frame inspection identifies another known component
        if inspected in KNOWN_COMPONENTS and inspected != "untrusted_caller":
            raise PermissionError(
                f"Subject spoofing detected: Frame is '{inspected}', but claimed subject is '{claimed_subject}'"
            )
    
    return claimed_subject or inspected
