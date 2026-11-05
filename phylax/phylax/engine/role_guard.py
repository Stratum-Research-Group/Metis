"""
Role-Spoofing and Context Boundary Demarcation Guard (Item 113).

Inspired by 'Understanding and Mitigating Role Spoofing in LLMs' (arXiv:2610.40396).
Defends agent execution and ingestion pipelines against untrusted data streams
(files, web responses, tool outputs) that synthesize privileged system/operator
headers, control tokens, or role markers to hijack execution authority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Pattern, Tuple


class RoleSpoofingViolationError(ValueError):
    """Raised when untrusted content contains unpermitted role spoofing or control tokens."""
    pass


@dataclass(frozen=True)
class RoleSpoofingMatch:
    """Details of a detected role spoofing attempt."""
    pattern_name: str
    matched_text: str
    position: int


class RoleSpoofingGuard:
    """Detects, sanitizes, and envelops untrusted content to prevent role spoofing."""

    # Special control tokens used across popular chat templates (ChatML, LLaMA, Mistral, Command-R)
    CONTROL_TOKEN_PATTERNS: List[Tuple[str, Pattern[str]]] = [
        ("chatml_im_start", re.compile(r"<\|im_start\|>", re.IGNORECASE)),
        ("chatml_im_end", re.compile(r"<\|im_end\|>", re.IGNORECASE)),
        ("special_role_system", re.compile(r"<\|(?:system|assistant|user|developer)\|>", re.IGNORECASE)),
        ("llama_inst_open", re.compile(r"\[INST\]", re.IGNORECASE)),
        ("llama_inst_close", re.compile(r"\[/INST\]", re.IGNORECASE)),
        ("llama_sys_open", re.compile(r"<<SYS>>", re.IGNORECASE)),
        ("llama_sys_close", re.compile(r"<</SYS>>", re.IGNORECASE)),
        ("end_of_turn", re.compile(r"<\|end_of_turn\|>|<\|eot_id\|>", re.IGNORECASE)),
    ]

    # Synthesized authority headers in plain text at line boundaries
    LINE_AUTHORITY_PATTERNS: List[Tuple[str, Pattern[str]]] = [
        ("line_system_bracket", re.compile(r"(?:^|\n)\s*\[(SYSTEM|SYSTEM\s+INSTRUCTION|OPERATOR|ADMIN|SUPERVISOR)\]", re.IGNORECASE)),
        ("line_role_header", re.compile(r"(?:^|\n)\s*(?:Role|Authority)\s*:\s*(system|admin|operator|supervisor)\b", re.IGNORECASE)),
        ("line_system_prompt", re.compile(r"(?:^|\n)\s*System\s*:\s*", re.IGNORECASE)),
        ("line_instruction_override", re.compile(r"(?:^|\n)\s*Disregard\s+(?:all\s+)?previous\s+instructions\b", re.IGNORECASE)),
    ]

    @classmethod
    def scan(cls, text: str) -> List[RoleSpoofingMatch]:
        """Scan text and return all matches of role spoofing or control token injection."""
        matches: List[RoleSpoofingMatch] = []

        for name, pattern in cls.CONTROL_TOKEN_PATTERNS:
            for m in pattern.finditer(text):
                matches.append(RoleSpoofingMatch(pattern_name=name, matched_text=m.group(0), position=m.start()))

        for name, pattern in cls.LINE_AUTHORITY_PATTERNS:
            for m in pattern.finditer(text):
                matches.append(RoleSpoofingMatch(pattern_name=name, matched_text=m.group(0).strip(), position=m.start()))

        return sorted(matches, key=lambda x: x.position)

    @classmethod
    def sanitize(cls, text: str) -> Tuple[str, List[RoleSpoofingMatch]]:
        """
        Sanitize untrusted text by neutralizing control tokens and neutralizing authority headers.
        Returns the sanitized text and the list of detected matches.
        """
        matches = cls.scan(text)
        if not matches:
            return text, []

        sanitized = text

        # Neutralize control tokens
        for name, pattern in cls.CONTROL_TOKEN_PATTERNS:
            sanitized = pattern.sub(lambda m: f"[SANITIZED_TOKEN:{m.group(0).strip('<|>[]')}]", sanitized)

        # Neutralize line-level authority headers
        for name, pattern in cls.LINE_AUTHORITY_PATTERNS:
            sanitized = pattern.sub(lambda m: f"\n[UNTRUSTED_CONTENT_TEXT: {m.group(0).strip()}]", sanitized)

        return sanitized, matches

    @classmethod
    def wrap_data_boundary(cls, text: str, origin: str = "untrusted_input") -> str:
        """
        Sanitize and wrap untrusted content inside unambiguous boundary envelopes.
        Prevents downstream models and parsers from treating data payloads as instructions.
        """
        sanitized_body, _ = cls.sanitize(text)
        clean_origin = re.sub(r'["\r\n<>]', "_", origin.strip()) or "untrusted_input"
        return (
            f"<metis_untrusted_data origin=\"{clean_origin}\">\n"
            f"{sanitized_body}\n"
            f"</metis_untrusted_data>"
        )

    @classmethod
    def assert_no_role_spoofing(cls, text: str, source_description: str = "Input") -> None:
        """Assert that text contains zero role spoofing or control token injections."""
        matches = cls.scan(text)
        if matches:
            details = ", ".join(f"'{m.matched_text}' ({m.pattern_name})" for m in matches[:3])
            raise RoleSpoofingViolationError(
                f"{source_description} violated role safety boundary: detected {len(matches)} spoofing pattern(s): {details}"
            )
