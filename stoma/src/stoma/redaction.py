"""
Stoma Pre-Ingestion Privacy Redactor & Content Assessment (Phase 2 / WALDO model).
Features:
- Heuristic regex masking for API keys, bearer tokens, private IPs, emails, and phone numbers.
- Gopher-style content assessment: detects pathological repetition and boilerplate junk before chunking/embedding.
"""

from __future__ import annotations

import collections
import re
from typing import Dict, List, Pattern, Tuple


class PrivacyRedactor:
    """
    Scans raw text prior to chunking and replaces sensitive credentials,
    private network addresses, and PII with standard redacted tokens.
    """

    PATTERNS: List[Tuple[str, Pattern[str], str]] = [
        # Bearer tokens
        ("BEARER", re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE), "[REDACTED_BEARER_TOKEN]"),
        # OpenAI / generic sk- keys
        ("OPENAI_KEY", re.compile(r"\bsk-[a-zA-Z0-9_-]{20,}\b"), "[REDACTED_API_KEY]"),
        # GitHub tokens (classic and fine-grained)
        ("GITHUB_KEY", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36}\b"), "[REDACTED_API_KEY]"),
        # AWS Access Keys
        ("AWS_KEY", re.compile(r"\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_API_KEY]"),
        # Google API Keys
        ("GOOGLE_KEY", re.compile(r"\bAIza[0-9A-Za-z-_]{35}\b"), "[REDACTED_API_KEY]"),
        # Generic Secret / Key assignments in text (e.g. secret = "...")
        ("ASSIGNMENT_SECRET", re.compile(r'(?i)\b(api_key|access_token|client_secret|db_password)\s*[:=]\s*["\']([a-zA-Z0-9_\-\.]{12,})["\']'), r'\1="[REDACTED_SECRET]"'),
        # Private IPv4 addresses (RFC 1918 + loopback)
        ("PRIVATE_IP", re.compile(r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|127\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"), "[REDACTED_IP]"),
        # Email addresses
        ("EMAIL", re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"), "[REDACTED_EMAIL]"),
        # International Phone numbers (E.164-like)
        ("PHONE", re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
    ]

    @classmethod
    def redact(cls, text: str) -> Tuple[str, int]:
        """
        Applies all privacy masks to the input text.
        Returns (sanitized_text, total_redactions_count).
        """
        if not text:
            return "", 0

        redacted_text = text
        total_count = 0

        for name, pattern, replacement in cls.PATTERNS:
            matches = len(pattern.findall(redacted_text))
            if matches > 0:
                total_count += matches
                redacted_text = pattern.sub(replacement, redacted_text)

        return redacted_text, total_count


class ContentAssessment:
    """
    Gopher-style repetition and boilerplate filter (adapted from WALDO).
    Identifies low-quality, looped, or repetitive text before embedding.
    """

    @classmethod
    def character_repetition_ratio(cls, text: str) -> float:
        """Computes fraction of characters belonging to runs of 4+ identical characters."""
        if not text:
            return 0.0
        # Find runs of 4 or more repeated characters
        repeated_chars = 0
        for match in re.finditer(r"(.)\1{3,}", text):
            repeated_chars += len(match.group(0))
        return repeated_chars / len(text)

    @classmethod
    def duplicate_ngram_fraction(cls, text: str, n: int = 10) -> float:
        """
        Computes fraction of duplicate n-grams in word tokens.
        Detects repetitive boilerplate loops.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        if len(words) < n * 2:
            return 0.0

        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0

        counts = collections.Counter(ngrams)
        duplicate_ngram_count = sum(c - 1 for c in counts.values() if c > 1)
        return duplicate_ngram_count / len(ngrams)

    @classmethod
    def is_boilerplate_or_spam(
        cls,
        text: str,
        max_ngram_duplication: float = 0.30,
        max_char_repeat: float = 0.20,
    ) -> Tuple[bool, str]:
        """
        Evaluates whether a passage should be rejected or flagged as boilerplate/spam.
        Returns (is_spam, reason).
        """
        if not text or len(text.strip()) < 20:
            return False, "Short text"

        char_rep = cls.character_repetition_ratio(text)
        if char_rep > max_char_repeat:
            return True, f"Excessive character repetition ({char_rep:.2%} > {max_char_repeat:.2%})"

        ngram_rep = cls.duplicate_ngram_fraction(text, n=10)
        if ngram_rep > max_ngram_duplication:
            return True, f"Excessive 10-gram repetition ({ngram_rep:.2%} > {max_ngram_duplication:.2%})"

        return False, "Quality passed"
