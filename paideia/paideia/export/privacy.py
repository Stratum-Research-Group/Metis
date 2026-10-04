"""Privacy, retention, and dataset sanitization utilities."""

import hashlib
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetentionPolicy(BaseModel):
    """Data retention and privacy boundary policy."""
    allow_raw_prompts: bool = Field(default=False, description="Whether raw user text can be stored in train artifacts")
    redact_emails_and_ips: bool = Field(default=True, description="Automatically redact emails and IP addresses")
    retention_days: int = Field(default=30, ge=1, description="Days before uncompressed training logs expire")
    privacy_tier: str = Field(default="internal", description="Access boundary tier: public, internal, restricted")


def sanitize_text(text: str, policy: Optional[RetentionPolicy] = None) -> str:
    """Redact identifiable patterns (emails, IPs, numbers) according to policy."""
    if policy is None:
        policy = RetentionPolicy()

    if not policy.redact_emails_and_ips:
        return text

    # Email pattern
    redacted = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_REDACTED]", text)
    # IPv4 pattern
    redacted = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_REDACTED]", redacted)
    return redacted


def extract_sanitized_features(question: str) -> Dict[str, Any]:
    """
    Extract privacy-safe local features without persisting raw user strings.
    Useful for training routers on sanitized features only.
    """
    words = re.findall(r"\b\w+\b", question.lower())
    return {
        "token_count": len(words),
        "char_count": len(question),
        "question_hash": hashlib.sha256(question.strip().encode("utf-8")).hexdigest(),
        "starts_with_wh": bool(re.match(r"^(who|what|where|when|which|why|how)\b", question.strip().lower())),
        "has_or": "or" in words,
        "has_both": "both" in words,
    }
