"""Programmatic, deterministic instruction checkers for IFEval.

Inspired by Google Research's IFEval (instruction_following_eval, arXiv 2311.07911).
Evaluates strict and loose adherence to verifiable constraints (counts, keywords,
formatting, schemas, forbidden tokens) with zero LLM judge cost.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def clean_loose_text(text: str) -> str:
    """Normalize text for loose evaluation: remove markdown code fences, trailing whitespace, common preamble."""
    cleaned = text.strip()
    # Strip markdown code blocks if wrapped
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:-1]).strip()
    return cleaned


def count_words(text: str) -> int:
    """Count words using standard whitespace splitting after stripping punctuation boundaries."""
    return len(re.findall(r"\b\w+\b", text))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries (. ! ?)."""
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    # Match sentences ending with ., !, or ?
    raw_sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [s.strip() for s in raw_sentences if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs delimited by double newlines."""
    raw = [p.strip() for p in re.split(r"\n\s*\n", text.strip())]
    return [p for p in raw if p]


class InstructionChecker(ABC):
    """Abstract base class for deterministic instruction checkers."""

    @abstractmethod
    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        """Check adherence to the instruction rule.
        
        Returns:
            (passed: bool, message: str)
        """
        ...


class JsonFormatChecker(InstructionChecker):
    """Verifies that response parses as valid JSON and matches optional required keys."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = clean_loose_text(response) if loose else response.strip()
        try:
            data = json.loads(target)
        except Exception as e:
            # Under loose mode, try searching for any valid JSON object block {...}
            if loose:
                match = re.search(r"\{.*\}", target, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                    except Exception:
                        return False, f"Invalid JSON format: {e}"
                else:
                    return False, f"Invalid JSON format: {e}"
            else:
                return False, f"Invalid JSON format: {e}"

        required_keys = kwargs.get("required_keys", [])
        if required_keys:
            if not isinstance(data, dict):
                return False, "Parsed JSON is not an object/dict"
            missing = [k for k in required_keys if k not in data]
            if missing:
                return False, f"JSON missing required keys: {missing}"

        return True, "Valid JSON format"


class WordCountChecker(InstructionChecker):
    """Verifies word count satisfies minimum, maximum, or exact constraints."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = clean_loose_text(response) if loose else response
        n_words = count_words(target)

        min_words = kwargs.get("min_words")
        if min_words is not None and n_words < min_words:
            return False, f"Word count {n_words} < required minimum {min_words}"

        max_words = kwargs.get("max_words")
        if max_words is not None and n_words > max_words:
            return False, f"Word count {n_words} > allowed maximum {max_words}"

        exact_words = kwargs.get("exact_words")
        if exact_words is not None:
            # Under loose mode allow a small tolerance of +/- 2 words
            tolerance = 2 if loose else 0
            if abs(n_words - exact_words) > tolerance:
                return False, f"Word count {n_words} != exact target {exact_words} (tolerance={tolerance})"

        return True, f"Word count {n_words} satisfied"


class SentenceCountChecker(InstructionChecker):
    """Verifies sentence count satisfies minimum, maximum, or exact constraints."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = clean_loose_text(response) if loose else response
        sentences = split_sentences(target)
        n = len(sentences)

        min_s = kwargs.get("min_sentences")
        if min_s is not None and n < min_s:
            return False, f"Sentence count {n} < required minimum {min_s}"

        max_s = kwargs.get("max_sentences")
        if max_s is not None and n > max_s:
            return False, f"Sentence count {n} > allowed maximum {max_s}"

        exact_s = kwargs.get("exact_sentences")
        if exact_s is not None and n != exact_s:
            return False, f"Sentence count {n} != exact target {exact_s}"

        return True, f"Sentence count {n} satisfied"


class ParagraphCountChecker(InstructionChecker):
    """Verifies paragraph count separated by blank lines."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = clean_loose_text(response) if loose else response
        paras = split_paragraphs(target)
        n = len(paras)

        min_p = kwargs.get("min_paragraphs")
        if min_p is not None and n < min_p:
            return False, f"Paragraph count {n} < required minimum {min_p}"

        max_p = kwargs.get("max_paragraphs")
        if max_p is not None and n > max_p:
            return False, f"Paragraph count {n} > allowed maximum {max_p}"

        exact_p = kwargs.get("exact_paragraphs")
        if exact_p is not None and n != exact_p:
            return False, f"Paragraph count {n} != exact target {exact_p}"

        return True, f"Paragraph count {n} satisfied"


class KeywordPresenceChecker(InstructionChecker):
    """Verifies that all specified keywords appear in the response."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = response
        case_sensitive = kwargs.get("case_sensitive", False)
        keywords = kwargs.get("keywords", [])

        if not case_sensitive:
            target = target.lower()

        missing = []
        for kw in keywords:
            search_term = kw if case_sensitive else kw.lower()
            if search_term not in target:
                missing.append(kw)

        if missing:
            return False, f"Missing required keywords: {missing}"
        return True, "All required keywords present"


class ForbiddenWordsChecker(InstructionChecker):
    """Verifies that none of the forbidden words appear in the response."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = response
        case_sensitive = kwargs.get("case_sensitive", False)
        forbidden = kwargs.get("forbidden_words", [])

        if not case_sensitive:
            target = target.lower()

        exact_only = kwargs.get("exact_token_only", False)
        found = []
        for word in forbidden:
            search_term = word if case_sensitive else word.lower()
            if exact_only:
                pattern = rf"\b{re.escape(search_term)}\b"
            else:
                pattern = rf"\b{re.escape(search_term)}(?:s|es|ed|ing)?\b"

            if re.search(pattern, target, re.IGNORECASE if not case_sensitive else 0):
                found.append(word)

        if found:
            return False, f"Response contains forbidden words: {found}"
        return True, "No forbidden words found"


class KeywordFrequencyChecker(InstructionChecker):
    """Verifies that a specific keyword appears with exact or minimum frequency."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        keyword = kwargs.get("keyword", "")
        case_sensitive = kwargs.get("case_sensitive", False)
        target = response if case_sensitive else response.lower()
        search_kw = keyword if case_sensitive else keyword.lower()

        pattern = rf"\b{re.escape(search_kw)}\b"
        matches = len(re.findall(pattern, target))

        exact_freq = kwargs.get("exact_frequency")
        if exact_freq is not None and matches != exact_freq:
            return False, f"Keyword '{keyword}' appeared {matches} times, expected exactly {exact_freq}"

        min_freq = kwargs.get("min_frequency")
        if min_freq is not None and matches < min_freq:
            return False, f"Keyword '{keyword}' appeared {matches} times, expected at least {min_freq}"

        max_freq = kwargs.get("max_frequency")
        if max_freq is not None and matches > max_freq:
            return False, f"Keyword '{keyword}' appeared {matches} times, expected at most {max_freq}"

        return True, f"Keyword '{keyword}' frequency {matches} satisfied"


class SectionHeaderChecker(InstructionChecker):
    """Verifies presence of specific Markdown section headers."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        headers = kwargs.get("headers", [])
        missing = []
        for header in headers:
            # Pattern matches header at start of line with optional markdown hash prefixes
            pattern = rf"(?m)^#*\s*{re.escape(header.lstrip('#').strip())}\b"
            if not re.search(pattern, response, re.IGNORECASE if loose else 0):
                missing.append(header)

        if missing:
            return False, f"Missing required section headers: {missing}"
        return True, "All section headers present"


class StartEndConstraintChecker(InstructionChecker):
    """Verifies response starts with and/or ends with specific strings."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = clean_loose_text(response) if loose else response.strip()

        starts_with = kwargs.get("starts_with")
        if starts_with:
            check_start = starts_with.strip()
            if loose:
                if not target.lower().startswith(check_start.lower()):
                    return False, f"Response does not start with '{check_start}'"
            else:
                if not target.startswith(check_start):
                    return False, f"Response does not start with '{check_start}'"

        ends_with = kwargs.get("ends_with")
        if ends_with:
            check_end = ends_with.strip()
            if loose:
                if not target.lower().endswith(check_end.lower()):
                    return False, f"Response does not end with '{check_end}'"
            else:
                if not target.endswith(check_end):
                    return False, f"Response does not end with '{check_end}'"

        return True, "Start and end constraints satisfied"


class NumberedListChecker(InstructionChecker):
    """Verifies that response contains a numbered list with at least N items."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = clean_loose_text(response) if loose else response
        items = re.findall(r"(?m)^\s*\d+[\.\)]\s+", target)
        n = len(items)

        min_items = kwargs.get("min_items", 1)
        if n < min_items:
            return False, f"Found {n} numbered list items, expected at least {min_items}"

        exact_items = kwargs.get("exact_items")
        if exact_items is not None and n != exact_items:
            return False, f"Found {n} numbered list items, expected exactly {exact_items}"

        return True, f"Numbered list with {n} items satisfied"


class PostscriptChecker(InstructionChecker):
    """Verifies that response concludes with a postscript starting with P.S. or PS:."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = clean_loose_text(response)
        pattern = r"(?im)(P\.?S\.?[:\s]+.*)$"
        match = re.search(pattern, target)
        if not match:
            return False, "Response missing postscript marker (e.g. 'P.S.')"
        return True, "Postscript present"


class CapitalLettersChecker(InstructionChecker):
    """Verifies casing constraints: all uppercase, all lowercase, or count of capitalized words."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = clean_loose_text(response) if loose else response.strip()
        all_capital = kwargs.get("all_capital", False)
        all_lowercase = kwargs.get("all_lowercase", False)
        min_capital_words = kwargs.get("min_capital_words")

        if all_capital:
            alpha_chars = [c for c in target if c.isalpha()]
            if not alpha_chars or not all(c.isupper() for c in alpha_chars):
                return False, "Response is not entirely in uppercase characters"

        if all_lowercase:
            alpha_chars = [c for c in target if c.isalpha()]
            if not alpha_chars or not all(c.islower() for c in alpha_chars):
                return False, "Response is not entirely in lowercase characters"

        if min_capital_words is not None:
            words = re.findall(r"\b[A-Za-z0-9_]+\b", target)
            caps = [w for w in words if len(w) >= 2 and w.isalpha() and w.isupper()]
            if len(caps) < min_capital_words:
                return False, f"Found {len(caps)} uppercase words, expected at least {min_capital_words}"

        return True, "Casing constraint satisfied"


class QuotationChecker(InstructionChecker):
    """Verifies that the response is wrapped in quotation marks or contains quoted segments."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = clean_loose_text(response) if loose else response.strip()
        wrapped_in_quotes = kwargs.get("wrapped_in_quotes", False)
        exact_quotes = kwargs.get("exact_quotes", [])

        if wrapped_in_quotes:
            is_quoted = (
                (target.startswith('"') and target.endswith('"') and len(target) >= 2)
                or (target.startswith("'") and target.endswith("'") and len(target) >= 2)
                or (target.startswith("“") and target.endswith("”") and len(target) >= 2)
            )
            if not is_quoted:
                return False, "Response is not wrapped in quotation marks"

        if exact_quotes:
            found_quotes = set(re.findall(r'["“\']([^"”\']+)["”\']', target))
            missing = [q for q in exact_quotes if q not in found_quotes]
            if missing:
                return False, f"Missing required quoted phrases: {missing}"

        return True, "Quotation constraint satisfied"


class HighlightSectionChecker(InstructionChecker):
    """Verifies that designated sections or keywords are highlighted via Markdown asterisks."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = clean_loose_text(response) if loose else response
        keywords = kwargs.get("highlighted_keywords", [])
        min_highlighted = kwargs.get("min_highlighted")

        highlights = re.findall(r"\*{1,2}([^*\n]+)\*{1,2}", target)
        found_set = {h.strip().lower() for h in highlights if h.strip()}

        if keywords:
            missing = [k for k in keywords if k.strip().lower() not in found_set]
            if missing:
                return False, f"Missing highlighted markdown sections: {missing}"

        if min_highlighted is not None and len(highlights) < min_highlighted:
            return False, f"Found {len(highlights)} highlighted elements, expected at least {min_highlighted}"

        return True, "Highlighting constraint satisfied"


class TwoResponsesChecker(InstructionChecker):
    """Verifies that the response provides two distinct responses separated by a delimiter."""

    def check(self, response: str, kwargs: Dict[str, Any], loose: bool = False) -> Tuple[bool, str]:
        target = clean_loose_text(response) if loose else response
        delimiter = kwargs.get("delimiter", r"(?:\*{3,}|-{3,}|Response\s*2:)")

        parts = re.split(delimiter, target, maxsplit=2)
        if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
            p1 = re.search(r"Response\s*1:\s*(.+?)(?=Response\s*2:|$)", target, re.DOTALL | re.IGNORECASE)
            p2 = re.search(r"Response\s*2:\s*(.+?)$", target, re.DOTALL | re.IGNORECASE)
            if p1 and p2 and p1.group(1).strip() and p2.group(1).strip():
                return True, "Two labeled responses verified"
            return False, "Response does not contain two distinct responses separated by delimiter"

        return True, "Two distinct responses verified"


# Registry mapping rule names to checker implementations
CHECKER_REGISTRY: Dict[str, InstructionChecker] = {
    "json_format": JsonFormatChecker(),
    "word_count": WordCountChecker(),
    "sentence_count": SentenceCountChecker(),
    "paragraph_count": ParagraphCountChecker(),
    "keyword_presence": KeywordPresenceChecker(),
    "forbidden_words": ForbiddenWordsChecker(),
    "keyword_frequency": KeywordFrequencyChecker(),
    "section_headers": SectionHeaderChecker(),
    "start_end": StartEndConstraintChecker(),
    "numbered_list": NumberedListChecker(),
    "postscript": PostscriptChecker(),
    "letter_case": CapitalLettersChecker(),
    "quotation": QuotationChecker(),
    "highlight_section": HighlightSectionChecker(),
    "two_responses": TwoResponsesChecker(),
}


def verify_instruction(
    rule_type: str,
    response: str,
    kwargs: Dict[str, Any],
    loose: bool = False,
) -> Tuple[bool, str]:
    """Helper to verify a single rule against a response."""
    checker = CHECKER_REGISTRY.get(rule_type)
    if not checker:
        return False, f"Unknown instruction rule type: '{rule_type}'"
    return checker.check(response, kwargs, loose=loose)
