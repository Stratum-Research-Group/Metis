"""Reference ARC task implementation with synthetic 25-item dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dokime.tasks.protocol import Query


class ArcEasy25Task:
    """Reference task containing ~25 synthetic ARC-style multiple-choice questions."""

    task_id: str = "arc_easy_25"

    def __init__(self, dataset_path: Path | None = None) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "arc_easy_25.json"
        self._dataset_path = dataset_path

    @property
    def dataset_path(self) -> Path | None:
        return self._dataset_path

    def load(self) -> list[Query]:
        """Load 25 synthetic ARC queries from disk."""
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Task dataset not found at: {self._dataset_path}")

        with open(self._dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries: list[Query] = []
        for item in data:
            queries.append(
                Query(
                    id=str(item["id"]),
                    prompt=str(item["prompt"]),
                    choices=dict(item.get("choices", {})),
                    expected=str(item.get("expected", "")).strip().upper(),
                    metadata=item.get("metadata", {}),
                )
            )
        return queries

    def format_prompt(self, query: Query, template: str | None = None) -> str:
        """Format the multiple-choice prompt.

        If a template string is provided, supported placeholders are:
          {question} or {prompt}: the question text
          {choices}: formatted list of choices
          {id}: query id
        Otherwise, a standard multiple-choice format is returned.
        """
        choice_lines = [f"{key}: {val}" for key, val in sorted(query.choices.items())]
        choices_text = "\n".join(choice_lines)

        if template:
            # Check if template has placeholders
            formatted = template
            if "{choices}" in template:
                formatted = formatted.replace("{choices}", choices_text)
            if "{question}" in template:
                formatted = formatted.replace("{question}", query.prompt)
            if "{prompt}" in template:
                formatted = formatted.replace("{prompt}", query.prompt)
            if "{id}" in template:
                formatted = formatted.replace("{id}", query.id)
            if formatted == template and not any(k in template for k in ("{choices}", "{question}", "{prompt}")):
                # Prefix template to standard prompt
                return f"{template.strip()}\n\nQuestion: {query.prompt}\n{choices_text}\nAnswer:"
            return formatted

        return (
            f"Question: {query.prompt}\n"
            f"{choices_text}\n"
            f"Answer with only the correct choice letter (e.g. A, B, C, or D).\nAnswer:"
        )

    @staticmethod
    def _extract_from_clause(tail: str, choices: dict[str, str]) -> str | None:
        """Extract choice letter or unambiguous named choice from text following an explicit phrase."""
        after = tail.lstrip()
        if not after:
            return None

        # Strip optional choice/option prefix, e.g. "Choice A" -> "A"
        prefix_m = re.match(r"^(?:choice|option)\s*[:]?\s*", after, re.IGNORECASE)
        if prefix_m:
            after = after[prefix_m.end():].lstrip()

        # 1. Bracketed or markdown-formatted choice letter token: (A), [A], **A**, etc.
        bracketed_m = re.match(r"^[\(\[\*\'\"\`]+([A-D])[\)\]\*\'\"\`]+", after, re.IGNORECASE)
        if bracketed_m:
            cand = bracketed_m.group(1).upper()
            rest = after[bracketed_m.end():]
            if re.match(r"^\s*(?:or|and|/)\s*[\(\[\*\'\"\`]*([A-D])", rest, re.IGNORECASE):
                return None
            return cand if cand in choices else None

        # 2. Choice letter token [A-D]
        letter_m = re.match(r"^([A-D])(?![a-zA-Z])", after, re.IGNORECASE)
        if letter_m:
            cand = letter_m.group(1).upper()
            rest = after[letter_m.end():]

            # Check if this 'a' is an indefinite article followed by whitespace and a word
            if cand == "A" and re.match(r"^\s+[a-zA-Z]", rest):
                # Not a standalone letter token! Article followed by noun or named choice.
                # Where a named choice appears unambiguously, extract it instead; otherwise return None
                named_cands = []
                for k, text in choices.items():
                    raw = text.strip()
                    if raw and re.search(rf"(?<!\w){re.escape(raw)}(?!\w)", after, re.IGNORECASE):
                        named_cands.append(k.upper())
                if len(named_cands) == 1:
                    return named_cands[0]
                return None

            # Standalone letter token: check for immediate conflict (e.g. A or B)
            if re.match(r"^\s*(?:or|and|/)\s*(?:(?:choice|option)\s*)?[\(\[\*]*([A-D])", rest, re.IGNORECASE):
                return None
            return cand if cand in choices else None

        # 3. No letter token, look for unambiguous named choice in clause
        named_cands = []
        for k, text in choices.items():
            raw = text.strip()
            if raw and re.search(rf"(?<!\w){re.escape(raw)}(?!\w)", after, re.IGNORECASE):
                named_cands.append(k.upper())
        if len(named_cands) == 1:
            return named_cands[0]

        return None

    def extract_prediction(self, query: Query, response: str) -> str | None:
        """Extract choice letter (A-D) from model response.

        Parses explicit answer syntax (e.g. bare letter, 'Answer: B',
        'The answer is (B)', 'Choice B', '(B)', or unambiguous named choice text).
        Ensures choice letters after explicit-answer phrases are unambiguous standalone
        letter tokens, not indefinite articles followed by nouns or named choices.
        Returns None for ambiguous or conflicting answers unless resolved by an
        explicit final-answer declaration.
        """
        if not response or not response.strip():
            return None

        clean = response.strip()

        # 1. Bare single-letter or bracketed response (e.g. "B", "(B)", "[B]", "**B**", "B.", "B:")
        bare_match = re.fullmatch(
            r"^[\s\(\[\{\*\_\"\'\`]*([A-Za-z])[\s\)\]\}\*\_\.\:\;\"\'\`]*$",
            clean,
        )
        if bare_match:
            letter = bare_match.group(1).upper()
            if letter in query.choices:
                return letter

        # 2. Explicit final answer rule
        # e.g. "Final answer: B", "The final answer is (B)", "My final answer is a blue object."
        pat_final = re.compile(
            r"(?:(?:the|my)\s+)?final\s+answer\s*(?:is\s*:?|:)\s*",
            re.IGNORECASE,
        )
        final_matches = list(pat_final.finditer(clean))
        if final_matches:
            final_cands = set()
            for m in final_matches:
                tail = clean[m.end():]
                cand = self._extract_from_clause(tail, query.choices)
                if cand:
                    final_cands.add(cand)
            if len(final_cands) == 1:
                return list(final_cands)[0]
            elif len(final_cands) > 1:
                return None
            return None

        # Check for general disjunction / conflict between explicit choices (e.g. "A or B")
        pat_disj = re.compile(
            r"(?:\b|[\(\[\*])([A-D])[\)\]\*]*\s*(?:or|and|/|vs\.?)\s*(?:(?:choice|option)\s*)?[\(\[\*]*([A-D])[\)\]\*]*(?!\w)",
            re.IGNORECASE,
        )
        disj_m = pat_disj.search(clean)
        if disj_m:
            c1, c2 = disj_m.group(1).upper(), disj_m.group(2).upper()
            if c1 in query.choices and c2 in query.choices and c1 != c2:
                return None

        # 3. Explicit answer declarations
        # e.g. "Answer: B", "The answer is (B)", "The correct answer is B", "Answer: a blue object."
        pat_answer = re.compile(
            r"(?:(?:the\s+)?(?:correct|best)\s+)?answer\s*(?:is\s*:?|:)\s*",
            re.IGNORECASE,
        )
        answer_matches = list(pat_answer.finditer(clean))
        if answer_matches:
            answer_cands = set()
            has_article_unknown = False
            for m in answer_matches:
                tail = clean[m.end():]
                cand = self._extract_from_clause(tail, query.choices)
                if cand:
                    answer_cands.add(cand)
                else:
                    after = tail.lstrip()
                    if re.match(r"^[aA]\s+[a-zA-Z]", after):
                        has_article_unknown = True
            if len(answer_cands) == 1 and not has_article_unknown:
                return list(answer_cands)[0]
            elif len(answer_cands) > 1 or has_article_unknown:
                return None

        # 4. Explicit choice / option declarations
        # e.g. "Choice B", "Choice: B", "Option B", "Option: a blue object."
        pat_choice = re.compile(
            r"\b(?:choice|option)\s*(?:is\s*:?|:)?\s*",
            re.IGNORECASE,
        )
        choice_matches = list(pat_choice.finditer(clean))
        if choice_matches:
            choice_cands = set()
            has_article_unknown = False
            for m in choice_matches:
                tail = clean[m.end():]
                cand = self._extract_from_clause(tail, query.choices)
                if cand:
                    choice_cands.add(cand)
                else:
                    after = tail.lstrip()
                    if re.match(r"^[aA]\s+[a-zA-Z]", after):
                        has_article_unknown = True
            if len(choice_cands) == 1 and not has_article_unknown:
                return list(choice_cands)[0]
            elif len(choice_cands) > 1 or has_article_unknown:
                return None

        # 5. Parenthesized / bracketed choice letter
        # e.g. "(B)", "[B]"
        pat_paren = r"(?:^|[\s,;:\.])[\(\[]([A-D])[\)\]](?=[\s,;:\.]|$)"
        paren_matches = re.findall(pat_paren, clean, re.IGNORECASE)
        if paren_matches:
            paren_cands = {m.upper() for m in paren_matches if m.upper() in query.choices}
            if len(paren_cands) == 1:
                return list(paren_cands)[0]
            elif len(paren_cands) > 1:
                return None

        # 6. Unambiguous line-starting choice indicator
        # e.g. "B) Sound energy..." or "B. Sound energy..." or "B: Sound energy..."
        pat_prefix = r"(?:^|\n)\s*([A-D])[\)\.\:\-]\s+"
        prefix_matches = re.findall(pat_prefix, clean, re.IGNORECASE)
        if prefix_matches:
            prefix_cands = {m.upper() for m in prefix_matches if m.upper() in query.choices}
            if len(prefix_cands) == 1:
                return list(prefix_cands)[0]
            elif len(prefix_cands) > 1:
                return None

        # 7. Unambiguous named choice text
        # e.g. "A blue object is the answer."
        matching_named_keys: list[str] = []
        for key, text in query.choices.items():
            raw_text = text.strip()
            if not raw_text:
                continue
            escaped = re.escape(raw_text)
            pattern = rf"(?<!\w){escaped}(?!\w)"
            if re.search(pattern, clean, re.IGNORECASE):
                matching_named_keys.append(key.upper())

        if len(matching_named_keys) == 1:
            return matching_named_keys[0]

        return None

    def score(self, query: Query, response: str) -> float:
        """Score response: 1.0 if prediction matches expected, else 0.0."""
        pred = self.extract_prediction(query, response)
        if pred is not None and pred == query.expected.strip().upper():
            return 1.0
        return 0.0
