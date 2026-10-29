"""Tests for IFEval programmatic instruction checkers and evaluation task."""

import json
from pathlib import Path
import pytest

from dokime.tasks.ifeval import IFEvalTask
from dokime.tasks.ifeval_checkers import (
    CHECKER_REGISTRY,
    AllLowercaseChecker,
    BulletListChecker,
    CapitalLettersChecker,
    ForbiddenWordsChecker,
    HighlightSectionChecker,
    JsonFormatChecker,
    KeywordFrequencyChecker,
    KeywordPresenceChecker,
    LetterFrequencyChecker,
    NoCommaChecker,
    NumberedListChecker,
    ParagraphCountChecker,
    PostscriptChecker,
    QuotationChecker,
    SectionHeaderChecker,
    SentenceCountChecker,
    StartEndConstraintChecker,
    TitleWrapChecker,
    TwoResponsesChecker,
    WordCountChecker,
    verify_instruction,
)
from dokime.tasks.protocol import Query, Task
from dokime.tasks.registry import discover_tasks, get_task, list_task_info


# -------------------------------------------------------------------------
# Unit Tests for Individual Checkers
# -------------------------------------------------------------------------

def test_json_format_checker():
    checker = JsonFormatChecker()

    # Strict valid JSON
    valid_json = '{"component": "kernel", "role": "scheduler", "example": "linux"}'
    ok, _ = checker.check(valid_json, {"required_keys": ["component", "role", "example"]}, loose=False)
    assert ok is True

    # Missing required keys
    ok_missing, msg = checker.check(valid_json, {"required_keys": ["component", "missing_key"]}, loose=False)
    assert ok_missing is False
    assert "missing required keys" in msg

    # Invalid JSON
    ok_inv, _ = checker.check("Not JSON at all", {}, loose=False)
    assert ok_inv is False

    # Loose mode strips markdown fences
    markdown_wrapped = '```json\n{"component": "kernel", "role": "scheduler", "example": "linux"}\n```'
    ok_loose, _ = checker.check(markdown_wrapped, {"required_keys": ["component"]}, loose=True)
    assert ok_loose is True


def test_word_count_checker():
    checker = WordCountChecker()
    text = "The quick brown fox jumps over the lazy dog."  # 9 words

    # Max words
    ok, _ = checker.check(text, {"max_words": 15})
    assert ok is True
    fail_max, _ = checker.check(text, {"max_words": 5})
    assert fail_max is False

    # Min words
    ok_min, _ = checker.check(text, {"min_words": 5})
    assert ok_min is True
    fail_min, _ = checker.check(text, {"min_words": 20})
    assert fail_min is False

    # Exact words
    ok_exact, _ = checker.check(text, {"exact_words": 9})
    assert ok_exact is True
    fail_exact, _ = checker.check(text, {"exact_words": 10}, loose=False)
    assert fail_exact is False


def test_sentence_count_checker():
    checker = SentenceCountChecker()
    text = "First sentence here. Second sentence follows! Is this the third sentence?"

    ok_exact, _ = checker.check(text, {"exact_sentences": 3})
    assert ok_exact is True

    fail_exact, _ = checker.check(text, {"exact_sentences": 2})
    assert fail_exact is False

    ok_min, _ = checker.check(text, {"min_sentences": 2})
    assert ok_min is True


def test_paragraph_count_checker():
    checker = ParagraphCountChecker()
    text = "Paragraph one content.\n\nParagraph two content.\n\nParagraph three content."

    ok, _ = checker.check(text, {"exact_paragraphs": 3})
    assert ok is True

    fail, _ = checker.check(text, {"exact_paragraphs": 2})
    assert fail is False


def test_keyword_presence_checker():
    checker = KeywordPresenceChecker()
    text = "Public-key cryptography provides asymmetric encryption and digital signatures."

    # Case-insensitive pass
    ok, _ = checker.check(text, {"keywords": ["asymmetric", "encryption", "digital"], "case_sensitive": False})
    assert ok is True

    # Missing keyword
    fail, msg = checker.check(text, {"keywords": ["asymmetric", "blockchain"], "case_sensitive": False})
    assert fail is False
    assert "blockchain" in msg


def test_forbidden_words_checker():
    checker = ForbiddenWordsChecker()
    text = "Garbage collection automatically frees unreferenced heap allocations."

    # Clean pass
    ok, _ = checker.check(text, {"forbidden_words": ["pointer", "leak", "manual"], "case_sensitive": False})
    assert ok is True

    # Fails when forbidden word is present
    fail_text = "This algorithm prevents memory leaks."
    fail, msg = checker.check(fail_text, {"forbidden_words": ["leak"], "case_sensitive": False})
    assert fail is False
    assert "leak" in msg

    # Word boundary guard: 'leakage' should not trigger exact '\bleak\b' if word boundary is respected
    ok_bound, _ = checker.check("System prevented leakage.", {"forbidden_words": ["leak"], "case_sensitive": False})
    assert ok_bound is True


def test_keyword_frequency_checker():
    checker = KeywordFrequencyChecker()
    text = "A fault in subsystem A caused another fault, resulting in a cascade fault."

    ok, _ = checker.check(text, {"keyword": "fault", "exact_frequency": 3})
    assert ok is True

    fail, _ = checker.check(text, {"keyword": "fault", "exact_frequency": 2})
    assert fail is False


def test_section_headers_checker():
    checker = SectionHeaderChecker()
    text = "## Incident Overview\nThe service went down.\n\n## Corrective Actions\nPatched the vulnerability."

    ok, _ = checker.check(text, {"headers": ["## Incident Overview", "## Corrective Actions"]})
    assert ok is True

    fail, _ = checker.check(text, {"headers": ["## Incident Overview", "## Root Cause"]})
    assert fail is False


def test_start_end_constraint_checker():
    checker = StartEndConstraintChecker()
    text = "STATUS_CONFIRMED: System initialized properly. VERIFICATION_COMPLETE."

    ok, _ = checker.check(text, {"starts_with": "STATUS_CONFIRMED:", "ends_with": "VERIFICATION_COMPLETE."})
    assert ok is True

    fail_start, _ = checker.check(text, {"starts_with": "ERROR:"})
    assert fail_start is False

    fail_end, _ = checker.check(text, {"ends_with": "DONE."})
    assert fail_end is False


def test_numbered_list_checker():
    checker = NumberedListChecker()
    text = (
        "Deployment steps:\n"
        "1. Build the binary\n"
        "2. Run unit tests\n"
        "3. Deploy container\n"
        "4. Verify health check\n"
    )

    ok_min, _ = checker.check(text, {"min_items": 4})
    assert ok_min is True

    ok_exact, _ = checker.check(text, {"exact_items": 4})
    assert ok_exact is True

    fail_exact, _ = checker.check(text, {"exact_items": 5})
    assert fail_exact is False


def test_postscript_checker():
    checker = PostscriptChecker()
    text = "Zero-trust requires continuous authentication.\n\nP.S. Refer to NIST SP 800-207 for guidelines."

    ok, _ = checker.check(text, {})
    assert ok is True

    fail, _ = checker.check("Standard message with no postscript attached.", {})
    assert fail is False


def test_capital_letters_checker():
    checker = CapitalLettersChecker()

    # All capital
    ok_caps, _ = checker.check("EMERGENCY FAILOVER IN PROGRESS", {"all_capital": True})
    assert ok_caps is True

    fail_caps, _ = checker.check("Emergency failover in progress", {"all_capital": True})
    assert fail_caps is False

    # All lowercase
    ok_lower, _ = checker.check("all lower text without any caps", {"all_lowercase": True})
    assert ok_lower is True

    fail_lower, _ = checker.check("All lower text with Cap", {"all_lowercase": True})
    assert fail_lower is False

    # Min capital words
    ok_min, _ = checker.check("The CPU and GPU must synchronize", {"min_capital_words": 2})
    assert ok_min is True

    fail_min, _ = checker.check("The CPU must synchronize", {"min_capital_words": 2})
    assert fail_min is False


def test_quotation_checker():
    checker = QuotationChecker()

    # Wrapped in double quotes
    ok_wrapped, _ = checker.check('"An operation is idempotent if f(f(x)) = f(x)."', {"wrapped_in_quotes": True})
    assert ok_wrapped is True

    fail_wrapped, _ = checker.check('An operation without wrapping quotes.', {"wrapped_in_quotes": True})
    assert fail_wrapped is False

    # Exact quotes
    text = 'Security guidelines require "zero trust" and "least privilege" controls.'
    ok_exact, _ = checker.check(text, {"exact_quotes": ["zero trust", "least privilege"]})
    assert ok_exact is True

    fail_exact, _ = checker.check(text, {"exact_quotes": ["zero trust", "defense in depth"]})
    assert fail_exact is False


def test_highlight_section_checker():
    checker = HighlightSectionChecker()
    text = "Linux supports *epoll* while BSD uses *kqueue* for event demuxing."

    ok_hi, _ = checker.check(text, {"highlighted_keywords": ["epoll", "kqueue"]})
    assert ok_hi is True

    fail_hi, _ = checker.check(text, {"highlighted_keywords": ["epoll", "iocp"]})
    assert fail_hi is False


def test_two_responses_checker():
    checker = TwoResponsesChecker()

    # Delimited by ***
    text = (
        "Microservices introduce distributed consensus challenges.\n"
        "***\n"
        "Alternatively, network partitioning degrades latency budgets."
    )
    ok_delim, _ = checker.check(text, {"delimiter": r"\*\*\*"})
    assert ok_delim is True

    fail_delim, _ = checker.check("Only one single response provided.", {"delimiter": r"\*\*\*"})
    assert fail_delim is False

    # Labeled Response 1 and Response 2
    labeled = (
        "Response 1: Monoliths simplify transactional consistency.\n"
        "Response 2: Microservices offer team deployment autonomy."
    )
    ok_labeled, _ = checker.check(labeled, {})
    assert ok_labeled is True


# -------------------------------------------------------------------------
# Integration Tests for IFEvalTask
# -------------------------------------------------------------------------

def test_ifeval_task_load_and_properties():
    task = IFEvalTask()
    assert task.task_id == "ifeval_instruction_following"
    assert task.dataset_path is not None
    assert task.dataset_path.exists()

    queries = task.load()
    assert len(queries) == 30
    assert all(isinstance(q, Query) for q in queries)
    assert queries[0].id == "ifeval-01"
    assert "rules" in queries[0].metadata


def test_ifeval_task_score_satisfaction():
    task = IFEvalTask()
    queries = task.load()
    q_mutex = next(q for q in queries if q.id == "ifeval-02")

    # Correct response (exactly 2 sentences)
    correct_resp = "A mutex is a synchronization primitive that grants exclusive resource access to one thread. It prevents race conditions during concurrent execution."
    assert task.score(q_mutex, correct_resp) == 1.0

    # Incorrect response (1 sentence)
    incorrect_resp = "A mutex prevents concurrent threads from accessing shared state simultaneously."
    assert task.score(q_mutex, incorrect_resp) == 0.0

    # Empty response
    assert task.score(q_mutex, "") == 0.0


def test_ifeval_task_score_detailed():
    task = IFEvalTask()
    queries = task.load()
    q_json = next(q for q in queries if q.id == "ifeval-01")

    # Wrapped in markdown
    resp = "```json\n{\"component\": \"kernel\", \"role\": \"resource manager\", \"example\": \"Linux\"}\n```"
    detailed = task.score_detailed(q_json, resp)

    assert detailed["query_id"] == "ifeval-01"
    assert detailed["loose_prompt_pass"] is True
    assert detailed["loose_rule_accuracy"] == 1.0


def test_ifeval_task_score_new_suite_items():
    task = IFEvalTask()
    queries = {q.id: q for q in task.load()}

    # ifeval-26 (no_comma + sentence_count >= 2)
    q26 = queries["ifeval-26"]
    valid_26 = "Write-ahead logging guarantees durability in storage systems. Changes are appended to an immutable disk log before applying updates."
    invalid_26 = "Write-ahead logging guarantees durability, and it ensures consistency."
    assert task.score(q26, valid_26) == 1.0
    assert task.score(q26, invalid_26) == 0.0

    # ifeval-27 (bullet_list exact 4)
    q27 = queries["ifeval-27"]
    valid_27 = "* Early compile-time bug detection\n* Improved refactoring confidence\n* Machine-verifiable code documentation\n* Enhanced IDE autocompletion and navigation"
    invalid_27 = "* Early compile-time bug detection\n* Improved refactoring confidence"
    assert task.score(q27, valid_27) == 1.0
    assert task.score(q27, invalid_27) == 0.0

    # ifeval-28 (all_lowercase)
    q28 = queries["ifeval-28"]
    valid_28 = "tail call optimization reuses the active stack frame when the final action of a function is a recursive invocation."
    invalid_28 = "Tail call optimization reuses the active stack frame."
    assert task.score(q28, valid_28) == 1.0
    assert task.score(q28, invalid_28) == 0.0

    # ifeval-29 (title_wrap angle_brackets)
    q29 = queries["ifeval-29"]
    valid_29 = "<<Zero-Copy Networking>>\nZero-copy networking transmits packet buffers directly between hardware network interfaces and application memory."
    invalid_29 = "Zero-copy networking transmits packet buffers directly between hardware network interfaces."
    assert task.score(q29, valid_29) == 1.0
    assert task.score(q29, invalid_29) == 0.0

    # ifeval-30 (letter_frequency 'h' >= 8 and word_count <= 60)
    q30 = queries["ifeval-30"]
    valid_30 = "Consistent hashing hashes cache keys onto a continuous hash ring. Each hash node handles hashes within its designated hash horizon. This healthy hashing approach smoothly shifts cached shards when nodes halt or join."
    invalid_30 = "Consistent hashing balances load across nodes."
    assert task.score(q30, valid_30) == 1.0
    assert task.score(q30, invalid_30) == 0.0


def test_ifeval_registry_discovery():
    available = discover_tasks()
    assert "ifeval_instruction_following" in available

    task = get_task("ifeval_instruction_following")
    assert isinstance(task, Task)
    assert task.task_id == "ifeval_instruction_following"

    info = list_task_info()
    ifeval_info = next(item for item in info if item["task_id"] == "ifeval_instruction_following")
    assert ifeval_info["items"] >= 25


def test_no_comma_checker():
    checker = NoCommaChecker()
    valid = "Operating systems allocate memory and schedule processes on physical cores."
    invalid = "Operating systems allocate memory, schedule processes, and handle interrupts."

    ok, _ = checker.check(valid, {})
    assert ok is True

    ok, msg = checker.check(invalid, {})
    assert ok is False
    assert "Found 2 commas" in msg

    # With max_commas allowance
    ok, _ = checker.check(invalid, {"max_commas": 2})
    assert ok is True


def test_bullet_list_checker():
    checker = BulletListChecker()
    valid = "Key components:\n* Memory management\n* CPU scheduler\n* File system"
    invalid = "Key components:\n1. Memory management\n2. CPU scheduler"

    ok, msg = checker.check(valid, {"exact_count": 3})
    assert ok is True
    assert "3 items" in msg

    ok, msg = checker.check(valid, {"min_count": 2})
    assert ok is True

    ok, msg = checker.check(invalid, {"exact_count": 2})
    assert ok is False


def test_all_lowercase_checker():
    checker = AllLowercaseChecker()
    valid = "this text is completely lowercase and contains no uppercase characters."
    invalid = "This text contains Uppercase Characters."

    ok, _ = checker.check(valid, {})
    assert ok is True

    ok, msg = checker.check(invalid, {})
    assert ok is False
    assert "Found uppercase character" in msg


def test_title_wrap_checker():
    checker = TitleWrapChecker()
    valid_angle = "<<Operating System Architecture>>\nThis overview explains microkernels."
    valid_markdown = "# Operating System Architecture\nThis overview explains microkernels."
    invalid = "Operating System Architecture\nThis overview explains microkernels."

    ok, msg = checker.check(valid_angle, {"wrapper": "angle_brackets"})
    assert ok is True
    assert "<<Operating System Architecture>>" in msg

    ok, msg = checker.check(valid_markdown, {"wrapper": "markdown"})
    assert ok is True
    assert "# Operating System Architecture" in msg

    ok, msg = checker.check(invalid, {"wrapper": "angle_brackets"})
    assert ok is False


def test_letter_frequency_checker():
    checker = LetterFrequencyChecker()
    text = "The quick brown fox jumps over the lazy dog."  # 'o' appears 4 times (brown, fox, over, dog)

    ok, _ = checker.check(text, {"letter": "o", "exact_frequency": 4})
    assert ok is True

    ok, _ = checker.check(text, {"letter": "o", "min_frequency": 3})
    assert ok is True

    ok, msg = checker.check(text, {"letter": "o", "exact_frequency": 2})
    assert ok is False
    assert "Found character 'o' 4 times, expected exactly 2" in msg

