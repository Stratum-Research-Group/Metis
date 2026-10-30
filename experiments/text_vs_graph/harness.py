"""Prompt formatting, model execution harness, and answer parsing.

Per test plan Section 13:
Prompt templates, output schema, answer parsing, and exact-match scoring.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple
import urllib.request
import urllib.error

from experiments.text_vs_graph.models import Question, Story

PASS1_TOKEN_CAP = 250

ANSWER_PASS_SYSTEM_BASE = """You answer questions from the evidence provided. Reason briefly if needed, then end with a single JSON object {"answer": <value>}. Use only the stated facts and rules.
Dependency rules: an event with rule AND occurs only if all its listed parents occur; with rule OR it occurs if at least one parent occurs; an event with no parents occurs unless explicitly denied.
Ordering questions use timestamps only."""

TEXT_AUTHORITY_RULE = 'If the structured facts conflict with the text, treat the text as correct.'

PASS1_SYSTEM_PROMPT = f"""You are preparing notes for a later answer. Do not answer the question. Write at most {PASS1_TOKEN_CAP} tokens."""


def get_answer_pass_system_prompt(arm: str) -> str:
    """Returns system prompt for the final answer pass.
    
    Arms C, D pass 2, and F pass 2 include the text-authority rule.
    Arms A and B omit it because they only present one evidence source.
    """
    if arm in ("C", "D", "F", "E"):
        return f"{ANSWER_PASS_SYSTEM_BASE}\n\n{TEXT_AUTHORITY_RULE}"
    return ANSWER_PASS_SYSTEM_BASE


def format_user_prompt(
    story: Story,
    question: Question,
    arm: Literal["A", "B", "C", "D", "F", "E"],
    graph_block: str,
    pass_number: int = 2,
    pass1_notes: Optional[str] = None,
) -> Tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the specified arm and pass."""
    
    format_instr = f"Format required: {question.answer_format}. Final line must be JSON: {{\"answer\": <value>}}."
    
    if arm == "A":
        sys_p = get_answer_pass_system_prompt("A")
        user_p = (
            f"EVIDENCE (prose):\n{story.prose}\n\n"
            f"QUESTION: {question.prompt}\n"
            f"{format_instr}"
        )
        return sys_p, user_p
        
    elif arm == "B":
        sys_p = get_answer_pass_system_prompt("B")
        user_p = (
            f"EVIDENCE (structured facts):\n{graph_block}\n\n"
            f"QUESTION: {question.prompt}\n"
            f"{format_instr}"
        )
        return sys_p, user_p
        
    elif arm in ("C", "E"):
        sys_p = get_answer_pass_system_prompt(arm)
        user_p = (
            f"TEXT:\n{story.prose}\n\n"
            f"STRUCTURED FACTS:\n{graph_block}\n\n"
            f"QUESTION: {question.prompt}\n"
            f"{format_instr}"
        )
        return sys_p, user_p
        
    elif arm == "D":
        if pass_number == 1:
            sys_p = PASS1_SYSTEM_PROMPT
            user_p = (
                f"TEXT:\n{story.prose}\n\n"
                f"STRUCTURED FACTS:\n{graph_block}\n\n"
                f"QUESTION: {question.prompt}\n\n"
                f"Task: list each structured fact that the text does not support or contradicts, "
                f"as a JSON list of fact ids, plus brief reasons."
            )
            return sys_p, user_p
        else:
            sys_p = get_answer_pass_system_prompt("D")
            notes = pass1_notes or "No discrepancies noted."
            user_p = (
                f"TEXT:\n{story.prose}\n\n"
                f"STRUCTURED FACTS:\n{graph_block}\n\n"
                f"Notes from the earlier check:\n{notes}\n\n"
                f"QUESTION: {question.prompt}\n"
                f"{format_instr}"
            )
            return sys_p, user_p
            
    elif arm == "F":
        if pass_number == 1:
            sys_p = PASS1_SYSTEM_PROMPT
            user_p = (
                f"TEXT:\n{story.prose}\n\n"
                f"STRUCTURED FACTS:\n{graph_block}\n\n"
                f"QUESTION: {question.prompt}\n\n"
                f"Task: restate the facts relevant to this question and your plan for answering."
            )
            return sys_p, user_p
        else:
            sys_p = get_answer_pass_system_prompt("F")
            notes = pass1_notes or "No notes provided."
            user_p = (
                f"TEXT:\n{story.prose}\n\n"
                f"STRUCTURED FACTS:\n{graph_block}\n\n"
                f"Notes:\n{notes}\n\n"
                f"QUESTION: {question.prompt}\n"
                f"{format_instr}"
            )
            return sys_p, user_p

    raise ValueError(f"Unknown arm: {arm}")


def parse_model_answer(raw_text: str, answer_format: str) -> Tuple[bool, Any]:
    """Parses model response to extract the JSON answer.
    
    Returns (parse_ok, parsed_value).
    """
    raw_text = raw_text.strip()
    
    # Attempt 1: Search for JSON object {"answer": ...}
    json_candidates = re.findall(r"\{[^{}]*\"answer\"\s*:[^{}]*\}", raw_text, flags=re.DOTALL)
    val = None
    parse_ok = False
    
    for candidate in reversed(json_candidates):
        try:
            data = json.loads(candidate)
            if "answer" in data:
                val = data["answer"]
                parse_ok = True
                break
        except Exception:
            continue
            
    # Attempt 2: Last line or code block containing json
    if not parse_ok:
        code_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
        for cb in reversed(code_blocks):
            try:
                data = json.loads(cb)
                if "answer" in data:
                    val = data["answer"]
                    parse_ok = True
                    break
            except Exception:
                continue

    # Attempt 3: Direct regex match on "answer": <val>
    if not parse_ok:
        direct = re.search(r"\"answer\"\s*:\s*(\[[^\]]*\]|\"[^\"]*\"|\d+|[a-zA-Z0-9_]+)", raw_text)
        if direct:
            raw_val = direct.group(1).strip()
            try:
                val = json.loads(raw_val)
                parse_ok = True
            except Exception:
                val = raw_val.strip("\"'")
                parse_ok = True

    if not parse_ok:
        return False, None

    # Normalization by answer format
    try:
        if answer_format == "event_id":
            if isinstance(val, str):
                # Extract event token like E1, E2
                m = re.search(r"\b(E\d+)\b", val)
                return True, m.group(1) if m else val.strip().upper()
            return True, str(val).strip().upper()

        elif answer_format == "integer":
            if isinstance(val, int):
                return True, val
            if isinstance(val, (float, str)):
                m = re.search(r"-?\d+", str(val))
                if m:
                    return True, int(m.group(0))
            return False, None

        elif answer_format == "id_list":
            if isinstance(val, list):
                norm_list = sorted([str(item).strip().upper() for item in val if str(item).strip()])
                return True, norm_list
            elif isinstance(val, str):
                # Try parsing string list or tokens
                tokens = re.findall(r"\bE\d+\b", val.upper())
                return True, sorted(list(set(tokens)))
            return False, None

        elif answer_format == "value":
            return True, str(val).strip()

    except Exception:
        return False, None

    return parse_ok, val


def score_answer(parsed: Any, gold: Any, answer_format: str) -> bool:
    """Exact-match comparison between parsed output and gold answer."""
    if parsed is None or gold is None:
        return False
        
    if answer_format == "event_id":
        return str(parsed).strip().upper() == str(gold).strip().upper()
        
    elif answer_format == "integer":
        try:
            return int(parsed) == int(gold)
        except Exception:
            return False
            
    elif answer_format == "id_list":
        if not isinstance(parsed, list) or not isinstance(gold, list):
            return False
        return sorted([str(x).upper() for x in parsed]) == sorted([str(x).upper() for x in gold])
        
    elif answer_format == "value":
        return str(parsed).strip().lower() == str(gold).strip().lower()
        
    return False


def call_local_hestia_chat(
    system_prompt: str,
    user_prompt: str,
    model: str = "gemma-4-26b-a4b-abliterated",
    endpoint: str = "http://127.0.0.1:8080/v1/chat/completions",
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Dispatches a chat completion call to local Hestia endpoint."""
    t0 = time.perf_counter()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t0
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {
                "success": True,
                "content": content,
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "latency_seconds": elapsed,
            }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "success": False,
            "error": str(e),
            "content": "",
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_seconds": elapsed,
        }

