"""Dokime evaluation task for Visually-Rich Document Understanding (VRDU).

Inspired by Google Research VRDU (arXiv:2211.15416) and FormNet / DocKu benchmarks.
Evaluates structured extraction from dense semi-structured documents (invoices,
purchase orders, clinical protocols, balance sheets, patents, lab specs, lease agreements)
against ground truth entity graphs and target key-value schemas.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dokime.tasks.protocol import Query


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract JSON object from LLM response text."""
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Case 1: Markdown code block ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Case 2: Direct JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Case 3: Find outermost { ... }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _normalize_string(val: str) -> str:
    """Normalize string for fuzzy robust field matching."""
    s = val.lower().strip()
    # Strip common punctuation / currency symbols
    s = re.sub(r"[\$€£,;\"']", "", s)
    # Collapse multiple whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _normalize_date(val: str) -> str:
    """Normalize dates in YYYY-MM-DD or Month DD, YYYY formats."""
    s = val.strip()
    # If already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s

    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10",
        "nov": "11", "dec": "12",
    }
    # Pattern: October 14, 2026 or 14 October 2026
    m1 = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$", s)
    if m1:
        month_str, day_str, year_str = m1.group(1).lower(), m1.group(2).zfill(2), m1.group(3)
        if month_str in months:
            return f"{year_str}-{months[month_str]}-{day_str}"

    m2 = re.match(r"^(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})$", s)
    if m2:
        day_str, month_str, year_str = m2.group(1).zfill(2), m2.group(2).lower(), m2.group(3)
        if month_str in months:
            return f"{year_str}-{months[month_str]}-{day_str}"

    return s


def _values_match(pred: Any, target: Any) -> bool:
    """Check if predicted value matches target value with flexible type casting and normalization."""
    if pred is None and target is None:
        return True
    if pred is None or target is None:
        return False

    # Numeric comparison
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        try:
            pred_num = float(str(pred).replace(",", "").replace("$", "").replace("€", "").strip())
            target_num = float(target)
            if math.isclose(pred_num, target_num, rel_tol=0.01, abs_tol=0.01):
                return True
        except (ValueError, TypeError):
            pass

    # Boolean comparison
    if isinstance(target, bool):
        if isinstance(pred, bool):
            return pred == target
        return str(pred).lower() == str(target).lower()

    # List comparison
    if isinstance(target, list):
        if not isinstance(pred, list):
            return False
        if len(target) == 0:
            return len(pred) == 0
        # If list of dicts (line items)
        if isinstance(target[0], dict):
            # Evaluate how many target items have a match in pred
            matched_count = 0
            unmatched_preds = list(pred)
            for t_item in target:
                found = False
                for p_item in unmatched_preds:
                    if isinstance(p_item, dict) and _dict_match_ratio(p_item, t_item) >= 0.7:
                        matched_count += 1
                        unmatched_preds.remove(p_item)
                        found = True
                        break
            return (matched_count / len(target)) >= 0.6
        else:
            # List of primitives
            norm_target = [_normalize_string(str(x)) for x in target]
            norm_pred = [_normalize_string(str(x)) for x in pred]
            return set(norm_target) == set(norm_pred) or (
                len(norm_target) > 0 and len(set(norm_target).intersection(norm_pred)) / len(norm_target) >= 0.8
            )

    # Dict comparison
    if isinstance(target, dict):
        if not isinstance(pred, dict):
            return False
        return _dict_match_ratio(pred, target) >= 0.7

    # Date normalization check
    t_str = str(target).strip()
    p_str = str(pred).strip()
    if _normalize_date(p_str) == _normalize_date(t_str):
        return True

    # String normalization and partial containment
    norm_t = _normalize_string(t_str)
    norm_p = _normalize_string(p_str)

    if norm_t == norm_p:
        return True

    # Substring inclusion for entity names (e.g. "Global Foundries Fab-21" vs "Global Foundries Fab-21, Phoenix AZ")
    if len(norm_t) >= 4 and len(norm_p) >= 4:
        if norm_t in norm_p or norm_p in norm_t:
            return True

    return False


def _dict_match_ratio(pred: Dict[str, Any], target: Dict[str, Any]) -> float:
    """Compute key-value match ratio for nested dictionaries."""
    if not target:
        return 1.0 if not pred else 0.0

    matches = 0
    for k, v in target.items():
        if k in pred and _values_match(pred[k], v):
            matches += 1
    return matches / len(target)


class VRDUTask:
    """Visually-Rich Document Understanding extraction benchmark task."""

    task_id: str = "vrdu_document_extraction"

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "vrdu_suite.json"
        self._dataset_path = dataset_path

    @property
    def dataset_path(self) -> Optional[Path]:
        return self._dataset_path

    def load(self) -> List[Query]:
        """Load document extraction benchmark queries from disk."""
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"VRDU dataset not found at: {self._dataset_path}")

        with open(self._dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries: List[Query] = []
        for item in data:
            queries.append(
                Query(
                    id=str(item["id"]),
                    prompt=str(item["prompt"]),
                    expected=json.dumps(item.get("target_fields", {}), indent=2),
                    metadata={
                        "target_fields": item.get("target_fields", {}),
                        "id": item["id"],
                    },
                )
            )
        return queries

    def format_prompt(self, query: Query, template: str | None = None) -> str:
        """Format the document extraction prompt."""
        if template:
            return template.replace("{question}", query.prompt).replace("{prompt}", query.prompt)
        return f"{query.prompt}\n\nRespond ONLY with valid JSON containing the requested fields."

    def extract_prediction(self, query: Query, response: str) -> Optional[str]:
        """Extract prediction JSON string or raw text."""
        extracted = _extract_json_from_text(response)
        if extracted is not None:
            return json.dumps(extracted, sort_keys=True)
        return response.strip() if response else None

    def evaluate_extraction(self, query: Query, response: str) -> Dict[str, Any]:
        """Compute precision, recall, and F1 across target fields."""
        target_fields = query.metadata.get("target_fields", {})
        extracted = _extract_json_from_text(response)

        if not target_fields:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "field_matches": {},
                "extracted": extracted,
            }

        if extracted is None:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "field_matches": {k: False for k in target_fields},
                "extracted": None,
                "error": "Failed to parse JSON from response",
            }

        field_matches: Dict[str, bool] = {}
        tp = 0
        fn = 0

        for key, target_val in target_fields.items():
            if key in extracted and _values_match(extracted[key], target_val):
                field_matches[key] = True
                tp += 1
            else:
                field_matches[key] = False
                fn += 1

        # Spurious predicted fields not in target_fields
        fp = sum(1 for k in extracted if k not in target_fields)
        # Also count target fields that had mismatch as false positive if key was present
        fp += sum(1 for k in target_fields if k in extracted and not field_matches[k])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "field_matches": field_matches,
            "extracted": extracted,
        }

    def score(self, query: Query, response: str) -> float:
        """Returns the micro-averaged F1 score for document extraction in [0.0, 1.0]."""
        results = self.evaluate_extraction(query, response)
        return float(results["f1"])
