"""Question class normalization utilities."""

from typing import Any

DEFAULT_KNOWN_CLASSES = [
    "closed_form_number",
    "closed_form_entity",
    "closed_form_choice",
    "unclassified",
]


def normalize_question_class(
    raw_class: Any,
    known_classes: list[str] | None = None,
) -> list[str]:
    """Normalize question class inputs to known classes or 'unclassified'."""
    known = known_classes or DEFAULT_KNOWN_CLASSES

    if raw_class is None:
        return ["unclassified"]

    if isinstance(raw_class, str):
        cleaned = raw_class.strip()
        if not cleaned:
            return ["unclassified"]
        if cleaned in known:
            return [cleaned]
        return ["unclassified"]

    if isinstance(raw_class, (list, tuple, set)):
        if not raw_class:
            return ["unclassified"]
        normalized: list[str] = []
        for item in raw_class:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if not cleaned:
                target = "unclassified"
            elif cleaned in known:
                target = cleaned
            else:
                target = "unclassified"
            
            if target not in normalized:
                normalized.append(target)
        
        return normalized if normalized else ["unclassified"]

    return ["unclassified"]
