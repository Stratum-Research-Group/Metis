"""Krisis routing package."""

from krisis.routing.normalizer import normalize_question_class
from krisis.routing.router import ClassRouter

__all__ = ["normalize_question_class", "ClassRouter"]
