"""
OpenReview Peer-Review Boundary Crawler for Heuresis Discovery Engine.

Extracts disciplinary boundaries, reviewer objections, and contested paper frontiers
(mean rating in [4.0, 6.5], variance >= 1.5) to construct community availability
proxies and identify unbridged theoretical intersections.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReviewerObjectionRecord(BaseModel):
    """Structured objection or boundary critique extracted from a peer review comment."""
    reviewer_id: str
    comment: str
    cited_boundary_concepts: List[str] = Field(default_factory=list)
    severity: str = "missing_bridge"  # "fatal_flaw", "missing_bridge", "unsubstantiated"


class PeerReviewForumRecord(BaseModel):
    """Structured record of a peer-reviewed submission and its evaluation metadata."""
    forum_id: str
    paper_title: str
    abstract: str
    primary_area: str
    decision: str = "Pending"  # "Accept", "Reject", "Pending"
    ratings: List[float] = Field(default_factory=list)
    mean_score: float = 0.0
    score_variance: float = 0.0
    is_contested_boundary: bool = False
    objections: List[ReviewerObjectionRecord] = Field(default_factory=list)
    boundary_pairs: List[Tuple[str, str]] = Field(default_factory=list)


class OpenReviewCrawlerConfig(BaseModel):
    """Configuration governing peer-review boundary extraction."""
    min_variance_contested: float = 1.5
    score_min_contested: float = 4.0
    score_max_contested: float = 6.5
    max_forums: int = 200


class CommunityAvailabilityIndex:
    """
    Tracks community availability and unbridged boundaries between concepts.
    High availability indicates established, accepted research paradigms.
    Low availability with reviewer boundary objections indicates unmined frontiers.
    """

    def __init__(self):
        self.accepted_co_occurrences: Dict[Tuple[str, str], int] = {}
        self.contested_boundary_mentions: Dict[Tuple[str, str], int] = {}
        self.concept_frequencies: Dict[str, int] = {}

    def _canonical_pair(self, a: str, b: str) -> Tuple[str, str]:
        a_clean = a.strip().lower()
        b_clean = b.strip().lower()
        return (a_clean, b_clean) if a_clean <= b_clean else (b_clean, a_clean)

    def record_co_occurrence(self, concept_a: str, concept_b: str, is_accepted: bool) -> None:
        pair = self._canonical_pair(concept_a, concept_b)
        self.concept_frequencies[pair[0]] = self.concept_frequencies.get(pair[0], 0) + 1
        self.concept_frequencies[pair[1]] = self.concept_frequencies.get(pair[1], 0) + 1
        if is_accepted:
            self.accepted_co_occurrences[pair] = self.accepted_co_occurrences.get(pair, 0) + 1
        else:
            self.contested_boundary_mentions[pair] = self.contested_boundary_mentions.get(pair, 0) + 1

    def get_availability_score(self, concept_a: str, concept_b: str) -> float:
        """
        Compute community availability score S_avail in [0.0, 1.0].
        - Accepted papers co-occurring: S_avail -> 1.0 (crowded, established).
        - Contested/rejected boundary objections: S_avail -> 0.05-0.20 (under-explored frontier).
        - Never co-occurring: S_avail -> 0.0 (alien / unbridged).
        """
        pair = self._canonical_pair(concept_a, concept_b)
        accepted_cnt = self.accepted_co_occurrences.get(pair, 0)
        contested_cnt = self.contested_boundary_mentions.get(pair, 0)

        if accepted_cnt > 0:
            # Diminishing returns scaling towards 1.0
            return 1.0 - math.exp(-0.5 * accepted_cnt)
        elif contested_cnt > 0:
            # Active boundary friction: known problem, but no accepted bridge exists
            return max(0.05, 0.25 / (1.0 + contested_cnt))
        else:
            # Completely unbridged alien space
            return 0.01


class OpenReviewBoundaryCrawler:
    """
    Crawler and parser for OpenReview peer review forums.
    Isolates high-variance contested submissions and extracts reviewer boundary objections.
    """

    # Regex patterns identifying disciplinary bridge critiques in reviewer comments
    OBJECTION_PATTERNS = [
        re.compile(r"failed to connect with (?:the )?([a-z\s_-]+) literature", re.IGNORECASE),
        re.compile(r"lacks? (?:a )?theoretical bridge (?:between|to) ([a-z\s_-]+)", re.IGNORECASE),
        re.compile(r"unclear how this relates to ([a-z\s_-]+)", re.IGNORECASE),
        re.compile(r"missing reference to ([a-z\s_-]+)", re.IGNORECASE),
        re.compile(r"disconnect between ([a-z\s_-]+) and ([a-z\s_-]+)", re.IGNORECASE),
    ]

    def __init__(self, config: Optional[OpenReviewCrawlerConfig] = None):
        self.config = config or OpenReviewCrawlerConfig()

    def parse_forum_dict(self, data: Dict[str, Any]) -> PeerReviewForumRecord:
        """Parse raw OpenReview forum API / export JSON into PeerReviewForumRecord."""
        forum_id = str(data.get("forum_id", data.get("id", "")))
        content = data.get("content", {})
        title = content.get("title", data.get("title", "Untitled"))
        abstract = content.get("abstract", data.get("abstract", ""))
        primary_area = content.get("primary_area", data.get("primary_area", "General ML"))
        decision = data.get("decision", "Pending")

        # Extract ratings
        raw_ratings = data.get("ratings", [])
        if not raw_ratings and "reviews" in data:
            for rev in data["reviews"]:
                rating_val = rev.get("rating")
                if isinstance(rating_val, (int, float)):
                    raw_ratings.append(float(rating_val))
                elif isinstance(rating_val, str):
                    m = re.search(r"(\d+(?:\.\d+)?)", rating_val)
                    if m:
                        raw_ratings.append(float(m.group(1)))

        # Compute mean and sample variance
        n = len(raw_ratings)
        if n >= 2:
            mean = sum(raw_ratings) / n
            var = sum((r - mean) ** 2 for r in raw_ratings) / (n - 1)
        elif n == 1:
            mean = raw_ratings[0]
            var = 0.0
        else:
            mean = 0.0
            var = 0.0

        # Contested submission criteria: rating in [score_min, score_max] and variance >= min_variance
        is_contested = (
            (self.config.score_min_contested <= mean <= self.config.score_max_contested)
            and (var >= self.config.min_variance_contested)
        )

        # Extract objections from reviews
        objections: List[ReviewerObjectionRecord] = []
        boundary_pairs: Set[Tuple[str, str]] = set()

        raw_reviews = data.get("reviews", [])
        for idx, rev in enumerate(raw_reviews):
            rev_id = rev.get("reviewer_id", f"Reviewer_{idx+1}")
            text = rev.get("review", rev.get("comment", ""))
            cited_concepts: List[str] = []

            for pat in self.OBJECTION_PATTERNS:
                for match in pat.finditer(text):
                    groups = match.groups()
                    for g in groups:
                        if g:
                            c_clean = g.strip().lower()
                            if len(c_clean) > 3 and c_clean not in cited_concepts:
                                cited_concepts.append(c_clean)

            if cited_concepts:
                objections.append(
                    ReviewerObjectionRecord(
                        reviewer_id=rev_id,
                        comment=text[:300],
                        cited_boundary_concepts=cited_concepts,
                        severity="missing_bridge",
                    )
                )
                # Form boundary pairs between paper primary area and cited concept
                area_clean = primary_area.strip().lower()
                for c in cited_concepts:
                    pair = (area_clean, c) if area_clean <= c else (c, area_clean)
                    boundary_pairs.add(pair)

        return PeerReviewForumRecord(
            forum_id=forum_id,
            paper_title=title,
            abstract=abstract,
            primary_area=primary_area,
            decision=decision,
            ratings=raw_ratings,
            mean_score=round(mean, 2),
            score_variance=round(var, 3),
            is_contested_boundary=is_contested,
            objections=objections,
            boundary_pairs=list(boundary_pairs),
        )

    def crawl_local_fixture(self, fixture_path: Path) -> List[PeerReviewForumRecord]:
        """Load and parse local peer-review JSON fixture file."""
        path = Path(fixture_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Fixture not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        records: List[PeerReviewForumRecord] = []
        items = raw if isinstance(raw, list) else raw.get("forums", [raw])
        for item in items[: self.config.max_forums]:
            records.append(self.parse_forum_dict(item))

        return records

    def build_availability_index(
        self,
        forums: List[PeerReviewForumRecord],
        concept_extract_fn: Optional[Any] = None,
    ) -> CommunityAvailabilityIndex:
        """
        Build a CommunityAvailabilityIndex from a collection of peer review forum records.
        """
        index = CommunityAvailabilityIndex()

        for forum in forums:
            is_accepted = "accept" in forum.decision.lower()
            
            # Concepts extracted from boundary pairs
            for a, b in forum.boundary_pairs:
                index.record_co_occurrence(a, b, is_accepted=is_accepted)

            # If external extractor supplied (e.g. keywords from abstract)
            if concept_extract_fn is not None:
                concepts = concept_extract_fn(forum.paper_title + " " + forum.abstract)
                for i in range(len(concepts)):
                    for j in range(i + 1, len(concepts)):
                        index.record_co_occurrence(concepts[i], concepts[j], is_accepted=is_accepted)

        return index

