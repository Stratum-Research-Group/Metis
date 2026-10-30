"""Unit tests for OpenReviewBoundaryCrawler and CommunityAvailabilityIndex."""

import json
from pathlib import Path
import pytest

from stoma.crawlers.openreview import (
    CommunityAvailabilityIndex,
    OpenReviewBoundaryCrawler,
    OpenReviewCrawlerConfig,
    PeerReviewForumRecord,
)


def test_parse_forum_dict_contested_boundary():
    crawler = OpenReviewBoundaryCrawler(
        config=OpenReviewCrawlerConfig(
            min_variance_contested=1.5,
            score_min_contested=4.0,
            score_max_contested=6.5,
        )
    )

    # 1. Contested paper (mean ~ 5.33, var ~ 6.33 >= 1.5)
    contested_raw = {
        "forum_id": "forum_101",
        "title": "Topological Gauge Invariants in Graph Transformers",
        "abstract": "We explore non-abelian gauge groups for graph representations.",
        "primary_area": "graph_neural_networks",
        "decision": "Reject - lack of theoretical bridge",
        "reviews": [
            {
                "reviewer_id": "AnonReviewer1",
                "rating": 8,
                "review": "Fascinating bridge between topology and graph neural networks.",
            },
            {
                "reviewer_id": "AnonReviewer2",
                "rating": 3,
                "review": "The authors failed to connect with the riemannian geometry literature.",
            },
            {
                "reviewer_id": "AnonReviewer3",
                "rating": 5,
                "review": "Lacks a theoretical bridge between graph_neural_networks and differential_geometry.",
            },
        ],
    }

    forum = crawler.parse_forum_dict(contested_raw)
    assert forum.forum_id == "forum_101"
    assert forum.is_contested_boundary is True
    assert forum.mean_score == pytest.approx(5.33, abs=0.1)
    assert forum.score_variance >= 1.5
    assert len(forum.objections) == 2
    assert ("graph_neural_networks", "riemannian geometry") in forum.boundary_pairs or (
        "riemannian geometry", "graph_neural_networks"
    ) in forum.boundary_pairs


def test_parse_forum_dict_consensus_accept():
    crawler = OpenReviewBoundaryCrawler()

    consensus_raw = {
        "forum_id": "forum_202",
        "title": "Scaling Laws for Transformer Memory",
        "primary_area": "natural_language_processing",
        "decision": "Accept (Oral)",
        "reviews": [
            {"rating": 8, "review": "Thorough and sound."},
            {"rating": 9, "review": "Excellent empirical results."},
            {"rating": 8, "review": "Clear acceptance."},
        ],
    }

    forum = crawler.parse_forum_dict(consensus_raw)
    assert forum.is_contested_boundary is False
    assert forum.mean_score >= 8.0
    assert forum.score_variance < 1.0


def test_community_availability_index_scoring():
    index = CommunityAvailabilityIndex()

    # Case 1: Accepted pair co-occurring multiple times
    index.record_co_occurrence("transformers", "attention", is_accepted=True)
    index.record_co_occurrence("transformers", "attention", is_accepted=True)
    index.record_co_occurrence("transformers", "attention", is_accepted=True)

    # Case 2: Contested boundary pair from rejected/disputed reviews
    index.record_co_occurrence("graph_neural_networks", "riemannian_geometry", is_accepted=False)
    index.record_co_occurrence("graph_neural_networks", "riemannian_geometry", is_accepted=False)

    # Case 3: Completely unobserved alien pair
    score_accepted = index.get_availability_score("transformers", "attention")
    score_contested = index.get_availability_score("graph_neural_networks", "riemannian_geometry")
    score_alien = index.get_availability_score("cellular_sheaves", "high_frequency_trading")

    assert score_accepted > 0.7  # Established, crowded research paradigm
    assert score_contested <= 0.25  # High boundary friction, no accepted bridge
    assert score_contested > score_alien  # Known contested problem > completely unobserved
    assert score_alien == pytest.approx(0.01)  # Virgin frontier


def test_crawl_local_fixture_e2e(tmp_path: Path):
    fixture_file = tmp_path / "mock_openreview.json"
    mock_data = [
        {
            "forum_id": "openreview_001",
            "title": "Quantum Invariants in Diffusion Models",
            "primary_area": "generative_models",
            "decision": "Reject",
            "reviews": [
                {"rating": 7, "review": "Novel formulation."},
                {"rating": 3, "review": "Authors failed to connect with the quantum mechanics literature."},
            ],
        },
        {
            "forum_id": "openreview_002",
            "title": "Standard Diffusion Guidance",
            "primary_area": "generative_models",
            "decision": "Accept",
            "reviews": [
                {"rating": 8, "review": "Good solid contribution."},
                {"rating": 8, "review": "Matches expectations."},
            ],
        },
    ]
    with open(fixture_file, "w", encoding="utf-8") as f:
        json.dump(mock_data, f)

    crawler = OpenReviewBoundaryCrawler()
    records = crawler.crawl_local_fixture(fixture_file)
    assert len(records) == 2
    assert records[0].is_contested_boundary is True
    assert len(records[0].objections) == 1

    avail_index = crawler.build_availability_index(records)
    score = avail_index.get_availability_score("generative_models", "quantum mechanics")
    assert score <= 0.25

