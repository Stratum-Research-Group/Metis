"""Tests for Dianoia walk causal, temporal, and access boundary enforcement."""

import unittest
from unittest.mock import MagicMock, patch
from dianoia.walk import WalkController


class TestWalkCausalBounds(unittest.TestCase):
    def setUp(self):
        self.mock_engine = MagicMock()
        self.mock_embedder = MagicMock()
        self.controller = WalkController(
            engine=self.mock_engine,
            embedder=self.mock_embedder,
            llm_base_url="http://127.0.0.1:9999",
            client=MagicMock(),
        )

    def test_seed_candidates_filters_future_dated(self):
        """Verify _seed_candidates filters out seeds created after as_of cutoff."""
        raw_candidates = [
            {"doc_id": "doc1", "content": "Valid historical fact", "created_epoch": 1000.0},
            {"doc_id": "doc2", "content": "Future fact", "created_epoch": 2000.0},
        ]
        self.mock_engine.superposition.retrieve.return_value = raw_candidates
        self.controller.use_stochastic_resonance = False

        filtered = self.controller._seed_candidates(MagicMock(), as_of=1500.0)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["doc_id"], "doc1")

    def test_seed_candidates_filters_access_boundaries(self):
        """Verify _seed_candidates filters out seeds rejected by access_filter."""
        raw_candidates = [
            {"doc_id": "public_doc", "content": "Public knowledge", "access": "public"},
            {"doc_id": "secret_doc", "content": "Confidential internal fact", "access": "restricted"},
        ]
        self.mock_engine.superposition.retrieve.return_value = raw_candidates
        self.controller.use_stochastic_resonance = False

        access_filter = lambda doc: doc.get("access") == "public"
        filtered = self.controller._seed_candidates(MagicMock(), access_filter=access_filter)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["doc_id"], "public_doc")

    def test_get_neighbours_filters_future_and_restricted(self):
        """Verify _get_neighbours filters out future-dated and restricted neighbors."""
        self.mock_engine.wormhole.get_bridges.return_value = [
            {"target": "node_past"},
            {"target": "node_future"},
            {"target": "node_restricted"},
        ]
        self.mock_engine.wormhole.node_metadata = {
            "node_past": {"content": "Historical data", "created_epoch": 500.0, "level": 1},
            "node_future": {"content": "Future prediction", "created_epoch": 3000.0, "level": 1},
            "node_restricted": {"content": "Confidential", "created_epoch": 500.0, "level": 9},
        }
        self.mock_engine.wormhole.graph = {}

        access_filter = lambda m: m.get("level", 0) <= 1
        neighbours = self.controller._get_neighbours(
            "seed_node", as_of=1000.0, access_filter=access_filter
        )
        self.assertEqual(len(neighbours), 1)
        self.assertEqual(neighbours[0]["node_id"], "node_past")


if __name__ == "__main__":
    unittest.main()
