"""Unit tests for the answer-vs-evidence verification module."""

import unittest
from unittest.mock import MagicMock

try:
    import torch
except ImportError:
    torch = None

from dianoia.verify import verify_answer_against_evidence


class TestVerifyAnswer(unittest.TestCase):
    def test_verify_with_answer_conflict(self):
        if torch is None:
            self.skipTest("torch not installed")

        mock_sheaf = MagicMock()
        mock_sheaf.evaluate_passages.return_value = {
            "is_consistent": False,
            "total_obstruction": 1.45,
            "conflicting_pairs": [
                {
                    "doc_a": "__dianoia_answer__",
                    "doc_b": "evidence_doc_1",
                    "content_a": "The model was trained on 1000 steps.",
                    "content_b": "The model was trained on 50000 steps.",
                    "obstruction_energy": 1.45,
                }
            ],
        }

        answer_text = "The model was trained on 1000 steps."
        evidence = [{"doc_id": "evidence_doc_1", "content": "The model was trained on 50000 steps."}]
        answer_emb = torch.zeros(64)
        evidence_embs = torch.zeros(1, 64)

        result = verify_answer_against_evidence(
            answer_text=answer_text,
            evidence_passages=evidence,
            answer_embedding=answer_emb,
            evidence_embeddings=evidence_embs,
            sheaf_filter=mock_sheaf,
            use_nli=True,
        )

        self.assertTrue(result["conflicts_found"])
        self.assertIsNotNone(result["conflict_detail"])
        self.assertIn("Answer conflicts with [evidence_doc_1]", result["conflict_detail"])
        self.assertIn("obstruction energy: 1.4500", result["conflict_detail"])
        self.assertIn("Answer-evidence conflicts found: 1", result["report"])

    def test_verify_with_passage_only_conflict(self):
        if torch is None:
            self.skipTest("torch not installed")

        mock_sheaf = MagicMock()
        mock_sheaf.evaluate_passages.return_value = {
            "is_consistent": False,
            "total_obstruction": 0.85,
            "conflicting_pairs": [
                {
                    "doc_a": "evidence_doc_1",
                    "doc_b": "evidence_doc_2",
                    "content_a": "Passage 1 content",
                    "content_b": "Passage 2 conflicting content",
                    "obstruction_energy": 0.85,
                }
            ],
        }

        answer_text = "Consistent synthesis."
        evidence = [
            {"doc_id": "evidence_doc_1", "content": "Passage 1 content"},
            {"doc_id": "evidence_doc_2", "content": "Passage 2 conflicting content"},
        ]
        answer_emb = torch.zeros(64)
        evidence_embs = torch.zeros(2, 64)

        result = verify_answer_against_evidence(
            answer_text=answer_text,
            evidence_passages=evidence,
            answer_embedding=answer_emb,
            evidence_embeddings=evidence_embs,
            sheaf_filter=mock_sheaf,
            use_nli=True,
        )

        # Conflict is passage-vs-passage, NOT answer-vs-evidence
        self.assertFalse(result["conflicts_found"])
        self.assertIsNone(result["conflict_detail"])
        self.assertIn("No answer-evidence conflicts detected.", result["report"])

    def test_verify_no_conflicts(self):
        if torch is None:
            self.skipTest("torch not installed")

        mock_sheaf = MagicMock()
        mock_sheaf.evaluate_passages.return_value = {
            "is_consistent": True,
            "total_obstruction": 0.0,
            "conflicting_pairs": [],
        }

        result = verify_answer_against_evidence(
            answer_text="Accurate answer.",
            evidence_passages=[{"doc_id": "doc1", "content": "Evidence content"}],
            answer_embedding=torch.zeros(64),
            evidence_embeddings=torch.zeros(1, 64),
            sheaf_filter=mock_sheaf,
            use_nli=True,
        )

    def test_nli_honesty_reporting_active(self):
        if torch is None:
            self.skipTest("torch not installed")

        mock_sheaf = MagicMock()
        mock_sheaf.nli_model = MagicMock()  # Active model instance
        mock_sheaf.nli_model_name = "cross-encoder/nli-deberta-v3-xsmall"
        mock_sheaf.evaluate_passages.return_value = {
            "is_consistent": True,
            "total_obstruction": 0.0,
            "conflicting_pairs": [],
        }

        result = verify_answer_against_evidence(
            answer_text="Answer",
            evidence_passages=[{"doc_id": "d1", "content": "Evidence"}],
            answer_embedding=torch.zeros(64),
            evidence_embeddings=torch.zeros(1, 64),
            sheaf_filter=mock_sheaf,
            use_nli=True,
        )
        self.assertIn("NLI active: cross-encoder/nli-deberta-v3-xsmall", result["report"])

    def test_nli_honesty_reporting_inactive(self):
        if torch is None:
            self.skipTest("torch not installed")

        mock_sheaf = MagicMock()
        mock_sheaf.nli_model = None  # Model NOT loaded/available
        mock_sheaf.nli_model_name = "cross-encoder/nli-deberta-v3-xsmall"
        mock_sheaf.evaluate_passages.return_value = {
            "is_consistent": True,
            "total_obstruction": 0.0,
            "conflicting_pairs": [],
        }

        result = verify_answer_against_evidence(
            answer_text="Answer",
            evidence_passages=[{"doc_id": "d1", "content": "Evidence"}],
            answer_embedding=torch.zeros(64),
            evidence_embeddings=torch.zeros(1, 64),
            sheaf_filter=mock_sheaf,
            use_nli=True,  # Even though flag is True, report must say inactive
        )
        self.assertIn("NLI inactive: vector sheaf Laplacian only", result["report"])


if __name__ == "__main__":
    unittest.main()
