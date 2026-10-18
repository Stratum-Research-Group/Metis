"""Unit and integration tests for Schedion candidate generation adapter."""

from __future__ import annotations

import unittest
from pathlib import Path

from krisis.contracts.candidate import Candidate, Claim
from krisis.contracts.decision import DecisionStatus
from krisis.contracts.policy import PolicyManifest
from krisis.contracts.request import KrisisRequest
from krisis.engine.selector import KrisisSelector
from schedion.claim_mapper import SentenceBoundaryClaimMapper
from schedion.contracts import SchedionProvenance
from schedion.generator import SchedionGenerator


class TestSentenceBoundaryClaimMapper(unittest.TestCase):
    """Test deterministic sentence-boundary claim extraction for Krisis."""

    def setUp(self):
        self.mapper = SentenceBoundaryClaimMapper()

    def test_multi_sentence_extraction(self):
        text = "First claim about Metis. Second claim with numbers like 42! Third claim asking questions?"
        claims = self.mapper.map_claims(text, candidate_id="cand_1", passage_id="p_100")

        self.assertEqual(len(claims), 3)
        self.assertEqual(claims[0].claim_id, "cand_1_c0")
        self.assertEqual(claims[0].text, "First claim about Metis.")
        self.assertEqual(claims[0].verdict, "unverified")
        self.assertEqual(claims[0].cited_passage_ids, ["p_100"])

        self.assertEqual(claims[1].claim_id, "cand_1_c1")
        self.assertEqual(claims[1].text, "Second claim with numbers like 42!")
        self.assertEqual(claims[1].cited_passage_ids, ["p_100"])

        self.assertEqual(claims[2].claim_id, "cand_1_c2")
        self.assertEqual(claims[2].text, "Third claim asking questions?")

    def test_single_sentence_no_terminal_punct(self):
        text = "This is a single sentence without terminal punctuation"
        claims = self.mapper.map_claims(text, candidate_id="cand_2", passage_id="p_200")

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].claim_id, "cand_2_c0")
        self.assertEqual(claims[0].text, text)
        self.assertEqual(claims[0].cited_passage_ids, ["p_200"])

    def test_empty_string(self):
        claims = self.mapper.map_claims("", candidate_id="cand_3")
        self.assertEqual(len(claims), 0)


class TestSchedionGenerator(unittest.TestCase):
    """Test Schedion candidate generator and Krisis integration."""

    def setUp(self):
        self.generator = SchedionGenerator(use_mock=True)

    def test_truncation_discipline(self):
        """Documents exceeding 512 GPT-2 tokens must be explicitly truncated and recorded."""
        short_text = "Metis is a local cognitive stack. " * 10
        _, _, was_trunc_short = self.generator.prepare_conditioned_span(short_text)
        self.assertFalse(was_trunc_short)

        long_text = "Metis deliberative topological memory reasoning agent. " * 80  # ~560 tokens
        cond_tokens, _, was_trunc_long = self.generator.prepare_conditioned_span(long_text)
        self.assertTrue(was_trunc_long)
        self.assertEqual(len(cond_tokens), 512)

    def test_candidate_generation_schema_conformance(self):
        """Generated candidates must be valid Krisis Candidate instances."""
        doc = (
            "Apollo 11 launched on July 16, 1969. "
            "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle. "
            "Armstrong became the first person to step onto the lunar surface six hours later."
        )

        candidates = self.generator.generate_candidates(
            document_text=doc,
            passage_id="apollo_doc_01",
            n_candidates=2,
            seed=101,
        )

        self.assertEqual(len(candidates), 2)
        for cand in candidates:
            # Krisis Candidate schema validation
            self.assertIsInstance(cand, Candidate)
            self.assertTrue(cand.id.startswith("schedion_101_"))
            self.assertTrue(cand.text)
            self.assertEqual(cand.citations, ["apollo_doc_01"])
            self.assertEqual(cand.cited_passage_ids, ["apollo_doc_01"])
            self.assertGreaterEqual(len(cand.claims), 1)

            # Check claim formatting
            for cl in cand.claims:
                self.assertIsInstance(cl, Claim)
                self.assertEqual(cl.cited_passage_ids, ["apollo_doc_01"])
                self.assertEqual(cl.verdict, "unverified")

            # Check provenance attached to candidate
            prov = cand.provenance
            self.assertIsNotNone(prov)
            self.assertEqual(prov["ncm_commit"], "c70274273f489abc73832b2b98584a8483b6b203")
            self.assertEqual(prov["task_config"], "xsum")
            self.assertEqual(prov["seed"], 101)
            self.assertIn("latency_ms", prov)
            self.assertIn("per_level_previews", prov)
            self.assertEqual(len(prov["per_level_previews"]), 9)

    def test_krisis_selector_ingestion(self):
        """
        Krisis Stage 2 Gate: Feed Schedion candidates through KrisisSelector.evaluate().
        Candidates must be processed cleanly without throwing schema or validation errors.
        """
        policy_path = Path("krisis/krisis-policy-0.1.json")
        self.assertTrue(policy_path.exists(), "krisis-policy-0.1.json must exist")

        selector = KrisisSelector(policy_path)

        doc = "Quantum computing harnesses superposition and entanglement to solve discrete optimization problems."
        candidates = self.generator.generate_candidates(
            document_text=doc,
            passage_id="quantum_doc",
            n_candidates=1,
            seed=42,
        )

        request = KrisisRequest(
            question="What does quantum computing harness?",
            question_class="FACTUAL",
            requires_grounding=False,
            candidates=candidates,
        )

        decision = selector.evaluate(request)

        # Must return valid KrisisDecision
        self.assertIsNotNone(decision)
        self.assertIn(decision.status, [s.value for s in DecisionStatus])

    def test_stage3_controlled_comparison_study(self):
        """
        Schedion Stage 3 Gate: Controlled comparison between Hestia summary baseline
        and Schedion NCM generator under KrisisSelector gating.
        """
        from schedion.study import SchedionStage3Study, SchedionStage3Report

        study = SchedionStage3Study(schedion_generator=self.generator)
        report = study.run_study()

        self.assertIsInstance(report, SchedionStage3Report)
        self.assertEqual(report.total_probes, 5)
        self.assertEqual(len(report.hestia_results), 5)
        self.assertEqual(len(report.schedion_results), 5)

        # Check metrics computed
        self.assertGreaterEqual(report.hestia_pass_rate, 0.0)
        self.assertGreaterEqual(report.schedion_pass_rate, 0.0)
        self.assertGreaterEqual(report.schedion_mean_grounding, 0.5)

        # Check decision gate verdict string
        self.assertIn(
            report.decision_gate_verdict,
            ["ADOPT_OPTIONAL_SOURCE", "PARITY_OFF_BY_DEFAULT", "ARCHIVE_NO_ADOPTION"],
        )
        self.assertTrue(report.decision_gate_rationale)
        self.assertIn("delta", report.paired_pass_bootstrap)


if __name__ == "__main__":
    unittest.main()

