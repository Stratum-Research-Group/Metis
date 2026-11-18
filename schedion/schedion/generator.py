"""Schedion candidate generator wrapping the Next-Concept Modeling (NCM) pipeline."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from krisis.contracts.candidate import Candidate
from schedion.claim_mapper import SentenceBoundaryClaimMapper
from schedion.contracts import SchedionProvenance, create_krisis_candidate

logger = logging.getLogger(__name__)

DEFAULT_NCM_PATH = Path(__file__).resolve().parents[2] / ".metis" / "scratch" / "ncm"
XSUM_PREFIX_LEN = 512
XSUM_LEVELS = [4, 6, 10, 14, 20, 26, 32, 40, 48]


class SchedionGenerator:
    """
    Candidate answer generator powered by NCM coarse-to-fine parallel hierarchy.
    Generates summary candidates from a conditioned document prefix, captures
    per-level previews, and attaches structured claims and provenance for Krisis.
    """

    def __init__(
        self,
        ncm_root: Optional[str | Path] = None,
        device: Optional[str] = None,
        use_mock: bool = False,
    ):
        self.ncm_root = Path(ncm_root) if ncm_root else DEFAULT_NCM_PATH
        self.use_mock = use_mock
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.claim_mapper = SentenceBoundaryClaimMapper()
        self._gpt2_tokenizer = None
        self._ncm_tokenizer = None
        self._ncm_generator = None
        self._models_loaded = False

    @property
    def gpt2_tokenizer(self):
        if self._gpt2_tokenizer is None:
            # Use local gpt2 tokenizer or standard huggingface tokenizer
            self._gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return self._gpt2_tokenizer

    def _ensure_models_loaded(self) -> None:
        """Lazily load NCM VQ-VAE tokenizer and generator model onto device."""
        if self._models_loaded or self.use_mock:
            return

        if not self.ncm_root.exists():
            logger.warning("NCM root %s does not exist; running in mock mode.", self.ncm_root)
            self.use_mock = True
            return

        artifacts_dir = self.ncm_root / "artifacts" / "xsum"
        cfg_file = self.ncm_root / "config" / "eval" / "xsum.yaml"
        if not artifacts_dir.exists() or not cfg_file.exists():
            logger.warning("NCM xsum config or artifacts missing; running in mock mode.")
            self.use_mock = True
            return

        try:
            from omegaconf import OmegaConf

            # Add NCM root to sys.path if not present
            ncm_root_str = str(self.ncm_root.resolve())
            if ncm_root_str not in sys.path:
                sys.path.insert(0, ncm_root_str)

            from models.next_concept import NextConceptTransformer
            from tokenization.vqvae_tokenizer import VQVAETokenizer

            cfg = OmegaConf.load(str(cfg_file))
            cfg.tokenizer.vqvae.checkpoint_path = str(artifacts_dir / cfg.checkpoints.tokenizer)

            logger.info("Loading NCM VQ-VAE tokenizer...")
            self._ncm_tokenizer = VQVAETokenizer.from_config(cfg.tokenizer)
            if self.device == "cuda":
                self._ncm_tokenizer.vqvae = self._ncm_tokenizer.vqvae.cuda().eval()

            logger.info("Loading NCM generator...")
            self._ncm_generator = NextConceptTransformer(
                vqvae_local=self._ncm_tokenizer.vqvae,
                config=cfg.model,
            )
            if self.device == "cuda":
                self._ncm_generator = self._ncm_generator.cuda()

            gen_ckpt = torch.load(
                str(artifacts_dir / cfg.checkpoints.generator),
                map_location="cpu",
                weights_only=True,
            )
            self._ncm_generator.load_state_dict(gen_ckpt["model"], strict=True)
            self._ncm_generator.eval()

            self._models_loaded = True
            logger.info("NCM models loaded successfully on %s.", self.device)
        except Exception as e:
            logger.warning("Failed to load real NCM models (%s); falling back to mock mode.", e)
            self.use_mock = True

    def prepare_conditioned_span(self, document_text: str) -> Tuple[List[int], str, bool]:
        """
        Tokenize document text and truncate strictly to 512 GPT-2 tokens (XSum context).
        Returns (token_ids, conditioned_text, was_truncated).
        """
        enc = self.gpt2_tokenizer(document_text, add_special_tokens=False)["input_ids"]
        was_truncated = len(enc) > XSUM_PREFIX_LEN
        conditioned_tokens = enc[:XSUM_PREFIX_LEN]
        conditioned_text = self.gpt2_tokenizer.decode(conditioned_tokens)
        return conditioned_tokens, conditioned_text, was_truncated

    def _generate_mock(
        self,
        conditioned_text: str,
        n_candidates: int,
        seed: int,
    ) -> List[Tuple[str, List[str]]]:
        """Generate deterministic mock candidates with 9 level previews."""
        results = []
        # Extract a few sentences from conditioned text for the mock summary
        sentences = [s.strip() for s in conditioned_text.split(".") if s.strip()]
        base_summary = sentences[0] + "." if sentences else "A concise summary of the document."

        for i in range(n_candidates):
            candidate_text = f"{base_summary} (Candidate {i+1}, seed {seed})"
            # Synthetic 9-level previews progressing from coarse fragments to fine summary
            words = candidate_text.split()
            previews = []
            for lvl_idx, lvl_len in enumerate(XSUM_LEVELS):
                n_words = max(1, min(len(words), int(len(words) * (lvl_idx + 1) / len(XSUM_LEVELS))))
                preview_str = " ".join(words[:n_words])
                if lvl_idx == len(XSUM_LEVELS) - 1:
                    preview_str = candidate_text
                previews.append(preview_str)
            results.append((candidate_text, previews))
        return results

    def _generate_ncm(
        self,
        conditioned_tokens: List[int],
        n_candidates: int,
        seed: int,
        temperature: float,
        top_k: int,
        top_p: float,
        cfg_scale: float,
    ) -> List[Tuple[str, List[str]]]:
        """Run actual NCM coarse-to-fine generation and return (final_text, per_level_previews)."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Pad tokens to prefix_len 512
        pad_id = self.gpt2_tokenizer.pad_token_id or self.gpt2_tokenizer.eos_token_id
        if len(conditioned_tokens) < XSUM_PREFIX_LEN:
            padded = conditioned_tokens + [pad_id] * (XSUM_PREFIX_LEN - len(conditioned_tokens))
        else:
            padded = conditioned_tokens[:XSUM_PREFIX_LEN]

        batch_tokens = torch.tensor([padded] * n_candidates, dtype=torch.long)
        if self.device == "cuda":
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            out_codes = self._ncm_generator.generate(
                idx=None,
                prefix_tokens=batch_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cfg_scale=cfg_scale,
            )
            # Decode multiscales across all levels
            multiscale_reconstructions = self._ncm_tokenizer.decode_multiscales(out_codes)

        results = []
        for i in range(n_candidates):
            level_texts = multiscale_reconstructions[i]  # List of 9 strings
            final_text = level_texts[-1] if level_texts else ""
            results.append((final_text, level_texts))
        return results

    def generate_candidates(
        self,
        document_text: str,
        passage_id: str = "doc_span_512",
        n_candidates: int = 1,
        seed: int = 42,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        cfg_scale: float = 0.0,
    ) -> List[Candidate]:
        """
        Execute Schedion candidate generation over the supplied document text.

        Args:
            document_text: Raw input text.
            passage_id: Identifier of the source passage for citation anchoring.
            n_candidates: Number of candidate drafts to generate.
            seed: RNG seed for reproducible generation.
            temperature: Sampling temperature.
            top_k: Top-k vocabulary truncation.
            top_p: Nucleus sampling cutoff.
            cfg_scale: Classifier-free guidance scale.

        Returns:
            List of Krisis-compatible Candidate objects.
        """
        t0 = time.perf_counter()
        self._ensure_models_loaded()

        cond_tokens, cond_text, was_truncated = self.prepare_conditioned_span(document_text)

        if self.use_mock or self._ncm_generator is None:
            generated = self._generate_mock(cond_text, n_candidates, seed)
        else:
            generated = self._generate_ncm(
                cond_tokens,
                n_candidates=n_candidates,
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cfg_scale=cfg_scale,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        peak_vram = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if (torch.cuda.is_available() and self.device == "cuda")
            else 0.0
        )

        candidates: List[Candidate] = []
        for idx, (cand_text, previews) in enumerate(generated):
            cand_id = f"schedion_{seed}_{idx}"
            claims = self.claim_mapper.map_claims(cand_text, cand_id, passage_id)

            prov = SchedionProvenance(
                ncm_commit="c70274273f489abc73832b2b98584a8483b6b203",
                checkpoint_sha256="xsum_generator_c702742",
                task_config="xsum",
                sampling_params={
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "cfg_scale": cfg_scale,
                },
                seed=seed,
                latency_ms=round(latency_ms, 2),
                peak_vram_gb=round(peak_vram, 3),
                conditioned_span_tokens=len(cond_tokens),
                conditioned_span_text=cond_text[:200] + ("..." if len(cond_text) > 200 else ""),
                was_truncated=was_truncated,
                per_level_previews=previews,
            )

            cand = create_krisis_candidate(
                candidate_id=cand_id,
                text=cand_text,
                claims=claims,
                cited_passage_ids=[passage_id],
                provenance=prov,
            )
            candidates.append(cand)

        return candidates
