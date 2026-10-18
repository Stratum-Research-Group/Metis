"""Tokenizer adapter for Symbolon: multiscale VQ-VAE code extraction with window scoping."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from symbolon.codes import ScopedCode

logger = logging.getLogger(__name__)

DEFAULT_NCM_PATH = Path(__file__).resolve().parents[2] / ".metis" / "scratch" / "ncm"
XSUM_WINDOW_LEN = 48
XSUM_LEVELS = [4, 6, 10, 14, 20, 26, 32, 40, 48]


class SymbolonTokenizerAdapter:
    """
    Adapts NCM's multiscale VQ-VAE tokenizer for LTHCM indexing.
    Chunks text passages into 48-token context windows, tags discrete codes
    with strict (checkpoint, tokenizer, level, window_idx, pos) scope,
    and supports multi-scale reconstruction diagnostics.
    """

    def __init__(
        self,
        ncm_root: Optional[str | Path] = None,
        checkpoint_id: str = "xsum_v1",
        tokenizer_id: str = "xsum_vqvae_48",
        device: Optional[str] = None,
        use_mock: bool = False,
    ):
        self.ncm_root = Path(ncm_root) if ncm_root else DEFAULT_NCM_PATH
        self.checkpoint_id = checkpoint_id
        self.tokenizer_id = tokenizer_id
        self.use_mock = use_mock
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.levels = XSUM_LEVELS
        self._gpt2_tokenizer = None
        self._ncm_tokenizer = None
        self._models_loaded = False

    @property
    def gpt2_tokenizer(self):
        if self._gpt2_tokenizer is None:
            self._gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return self._gpt2_tokenizer

    def _ensure_models_loaded(self) -> None:
        """Lazily load NCM VQ-VAE tokenizer onto device."""
        if self._models_loaded or self.use_mock:
            return

        artifacts_dir = self.ncm_root / "artifacts" / "xsum"
        cfg_file = self.ncm_root / "config" / "eval" / "xsum.yaml"
        if not artifacts_dir.exists() or not cfg_file.exists():
            logger.warning("NCM xsum artifacts missing; running Symbolon in mock mode.")
            self.use_mock = True
            return

        try:
            from omegaconf import OmegaConf

            ncm_root_str = str(self.ncm_root.resolve())
            if ncm_root_str not in sys.path:
                sys.path.insert(0, ncm_root_str)

            from tokenization.vqvae_tokenizer import VQVAETokenizer

            cfg = OmegaConf.load(str(cfg_file))
            cfg.tokenizer.vqvae.checkpoint_path = str(artifacts_dir / cfg.checkpoints.tokenizer)

            self._ncm_tokenizer = VQVAETokenizer.from_config(cfg.tokenizer)
            if self.device == "cuda":
                self._ncm_tokenizer.vqvae = self._ncm_tokenizer.vqvae.cuda().eval()
            else:
                self._ncm_tokenizer.vqvae = self._ncm_tokenizer.vqvae.eval()

            self._models_loaded = True
            logger.info("Symbolon VQ-VAE tokenizer loaded successfully on %s.", self.device)
        except Exception as e:
            logger.warning("Failed to load real NCM tokenizer (%s); falling back to mock mode.", e)
            self.use_mock = True

    def chunk_passage_tokens(self, text: str) -> List[List[int]]:
        """
        Tokenize passage into discrete chunks of exactly 48 tokens.
        Pads the final window to 48 tokens per NCM convention.
        """
        tokens = self.gpt2_tokenizer(text, add_special_tokens=False)["input_ids"]
        if not tokens:
            return []

        pad_id = self.gpt2_tokenizer.pad_token_id or self.gpt2_tokenizer.eos_token_id or 50256
        windows = []
        for i in range(0, len(tokens), XSUM_WINDOW_LEN):
            chunk = tokens[i : i + XSUM_WINDOW_LEN]
            if len(chunk) < XSUM_WINDOW_LEN:
                chunk = chunk + [pad_id] * (XSUM_WINDOW_LEN - len(chunk))
            windows.append(chunk)
        return windows

    def _encode_window_mock(self, window_idx: int, window_tokens: List[int]) -> List[ScopedCode]:
        """Generate deterministic mock codes across all 9 levels."""
        codes = []
        for lvl_idx, lvl_len in enumerate(self.levels):
            for pos in range(lvl_len):
                # Deterministic hash function of token slice and level/pos
                base_val = sum(window_tokens[: pos + 1]) if window_tokens else 42
                code_int = (base_val * 31 + lvl_idx * 17 + pos * 13) % 2048
                codes.append(
                    ScopedCode(
                        checkpoint_id=self.checkpoint_id,
                        tokenizer_id=self.tokenizer_id,
                        level=lvl_idx,
                        window_idx=window_idx,
                        pos=pos,
                        code_int=code_int,
                    )
                )
        return codes

    def _encode_window_ncm(self, window_idx: int, window_tokens: List[int]) -> List[ScopedCode]:
        """Encode a 48-token window through NCM VQ-VAE into strictly scoped discrete codes."""
        tensor_in = torch.tensor([window_tokens], dtype=torch.long)
        if self.device == "cuda":
            tensor_in = tensor_in.cuda()

        with torch.no_grad():
            z = self._ncm_tokenizer.vqvae._encode_pre_quant(tensor_in)
            # z_to_idx returns the concatenated index tensor
            vq = self._ncm_tokenizer.vqvae.vq
            z_no_grad = z.detach()
            B, L, C = z.shape
            num_levels = len(vq.levels)
            z_rest = z_no_grad.clone()
            z_hat = torch.zeros_like(z_rest)

            codes: List[ScopedCode] = []
            for lvl_idx, level_len in enumerate(vq.levels):
                emb_i = vq.embeddings[lvl_idx]
                if lvl_idx != num_levels - 1:
                    flat_input = vq._down_for_quant(z_rest, lvl_idx, level_len).reshape(-1, C)
                else:
                    flat_input = z_rest.reshape(-1, C)

                distances = vq._distances(flat_input, emb_i.weight)
                indices = torch.argmin(distances, dim=1).view(B, level_len)

                # Quantize residual
                if lvl_idx != num_levels - 1:
                    quantized = vq._up_to_full(emb_i(indices), lvl_idx, L)
                else:
                    quantized = emb_i(indices)
                quantized = vq._apply_quant_resi(quantized, lvl_idx)
                z_hat = z_hat + quantized
                z_rest -= quantized

                # Extract code values for batch item 0
                level_codes = indices[0].tolist()
                for pos, code_val in enumerate(level_codes):
                    codes.append(
                        ScopedCode(
                            checkpoint_id=self.checkpoint_id,
                            tokenizer_id=self.tokenizer_id,
                            level=lvl_idx,
                            window_idx=window_idx,
                            pos=pos,
                            code_int=int(code_val),
                        )
                    )
        return codes

    def encode_passage(self, text: str) -> List[ScopedCode]:
        """
        Encode an arbitrary-length passage into a collection of ScopedCodes.
        Chunks into 48-token windows and assigns passage-relative window_idx.
        """
        self._ensure_models_loaded()
        windows = self.chunk_passage_tokens(text)
        if not windows:
            return []

        all_codes: List[ScopedCode] = []
        for w_idx, w_tokens in enumerate(windows):
            if self.use_mock or self._ncm_tokenizer is None:
                w_codes = self._encode_window_mock(w_idx, w_tokens)
            else:
                w_codes = self._encode_window_ncm(w_idx, w_tokens)
            all_codes.extend(w_codes)
        return all_codes

    def reconstruct_window(self, text: str) -> List[str]:
        """
        Single-window multiscale decode for diagnostic inspection (Spec Section 2).
        Returns a list of 9 decoded strings corresponding to levels 0..8.
        """
        self._ensure_models_loaded()
        if self.use_mock or self._ncm_tokenizer is None:
            # Deterministic mock reconstruction
            words = text.split()
            previews = []
            for lvl_idx in range(len(self.levels)):
                frac = (lvl_idx + 1) / len(self.levels)
                n = max(1, min(len(words), int(len(words) * frac)))
                previews.append(" ".join(words[:n]))
            return previews

        # Chunk to single 48-token window
        tokens = self.gpt2_tokenizer(text, add_special_tokens=False)["input_ids"][:XSUM_WINDOW_LEN]
        pad_id = self.gpt2_tokenizer.pad_token_id or self.gpt2_tokenizer.eos_token_id or 50256
        if len(tokens) < XSUM_WINDOW_LEN:
            tokens = tokens + [pad_id] * (XSUM_WINDOW_LEN - len(tokens))

        tensor_in = torch.tensor([tokens], dtype=torch.long)
        if self.device == "cuda":
            tensor_in = tensor_in.cuda()

        with torch.no_grad():
            z = self._ncm_tokenizer.vqvae._encode_pre_quant(tensor_in)
            _, z_levels = self._ncm_tokenizer.vqvae.vq.z_to_idx(z, return_zlevels=True)
            decoded_levels = []
            for zl in z_levels:
                logits = self._ncm_tokenizer.vqvae.decoder(zl)
                toks = torch.argmax(logits, dim=-1)[0].tolist()
                text_out = self.gpt2_tokenizer.decode(toks, skip_special_tokens=True)
                decoded_levels.append(text_out)
        return decoded_levels
