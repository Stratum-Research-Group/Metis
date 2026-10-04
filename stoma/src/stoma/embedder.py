"""
Stoma Embedder Wrapper.
Enforces runtime import of pinned dianoia.embedding.TextEmbedder.
Binds strict model identity: measured tokenizer limits, special tokens overhead,
exact state-dict weights checksum, and git revision.
Disallows silent network downloading during ingest/query.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, List, Optional
import torch

from dianoia.embedding import TextEmbedder
from stoma.config import verify_runtime_dependencies
from stoma.provenance import ModelMetadata

logger = logging.getLogger(__name__)


class ModelInspectionError(RuntimeError):
    """Raised when model identity, tokenizer, weights, or limits cannot be verified."""
    pass


class OfflineCacheMissError(ModelInspectionError):
    """Raised when the embedding model is not present in local cache during offline operation."""
    pass


def prefetch_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> None:
    """
    Explicit, separate setup operation that accesses HuggingFace Hub ONLY to prefetch declared weights.
    No ingestion or query startup may fetch weights across the network.
    Temporarily allows online access during download, then re-enforces offline mode.
    """
    verify_runtime_dependencies()
    prev_tf_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)
    prev_hf_offline = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        from dianoia.embedding import TextEmbedder
        embedder = TextEmbedder(model_name=model_name, device=device or "cpu")
        logger.info("Successfully prefetched model '%s' (d_model=%s)", model_name, getattr(embedder, "d_model", None))
    finally:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"


class StomaEmbedder:
    """
    Wrapper around dianoia.embedding.TextEmbedder.
    Inspects and binds effective sequence limits, special-token overhead,
    weights checksum, and exact tokenizer at initialization.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        embedder_instance: Optional[Any] = None,
    ):
        # Verify submodule dependencies and exact commit checkouts first
        verify_runtime_dependencies()

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Enforce offline mode BEFORE model initialization
        # Prevents any network requests during ingestion or query startup
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        # Load dianoia TextEmbedder instance
        if embedder_instance is not None:
            self._dianoia_embedder = embedder_instance
        else:
            try:
                self._dianoia_embedder = TextEmbedder(model_name=self.model_name, device=self.device)
            except Exception as e:
                raise OfflineCacheMissError(
                    f"Failed to load required TextEmbedder for model '{self.model_name}' in offline mode (cache miss): {e}. "
                    "Please run the setup command 'python -m stoma.embedder prefetch' or 'stoma setup-model' before running offline."
                ) from e

        # Model inspection: dimensionality
        if hasattr(self._dianoia_embedder, "d_model"):
            self.d_model = int(self._dianoia_embedder.d_model)
        else:
            dim_fn = getattr(
                self._dianoia_embedder.model,
                "get_sentence_embedding_dimension",
                getattr(self._dianoia_embedder.model, "get_embedding_dimension", None)
            )
            if not callable(dim_fn):
                raise ModelInspectionError("Cannot determine d_model from embedder.")
            self.d_model = int(dim_fn())

        # Inspect underlying SentenceTransformer model and tokenizer
        raw_st_model = getattr(self._dianoia_embedder, "model", None)
        if raw_st_model is None:
            raise ModelInspectionError("Underlying SentenceTransformer model could not be accessed.")

        self.tokenizer = getattr(raw_st_model, "tokenizer", None)
        if self.tokenizer is None:
            raise ModelInspectionError("Tokenizer is missing from the underlying SentenceTransformer model.")

        self.tokenizer_name = type(self.tokenizer).__name__

        # Inspect max sequence length
        max_seq = getattr(raw_st_model, "max_seq_length", None)
        if max_seq is None:
            max_seq = getattr(self.tokenizer, "model_max_length", None)
        if max_seq is None or max_seq <= 0 or max_seq > 100000:
            raise ModelInspectionError(f"Effective sequence limit could not be inspected (got {max_seq}).")
        self.max_seq_length = int(max_seq)

        # Inspect special-token overhead
        overhead_fn = getattr(self.tokenizer, "num_special_tokens_to_add", None)
        if callable(overhead_fn):
            self.special_tokens_overhead = int(overhead_fn(pair=False))
        else:
            # Measure directly by encoding an empty string with and without special tokens
            enc_with = self.tokenizer.encode("", add_special_tokens=True)
            enc_without = self.tokenizer.encode("", add_special_tokens=False)
            self.special_tokens_overhead = len(enc_with) - len(enc_without)
        
        if self.special_tokens_overhead < 0:
            self.special_tokens_overhead = 0
            
        self.effective_max_tokens = self.max_seq_length - self.special_tokens_overhead
        if self.effective_max_tokens <= 0:
            raise ModelInspectionError(
                f"Effective max tokens ({self.effective_max_tokens}) must be > 0. "
                f"max_seq_length={self.max_seq_length}, overhead={self.special_tokens_overhead}"
            )

        # Compute exact weights checksum over state dict
        try:
            state_dict = raw_st_model.state_dict()
            if not state_dict:
                raise ModelInspectionError("Model state_dict is empty.")
            hasher = hashlib.sha256()
            for k, v in sorted(state_dict.items()):
                hasher.update(k.encode("utf-8"))
                hasher.update(v.cpu().numpy().tobytes())
            self.weights_checksum = hasher.hexdigest()
        except Exception as e:
            raise ModelInspectionError(f"Failed to calculate model weights checksum: {e}") from e

        # Extract git commit revision
        self.revision = self._extract_revision(raw_st_model)

        # Enforce offline mode to prevent silent network downloads during ingest / query
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        logger.info(
            "StomaEmbedder verified: model=%s, d_model=%d, max_seq=%d, overhead=%d, effective_max=%d, "
            "tokenizer=%s, revision=%s, weights_sha256=%s",
            self.model_name,
            self.d_model,
            self.max_seq_length,
            self.special_tokens_overhead,
            self.effective_max_tokens,
            self.tokenizer_name,
            self.revision,
            self.weights_checksum[:16] + "...",
        )

    def _extract_revision(self, raw_st_model: Any) -> Optional[str]:
        """Attempt to extract git commit / revision hash from the underlying model config."""
        if raw_st_model is None:
            return None
        try:
            for module in raw_st_model:
                auto_model = getattr(module, "auto_model", None)
                if auto_model and hasattr(auto_model, "config"):
                    commit_hash = getattr(auto_model.config, "_commit_hash", None)
                    if commit_hash:
                        return str(commit_hash)
        except Exception:
            pass
        return None

    def get_metadata(self) -> ModelMetadata:
        """Return ModelMetadata record for durable storage in the manifest."""
        return ModelMetadata(
            model_name=self.model_name,
            d_model=self.d_model,
            max_seq_length=self.max_seq_length,
            tokenizer_name=self.tokenizer_name,
            revision=self.revision,
            weights_checksum=self.weights_checksum,
            special_tokens_overhead=self.special_tokens_overhead,
            effective_max_tokens=self.effective_max_tokens,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using the loaded model's tokenizer. Never guesses."""
        if not text:
            return 0
        if self.tokenizer is None:
            raise ModelInspectionError("Cannot count tokens: tokenizer is not available.")
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
            raise ModelInspectionError(f"Tokenizer encoding failed: {e}") from e

    def encode_single(self, text: str) -> torch.Tensor:
        """Encode a single text into a 1D tensor [d_model]."""
        emb = self._dianoia_embedder.encode_single(text)
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float32)
        if emb.dim() == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        return emb

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts into a 2D tensor [batch_size, d_model]."""
        if not texts:
            return torch.empty((0, self.d_model), dtype=torch.float32)
        embs = self._dianoia_embedder.encode(texts)
        if not isinstance(embs, torch.Tensor):
            embs = torch.tensor(embs, dtype=torch.float32)
        return embs

    def create_facet_vector(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Produce single-facet tensor [1, d_model] as required by LTHCMEngine(num_facets=1).
        Matches dianoia/cli.py:94: facet_vectors = emb.unsqueeze(0).
        Fails loudly if shape does not match [1, d_model].
        """
        if embedding.dim() == 1:
            facet_vectors = embedding.unsqueeze(0)
        elif embedding.dim() == 2 and embedding.shape[0] == 1:
            facet_vectors = embedding
        else:
            raise ValueError(
                f"Invalid embedding dimension {embedding.shape} for single facet. "
                f"Expected [d_model] or [1, d_model]."
            )

        if facet_vectors.shape[1] != self.d_model:
            raise ValueError(
                f"Facet vector d_model mismatch: expected {self.d_model}, got {facet_vectors.shape[1]}"
            )
        if facet_vectors.shape[0] != 1:
            raise ValueError(
                f"Facet vector num_facets mismatch: expected 1, got {facet_vectors.shape[0]}"
            )

        return facet_vectors


if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "prefetch"
    if action == "inspect":
        embedder = StomaEmbedder()
        print(f"selected_model_identifier: {embedder.model_name}")
        print(f"import_origin: {embedder._dianoia_embedder.__class__.__module__}")
        print(f"revision: {embedder.revision}")
        print(f"state_dict_checksum: {embedder.weights_checksum}")
        print(f"tokenizer_identity: {embedder.tokenizer_name}")
        print(f"raw_max_length: {embedder.max_seq_length}")
        print(f"overhead: {embedder.special_tokens_overhead}")
        print(f"effective_limit: {embedder.effective_max_tokens}")
        print(f"dimension: {embedder.d_model}")
    else:
        model = sys.argv[2] if len(sys.argv) > 2 else (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "prefetch" else "all-MiniLM-L6-v2")
        prefetch_model(model_name=model)
        print(f"Successfully prefetched model '{model}'. Offline cache is populated.")
