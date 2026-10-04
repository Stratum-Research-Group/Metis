"""
Text embedding utilities for Dianoia.

Wraps sentence-transformers to produce query/document vectors compatible
with LTHCM's d_model dimensionality.
"""

from typing import List, Optional

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore


class TextEmbedder:
    """
    Encodes text into dense vectors using a sentence-transformers model.

    LTHCM's engine works with raw tensors; Dianoia needs to bridge from
    natural language to those tensors. This class handles that conversion.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.d_model = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Encode a list of texts into a [N, d_model] tensor.

        Args:
            texts: List of strings to encode.
            normalize: If True, L2-normalize embeddings.

        Returns:
            Tensor of shape [N, d_model].
        """
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.to(self.device)

    def encode_single(self, text: str, normalize: bool = True) -> torch.Tensor:
        """Encode a single text string into a [d_model] tensor."""
        return self.encode([text], normalize=normalize).squeeze(0)
