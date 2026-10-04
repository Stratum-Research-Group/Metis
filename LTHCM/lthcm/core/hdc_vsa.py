"""
Hyperdimensional Computing & Vector Symbolic Architectures (HDC / VSA)
======================================================================
Implements Holographic Reduced Representations (HRR) and Vector Symbolic
Architectures on continuous / real-valued hypervectors in R^D.

Provides algebraically exact symbolic composition operations:
  1. Binding (Circular Convolution via FFT):
         c = a (*) b = F^{-1}(F(a) . F(b))
     Creates role-filler pairs. c is quasi-orthogonal to both a and b.
     Invertible via Unbinding (Circular Correlation):
         a' = c (*)~ b = F^{-1}(F(c) . conj(F(b)))
  2. Bundling (Superposition):
         s = normalize(sum_i w_i * v_i)
     Creates algebraic sets/unions of facts.
  3. Permutation (Circular Roll Pi^k):
         Pi^k(v) = roll(v, k)
     Encodes sequence, temporal order, or tree hierarchy without positional parameters.
  4. Projection Bridge:
     Maps continuous neural embeddings in R^{d_model} to and from holographic R^{d_hdc}.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorSymbolicArchitecture(nn.Module):
    """
    Holographic Vector Symbolic Architecture (HRR) engine for symbolic reasoning in LTHCM.
    """

    def __init__(
        self,
        d_hdc: int = 512,
        d_model: Optional[int] = None,
    ):
        """
        Args:
            d_hdc: Dimensionality of the hyperdimensional space (typically 512, 1024, or 2048).
            d_model: Dimensionality of base neural embeddings (if bridging to continuous space).
        """
        super().__init__()
        self.d_hdc = d_hdc
        self.d_model = d_model or d_hdc

        # Neural-to-HDC projection bridge
        if self.d_model != self.d_hdc:
            self.embed_to_hdc = nn.Linear(self.d_model, self.d_hdc, bias=False)
            self.hdc_to_embed = nn.Linear(self.d_hdc, self.d_model, bias=False)
            nn.init.orthogonal_(self.embed_to_hdc.weight)
            nn.init.orthogonal_(self.hdc_to_embed.weight)
        else:
            self.embed_to_hdc = None
            self.hdc_to_embed = None

        # Codebook for symbolic atom grounding
        self.atom_codebook: Dict[str, torch.Tensor] = {}

    def to_hdc(self, x: torch.Tensor) -> torch.Tensor:
        """Project neural embedding to HDC hypervector."""
        if self.embed_to_hdc is not None:
            h = self.embed_to_hdc(x)
        else:
            h = x
        return F.normalize(h, p=2, dim=-1)

    def to_embedding(self, h: torch.Tensor) -> torch.Tensor:
        """Project HDC hypervector back to neural embedding space."""
        if self.hdc_to_embed is not None:
            x = self.hdc_to_embed(h)
        else:
            x = h
        return F.normalize(x, p=2, dim=-1)

    def get_or_create_atom(self, symbol_name: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Fetch or deterministically generate a quasi-orthogonal atomic basis vector."""
        if symbol_name not in self.atom_codebook:
            seed = abs(hash(symbol_name)) % (2**31 - 1)
            g = torch.Generator().manual_seed(seed)
            v = torch.randn(self.d_hdc, generator=g, device="cpu")
            self.atom_codebook[symbol_name] = F.normalize(v, p=2, dim=-1)
        atom = self.atom_codebook[symbol_name]
        if device is not None:
            return atom.to(device)
        return atom

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Binding: Circular Convolution via Real Fast Fourier Transform (RFFT).
        Properties: Commutative, associative, distributes over bundling.
        a (*) b produces a hypervector that is quasi-orthogonal to both a and b.
        """
        b = b.to(a.device)
        A = torch.fft.rfft(a, dim=-1)
        B = torch.fft.rfft(b, dim=-1)
        # Elementwise multiplication in frequency domain
        C = A * B
        c = torch.fft.irfft(C, n=self.d_hdc, dim=-1)
        return F.normalize(c, p=2, dim=-1)

    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Unbinding: Circular Correlation via complex conjugation in frequency domain.
        Given c = a (*) b, unbind(c, b) recovers an approximate reconstruction of a.
        """
        key = key.to(bound.device)
        C = torch.fft.rfft(bound, dim=-1)
        K = torch.fft.rfft(key, dim=-1)
        # Multiplication by complex conjugate
        A_approx = C * torch.conj(K)
        a = torch.fft.irfft(A_approx, n=self.d_hdc, dim=-1)
        return F.normalize(a, p=2, dim=-1)

    def bundle(self, vectors: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        Bundling: Superposition of concepts (algebraic union).
        The bundled vector is simultaneously similar to all its components.
        """
        if len(vectors) == 0:
            raise ValueError("Cannot bundle an empty list of vectors.")

        dev = vectors[0].device
        vecs = [v.to(dev) for v in vectors]
        if weights is None:
            summed = torch.stack(vecs, dim=0).sum(dim=0)
        else:
            w_tensor = torch.tensor(weights, device=dev, dtype=vecs[0].dtype)
            stacked = torch.stack(vecs, dim=0)  # [N, ...]
            for _ in range(stacked.dim() - 1):
                w_tensor = w_tensor.unsqueeze(-1)
            summed = (stacked * w_tensor).sum(dim=0)

        return F.normalize(summed, p=2, dim=-1)

    def permute(self, v: torch.Tensor, shift: int = 1) -> torch.Tensor:
        """
        Permutation: Circular roll by `shift` positions.
        Encodes sequence positions or tree child indices without learned parameters.
        """
        return torch.roll(v, shifts=shift, dims=-1)

    def inverse_permute(self, v: torch.Tensor, shift: int = 1) -> torch.Tensor:
        """Inverse permutation: Circular roll by `-shift`."""
        return torch.roll(v, shifts=-shift, dims=-1)

    def create_relation(self, role: str, filler: Union[str, torch.Tensor]) -> torch.Tensor:
        """
        Create a role-filler pair: ROLE (*) FILLER.
        """
        dev = filler.device if isinstance(filler, torch.Tensor) else None
        r_vec = self.get_or_create_atom(role, device=dev)
        if isinstance(filler, str):
            f_vec = self.get_or_create_atom(filler, device=dev)
        else:
            f_vec = filler
        return self.bind(r_vec, f_vec)

    def create_proposition(self, predicate: str, subject: Union[str, torch.Tensor], object_: Union[str, torch.Tensor]) -> torch.Tensor:
        """
        Construct a structured proposition:
            P = PREDICATE (*) (Pi^0(SUBJECT) + Pi^1(OBJECT))
        """
        dev = subject.device if isinstance(subject, torch.Tensor) else (object_.device if isinstance(object_, torch.Tensor) else None)
        pred_vec = self.get_or_create_atom(predicate, device=dev)
        s_vec = self.get_or_create_atom(subject, device=dev) if isinstance(subject, str) else subject
        o_vec = self.get_or_create_atom(object_, device=dev) if isinstance(object_, str) else object_

        args_bundled = self.bundle([self.permute(s_vec, 0), self.permute(o_vec, 1)])
        return self.bind(pred_vec, args_bundled)

    def extract_role_filler(self, proposition: torch.Tensor, predicate: str, role_index: int = 0) -> torch.Tensor:
        """
        Query a structured proposition to unbind an argument:
            args = unbind(P, PREDICATE)
            arg_i = Pi^{-i}(args)
        """
        pred_vec = self.get_or_create_atom(predicate, device=proposition.device)
        args_approx = self.unbind(proposition, pred_vec)
        return self.inverse_permute(args_approx, shift=role_index)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity between two hypervectors."""
        b = b.to(a.device)
        a_n = F.normalize(a, p=2, dim=-1)
        b_n = F.normalize(b, p=2, dim=-1)
        return float(torch.sum(a_n * b_n, dim=-1).item())
