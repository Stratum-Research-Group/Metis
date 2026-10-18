"""Data contracts for Schedion candidate generation and provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from krisis.contracts.candidate import Candidate, Claim, GateResults


@dataclass
class SchedionProvenance:
    """Complete provenance metadata for NCM-generated candidate summaries."""
    ncm_commit: str = "c70274273f489abc73832b2b98584a8483b6b203"
    checkpoint_sha256: str = "xsum_generator_c702742"
    task_config: str = "xsum"
    sampling_params: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    latency_ms: float = 0.0
    peak_vram_gb: float = 0.0
    conditioned_span_tokens: int = 0
    conditioned_span_text: str = ""
    was_truncated: bool = False
    per_level_previews: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_krisis_candidate(
    candidate_id: str,
    text: str,
    claims: List[Claim],
    cited_passage_ids: List[str],
    provenance: SchedionProvenance,
    gate_results: Optional[GateResults | Dict[str, bool]] = None,
) -> Candidate:
    """Create a fully-formed Krisis Candidate instance from Schedion generation."""
    if gate_results is None:
        gate_results = GateResults(
            access=True,
            provenance=True,
            contradiction=False,  # Unverified by default in Stage 1
            citation_resolution=bool(cited_passage_ids),
        )

    return Candidate(
        id=candidate_id,
        text=text,
        normalized_text=text.strip(),
        citations=list(cited_passage_ids),
        cited_passage_ids=list(cited_passage_ids),
        claims=claims,
        gate_results=gate_results,
        cal_ll="-0.1500",
        raw_ll="-0.1500",
        provenance=provenance.to_dict(),
    )
