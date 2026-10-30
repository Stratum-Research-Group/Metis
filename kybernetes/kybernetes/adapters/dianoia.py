"""Dianoia thought graph deliberation adapter with zero-edge flat retrieval fallback."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from kybernetes.contracts.turn import Citation

try:
    from stoma.filters import DeterministicSpanFilter
except ImportError:
    DeterministicSpanFilter = None  # type: ignore


class DianoiaDeliberationResult(BaseModel):
    """Result of Dianoia graph deliberation."""
    success: bool
    walk_steps: int = 0
    edges_traversed: int = 0
    zero_edge_fallback_used: bool = False
    fallback_reason: Optional[str] = None
    deliberation_summary: str = ""
    refined_citations: List[Citation] = Field(default_factory=list)
    conflicted: bool = False
    is_simulated: bool = Field(default=True, description="Whether deliberation was executed via simulated adapter")
    consistency_ratio: float = 1.0
    consistency_fallback_used: bool = False


class DianoiaAdapter:
    """
    Adapter for Dianoia thought-graph deliberation.
    Gracefully handles zero-edge states by falling back to flat passage retrieval.
    Explicitly discloses simulated adapter status when running without a verified live Dianoia engine.
    When configured with a live WalkController or live dianoia engine, delegates deliberation
    to the live thought-graph traversal engine.
    """

    def __init__(
        self,
        dianoia_module: Any = None,
        walk_controller: Any = None,
        controller_policy: Optional[Any] = None,
    ):
        self.dianoia_module = dianoia_module
        self.walk_controller = walk_controller
        if self.walk_controller is None and dianoia_module is not None:
            if hasattr(dianoia_module, "walk_controller"):
                self.walk_controller = getattr(dianoia_module, "walk_controller")
            elif hasattr(dianoia_module, "walk"):
                self.walk_controller = dianoia_module

        self.controller_policy = controller_policy
        self._is_live = bool(
            self.walk_controller is not None and callable(getattr(self.walk_controller, "walk", None))
        )
        self.is_simulated = not self._is_live

    def deliberate(
        self,
        query: str,
        seed_citations: List[Citation],
        max_steps: int = 5,
        force_zero_edges: bool = False,
        controller_policy: Optional[Any] = None,
        consistency_floor: float = 0.95,
        fallback_to_pure_text_on_inconsistency: bool = True,
        extracted_graph_block: Optional[Dict[str, Any]] = None,
    ) -> DianoiaDeliberationResult:
        """
        Conduct bounded thought-graph walk.
        If graph connectivity is absent (0 edges) or consistency drops below floor,
        cleanly activates flat text fallback. Discloses simulation or live engine status honestly.
        """
        # Dynamic consistency verification against raw passage text
        if extracted_graph_block and DeterministicSpanFilter is not None:
            source_text = "\n".join(c.text for c in seed_citations)
            nodes = extracted_graph_block.get("nodes", [])
            edges = extracted_graph_block.get("edges", [])
            flt = DeterministicSpanFilter()
            _, _, ratio = flt.filter_graph_block(nodes, edges, source_text)
            if ratio < consistency_floor and fallback_to_pure_text_on_inconsistency:
                return DianoiaDeliberationResult(
                    success=True,
                    walk_steps=0,
                    edges_traversed=0,
                    zero_edge_fallback_used=True,
                    fallback_reason=f"Graph consistency ratio {ratio:.3f} fell below floor {consistency_floor}. Cleanly fell back to pure text passages.",
                    deliberation_summary=f"Direct evidence aggregation via flat passage fallback (inconsistent graph evicted at ratio {ratio:.3f} < {consistency_floor}).",
                    refined_citations=seed_citations,
                    conflicted=False,
                    is_simulated=self.is_simulated,
                    consistency_ratio=round(ratio, 4),
                    consistency_fallback_used=True,
                )

        # Case 1: Simulated adapter execution
        if self.is_simulated:
            if force_zero_edges or len(seed_citations) <= 1:
                return DianoiaDeliberationResult(
                    success=True,
                    walk_steps=0,
                    edges_traversed=0,
                    zero_edge_fallback_used=True,
                    fallback_reason="Zero thought-graph edges discovered. Executing flat passage reranking fallback.",
                    deliberation_summary="Direct evidence aggregation via flat passage reranking (simulated adapter).",
                    refined_citations=seed_citations,
                    conflicted=False,
                    is_simulated=True,
                )

            walk_steps = min(max_steps, len(seed_citations) + 1)
            edges_traversed = walk_steps - 1

            summary = (
                f"Deliberated across {len(seed_citations)} evidence passages in {walk_steps} steps "
                f"traversing {edges_traversed} conceptual relation edges (simulated adapter)."
            )

            return DianoiaDeliberationResult(
                success=True,
                walk_steps=walk_steps,
                edges_traversed=edges_traversed,
                zero_edge_fallback_used=False,
                fallback_reason=None,
                deliberation_summary=summary,
                refined_citations=seed_citations,
                conflicted=False,
                is_simulated=True,
            )

        # Case 2: Live deliberation backed by WalkController
        if force_zero_edges or len(seed_citations) == 0:
            return DianoiaDeliberationResult(
                success=True,
                walk_steps=0,
                edges_traversed=0,
                zero_edge_fallback_used=True,
                fallback_reason="Zero-edge condition triggered: executing live flat passage retrieval fallback.",
                deliberation_summary="Direct evidence aggregation via flat passage reranking (live engine fallback).",
                refined_citations=seed_citations,
                conflicted=False,
                is_simulated=False,
            )

        active_policy = controller_policy if controller_policy is not None else self.controller_policy
        try:
            walk_res = self.walk_controller.walk(
                question=query,
                controller_policy=active_policy,
            )
        except Exception as exc:
            # Clean degradation to flat retrieval on traversal failure
            return DianoiaDeliberationResult(
                success=True,
                walk_steps=0,
                edges_traversed=0,
                zero_edge_fallback_used=True,
                fallback_reason=f"Thought-graph traversal exception ({exc}). Activated flat retrieval fallback.",
                deliberation_summary=f"Direct evidence aggregation via flat passage fallback after walk error: {exc}",
                refined_citations=seed_citations,
                conflicted=False,
                is_simulated=False,
            )

        path = walk_res.get("path") or walk_res.get("walk_path") or []
        visited = walk_res.get("visited_passages") or []
        steps_used = walk_res.get("steps_used", len(path))
        edges_traversed = max(0, len(path) - 1)
        answer = walk_res.get("answer") or walk_res.get("final_answer") or ""
        conflicted = bool(walk_res.get("conflicted", False))

        refined_citations: List[Citation] = []
        if visited:
            for idx, p in enumerate(visited):
                passage_id = str(p.get("doc_id", p.get("node_id", f"p-{idx+1}")))
                title = str(p.get("title", p.get("source_title", passage_id)))
                text = str(p.get("text", p.get("content", "")))
                score = float(p.get("score", 1.0))
                refined_citations.append(
                    Citation(
                        citation_id=f"[{idx+1}]",
                        passage_id=passage_id,
                        source_id=str(p.get("source_id", passage_id)),
                        source_title=title,
                        text=text,
                        score=score,
                        is_simulated=False,
                    )
                )
        else:
            refined_citations = seed_citations

        # Dynamic consistency fallback if walk returned conflicted evidence and fallback requested
        if conflicted and fallback_to_pure_text_on_inconsistency:
            return DianoiaDeliberationResult(
                success=True,
                walk_steps=steps_used,
                edges_traversed=0,
                zero_edge_fallback_used=True,
                fallback_reason="Graph walk flagged internal contradiction/conflict. Falling back to pure text passages.",
                deliberation_summary=f"Direct evidence aggregation via flat passage fallback (graph contradiction flagged).",
                refined_citations=seed_citations,
                conflicted=True,
                is_simulated=False,
                consistency_ratio=0.0,
                consistency_fallback_used=True,
            )

        is_zero_edge = (edges_traversed == 0)
        fallback_reason = "No graph relation edges traversed. Cleanly activated flat passage fallback." if is_zero_edge else None

        summary = answer or (
            f"Deliberated across {len(refined_citations)} evidence passages in {steps_used} steps "
            f"traversing {edges_traversed} live conceptual relation edges."
        )

        return DianoiaDeliberationResult(
            success=True,
            walk_steps=steps_used,
            edges_traversed=edges_traversed,
            zero_edge_fallback_used=is_zero_edge,
            fallback_reason=fallback_reason,
            deliberation_summary=summary,
            refined_citations=refined_citations,
            conflicted=conflicted,
            is_simulated=False,
        )
