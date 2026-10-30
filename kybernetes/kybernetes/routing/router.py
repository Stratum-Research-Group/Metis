"""Rule-based plan router generating inspectable, multi-stage execution plans."""

import re
import uuid
from typing import List, Tuple

from kybernetes.contracts.enums import RouteTag, StageName
from kybernetes.contracts.plan import ExecutionPlan, StagePlan
from kybernetes.contracts.turn import TurnRequest


class PlanRouter:
    """
    Deterministic rule-based router.
    Evaluates evidence need, temporal bounds, multi-hop complexity, and tool need.
    Emits an ordered, inspectable ExecutionPlan without siloed execution traps.
    """

    MATH_REGEX = re.compile(r"(\bcalculate\b|\bcompute\b|\bsqrt\b|\bmath\b|[0-9]+[\s]*[\+\-\*\/\^][\s]*[0-9]+)")
    FILE_SEARCH_REGEX = re.compile(r"(\bsearch file\b|\bfind in files\b|\bgrep\b|\bwhere is\b|\bsearch code\b)")
    ACTION_APPROVAL_REGEX = re.compile(r"(\bwrite file\b|\bsave\b|\bdelete\b|\bexport report\b|\bmodify system\b)")
    FACTUAL_KEYWORDS = [
        "what is", "who is", "how does", "explain", "metis", "kybernetes",
        "phylax", "dianoia", "lthcm", "krisis", "prosopon", "paideia", "architecture"
    ]
    MULTI_HOP_KEYWORDS = [
        "relationship", "compare", "contrast", "both", "how do", "connect",
        "deliberate", "multi-hop", "flow", "interact"
    ]

    def build_plan(self, request: TurnRequest) -> ExecutionPlan:
        """Analyze turn query and construct typed ExecutionPlan."""
        q_lower = request.query.lower().strip()
        turn_id = request.turn_id or f"turn-{uuid.uuid4().hex[:8]}"
        plan_id = f"plan-{uuid.uuid4().hex[:8]}"

        route_tags: List[RouteTag] = []
        stages: List[StagePlan] = []

        # 1. AUTHORIZE STAGE: Always active to enforce Phylax permissions and pre-dispatch budget
        stages.append(
            StagePlan(
                stage=StageName.AUTHORIZE,
                description="Enforce Phylax capability boundaries and pre-dispatch budget checkpoint.",
                timeout_ms=5000,
                required_capabilities=["budget_check"],
                enabled=True,
            )
        )

        # Detect tool necessity
        needs_math = bool(self.MATH_REGEX.search(q_lower))
        needs_file_search = bool(self.FILE_SEARCH_REGEX.search(q_lower))
        needs_approval_action = bool(self.ACTION_APPROVAL_REGEX.search(q_lower))
        needs_tool = needs_math or needs_file_search or needs_approval_action

        # Detect evidence / memory retrieval necessity
        is_greeting = any(g in q_lower for g in ["hello", "hi", "greeting", "hey", "how are you"])
        has_factual = any(kw in q_lower for kw in self.FACTUAL_KEYWORDS)
        needs_retrieval = has_factual and not (is_greeting and not any(k in q_lower for k in ["metis", "architecture", "component"]))

        # Detect multi-hop deliberation necessity
        needs_deliberation = any(kw in q_lower for kw in self.MULTI_HOP_KEYWORDS) and needs_retrieval

        # 2. RETRIEVE STAGE (if evidence needed)
        if needs_retrieval:
            route_tags.append(RouteTag.MEMORY)
            stages.append(
                StagePlan(
                    stage=StageName.RETRIEVE,
                    description="Retrieve grounded evidence passages from LTHCM/Stoma index.",
                    timeout_ms=8000,
                    required_capabilities=["read_memory"],
                    enabled=True,
                )
            )

        # 3. DELIBERATE STAGE (if multi-hop synthesis needed)
        if needs_deliberation:
            route_tags.append(RouteTag.DELIBERATE)
            stages.append(
                StagePlan(
                    stage=StageName.DELIBERATE,
                    description="Conduct bounded thought-graph walk via Dianoia with zero-edge fallback and consistency gating.",
                    timeout_ms=10000,
                    required_capabilities=["deliberate"],
                    enabled=True,
                    params={
                        "graph_consistency_floor": 0.95,
                        "fallback_to_pure_text_on_inconsistency": True,
                    },
                )
            )

        # 4. TOOL STAGE (if tool needed)
        if needs_tool:
            route_tags.append(RouteTag.TOOL)
            tool_name = "calculator" if needs_math else ("file_search" if needs_file_search else "write_file")
            stages.append(
                StagePlan(
                    stage=StageName.TOOL,
                    description=f"Execute guarded tool '{tool_name}' under Phylax supervision.",
                    timeout_ms=12000,
                    required_capabilities=["compute" if needs_math else "read_file"],
                    enabled=True,
                    params={
                        "tool_name": tool_name,
                        "needs_interactive_approval": needs_approval_action,
                    },
                )
            )

        # If no memory, deliberation, or tools were needed, it is a DIRECT conversational turn
        if not route_tags:
            route_tags.append(RouteTag.DIRECT)
            route_label = "direct"
        elif RouteTag.DELIBERATE in route_tags:
            route_label = "deliberate_graph"
        elif RouteTag.TOOL in route_tags:
            route_label = "guarded_tool"
        else:
            route_label = "retrieval_augmented"

        # 5. SYNTHESIZE STAGE: Always present for generating turn response
        stages.append(
            StagePlan(
                stage=StageName.SYNTHESIZE,
                description="Synthesize coherent answer through Hestia local model.",
                timeout_ms=25000,
                required_capabilities=["infer"],
                enabled=True,
            )
        )

        # 6. VERIFY STAGE: Krisis quality gating and factual verification
        stages.append(
            StagePlan(
                stage=StageName.VERIFY,
                description="Verify grounded claims against citations using Krisis gates.",
                timeout_ms=5000,
                required_capabilities=["verify"],
                enabled=True,
            )
        )

        # 7. RESPOND STAGE: Final emission of citations, stream completion, and show_work metadata
        stages.append(
            StagePlan(
                stage=StageName.RESPOND,
                description="Package final response tokens, citations, and show_work transparency drawer.",
                timeout_ms=2000,
                enabled=True,
            )
        )

        return ExecutionPlan(
            plan_id=plan_id,
            turn_id=turn_id,
            route_label=route_label,
            route_tags=route_tags,
            stages=stages,
            budget_limits={
                "input_tokens": 1500.0,
                "output_tokens": 1000.0,
                "time_seconds": 30.0,
            },
        )
