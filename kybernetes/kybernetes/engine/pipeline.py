"""Sequential, inspectable execution pipeline driving Kybernetes conversational turns."""

import asyncio
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from kybernetes.adapters.dianoia import DianoiaAdapter
from kybernetes.adapters.hestia import HestiaAdapter
from kybernetes.adapters.krisis import KrisisAdapter, KrisisVerificationReport
from kybernetes.adapters.lthcm import LTHCMAdapter
from kybernetes.adapters.phylax import PhylaxAdapter
from kybernetes.budget.engine import BudgetEngine
from kybernetes.budget.models import (
    PhasePlan,
    TaskPlan,
    UsageEvent,
    UsageMeasurement,
)
from kybernetes.contracts.enums import (
    AgentSemanticState,
    BudgetUnit,
    CheckpointAction,
    EventType,
    RouteTag,
    StageName,
    TurnOutcomeStatus,
)
from kybernetes.contracts.plan import ExecutionPlan, StageOutcome
from kybernetes.contracts.tool_spec import ToolCallRequest
from kybernetes.contracts.trace import TraceRecord
from kybernetes.contracts.turn import (
    Citation,
    ShowWorkInfo,
    TurnEvent,
    TurnRequest,
)
from kybernetes.engine.session import SessionManager, SessionTurn
from kybernetes.routing.router import PlanRouter
from kybernetes.tools.registry import ToolRegistry


class ExecutionPipeline:
    """
    Coordinates end-to-end turn execution across the Metis stack.
    Enforces budget boundaries, capability authorization, deterministic tool safety,
    evidence citations, and honest deliberative status.
    """

    def __init__(
        self,
        router: PlanRouter,
        budget_engine: BudgetEngine,
        tool_registry: ToolRegistry,
        hestia_adapter: HestiaAdapter,
        lthcm_adapter: LTHCMAdapter,
        dianoia_adapter: DianoiaAdapter,
        phylax_adapter: PhylaxAdapter,
        krisis_adapter: KrisisAdapter,
        session_manager: SessionManager,
        thinking_budget_steps: int = 10,
    ):
        self.router = router
        self.budget_engine = budget_engine
        self.tool_registry = tool_registry
        self.hestia = hestia_adapter
        self.lthcm = lthcm_adapter
        self.dianoia = dianoia_adapter
        self.phylax = phylax_adapter
        self.krisis = krisis_adapter
        self.sessions = session_manager
        self.thinking_budget_steps = thinking_budget_steps
        self.max_traces: int = 1000
        self._traces: Dict[str, TraceRecord] = {}

    def _record_trace(self, turn_id: str, trace: TraceRecord) -> None:
        self._traces[turn_id] = trace
        if len(self._traces) > self.max_traces:
            oldest_key = next(iter(self._traces))
            self._traces.pop(oldest_key, None)

    def _record_terminal_state(
        self,
        trace_id: str,
        turn_id: str,
        session_id: str,
        plan: Any,
        components_used: List[str],
        stage_outcomes: List[StageOutcome],
        total_lat: float,
        final_status: TurnOutcomeStatus,
        zero_edge_fallback: bool,
        is_turn_simulated: bool,
        v_report: Optional[KrisisVerificationReport],
        query: str,
        response_text: str,
        emitted_events: List[TurnEvent],
        budget_usage: Optional[Any] = None,
    ) -> None:
        trace = TraceRecord(
            trace_id=trace_id,
            turn_id=turn_id,
            session_id=session_id,
            route_label=plan.route_label if plan else "unknown",
            route_tags=plan.route_tags if plan else [],
            components_used=list(set(components_used)),
            dependency_versions={
                "kybernetes": "0.1.0",
                "hestia": getattr(self.hestia, "default_model", "local"),
                "lthcm": getattr(self.lthcm, "generation_id", None) or ("simulated_demo" if getattr(self.lthcm, "is_simulated", True) else "live_index"),
                "dianoia": "live_engine" if not getattr(self.dianoia, "is_simulated", True) else "simulated_adapter",
                "phylax": "guarded_gateway" if not getattr(self.phylax, "is_simulated", True) else "local_approval_shim",
                "krisis": "standalone_selector" if not getattr(self.krisis, "is_simulated", True) else "heuristic_half_even",
            },
            stage_outcomes=stage_outcomes,
            total_latency_ms=round(total_lat, 2),
            final_status=final_status,
            zero_edge_fallback_used=zero_edge_fallback,
            budget_usage=budget_usage or {},
            metadata={
                "served_model": getattr(self.hestia, "default_model", "unknown"),
                "verifier_mode": v_report.verifier_mode if v_report else None,
                "is_simulated": is_turn_simulated,
                "verifier_status": final_status.value,
                "verifier_score": v_report.score if v_report else None,
            },
        )
        self._record_trace(turn_id, trace)
        self.sessions.record_turn_events(
            session_id,
            SessionTurn(
                turn_id=turn_id,
                query=query,
                response_text=response_text,
                events=emitted_events,
                status=final_status.value,
            ),
        )
        self.sessions.unregister_active_turn(turn_id)

    def get_trace(self, turn_id: str) -> Optional[TraceRecord]:
        return self._traces.get(turn_id)

    async def execute_turn_stream(
        self,
        request: TurnRequest,
    ) -> AsyncIterator[TurnEvent]:
        """
        Execute turn plan stage by stage, streaming TurnEvent items over SSE.
        """
        start_time = time.perf_counter()
        turn_id = request.turn_id or f"turn-{uuid.uuid4().hex[:8]}"
        self.sessions.register_active_turn(turn_id)
        trace_id = f"trace-{uuid.uuid4().hex[:8]}"
        seq_num = 0
        emitted_events: List[TurnEvent] = []

        def _record_event(ev: TurnEvent) -> None:
            emitted_events.append(ev)
            self.sessions.append_turn_event(turn_id, ev)

        is_turn_simulated = bool(
            getattr(self.lthcm, "is_simulated", True)
            or getattr(self.dianoia, "is_simulated", True)
        )

        # 1. Event: STARTED
        e_started = TurnEvent(
            turn_id=turn_id,
            seq_num=seq_num,
            event_type=EventType.STARTED.value,
            payload={"session_id": request.session_id, "query": request.query},
            status=TurnOutcomeStatus.DRAFT.value,
        )
        _record_event(e_started)
        yield e_started
        seq_num += 1

        # 2. Build Plan & Event: PLAN_SELECTED
        plan: ExecutionPlan = self.router.build_plan(request)
        plan.turn_id = turn_id

        e_plan = TurnEvent(
            turn_id=turn_id,
            seq_num=seq_num,
            event_type=EventType.PLAN_SELECTED.value,
            payload={
                "route": plan.route_label,
                "route_tags": [t.value for t in plan.route_tags],
                "stages": [s.stage.value for s in plan.stages],
                "components": ["Hestia", "Phylax", "Krisis"]
                + (["LTHCM"] if RouteTag.MEMORY in plan.route_tags else [])
                + (["Dianoia"] if RouteTag.DELIBERATE in plan.route_tags else []),
            },
            status=TurnOutcomeStatus.DRAFT.value,
        )
        _record_event(e_plan)
        yield e_plan
        seq_num += 1

        # Emit initial PLANNING state (Herdr FSM)
        e_state_planning = TurnEvent(
            turn_id=turn_id,
            seq_num=seq_num,
            event_type=EventType.STATE_CHANGED.value,
            payload={
                "to_state": AgentSemanticState.PLANNING.value,
                "reason": f"Plan selected ({plan.route_label}). Beginning stage execution.",
            },
            status=TurnOutcomeStatus.DRAFT.value,
        )
        _record_event(e_state_planning)
        yield e_state_planning
        seq_num += 1

        # Context state across pipeline stages
        citations: List[Citation] = []
        deliberation_summary = ""
        zero_edge_fallback = False
        tool_output: Optional[Dict[str, Any]] = None
        v_report: Optional[KrisisVerificationReport] = None
        response_text = ""
        succeeded = False
        stage_outcomes: List[StageOutcome] = []
        components_used = ["Hestia", "Phylax", "Krisis"]
        timings_ms: Dict[str, float] = {}

        # Set up TaskPlan in budget engine
        budget_phases = [
            PhasePlan(
                phase_id=f"phase-{s.stage.value.lower()}",
                phase_name=s.stage.value,
                target_criteria=s.description,
                allocated_input_tokens=300.0,
                allocated_output_tokens=200.0,
                allocated_time_seconds=5.0,
            )
            for s in plan.stages
        ]
        task_plan = TaskPlan(
            task_id=f"task-{turn_id}",
            turn_id=turn_id,
            cohort="standard_turn",
            phases=budget_phases,
            limit_input_tokens=plan.budget_limits.get("input_tokens", 1500.0),
            limit_output_tokens=plan.budget_limits.get("output_tokens", 1000.0),
            limit_time_seconds=plan.budget_limits.get("time_seconds", 30.0),
            hard_stop_enabled=False,
        )
        self.budget_engine.prepare_task_estimate(task_plan)

        executed_tool_name: Optional[str] = None

        # Iterate through stages
        for stage_plan in plan.stages:
            stage_start = time.perf_counter()

            # Check cancellation
            if self.sessions.is_turn_cancelled(turn_id):
                e_cancel = TurnEvent(
                    turn_id=turn_id,
                    seq_num=seq_num,
                    event_type=EventType.CANCELLED.value,
                    payload={"reason": "Turn was cancelled by user"},
                    status=TurnOutcomeStatus.ABSTAINED.value,
                )
                _record_event(e_cancel)
                yield e_cancel
                return

            if stage_plan.stage == StageName.AUTHORIZE:
                # Pre-dispatch budget checkpoint
                decision = self.budget_engine.evaluate_checkpoint(task_plan.task_id)
                allowed = self.phylax.enforce_budget_guardrail(task_plan.task_id, decision.action, decision.reason)
                if not allowed:
                    e_fail = TurnEvent(
                        turn_id=turn_id,
                        seq_num=seq_num,
                        event_type=EventType.FAILED.value,
                        payload={"error": f"Execution halted by budget guardrail: {decision.reason}"},
                        status=TurnOutcomeStatus.FAILED.value,
                    )
                    _record_event(e_fail)
                    yield e_fail
                    return
                stage_lat = (time.perf_counter() - stage_start) * 1000
                timings_ms["authorize"] = round(stage_lat, 2)
                stage_outcomes.append(StageOutcome(stage=StageName.AUTHORIZE, success=True, latency_ms=stage_lat))

            elif stage_plan.stage == StageName.RETRIEVE:
                components_used.append("LTHCM")
                citations = self.lthcm.retrieve(request.query, top_k=3)
                if citations:
                    e_ev = TurnEvent(
                        turn_id=turn_id,
                        seq_num=seq_num,
                        event_type=EventType.EVIDENCE.value,
                        payload={"citations": [c.model_dump() for c in citations]},
                        status=TurnOutcomeStatus.DRAFT.value,
                    )
                    _record_event(e_ev)
                    yield e_ev
                    seq_num += 1

                stage_lat = (time.perf_counter() - stage_start) * 1000
                timings_ms["retrieve"] = round(stage_lat, 2)
                stage_outcomes.append(
                    StageOutcome(
                        stage=StageName.RETRIEVE,
                        success=True,
                        latency_ms=stage_lat,
                        output={"citation_count": len(citations)},
                    )
                )

            elif stage_plan.stage == StageName.DELIBERATE:
                components_used.append("Dianoia")
                delib_steps = stage_plan.params.get("max_steps", self.thinking_budget_steps)
                consistency_floor = stage_plan.params.get("graph_consistency_floor", 0.95)
                fallback_on_inconsistency = stage_plan.params.get("fallback_to_pure_text_on_inconsistency", True)
                constructive_plan = stage_plan.params.get("two_pass_constructive_planning", True)
                d_result = self.dianoia.deliberate(
                    query=request.query,
                    seed_citations=citations,
                    max_steps=delib_steps,
                    consistency_floor=consistency_floor,
                    fallback_to_pure_text_on_inconsistency=fallback_on_inconsistency,
                    two_pass_constructive_planning=constructive_plan,
                    hestia_client=self.hestia,
                )
                zero_edge_fallback = d_result.zero_edge_fallback_used
                deliberation_summary = d_result.deliberation_summary

                stage_lat = (time.perf_counter() - stage_start) * 1000
                timings_ms["deliberate"] = round(stage_lat, 2)
                stage_outcomes.append(
                    StageOutcome(
                        stage=StageName.DELIBERATE,
                        success=True,
                        latency_ms=stage_lat,
                        output={
                            "zero_edge_fallback": zero_edge_fallback,
                            "consistency_fallback": getattr(d_result, "consistency_fallback_used", False),
                            "consistency_ratio": getattr(d_result, "consistency_ratio", 1.0),
                            "constructive_planning": getattr(d_result, "constructive_planning_used", False),
                            "walk_steps": d_result.walk_steps,
                            "edges_traversed": d_result.edges_traversed,
                            "max_deliberation_steps": delib_steps,
                        },
                    )
                )

            elif stage_plan.stage == StageName.TOOL:
                tool_name = stage_plan.params.get("tool_name", "calculator")
                needs_interactive = stage_plan.params.get("needs_interactive_approval", False)

                # Determine actual tool capability, target, and arguments to be executed
                tool_args = {}
                if tool_name == "calculator":
                    import re
                    m = re.search(r"(?:compute|calculate|evaluate)\s+(.+)", request.query, re.IGNORECASE)
                    if m:
                        expr = m.group(1).rstrip(".?!")
                    else:
                        match = PlanRouter.MATH_REGEX.search(request.query)
                        expr = match.group(0) if match else "2 + 2"
                        expr = expr.replace("calculate", "").replace("compute", "").strip() or "2 + 2"
                    tool_args = {"expression": expr.strip()}
                    capability = "compute"
                    target = "math_engine"
                else:
                    tool_args = stage_plan.params.get("tool_args", {"pattern": request.query.split()[-1] if request.query.split() else "metis"})
                    capability = stage_plan.params.get("capability", "write_file")
                    target = stage_plan.params.get("target", tool_args.get("path", tool_args.get("target", "/tmp/metis_task.txt")))

                # Check Phylax permission gate with exact capability, target, and arguments
                permitted, app_req = self.phylax.is_action_permitted(
                    capability=capability,
                    target=target,
                    arguments=tool_args,
                )
                if not permitted:
                    if app_req and needs_interactive:
                        e_state_block = TurnEvent(
                            turn_id=turn_id,
                            seq_num=seq_num,
                            event_type=EventType.STATE_CHANGED.value,
                            payload={
                                "to_state": AgentSemanticState.AWAITING_APPROVAL.value,
                                "reason": f"Phylax gating approval required for: {capability} -> {target}",
                                "approval_id": app_req.get("approval_id"),
                            },
                            status=TurnOutcomeStatus.DRAFT.value,
                        )
                        _record_event(e_state_block)
                        yield e_state_block
                        seq_num += 1

                        e_ask = TurnEvent(
                            turn_id=turn_id,
                            seq_num=seq_num,
                            event_type=EventType.ASK_APPROVAL.value,
                            payload={"approval": app_req},
                            status=TurnOutcomeStatus.DRAFT.value,
                        )
                        _record_event(e_ask)
                        yield e_ask
                        seq_num += 1

                        # Await decision or cancellation concurrently
                        cancel_event = self.sessions.get_cancellation_event(turn_id)
                        approval_task = asyncio.create_task(self.phylax.wait_for_approval(app_req["approval_id"]))
                        cancel_task = asyncio.create_task(cancel_event.wait())

                        done, pending = await asyncio.wait(
                            [approval_task, cancel_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for t in pending:
                            t.cancel()

                        if cancel_task in done or self.sessions.is_turn_cancelled(turn_id):
                            cancel_msg = "Turn was cancelled while awaiting user approval."
                            e_comp = TurnEvent(
                                turn_id=turn_id,
                                seq_num=seq_num,
                                event_type=EventType.COMPLETED.value,
                                payload={"status": "cancelled", "reason": cancel_msg},
                                status=TurnOutcomeStatus.FAILED.value,
                            )
                            _record_event(e_comp)
                            yield e_comp
                            self._record_terminal_state(
                                trace_id=trace_id,
                                turn_id=turn_id,
                                session_id=request.session_id,
                                plan=plan,
                                components_used=components_used,
                                stage_outcomes=stage_outcomes,
                                total_lat=(time.perf_counter() - start_time) * 1000,
                                final_status=TurnOutcomeStatus.FAILED,
                                zero_edge_fallback=False,
                                is_turn_simulated=is_turn_simulated,
                                v_report=None,
                                query=request.query,
                                response_text=cancel_msg,
                                emitted_events=emitted_events,
                            )
                            return

                        user_dec = approval_task.result()
                        e_res = TurnEvent(
                            turn_id=turn_id,
                            seq_num=seq_num,
                            event_type=EventType.APPROVAL_RESOLVED.value,
                            payload={"approval_id": app_req["approval_id"], "decision": user_dec},
                            status=TurnOutcomeStatus.DRAFT.value,
                        )
                        _record_event(e_res)
                        yield e_res
                        seq_num += 1

                        if user_dec != "allow_once":
                            denied_msg = "Action was denied by user. Operation safely cancelled without side-effects."
                            e_delta = TurnEvent(
                                turn_id=turn_id,
                                seq_num=seq_num,
                                event_type=EventType.DELTA.value,
                                payload={"text": denied_msg},
                                status=TurnOutcomeStatus.DENIED.value,
                            )
                            _record_event(e_delta)
                            yield e_delta
                            seq_num += 1

                            denied_lat = round((time.perf_counter() - start_time) * 1000, 2)
                            e_comp = TurnEvent(
                                turn_id=turn_id,
                                seq_num=seq_num,
                                event_type=EventType.COMPLETED.value,
                                payload={
                                    "route_summary": {
                                        "route": plan.route_label,
                                        "action_status": "denied",
                                        "components_used": list(set(components_used)),
                                        "latency_ms": denied_lat,
                                    }
                                },
                                status=TurnOutcomeStatus.DENIED.value,
                            )
                            _record_event(e_comp)
                            yield e_comp
                            self._record_terminal_state(
                                trace_id=trace_id,
                                turn_id=turn_id,
                                session_id=request.session_id,
                                plan=plan,
                                components_used=components_used,
                                stage_outcomes=stage_outcomes,
                                total_lat=denied_lat,
                                final_status=TurnOutcomeStatus.DENIED,
                                zero_edge_fallback=False,
                                is_turn_simulated=is_turn_simulated,
                                v_report=None,
                                query=request.query,
                                response_text=denied_msg,
                                emitted_events=emitted_events,
                            )
                            return
                    else:
                        denied_msg = f"Action was denied by security policy: {capability} -> {target}"
                        e_delta = TurnEvent(
                            turn_id=turn_id,
                            seq_num=seq_num,
                            event_type=EventType.DELTA.value,
                            payload={"text": denied_msg},
                            status=TurnOutcomeStatus.DENIED.value,
                        )
                        _record_event(e_delta)
                        yield e_delta
                        seq_num += 1

                        denied_lat = round((time.perf_counter() - start_time) * 1000, 2)
                        e_comp = TurnEvent(
                            turn_id=turn_id,
                            seq_num=seq_num,
                            event_type=EventType.COMPLETED.value,
                            payload={
                                "route_summary": {
                                    "route": plan.route_label,
                                    "action_status": "denied",
                                    "components_used": list(set(components_used)),
                                    "latency_ms": denied_lat,
                                }
                            },
                            status=TurnOutcomeStatus.DENIED.value,
                        )
                        _record_event(e_comp)
                        yield e_comp
                        self._record_terminal_state(
                            trace_id=trace_id,
                            turn_id=turn_id,
                            session_id=request.session_id,
                            plan=plan,
                            components_used=components_used,
                            stage_outcomes=stage_outcomes,
                            total_lat=denied_lat,
                            final_status=TurnOutcomeStatus.DENIED,
                            zero_edge_fallback=False,
                            is_turn_simulated=is_turn_simulated,
                            v_report=None,
                            query=request.query,
                            response_text=denied_msg,
                            emitted_events=emitted_events,
                        )
                        return

                if self.sessions.is_turn_cancelled(turn_id):
                    cancel_msg = "Turn was cancelled before tool execution."
                    e_comp = TurnEvent(
                        turn_id=turn_id,
                        seq_num=seq_num,
                        event_type=EventType.COMPLETED.value,
                        payload={"status": "cancelled", "reason": cancel_msg},
                        status=TurnOutcomeStatus.FAILED.value,
                    )
                    _record_event(e_comp)
                    yield e_comp
                    self._record_terminal_state(
                        trace_id=trace_id,
                        turn_id=turn_id,
                        session_id=request.session_id,
                        plan=plan,
                        components_used=components_used,
                        stage_outcomes=stage_outcomes,
                        total_lat=(time.perf_counter() - start_time) * 1000,
                        final_status=TurnOutcomeStatus.FAILED,
                        zero_edge_fallback=False,
                        is_turn_simulated=is_turn_simulated,
                        v_report=None,
                        query=request.query,
                        response_text=cancel_msg,
                        emitted_events=emitted_events,
                    )
                    return

                executed_tool_name = tool_name
                approval_id = app_req.get("approval_id") if app_req else None
                t_res = self.tool_registry.execute(ToolCallRequest(tool_id=tool_name, arguments=tool_args, approval_id=approval_id))
                tool_output = t_res.data
                stage_lat = (time.perf_counter() - stage_start) * 1000
                timings_ms["tool"] = round(stage_lat, 2)
                stage_outcomes.append(
                    StageOutcome(
                        stage=StageName.TOOL,
                        success=t_res.success,
                        latency_ms=stage_lat,
                        output={"result": t_res.data, "error": t_res.error},
                    )
                )

            elif stage_plan.stage == StageName.SYNTHESIZE:
                # Compose synthesis prompt
                context_parts = []
                if citations:
                    context_parts.append("EVIDENCE CITATIONS:")
                    for c in citations:
                        context_parts.append(f"{c.citation_id} [{c.source_title}]: {c.text}")
                if deliberation_summary:
                    context_parts.append(f"DELIBERATION NOTE: {deliberation_summary}")
                if tool_output:
                    context_parts.append(f"TOOL EXECUTION RESULT: {tool_output}")

                context_str = "\n".join(context_parts)
                prompt = (
                    f"Context Information:\n{context_str}\n\n"
                    f"User Query: {request.query}\n"
                    f"State the key facts and chronological timeline constructively from the evidence passages before providing your direct answer. "
                    f"Provide a direct, accurate, and concise response using citations [1], [2] where applicable."
                )

                # Check if Hestia is live; if not, generate grounded synthetic response
                h_health = await self.hestia.check_health()
                succeeded = False
                streamed_delta_count = 0
                if h_health.get("available"):
                    async for chunk in self.hestia.stream_chat(prompt=prompt):
                        if chunk.get("error"):
                            # Hestia returned error (e.g. 401 Unauthorized or model error); fall back
                            err_msg = chunk.get("error")
                            if streamed_delta_count > 0:
                                # Finding B1: Deltas were already streamed to client.
                                # Prevent concatenating unrelated fallback text onto partial stream.
                                notice = f"\n\n[Upstream generation interrupted: {err_msg}]"
                                response_text += notice
                                e_notice = TurnEvent(
                                    turn_id=turn_id,
                                    seq_num=seq_num,
                                    event_type=EventType.DELTA.value,
                                    payload={"text": notice, "stream_interrupted": True, "error": err_msg},
                                    status=TurnOutcomeStatus.DRAFT.value,
                                )
                                _record_event(e_notice)
                                yield e_notice
                                seq_num += 1
                                succeeded = True
                            else:
                                response_text = ""
                                succeeded = False
                            break
                        delta = chunk.get("delta", "")
                        if delta:
                            response_text += delta
                            streamed_delta_count += 1
                            e_delta = TurnEvent(
                                turn_id=turn_id,
                                seq_num=seq_num,
                                event_type=EventType.DELTA.value,
                                payload={"text": delta},
                                status=TurnOutcomeStatus.DRAFT.value,
                            )
                            _record_event(e_delta)
                            yield e_delta
                            seq_num += 1
                            succeeded = True
                        usage = chunk.get("usage")
                        if usage:
                            # Ingest actual Hestia usage
                            self.budget_engine.ingest_usage(
                                UsageEvent(
                                    event_id=f"use-{turn_id}-{seq_num}",
                                    task_id=task_plan.task_id,
                                    phase_id="phase-synthesize",
                                    request_id=f"req-{turn_id}",
                                    measurements=[
                                        UsageMeasurement(
                                            unit=BudgetUnit.INPUT_TOKENS,
                                            quantity=float(usage.get("prompt_tokens", 100)),
                                        ),
                                        UsageMeasurement(
                                            unit=BudgetUnit.OUTPUT_TOKENS,
                                            quantity=float(usage.get("completion_tokens", 50)),
                                        ),
                                    ],
                                )
                            )
                if not succeeded:
                    # Deterministic local fallback generator matching evidence
                    if tool_output:
                        res_val = tool_output.get("result", tool_output)
                        response_text = f"The calculation result is {res_val}."
                    elif citations:
                        c_text = " ".join([f"{c.text} {c.citation_id}" for c in citations])
                        response_text = f"Based on local evidence: {c_text}"
                    else:
                        response_text = f"Processed query '{request.query}' via {plan.route_label} pipeline."

                    for word in response_text.split():
                        if self.sessions.is_turn_cancelled(turn_id):
                            break
                        await asyncio.sleep(0.01)
                        e_delta = TurnEvent(
                            turn_id=turn_id,
                            seq_num=seq_num,
                            event_type=EventType.DELTA.value,
                            payload={"text": word + " "},
                            status=TurnOutcomeStatus.DRAFT.value,
                        )
                        _record_event(e_delta)
                        yield e_delta
                        seq_num += 1

                if self.sessions.is_turn_cancelled(turn_id):
                    cancel_msg = "Turn was cancelled during synthesis."
                    e_comp = TurnEvent(
                        turn_id=turn_id,
                        seq_num=seq_num,
                        event_type=EventType.COMPLETED.value,
                        payload={"status": "cancelled", "reason": cancel_msg},
                        status=TurnOutcomeStatus.FAILED.value,
                    )
                    _record_event(e_comp)
                    yield e_comp
                    self._record_terminal_state(
                        trace_id=trace_id,
                        turn_id=turn_id,
                        session_id=request.session_id,
                        plan=plan,
                        components_used=components_used,
                        stage_outcomes=stage_outcomes,
                        total_lat=(time.perf_counter() - start_time) * 1000,
                        final_status=TurnOutcomeStatus.FAILED,
                        zero_edge_fallback=False,
                        is_turn_simulated=is_turn_simulated,
                        v_report=None,
                        query=request.query,
                        response_text=cancel_msg,
                        emitted_events=emitted_events,
                    )
                    return

                    # Ingest simulated usage for offline testing
                    self.budget_engine.ingest_usage(
                        UsageEvent(
                            event_id=f"use-{turn_id}-{seq_num}",
                            task_id=task_plan.task_id,
                            phase_id="phase-synthesize",
                            request_id=f"req-{turn_id}",
                            measurements=[
                                UsageMeasurement(unit=BudgetUnit.INPUT_TOKENS, quantity=150.0),
                                UsageMeasurement(unit=BudgetUnit.OUTPUT_TOKENS, quantity=60.0),
                                UsageMeasurement(unit=BudgetUnit.TIME_SECONDS, quantity=0.25),
                            ],
                        )
                    )

                stage_lat = (time.perf_counter() - stage_start) * 1000
                timings_ms["synthesize"] = round(stage_lat, 2)
                stage_outcomes.append(StageOutcome(stage=StageName.SYNTHESIZE, success=True, latency_ms=stage_lat))

            elif stage_plan.stage == StageName.VERIFY:
                is_tool = tool_output is not None
                v_report = self.krisis.verify_response(request.query, response_text, citations, is_tool_output=is_tool)
                stage_lat = (time.perf_counter() - stage_start) * 1000
                timings_ms["verify"] = round(stage_lat, 2)
                stage_outcomes.append(
                    StageOutcome(
                        stage=StageName.VERIFY,
                        success=(v_report.status not in (TurnOutcomeStatus.FAILED,)),
                        latency_ms=stage_lat,
                        output={
                            "status": v_report.status.value,
                            "score": v_report.score,
                            "supported_claims": v_report.supported_claims,
                            "unsupported_claims": v_report.unsupported_claims,
                            "conflict_detected": v_report.conflict_detected,
                            "is_simulated": v_report.is_simulated,
                            "verifier_mode": v_report.verifier_mode,
                            "explanation": v_report.explanation,
                        },
                    )
                )

        # Reconcile budget
        reconcile_report = self.budget_engine.reconcile_task(task_plan.task_id, final_status="completed")
        total_lat = (time.perf_counter() - start_time) * 1000
        timings_ms["total"] = round(total_lat, 2)

        # Determine final status following actual verification verdict and stage outcomes (Finding R5)
        if any(not so.success for so in stage_outcomes):
            final_status = TurnOutcomeStatus.FAILED
        elif v_report is not None:
            final_status = v_report.status
        else:
            final_status = TurnOutcomeStatus.DRAFT

        # Finding R6: if Hestia generation failed and fallback was used, is_turn_simulated must be True
        is_turn_simulated = (
            is_turn_simulated
            or any(c.is_simulated for c in citations)
            or (v_report.is_simulated if v_report else False)
            or (not succeeded)
        )

        # 7. Completed event with route_summary matching Prosopon
        show_work = ShowWorkInfo(
            turn_id=turn_id,
            route=plan.route_label,
            route_tags=[t.value for t in plan.route_tags],
            components_used=list(set(components_used)),
            citations=citations,
            timings_ms=timings_ms,
            verifier_status=final_status.value,
            verifier_mode=v_report.verifier_mode if v_report else None,
            is_simulated=is_turn_simulated,
            budget_summary=reconcile_report.total_actual,
            zero_edge_fallback_used=zero_edge_fallback,
        )

        e_state_done = TurnEvent(
            turn_id=turn_id,
            seq_num=seq_num,
            event_type=EventType.STATE_CHANGED.value,
            payload={
                "to_state": AgentSemanticState.COMPLETED.value if final_status == TurnOutcomeStatus.VERIFIED else AgentSemanticState.IDLE.value,
                "reason": "Turn execution finished",
            },
            status=final_status.value,
        )
        _record_event(e_state_done)
        yield e_state_done
        seq_num += 1

        e_comp = TurnEvent(
            turn_id=turn_id,
            seq_num=seq_num,
            event_type=EventType.COMPLETED.value,
            payload={
                "route_summary": {
                    "route": plan.route_label,
                    "components_used": list(set(components_used)),
                    "latency_ms": round(total_lat, 2),
                    "verifier_status": final_status.value,
                    "verifier_mode": v_report.verifier_mode if v_report else None,
                    "is_simulated": is_turn_simulated,
                    "zero_edge_fallback_used": zero_edge_fallback,
                    "citations_count": len(citations),
                },
                "show_work": show_work.model_dump(),
            },
            status=final_status.value,
        )
        _record_event(e_comp)
        yield e_comp

        # Record TraceRecord for Dokime and audit, and save turn in session manager
        self._record_terminal_state(
            trace_id=trace_id,
            turn_id=turn_id,
            session_id=request.session_id,
            plan=plan,
            components_used=components_used,
            stage_outcomes=stage_outcomes,
            total_lat=total_lat,
            final_status=final_status,
            zero_edge_fallback=zero_edge_fallback,
            is_turn_simulated=is_turn_simulated,
            v_report=v_report,
            query=request.query,
            response_text=response_text,
            emitted_events=emitted_events,
            budget_usage=reconcile_report.total_actual,
        )
