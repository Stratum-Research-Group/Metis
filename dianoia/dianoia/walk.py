"""
Deliberative Walk Controller for Dianoia.

The core of Dianoia v1: a WalkController that uses a deliberation model
(served by llama.cpp via OpenAI-compatible endpoint) to direct traversal
over an LTHCMEngine instance's memory graph.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    from lthcm.engine.pipeline import LTHCMEngine
except ImportError:
    LTHCMEngine = Any  # type: ignore

import hashlib

from dianoia.anadrome import (
    AnadromeLogger,
    AnadromePolicy,
    AnswerProbeCache,
    DecisionObservation,
    EarlyExitAnswerPolicy,
    EligibleAction,
    EnvironmentIdentity,
    GreedySimilarityPolicy,
    HighEntropyStochasticPolicy,
    RecordedTransition,
    ReplayTreeRecord,
    SheafContradictionBacktrackPolicy,
)

APPROVED_CONTROLLER_POLICIES: Dict[str, Any] = {
    "baseline_llm": None,
    "greedy_similarity": GreedySimilarityPolicy,
    "early_exit_answer": EarlyExitAnswerPolicy,
    "sheaf_contradiction_backtrack": SheafContradictionBacktrackPolicy,
    "stochastic_softmax": HighEntropyStochasticPolicy,
}
from dianoia.embedding import TextEmbedder
from dianoia.log import append_walk_log, build_walk_record
from dianoia.prompts import (
    ANSWER_PROMPT_TEMPLATE,
    CONTROLLER_SYSTEM_PROMPT,
    REWALK_CONFLICT_PREFIX,
    STEP_PROMPT_TEMPLATE,
    SYNTHESIS_SYSTEM_PROMPT,
)
from dianoia.verify import verify_answer_against_evidence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

@dataclass
class ParsedAction:
    """Result of parsing the model's controller output."""
    action: str = "BACKTRACK"  # "EXPAND", "BACKTRACK", or "ANSWER"
    target: Optional[str] = None  # neighbour_id for EXPAND
    reason: str = ""
    raw_output: str = ""
    parse_failed: bool = False


def parse_controller_output(
    raw: str,
    valid_neighbour_ids: List[str],
) -> ParsedAction:
    """
    Parse the model output into a structured action.

    Parse failure policy (from spec, line 30):
        If the model output is malformed, names a non-existent neighbour,
        or lists multiple actions — log the raw output, treat the step as
        BACKTRACK, and decrement the step budget.

    Args:
        raw: Raw model output string.
        valid_neighbour_ids: List of valid neighbour node IDs for EXPAND.

    Returns:
        ParsedAction with parse_failed=True if the output was malformed.
    """
    result = ParsedAction(raw_output=raw)

    if not raw:
        logger.warning("Parse failure: empty output.")
        result.action = "BACKTRACK"
        result.parse_failed = True
        return result

    # Output must be EXACTLY two lines in order:
    # Line 1: ^ACTION: (EXPAND \S+|BACKTRACK|ANSWER)$
    # Line 2: ^REASON: .+$
    # Any trailing/leading/intermediate blank lines or extra lines are malformed.
    lines = raw.split("\r\n") if "\r\n" in raw else raw.split("\n")
    if len(lines) != 2:
        logger.warning(
            "Parse failure: expected exactly two lines, got %d. Raw output: %r",
            len(lines),
            raw[:200],
        )
        result.action = "BACKTRACK"
        result.parse_failed = True
        return result

    line1, line2 = lines[0], lines[1]

    # Line 1: strict uppercase ACTION: and valid action keyword
    action_match = re.match(r"^ACTION: (EXPAND \S+|BACKTRACK|ANSWER)$", line1)
    if not action_match:
        logger.warning(
            "Parse failure: line 1 '%s' does not match '^ACTION: (EXPAND \\S+|BACKTRACK|ANSWER)$'.",
            line1,
        )
        result.action = "BACKTRACK"
        result.parse_failed = True
        return result

    # Line 2: strict uppercase REASON: and non-empty reason
    reason_match = re.match(r"^REASON: (.+)$", line2)
    if not reason_match:
        logger.warning(
            "Parse failure: line 2 '%s' does not match '^REASON: .+$'.",
            line2,
        )
        result.action = "BACKTRACK"
        result.parse_failed = True
        return result

    action_text = action_match.group(1)
    reason_text = reason_match.group(1).strip()
    result.reason = reason_text

    if action_text == "BACKTRACK":
        result.action = "BACKTRACK"
        result.parse_failed = False
        return result

    if action_text == "ANSWER":
        result.action = "ANSWER"
        result.parse_failed = False
        return result

    if action_text.startswith("EXPAND "):
        target_id = action_text[7:]
        if target_id in valid_neighbour_ids:
            result.action = "EXPAND"
            result.target = target_id
            result.parse_failed = False
            return result
        else:
            logger.warning(
                "Parse failure: EXPAND target '%s' is not a valid neighbour.",
                target_id,
            )
            result.action = "BACKTRACK"
            result.parse_failed = True
            return result

    logger.warning("Parse failure: unrecognised action '%s'.", action_text)
    result.action = "BACKTRACK"
    result.parse_failed = True
    return result


# ---------------------------------------------------------------------------
# Walk Controller
# ---------------------------------------------------------------------------

class WalkController:
    """
    Deliberative walk controller over an LTHCMEngine's memory graph.

    Uses a local LLM (via llama.cpp's OpenAI-compatible endpoint) to decide
    at each step whether to EXPAND to a neighbour, BACKTRACK, or ANSWER.

    Args:
        engine: An LTHCMEngine instance with indexed documents.
        embedder: A TextEmbedder for encoding text to vectors.
        llm_base_url: Base URL for the llama.cpp server's OpenAI endpoint.
        model_name: Model identifier for the OpenAI client.
        max_steps: Step budget per walk (default 8).
        top_k: Number of seed nodes from initial retrieval (default 5).
        temperature: Sampling temperature for controller decisions (default 0.2).
        top_p: Top-p sampling for controller decisions (default 0.9).
        fallback_model_name: Fallback model identifier if primary misbehaves.
        consecutive_failures_to_fallback: Number of consecutive parse failures
            before switching to fallback model (default 3).
        log_path: Path for JSONL walk log file (default ~/.dianoia/walks.jsonl).
        use_stochastic_resonance: Whether to use stochastic resonance seeding
            (True) or fall back to superposition.retrieve (False).
    """

    def __init__(
        self,
        engine: LTHCMEngine,
        embedder: TextEmbedder,
        llm_base_url: str = "http://localhost:8080/v1",
        model_name: str = "qwen2.5-7b-instruct",
        max_steps: int = 8,
        top_k: int = 5,
        temperature: float = 0.2,
        top_p: float = 0.9,
        fallback_model_name: Optional[str] = "llama-3.1-8b-instruct",
        consecutive_failures_to_fallback: int = 3,
        log_path: Optional[str] = None,
        use_stochastic_resonance: bool = True,
        client: Optional[Any] = None,
        api_key: Optional[str] = None,
        enable_anadrome_logging: bool = False,
        replay_tree_log_path: Optional[str] = None,
        probe_cache_path: Optional[str] = None,
        corpus_snapshot_hash: str = "default_corpus",
        index_build_id: str = "lthcm_v1",
        max_shadow_expansions: int = 1,
        controller_policy: Optional[Any] = None,
    ):
        self.engine = engine
        self.embedder = embedder
        self.max_steps = max_steps
        self.top_k = top_k
        self.temperature = temperature
        self.top_p = top_p
        self.log_path = log_path
        self.use_stochastic_resonance = use_stochastic_resonance

        # Controller policy (Anadrome Milestone 3.2: candidate walk policies)
        if isinstance(controller_policy, str) and controller_policy in APPROVED_CONTROLLER_POLICIES:
            policy_cls = APPROVED_CONTROLLER_POLICIES[controller_policy]
            self.controller_policy = policy_cls() if policy_cls is not None else None
        else:
            self.controller_policy = controller_policy

        # Anadrome Decision-Tree Replay Logging
        self.enable_anadrome_logging = enable_anadrome_logging or (replay_tree_log_path is not None)
        self.replay_tree_log_path = replay_tree_log_path
        self.corpus_snapshot_hash = corpus_snapshot_hash
        self.index_build_id = index_build_id
        self.max_shadow_expansions = max_shadow_expansions
        self.anadrome_logger = AnadromeLogger(replay_tree_log_path) if self.enable_anadrome_logging else None
        self.probe_cache = AnswerProbeCache(probe_cache_path) if self.enable_anadrome_logging else None

        # Model management
        self.primary_model = model_name
        self.fallback_model = fallback_model_name
        self.consecutive_failures_to_fallback = consecutive_failures_to_fallback
        self.active_model = model_name
        self.fallback_used = False
        self.fallback_available: Optional[bool] = None

        # OpenAI-compatible client for llama.cpp / Hestia server
        if client is not None:
            self.client = client
        elif OpenAI is not None:
            resolved_key = (
                api_key
                or os.environ.get("HESTIA_SERVICE_TOKEN")
                or os.environ.get("HESTIA_API_KEY")
                or "not-needed"
            )
            self.client = OpenAI(
                base_url=llm_base_url,
                api_key=resolved_key,
            )
        else:
            raise ImportError(
                "The 'openai' package is required for WalkController when no client is passed. "
                "Install it with: pip install openai"
            )

        # Check server model availability for fallback model
        self._verify_server_models()

    def _verify_server_models(self):
        """Check if models (both primary deliberation and fallback) are available on the server and warn if not."""
        if self.client is None or (not self.active_model and not self.fallback_model):
            return

        available = []
        try:
            if hasattr(self.client, "models") and hasattr(self.client.models, "list"):
                model_list = self.client.models.list()
                for m in model_list:
                    mid = getattr(m, "id", None)
                    if mid is None and isinstance(m, dict):
                        mid = m.get("id")
                    if mid:
                        available.append(str(mid))
            else:
                warn_msg = (
                    f"Warning: client has no models.list method. "
                    f"Availability of models (primary: '{self.active_model}', fallback: '{self.fallback_model}') is unconfirmed."
                )
                logger.warning(warn_msg)
                print(warn_msg, file=sys.stderr)
                return
        except Exception as e:
            warn_msg = (
                f"Warning: could not verify models at server {getattr(self.client, 'base_url', '')}: {e}. "
                f"Availability of models (primary: '{self.active_model}', fallback: '{self.fallback_model}') is unconfirmed."
            )
            logger.warning(warn_msg)
            print(warn_msg, file=sys.stderr)
            return

        if not available:
            warn_msg = (
                f"Warning: server {getattr(self.client, 'base_url', '')} returned an empty models list. "
                f"Availability of models (primary: '{self.active_model}', fallback: '{self.fallback_model}') is unconfirmed."
            )
            logger.warning(warn_msg)
            print(warn_msg, file=sys.stderr)
            return

        avail_lower = [a.lower() for a in available]

        # Verify primary deliberation model
        if self.active_model and self.active_model.lower() not in avail_lower:
            warn_msg = (
                f"Warning: primary model '{self.active_model}' was not found on the server (available: {available}). "
                "Deliberation steps may fail unless the server serves this model."
            )
            logger.warning(warn_msg)
            print(warn_msg, file=sys.stderr)

        # Verify fallback model
        if self.fallback_model:
            if self.fallback_model.lower() not in avail_lower:
                self.fallback_available = False
                warn_msg = (
                    f"Warning: fallback model '{self.fallback_model}' was not found on the server (available: {available}). "
                    "Operational fallback is disabled; walk will fail if fallback is triggered."
                )
                logger.warning(warn_msg)
                print(warn_msg, file=sys.stderr)
            else:
                self.fallback_available = True

    def _validate_graph(self):
        """Verify the memory graph contains edges. Fail loudly on 0 edges."""
        edge_count = 0
        if hasattr(self.engine, "wormhole") and hasattr(self.engine.wormhole, "graph"):
            if hasattr(self.engine.wormhole.graph, "number_of_edges"):
                edge_count = self.engine.wormhole.graph.number_of_edges()
            elif hasattr(self.engine.wormhole.graph, "edges"):
                edge_count = len(self.engine.wormhole.graph.edges)
        if edge_count == 0:
            raise ValueError(
                "LTHCM memory graph contains 0 edges. Deliberative walk requires "
                "an actual edge structure to navigate. Ensure documents were indexed with edges."
            )

    def _trigger_fallback(self, reason: str) -> bool:
        """Switch to fallback model if available and not already used."""
        if self.fallback_available is False:
            err_msg = (
                f"Fallback triggered ({reason}), but fallback model '{self.fallback_model}' "
                "is not served on the LLM server. Aborting walk."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        if self.fallback_model and not self.fallback_used:
            logger.warning(
                "Switching to fallback model '%s' due to: %s",
                self.fallback_model,
                reason,
            )
            self.active_model = self.fallback_model
            self.fallback_used = True
            if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        return False

    # ------------------------------------------------------------------
    # Seeding: pre-verification retrieval from LTHCM
    # ------------------------------------------------------------------

    def _seed_candidates(
        self,
        query_vec: torch.Tensor,
        as_of: Optional[float] = None,
        access_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Seed the walk from the engine's pre-verification retrieval.

        At the pinned commit (8a71605), this calls:
          - engine.stochastic_resonance.resonate_retrieval() (pipeline.py:204-209)
          - Fallback: engine.superposition.retrieve() (pipeline.py:211)

        Filters returned seeds strictly by temporal cutoff (as_of) and access_filter.
        """
        raw_results = []
        if self.use_stochastic_resonance:
            try:
                raw_results = self.engine.stochastic_resonance.resonate_retrieval(
                    query=query_vec,
                    memory_store=self.engine.superposition,
                    top_k=self.top_k,
                )
            except Exception as e:
                logger.warning(
                    "Stochastic resonance seeding failed, falling back to "
                    "superposition.retrieve: %s",
                    e,
                )
                raw_results = self.engine.superposition.retrieve(query_vec, top_k=self.top_k)
        else:
            raw_results = self.engine.superposition.retrieve(query_vec, top_k=self.top_k)

        filtered = []
        for r in raw_results:
            has_time = "created_epoch" in r or "timestamp" in r
            if as_of is not None:
                if not has_time:
                    continue
                created = float(r.get("created_epoch", r.get("timestamp", 0.0)))
                if created > as_of:
                    continue
            if access_filter is not None:
                try:
                    if not access_filter(r):
                        continue
                except Exception:
                    continue
            filtered.append(r)
        return filtered

    # ------------------------------------------------------------------
    # Graph neighbour access
    # ------------------------------------------------------------------

    def _get_neighbours(
        self,
        node_id: str,
        as_of: Optional[float] = None,
        access_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get graph neighbours for a node from LTHCM's wormhole graph.

        Returns a list of dicts with 'node_id' and 'content' keys.
        A node with no edges returns an empty list. No fabricated neighbours.
        Enforces as_of temporal cutoff and access boundary checks.
        """
        neighbours = []

        # Outgoing wormhole bridges
        bridges = self.engine.wormhole.get_bridges(node_id)
        for bridge in bridges:
            target_id = bridge["target"]
            meta = self.engine.wormhole.node_metadata.get(target_id, {})
            has_time = "created_epoch" in meta or "timestamp" in meta
            if as_of is not None:
                if not has_time:
                    continue
                created = float(meta.get("created_epoch", meta.get("timestamp", 0.0)))
                if created > as_of:
                    continue
            if access_filter is not None:
                try:
                    if not access_filter(meta):
                        continue
                except Exception:
                    continue
            bridge_score = bridge.get("weight", bridge.get("score", meta.get("similarity", 0.5)))
            try:
                b_score = float(bridge_score)
            except (TypeError, ValueError):
                b_score = 0.5
            b_edge_type = bridge.get("edge_type", "wormhole")
            neighbours.append({
                "node_id": target_id,
                "content": meta.get("content", "[content unavailable]"),
                "score": b_score,
                "edge_type": str(b_edge_type) if isinstance(b_edge_type, str) else "wormhole",
            })

        # Also check direct graph successors that might not be wormholes
        if node_id in self.engine.wormhole.graph:
            for successor in self.engine.wormhole.graph.successors(node_id):
                if not any(n["node_id"] == successor for n in neighbours):
                    meta = self.engine.wormhole.node_metadata.get(successor, {})
                    created = float(meta.get("created_epoch", meta.get("timestamp", 0.0)))
                    if as_of is not None and created > as_of:
                        continue
                    if access_filter is not None and not access_filter(meta):
                        continue
                    edge_data = {}
                    if hasattr(self.engine.wormhole.graph, "get_edge_data"):
                        raw_edge = self.engine.wormhole.graph.get_edge_data(node_id, successor)
                        if isinstance(raw_edge, dict):
                            edge_data = raw_edge
                    edge_score = edge_data.get("weight", edge_data.get("score", meta.get("similarity", 0.5)))
                    try:
                        e_score = float(edge_score)
                    except (TypeError, ValueError):
                        e_score = 0.5
                    raw_type = edge_data.get("edge_type", "directed")
                    neighbours.append({
                        "node_id": successor,
                        "content": meta.get("content", "[content unavailable]"),
                        "score": e_score,
                        "edge_type": str(raw_type) if isinstance(raw_type, str) else "directed",
                    })

        return neighbours

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
    ) -> str:
        """Call the deliberation model via the OpenAI-compatible endpoint."""
        try:
            response = self.client.chat.completions.create(
                model=self.active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err_msg = str(e).lower()
            if "out of memory" in err_msg or "oom" in err_msg:
                if self._trigger_fallback(f"LLM call OOM error: {e}"):
                    response = self.client.chat.completions.create(
                        model=self.active_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content or ""
            raise

    def _synthesize_answer(
        self,
        question: str,
        visited_passages: List[Dict[str, Any]],
    ) -> str:
        """Use the deliberation model to synthesize a final answer."""
        passages_text = "\n\n".join(
            f"[{p['node_id']}]: {p['content']}"
            for p in visited_passages
        )
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            question=question,
            visited_passages=passages_text,
        )
        try:
            return self._call_llm(SYNTHESIS_SYSTEM_PROMPT, prompt, max_tokens=1024)
        except Exception as e:
            err_msg = str(e).lower()
            is_oom = (
                "out of memory" in err_msg
                or "oom" in err_msg
                or (torch is not None and isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())))
            )
            if is_oom and self._trigger_fallback(f"Answer synthesis OOM error: {e}"):
                return self._call_llm(SYNTHESIS_SYSTEM_PROMPT, prompt, max_tokens=1024)
            raise

    # ------------------------------------------------------------------
    # Format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_neighbours_block(neighbours: List[Dict[str, Any]]) -> str:
        """Format neighbour list for the step prompt."""
        if not neighbours:
            return "(no neighbours available)"
        lines = []
        for n in neighbours:
            preview = n["content"][:200]
            lines.append(f"  [{n['node_id']}]: {preview}")
        return "\n".join(lines)

    @staticmethod
    def _format_visited_summary(visited: List[Dict[str, Any]]) -> str:
        """Format the visited-passages summary for the step prompt."""
        if not visited:
            return "(none yet)"
        lines = []
        for p in visited:
            preview = p["content"][:100]
            lines.append(f"  [{p['node_id']}]: {preview}...")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Anadrome Decision-Tree Replay Helpers
    # ------------------------------------------------------------------

    def _get_environment_identity(self) -> EnvironmentIdentity:
        """Construct the immutable environment identity for walk replay."""
        return EnvironmentIdentity(
            corpus_snapshot_hash=self.corpus_snapshot_hash,
            index_build_id=self.index_build_id,
            embedder_id=getattr(self.embedder, "model_name", "default_embedder"),
            controller_model_id=self.active_model,
            sampling_params={"temperature": self.temperature, "top_p": self.top_p},
            prompt_template_version="step_v1:ans_v1",
            answer_generator_config={"max_tokens": 1024, "model": self.active_model},
            seeding_config={
                "top_k": self.top_k,
                "use_stochastic_resonance": self.use_stochastic_resonance,
            },
            budget_config={"max_steps": self.max_steps},
        )

    def _execute_shadow_branch(
        self,
        question: str,
        branch_node: str,
        parent_walk_id: str,
        branch_step: int,
        walk_stack: List[str],
        visited_passages: List[Dict[str, Any]],
        max_shadow_depth: int,
        env_identity: EnvironmentIdentity,
        as_of: Optional[float] = None,
        access_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> None:
        """
        Execute an alternative expansion branch up to max_shadow_depth steps (Spec 4.5).
        Logs a separate continuation tree linked to parent_walk_id with is_shadow=True.
        """
        shadow_tree_id = str(uuid.uuid4())
        shadow_walk_id = f"shadow_{uuid.uuid4().hex[:12]}"

        # Clone state at the branch point
        shadow_stack = list(walk_stack)
        shadow_visited = list(visited_passages)

        # Branch onto target node
        shadow_stack.append(branch_node)
        meta = self.engine.wormhole.node_metadata.get(branch_node, {}) if getattr(self, "engine", None) and hasattr(self.engine, "wormhole") else {}
        current_content = meta.get("content", "[content unavailable]")
        if not any(p["node_id"] == branch_node for p in shadow_visited):
            shadow_visited.append({"node_id": branch_node, "content": current_content})

        current_node_id = branch_node
        shadow_transitions: List[RecordedTransition] = []
        terminal_status = "BUDGET_EXHAUSTED"

        for s_idx in range(max_shadow_depth):
            neighbours = self._get_neighbours(current_node_id, as_of=as_of, access_filter=access_filter)
            valid_ids = [n["node_id"] for n in neighbours]

            eligible_actions: List[EligibleAction] = [
                EligibleAction(
                    action_type="EXPAND",
                    target_node_id=n["node_id"],
                    score=n.get("score"),
                    edge_type=n.get("edge_type", "wormhole"),
                )
                for n in neighbours
            ]
            if len(shadow_stack) > 1:
                eligible_actions.append(EligibleAction(action_type="BACKTRACK"))
            eligible_actions.append(EligibleAction(action_type="ANSWER"))

            obs = DecisionObservation(
                env_hash=env_identity.compute_env_hash(),
                current_node_id=current_node_id,
                current_content=current_content,
                path_prefix=list(shadow_stack),
                visited_nodes=[p["node_id"] for p in shadow_visited],
                remaining_budget=max_shadow_depth - s_idx,
                eligible_actions=eligible_actions,
            )

            step_prompt = STEP_PROMPT_TEMPLATE.format(
                question=question,
                current_node_id=current_node_id,
                current_content=current_content,
                neighbours_block=self._format_neighbours_block(neighbours),
                steps_remaining=max_shadow_depth - s_idx,
                visited_summary=self._format_visited_summary(shadow_visited),
            )

            try:
                raw_output = self._call_llm(CONTROLLER_SYSTEM_PROMPT, step_prompt)
                parsed = parse_controller_output(raw_output, valid_ids)
            except Exception as e:
                parsed = ParsedAction(action="BACKTRACK", parse_failed=True, raw_output=f"[error: {e}]")

            taken_action_str = parsed.action + (f" {parsed.target}" if parsed.target else "")

            shadow_transitions.append(
                RecordedTransition(
                    step_index=s_idx,
                    observation_hash=obs.observation_hash,
                    observation=obs,
                    taken_action=taken_action_str,
                    raw_model_output=parsed.raw_output,
                    parsed_reason=parsed.reason,
                    parse_failed=parsed.parse_failed,
                    next_node_id=parsed.target if parsed.action == "EXPAND" else (shadow_stack[-2] if parsed.action == "BACKTRACK" and len(shadow_stack) > 1 else None),
                    answer_probe=None,
                )
            )

            if parsed.action == "ANSWER":
                terminal_status = "ANSWER"
                break
            elif parsed.action == "EXPAND" and parsed.target and parsed.target in valid_ids:
                current_node_id = parsed.target
                shadow_stack.append(current_node_id)
                meta = self.engine.wormhole.node_metadata.get(current_node_id, {}) if getattr(self, "engine", None) and hasattr(self.engine, "wormhole") else {}
                current_content = meta.get("content", "[content unavailable]")
                if not any(p["node_id"] == current_node_id for p in shadow_visited):
                    shadow_visited.append({"node_id": current_node_id, "content": current_content})
            elif parsed.action == "BACKTRACK" and len(shadow_stack) > 1:
                shadow_stack.pop()
                current_node_id = shadow_stack[-1]
                meta = self.engine.wormhole.node_metadata.get(current_node_id, {}) if getattr(self, "engine", None) and hasattr(self.engine, "wormhole") else {}
                current_content = meta.get("content", "[content unavailable]")

        try:
            shadow_answer = self._synthesize_answer(question, shadow_visited)
        except Exception:
            shadow_answer = "[shadow answer synthesis unavailable]"

        if self.anadrome_logger:
            tree_rec = ReplayTreeRecord(
                schema_version="1.0.0",
                tree_id=shadow_tree_id,
                walk_id=shadow_walk_id,
                question=question,
                seeded_nodes=[shadow_stack[0]],
                environment=env_identity,
                is_shadow=True,
                parent_walk_id=parent_walk_id,
                branch_point_step=branch_step,
                branch_point_node=branch_node,
                transitions=shadow_transitions,
                final_answer=shadow_answer,
                steps_used=len(shadow_transitions),
                terminal_status=terminal_status,
            )
            self.anadrome_logger.append_tree(tree_rec)

    # ------------------------------------------------------------------
    # Main walk loop
    # ------------------------------------------------------------------

    def walk(
        self,
        question: str,
        conflict_prefix: Optional[str] = None,
        as_of: Optional[float] = None,
        access_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        walk_id: Optional[str] = None,
        parent_walk_id: Optional[str] = None,
        is_shadow: bool = False,
        controller_policy: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single deliberative walk.

        Args:
            question: The user's question.
            conflict_prefix: If set, prepended to the system prompt (for re-walks
                informed by a conflict report).
            as_of: Optional POSIX epoch timestamp cutoff; nodes created after are excluded.
            access_filter: Optional predicate (dict -> bool) returning False for restricted nodes.
            walk_id: Optional unique identifier for this walk.
            parent_walk_id: Optional parent identifier for shadow branches.
            is_shadow: Whether this walk is a shadow branch.
            controller_policy: Optional AnadromePolicy instance or name override.

        Returns:
            Dict with:
                - answer: str
                - path: list of step records
                - visited_passages: list of passage dicts
                - seeded_nodes: list of seed node IDs
                - steps_used: int
        """
        self._validate_graph()

        active_walk_id = walk_id or str(uuid.uuid4())
        env_identity = self._get_environment_identity() if self.enable_anadrome_logging else None
        anadrome_transitions: List[RecordedTransition] = []
        shadow_expansions_done = 0

        # Resolve active controller policy (Anadrome Milestone 3.2)
        active_policy = controller_policy if controller_policy is not None else self.controller_policy
        if isinstance(active_policy, str) and active_policy in APPROVED_CONTROLLER_POLICIES:
            policy_cls = APPROVED_CONTROLLER_POLICIES[active_policy]
            active_policy = policy_cls() if policy_cls is not None else None

        # Encode the question
        query_vec = self.embedder.encode_single(question)

        # Seed candidates
        seed_results = self._seed_candidates(query_vec, as_of=as_of, access_filter=access_filter)
        seeded_node_ids = [r.get("doc_id", r.get("node_id", "")) for r in seed_results]

        if not seed_results:
            return {
                "answer": "No relevant documents found in the knowledge graph.",
                "path": [],
                "visited_passages": [],
                "seeded_nodes": [],
                "steps_used": 0,
            }

        # Build system prompt (with optional conflict prefix for re-walks)
        system_prompt = CONTROLLER_SYSTEM_PROMPT
        if conflict_prefix:
            system_prompt = conflict_prefix + "\n" + system_prompt

        # Walk state
        # Start from the top seed
        current_node_id = seeded_node_ids[0]
        current_content = seed_results[0].get("content", "")
        visited_passages: List[Dict[str, Any]] = [
            {"node_id": current_node_id, "content": current_content}
        ]
        path_records: List[Dict[str, Any]] = []
        walk_stack: List[str] = [current_node_id]  # for BACKTRACK
        consecutive_parse_failures = 0
        steps_used = 0

        for step in range(self.max_steps):
            steps_used = step + 1

            # Get neighbours of current node
            neighbours = self._get_neighbours(current_node_id, as_of=as_of, access_filter=access_filter)
            valid_ids = [n["node_id"] for n in neighbours]

            if active_policy is not None:
                # Direct policy deliberation (Anadrome Track B)
                policy_eligible_actions = [
                    {
                        "action_type": "EXPAND",
                        "target_node_id": n["node_id"],
                        "score": n.get("score", 0.5),
                        "edge_type": n.get("edge_type", "wormhole"),
                    }
                    for n in neighbours
                ]
                if len(walk_stack) > 1:
                    policy_eligible_actions.append({"action_type": "BACKTRACK"})
                policy_eligible_actions.append({"action_type": "ANSWER"})

                obs_dict = {
                    "current_node_id": current_node_id,
                    "current_content": current_content,
                    "path_prefix": list(walk_stack),
                    "visited_nodes": [p["node_id"] for p in visited_passages],
                    "remaining_budget": self.max_steps - step,
                    "eligible_actions": policy_eligible_actions,
                    "question": question,
                    "neighbours": neighbours,
                    "realization_ids": {
                        "step": step,
                        "sheaf_obstruction": (
                            getattr(self, "_last_conflict_detected", False)
                            or "[CONFLICT]" in current_content
                            or "CONTRADICTION" in current_content
                        ),
                        "has_conflict": (
                            getattr(self, "_last_conflict_detected", False)
                            or "[CONFLICT]" in current_content
                            or "CONTRADICTION" in current_content
                        ),
                    },
                }
                try:
                    policy_act_str = active_policy.select_action(obs_dict)
                except Exception as policy_err:
                    logger.warning(
                        "Policy execution failed (%s), defaulting to ANSWER: %s",
                        getattr(active_policy, "name", "policy"),
                        policy_err,
                    )
                    policy_act_str = "ANSWER"
                raw_output = f"ACTION: {policy_act_str}\nREASON: Policy {getattr(active_policy, 'name', 'custom')}"
                parsed = parse_controller_output(raw_output, valid_ids)
            else:
                # Execute step with step-level OOM recovery covering neighbours, prompt, and LLM call
                try:
                    # Build the step prompt
                    step_prompt = STEP_PROMPT_TEMPLATE.format(
                        question=question,
                        current_node_id=current_node_id,
                        current_content=current_content,
                        neighbours_block=self._format_neighbours_block(neighbours),
                        steps_remaining=self.max_steps - step,
                        visited_summary=self._format_visited_summary(visited_passages),
                    )

                    # Call the deliberation model
                    raw_output = self._call_llm(system_prompt, step_prompt)
                    parsed = parse_controller_output(raw_output, valid_ids)
                except Exception as e:
                    err_msg = str(e).lower()
                    is_oom = (
                        "out of memory" in err_msg
                        or "oom" in err_msg
                        or (torch is not None and isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())))
                    )
                    if is_oom and self._trigger_fallback(f"Walk step {step + 1} OOM error: {e}"):
                        neighbours = self._get_neighbours(current_node_id, as_of=as_of, access_filter=access_filter)
                        valid_ids = [n["node_id"] for n in neighbours]
                        step_prompt = STEP_PROMPT_TEMPLATE.format(
                            question=question,
                            current_node_id=current_node_id,
                            current_content=current_content,
                            neighbours_block=self._format_neighbours_block(neighbours),
                            steps_remaining=self.max_steps - step,
                            visited_summary=self._format_visited_summary(visited_passages),
                        )
                        raw_output = self._call_llm(system_prompt, step_prompt)
                        parsed = parse_controller_output(raw_output, valid_ids)
                    else:
                        raise

            # Track parse failures for fallback trigger
            if parsed.parse_failed:
                consecutive_parse_failures += 1
                logger.warning(
                    "Parse failure %d/%d. Raw output: %s",
                    consecutive_parse_failures,
                    self.consecutive_failures_to_fallback,
                    raw_output[:200],
                )
                # Fallback trigger: 3 consecutive unparseable actions
                if consecutive_parse_failures >= self.consecutive_failures_to_fallback:
                    self._trigger_fallback(f"{consecutive_parse_failures} consecutive unparseable actions")
                    consecutive_parse_failures = 0
            else:
                consecutive_parse_failures = 0

            # Record this step in live path records
            path_records.append({
                "node": current_node_id,
                "action": parsed.action,
                "raw_output": parsed.raw_output,
                "reason": parsed.reason,
                "parse_failed": parsed.parse_failed,
            })

            # Record decision observation and transition for Anadrome replay tree
            if self.enable_anadrome_logging and env_identity is not None:
                eligible_actions: List[EligibleAction] = [
                    EligibleAction(
                        action_type="EXPAND",
                        target_node_id=n["node_id"],
                        score=n.get("score"),
                        edge_type=n.get("edge_type", "wormhole"),
                    )
                    for n in neighbours
                ]
                if len(walk_stack) > 1:
                    eligible_actions.append(EligibleAction(action_type="BACKTRACK"))
                eligible_actions.append(EligibleAction(action_type="ANSWER"))

                obs = DecisionObservation(
                    env_hash=env_identity.compute_env_hash(),
                    current_node_id=current_node_id,
                    current_content=current_content,
                    path_prefix=list(walk_stack),
                    visited_nodes=[p["node_id"] for p in visited_passages],
                    remaining_budget=self.max_steps - step,
                    eligible_actions=eligible_actions,
                )

                # Intermediate answer probe (Spec 4.4 & Roadmap Gap 1: path depth >= 2, non-ANSWER action)
                answer_probe = None
                if (step + 1) >= 2 and parsed.action != "ANSWER":
                    prompt_hash = hashlib.sha256(
                        "\n".join(p["node_id"] + p["content"] for p in visited_passages).encode("utf-8")
                    ).hexdigest()
                    cached = self.probe_cache.get(obs.observation_hash, self.active_model, prompt_hash) if self.probe_cache else None
                    if cached:
                        answer_probe = cached
                    else:
                        try:
                            draft_ans = self._synthesize_answer(question, visited_passages)
                            ver = self._verify_answer(question, draft_ans, visited_passages)
                            answer_probe = {
                                "draft_answer": draft_ans,
                                "verified": not ver["conflicts_found"],
                                "conflict_detail": ver.get("conflict_detail"),
                            }
                            if self.probe_cache:
                                self.probe_cache.put(
                                    state_id_hash=obs.observation_hash,
                                    model_id=self.active_model,
                                    prompt_hash=prompt_hash,
                                    draft_answer=draft_ans,
                                    verified=not ver["conflicts_found"],
                                    conflict_detail=ver.get("conflict_detail"),
                                )
                        except Exception as probe_err:
                            logger.debug("Intermediate answer probe skipped: %s", probe_err)

                taken_act = parsed.action + (f" {parsed.target}" if parsed.target else "")
                anadrome_transitions.append(
                    RecordedTransition(
                        step_index=step,
                        observation_hash=obs.observation_hash,
                        observation=obs,
                        taken_action=taken_act,
                        raw_model_output=parsed.raw_output,
                        parsed_reason=parsed.reason,
                        parse_failed=parsed.parse_failed,
                        next_node_id=parsed.target if parsed.action == "EXPAND" else (walk_stack[-2] if parsed.action == "BACKTRACK" and len(walk_stack) > 1 else None),
                        answer_probe=answer_probe,
                    )
                )

                # Shadow expansion: exactly 1 per walk, depth <= 2 (Spec 4.5 & Roadmap Gap 1)
                if (
                    not is_shadow
                    and shadow_expansions_done < self.max_shadow_expansions
                    and parsed.action == "EXPAND"
                    and len(valid_ids) > 1
                ):
                    alt_targets = [vid for vid in valid_ids if vid != parsed.target]
                    if alt_targets:
                        shadow_expansions_done += 1
                        self._execute_shadow_branch(
                            question=question,
                            branch_node=alt_targets[0],
                            parent_walk_id=active_walk_id,
                            branch_step=step,
                            walk_stack=list(walk_stack),
                            visited_passages=list(visited_passages),
                            max_shadow_depth=2,
                            env_identity=env_identity,
                            as_of=as_of,
                            access_filter=access_filter,
                        )

            # Execute the action
            if parsed.action == "ANSWER":
                break

            elif parsed.action == "EXPAND" and parsed.target:
                # Revalidate selected action against valid, causally admitted neighbors
                if parsed.target not in valid_ids:
                    logger.warning(
                        "Action EXPAND target '%s' is not in causal/valid neighbor list %s. Refusing expansion.",
                        parsed.target, valid_ids
                    )
                else:
                    # Move to target node and push it onto the walk stack
                    current_node_id = parsed.target
                    walk_stack.append(current_node_id)
                    # Get content for the new node
                    meta = self.engine.wormhole.node_metadata.get(current_node_id, {})
                    current_content = meta.get("content", "[content unavailable]")
                    # Record as visited if not already
                    if not any(p["node_id"] == current_node_id for p in visited_passages):
                        visited_passages.append({
                            "node_id": current_node_id,
                            "content": current_content,
                        })

            elif parsed.action == "BACKTRACK":
                # Pop the current node, returning to the previous node (new top).
                # Previously visited nodes may be re-expanded.
                if len(walk_stack) > 1:
                    walk_stack.pop()
                    current_node_id = walk_stack[-1]
                    meta = self.engine.wormhole.node_metadata.get(current_node_id, {})
                    current_content = meta.get("content", "[content unavailable]")
                # If stack has only 1 element (the root seed), stay at current node (can't backtrack further)

        else:
            # Budget exhausted — force ANSWER from what has been seen
            path_records.append({
                "node": current_node_id,
                "action": "ANSWER",
                "raw_output": "[budget exhausted, forced ANSWER]",
                "reason": "Step budget exhausted.",
                "parse_failed": False,
            })
            steps_used = self.max_steps

            if self.enable_anadrome_logging and env_identity is not None and not (anadrome_transitions and anadrome_transitions[-1].taken_action == "ANSWER"):
                obs = DecisionObservation(
                    env_hash=env_identity.compute_env_hash(),
                    current_node_id=current_node_id,
                    current_content=current_content,
                    path_prefix=list(walk_stack),
                    visited_nodes=[p["node_id"] for p in visited_passages],
                    remaining_budget=0,
                    eligible_actions=[EligibleAction(action_type="ANSWER")],
                )
                anadrome_transitions.append(
                    RecordedTransition(
                        step_index=steps_used - 1,
                        observation_hash=obs.observation_hash,
                        observation=obs,
                        taken_action="ANSWER",
                        raw_model_output="[budget exhausted, forced ANSWER]",
                        parsed_reason="Step budget exhausted.",
                        parse_failed=False,
                        next_node_id=None,
                        answer_probe=None,
                    )
                )

        # Synthesize the final answer
        answer = self._synthesize_answer(question, visited_passages)

        # Log completed replay tree record if Anadrome is enabled
        if self.enable_anadrome_logging and self.anadrome_logger and env_identity is not None:
            tree_record = ReplayTreeRecord(
                schema_version="1.0.0",
                tree_id=str(uuid.uuid4()),
                walk_id=active_walk_id,
                question=question,
                seeded_nodes=seeded_node_ids,
                environment=env_identity,
                is_shadow=is_shadow,
                parent_walk_id=parent_walk_id,
                branch_point_step=None,
                branch_point_node=None,
                transitions=anadrome_transitions,
                final_answer=answer,
                steps_used=steps_used,
                terminal_status="ANSWER" if (path_records and path_records[-1]["action"] == "ANSWER") else "BUDGET_EXHAUSTED",
            )
            self.anadrome_logger.append_tree(tree_record)

        return {
            "answer": answer,
            "path": path_records,
            "visited_passages": visited_passages,
            "seeded_nodes": seeded_node_ids,
            "steps_used": steps_used,
            "walk_id": active_walk_id,
        }

    # ------------------------------------------------------------------
    # Full pipeline: walk + verify + optional re-walk + log
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        as_of: Optional[float] = None,
        access_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        controller_policy: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full Dianoia v1 pipeline:
            1. Deliberative walk
            2. Answer-time verification (sheaf/NLI)
            3. If conflicts: one re-walk informed by conflict report
            4. Log the walk

        Args:
            question: The user's question.
            as_of: Optional temporal cutoff.
            access_filter: Optional access predicate.
            controller_policy: Optional AnadromePolicy instance or name override.

        Returns:
            Dict with:
                - final_answer: str
                - walk_path: list of step records
                - verification: dict with verification report
                - seeded_nodes: list
                - steps_used: int
                - model: str
                - fallback_used: bool
        """
        walk_id = str(uuid.uuid4())

        # --- Step 1: Initial walk ---
        walk_result = self.walk(
            question,
            as_of=as_of,
            access_filter=access_filter,
            walk_id=walk_id,
            controller_policy=controller_policy,
        )

        # --- Step 2: Answer-time verification ---
        verification = self._verify_answer(
            question, walk_result["answer"], walk_result["visited_passages"]
        )

        final_answer = walk_result["answer"]
        final_path = walk_result["path"]
        final_steps = walk_result["steps_used"]
        rewalked = False
        initial_path = None

        # --- Step 3: If conflicts, log initial walk and re-walk once ---
        if verification["conflicts_found"]:
            logger.info("Conflicts detected, logging initial walk and executing re-walk.")
            initial_path = walk_result["path"]

            # Log the initial walk trajectory to preserve negative training signal
            initial_log_record = build_walk_record(
                question=question,
                seeded_nodes=walk_result["seeded_nodes"],
                path=initial_path,
                verification={
                    "conflicts_found": True,
                    "rewalked": False,
                    "report": verification["report"],
                    "conflict_detail": verification.get("conflict_detail"),
                },
                final_answer=walk_result["answer"],
                max_steps=self.max_steps,
                steps_used=walk_result["steps_used"],
                model=self.active_model,
                fallback_used=self.fallback_used,
                walk_id=walk_id,
            )
            append_walk_log(initial_log_record, log_path=self.log_path)

            # Prepend conflict report to the deliberation prompt
            conflict_prefix = REWALK_CONFLICT_PREFIX.format(
                conflict_report=verification["conflict_detail"] or "Unspecified conflict."
            )
            try:
                rewalk_id = f"{walk_id}_rewalk"
                rewalk_result = self.walk(
                    question,
                    conflict_prefix=conflict_prefix,
                    as_of=as_of,
                    access_filter=access_filter,
                    walk_id=rewalk_id,
                    controller_policy=controller_policy,
                )

                # Re-verify the new answer
                re_verification = self._verify_answer(
                    question, rewalk_result["answer"], rewalk_result["visited_passages"]
                )
                final_answer = rewalk_result["answer"]
                final_path = rewalk_result["path"]
                final_steps = rewalk_result["steps_used"]
                rewalked = True

                verification_record = {
                    "conflicts_found": re_verification["conflicts_found"],
                    "rewalked": True,
                    "report": re_verification["report"],
                    "conflict_detail": re_verification.get("conflict_detail"),
                }
            except Exception as e:
                logger.error("Re-walk failed: %s", e, exc_info=True)
                rewalked = False
                verification_record = {
                    "conflicts_found": True,
                    "rewalked": False,
                    "rewalk_failed": True,
                    "rewalk_error": str(e),
                    "report": f"{verification['report']} [Re-walk failed: {e}]",
                    "conflict_detail": verification.get("conflict_detail"),
                }
        else:
            verification_record = {
                "conflicts_found": False,
                "rewalked": False,
                "report": verification["report"],
                "conflict_detail": None,
            }

        # --- Step 4: Log the walk (final / re-walk record) ---
        log_record = build_walk_record(
            question=question,
            seeded_nodes=walk_result["seeded_nodes"],
            path=final_path,
            verification=verification_record,
            final_answer=final_answer,
            max_steps=self.max_steps,
            steps_used=final_steps,
            model=self.active_model,
            fallback_used=self.fallback_used,
            initial_path=initial_path,
            walk_id=walk_id,
        )
        log_file = append_walk_log(log_record, log_path=self.log_path)
        logger.info("Walk logged to %s", log_file)

        return {
            "final_answer": final_answer,
            "walk_path": final_path,
            "initial_walk_path": initial_path,
            "verification": verification_record,
            "seeded_nodes": walk_result["seeded_nodes"],
            "steps_used": final_steps,
            "model": self.active_model,
            "fallback_used": self.fallback_used,
            "walk_id": walk_id,
        }

    def _verify_answer(
        self,
        question: str,
        answer: str,
        visited_passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run sheaf/NLI verification on the answer against evidence."""
        if not visited_passages:
            return {
                "verified": False,
                "verifier_mode": "none",
                "provenance": "unverified",
                "conflicts_found": None,
                "conflict_detail": None,
                "report": "No evidence passages to verify against.",
            }

        # Embed the answer and evidence
        answer_emb = self.embedder.encode_single(answer)
        evidence_texts = [p["content"] for p in visited_passages]
        evidence_embs = self.embedder.encode(evidence_texts)

        # Build passage dicts with required keys for sheaf filter
        evidence_dicts = [
            {"doc_id": p["node_id"], "content": p["content"]}
            for p in visited_passages
        ]

        return verify_answer_against_evidence(
            answer_text=answer,
            evidence_passages=evidence_dicts,
            answer_embedding=answer_emb,
            evidence_embeddings=evidence_embs,
            sheaf_filter=self.engine.sheaf_filter,
            use_nli=True,
        )
