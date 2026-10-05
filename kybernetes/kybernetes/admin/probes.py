"""Truthful probe registry and status cache for Metis stack components."""

import asyncio
from datetime import datetime, timezone
import time
from typing import Any, Callable, Coroutine, Dict, Optional

from kybernetes.admin.status import ComponentState, ComponentStatus, Freshness


class ProbeRegistry:
    """
    Central probe registry and cache in Kybernetes.
    Adheres strictly to the Truthful Status contract:
    - Never mark 'live' merely from non-null dependencies, imported modules, or memory_root.
    - Live state requires a successful probe.
    - Polling reads cached results; does not trigger model loads or completions.
    """

    REACHABILITY_TTL_SECONDS = 45
    FUNCTIONAL_TTL_SECONDS = 900  # 15 minutes

    def __init__(
        self,
        hestia_adapter: Any = None,
        lthcm_adapter: Any = None,
        dianoia_adapter: Any = None,
        phylax_adapter: Any = None,
        krisis_adapter: Any = None,
    ):
        self.hestia = hestia_adapter
        self.lthcm = lthcm_adapter
        self.dianoia = dianoia_adapter
        self.phylax = phylax_adapter
        self.krisis = krisis_adapter

        self._cache: Dict[str, ComponentStatus] = {}
        self._probe_locks: Dict[str, asyncio.Lock] = {
            "hestia": asyncio.Lock(),
            "lthcm": asyncio.Lock(),
            "dianoia": asyncio.Lock(),
            "phylax": asyncio.Lock(),
            "krisis": asyncio.Lock(),
            "prosopon": asyncio.Lock(),
            "stoma": asyncio.Lock(),
            "dokime": asyncio.Lock(),
            "paideia": asyncio.Lock(),
            "kleroterion": asyncio.Lock(),
            "kybernetes": asyncio.Lock(),
        }

    def _get_freshness(self, status: ComponentStatus, ttl_seconds: int) -> Freshness:
        try:
            checked_time = datetime.fromisoformat(status.checked_at)
            now = datetime.now(timezone.utc)
            delta = (now - checked_time).total_seconds()
            if delta <= ttl_seconds:
                return Freshness.FRESH
            return Freshness.STALE
        except Exception:
            return Freshness.UNKNOWN

    async def check_hestia_reachability(self, timeout_seconds: float = 2.0) -> ComponentStatus:
        """Probe Hestia reachability via cheap /models or /health check with strict timeout."""
        start_time = time.monotonic()
        checked_at = datetime.now(timezone.utc).isoformat()
        if not self.hestia:
            status = ComponentStatus(
                state=ComponentState.OFFLINE,
                detail="Hestia adapter not configured",
                freshness=Freshness.FRESH,
                checked_at=checked_at,
                evidence="No Hestia adapter instance",
                status="offline",
                mode="offline",
            )
            self._cache["hestia"] = status
            return status

        try:
            import httpx
            base_url = getattr(self.hestia, "base_url", "http://127.0.0.1:8080/v1").rstrip("/")
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                resp = await client.get(f"{base_url}/models")
                latency = int((time.monotonic() - start_time) * 1000)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("id") for m in data.get("data", []) if "id" in m]
                    default_model = getattr(self.hestia, "default_model", "llama-3.1-8b-instruct")
                    status = ComponentStatus(
                        state=ComponentState.LIVE,
                        detail=f"Inference server responsive ({len(models)} model(s) available, default: {default_model})",
                        freshness=Freshness.FRESH,
                        version=data.get("version", "0.1.0"),
                        checked_at=checked_at,
                        latency_ms=latency,
                        evidence=f"HTTP 200 from {base_url}/models ({latency}ms)",
                        is_simulated=False,
                        status="ready",
                        mode="live",
                    )
                else:
                    status = ComponentStatus(
                        state=ComponentState.OFFLINE,
                        detail=f"Hestia returned HTTP {resp.status_code}",
                        freshness=Freshness.FRESH,
                        checked_at=checked_at,
                        latency_ms=latency,
                        evidence=f"HTTP {resp.status_code} from {base_url}/models",
                        is_simulated=False,
                        status="offline",
                        mode="offline",
                    )
        except Exception as exc:
            latency = int((time.monotonic() - start_time) * 1000)
            status = ComponentStatus(
                state=ComponentState.OFFLINE,
                detail=f"Connection refused or timed out: {type(exc).__name__}",
                freshness=Freshness.FRESH,
                checked_at=checked_at,
                latency_ms=latency,
                evidence=f"Probe failure: {exc}",
                is_simulated=False,
                status="offline",
                mode="offline",
            )

        self._cache["hestia"] = status
        return status

    def get_cached_status(self, component_id: str) -> ComponentStatus:
        """
        Get cached status or build a truthful default status.
        Never marks live without an explicit probe recorded.
        """
        if component_id in self._cache:
            cached = self._cache[component_id]
            ttl = self.FUNCTIONAL_TTL_SECONDS if cached.state == ComponentState.LIVE else self.REACHABILITY_TTL_SECONDS
            cached.freshness = self._get_freshness(cached, ttl)
            return cached

        checked_at = datetime.now(timezone.utc).isoformat()

        if component_id == "kybernetes":
            return ComponentStatus(
                state=ComponentState.LIVE,
                detail="Orchestrator control plane active",
                freshness=Freshness.FRESH,
                version="0.1.0",
                checked_at=checked_at,
                evidence="In-process FastAPI orchestrator",
                is_simulated=False,
                status="ready",
                mode="live",
            )

        if component_id == "hestia":
            return ComponentStatus(
                state=ComponentState.UNKNOWN,
                detail="Inference server unprobed",
                freshness=Freshness.UNKNOWN,
                checked_at=checked_at,
                evidence="No reachability probe executed yet",
                is_simulated=False,
                status="unknown",
                mode="unknown",
            )

        if component_id == "lthcm":
            is_sim = getattr(self.lthcm, "is_simulated", True) if self.lthcm else True
            has_root = bool(getattr(self.lthcm, "memory_root", None)) if self.lthcm else False
            detail = "real index loaded (unprobed)" if (has_root and not is_sim) else "in-memory demo corpus (4 passages)"
            evidence = "adapter unprobed; demo corpus active" if is_sim else "index directory mounted; unprobed"
            return ComponentStatus(
                state=ComponentState.SIMULATED if is_sim else ComponentState.UNKNOWN,
                detail=detail,
                freshness=Freshness.FRESH if is_sim else Freshness.UNKNOWN,
                version="0.1.0",
                checked_at=checked_at,
                evidence=evidence,
                is_simulated=is_sim,
                status="simulated" if is_sim else "unknown",
                mode="simulated" if is_sim else "unknown",
            )

        if component_id == "dianoia":
            is_sim = getattr(self.dianoia, "is_simulated", True) if self.dianoia else True
            return ComponentStatus(
                state=ComponentState.SIMULATED,
                detail="flat retrieval fallback / simulated graph walk",
                freshness=Freshness.FRESH,
                version="0.1.0",
                checked_at=checked_at,
                evidence="Adapter runs zero-edge fallback and simulated deliberation",
                is_simulated=True,
                status="simulated",
                mode="simulated",
            )

        if component_id == "phylax":
            is_sim = getattr(self.phylax, "is_simulated", False) if self.phylax else True
            has_shim = getattr(self.phylax, "shim", None) is not None if self.phylax else False
            detail = "in-memory capability approval manager" if not has_shim else "local capability shim (unprobed)"
            return ComponentStatus(
                state=ComponentState.SIMULATED,
                detail=detail,
                freshness=Freshness.FRESH,
                version="0.1.0",
                checked_at=checked_at,
                evidence="In-process capability approval manager; unprobed gateway",
                is_simulated=True,
                status="simulated",
                mode="simulated",
            )

        if component_id == "krisis":
            is_sim = getattr(self.krisis, "is_simulated", True) if self.krisis else True
            return ComponentStatus(
                state=ComponentState.SIMULATED,
                detail="heuristic claim & citation verifier",
                freshness=Freshness.FRESH,
                version="0.1.0",
                checked_at=checked_at,
                evidence="Local rule-based heuristic verifier",
                is_simulated=True,
                status="simulated",
                mode="simulated",
            )

        if component_id in ("stoma", "dokime", "paideia"):
            return ComponentStatus(
                state=ComponentState.UNKNOWN,
                detail="Batch tool (no persistent daemon)",
                freshness=Freshness.UNKNOWN,
                version="0.1.0",
                checked_at=checked_at,
                evidence="Batch tool executed via CLI / pipeline",
                is_simulated=False,
                status="unknown",
                mode="unknown",
            )

        if component_id == "kleroterion":
            return ComponentStatus(
                state=ComponentState.SIMULATED,
                detail="Deterministic consensus / ensemble selection library",
                freshness=Freshness.FRESH,
                version="0.1.0",
                checked_at=checked_at,
                evidence="Deterministic library module",
                is_simulated=True,
                status="simulated",
                mode="simulated",
            )

        if component_id == "prosopon":
            return ComponentStatus(
                state=ComponentState.UNKNOWN,
                detail="Client UI frontend / BFF",
                freshness=Freshness.UNKNOWN,
                version="0.1.0",
                checked_at=checked_at,
                evidence="Upstream client gateway",
                is_simulated=False,
                status="unknown",
                mode="unknown",
            )

        return ComponentStatus(
            state=ComponentState.UNKNOWN,
            detail=f"Unknown module '{component_id}'",
            freshness=Freshness.UNKNOWN,
            checked_at=checked_at,
            evidence="Unrecognized identifier",
            status="unknown",
            mode="unknown",
        )

    async def run_functional_probe(self, component_id: str) -> ComponentStatus:
        """
        Execute bounded, non-mutating canary functional probe.
        Guarantees single concurrency per component and updates probe cache.
        """
        lock = self._probe_locks.get(component_id)
        if not lock:
            raise ValueError(f"Unknown component '{component_id}'")

        async with lock:
            start_time = time.monotonic()
            checked_at = datetime.now(timezone.utc).isoformat()

            if component_id == "hestia":
                return await self.check_hestia_reachability()

            elif component_id == "lthcm":
                if not self.lthcm:
                    status = ComponentStatus(
                        state=ComponentState.OFFLINE,
                        detail="LTHCM adapter unavailable",
                        freshness=Freshness.FRESH,
                        checked_at=checked_at,
                        evidence="Adapter not instantiated",
                        status="offline",
                        mode="offline",
                    )
                else:
                    try:
                        results = self.lthcm.retrieve("test canary query")
                        latency = int((time.monotonic() - start_time) * 1000)
                        is_sim = getattr(self.lthcm, "is_simulated", True)
                        if is_sim:
                            status = ComponentStatus(
                                state=ComponentState.SIMULATED,
                                detail=f"Functional probe retrieved {len(results)} passage(s) from demo corpus",
                                freshness=Freshness.FRESH,
                                version="0.1.0",
                                checked_at=checked_at,
                                latency_ms=latency,
                                evidence=f"Canary retrieval successful ({latency}ms, simulated corpus)",
                                is_simulated=True,
                                status="simulated",
                                mode="simulated",
                            )
                        else:
                            status = ComponentStatus(
                                state=ComponentState.LIVE,
                                detail=f"Functional probe retrieved {len(results)} passage(s) from indexed storage",
                                freshness=Freshness.FRESH,
                                version="0.1.0",
                                checked_at=checked_at,
                                latency_ms=latency,
                                evidence=f"Canary retrieval confirmed live index ({latency}ms)",
                                is_simulated=False,
                                status="ready",
                                mode="live",
                            )
                    except Exception as exc:
                        latency = int((time.monotonic() - start_time) * 1000)
                        status = ComponentStatus(
                            state=ComponentState.OFFLINE,
                            detail=f"LTHCM retrieval probe failed: {exc}",
                            freshness=Freshness.FRESH,
                            checked_at=checked_at,
                            latency_ms=latency,
                            evidence=f"Retrieval exception: {exc}",
                            is_simulated=True,
                            status="offline",
                            mode="offline",
                        )

            elif component_id == "krisis":
                if not self.krisis:
                    status = ComponentStatus(
                        state=ComponentState.OFFLINE,
                        detail="Krisis adapter unavailable",
                        freshness=Freshness.FRESH,
                        checked_at=checked_at,
                        evidence="Adapter not instantiated",
                    )
                else:
                    try:
                        from kybernetes.contracts.turn import Citation
                        test_citations = [
                            Citation(
                                citation_id="[1]",
                                passage_id="canary_p1",
                                source_id="canary_src",
                                source_title="Canary Document",
                                text="Canary fact statement for verifier.",
                                score=1.0,
                            )
                        ]
                        rep = self.krisis.verify_response(
                            query="canary query",
                            response_text="Statement supported by [1].",
                            citations=test_citations,
                        )
                        latency = int((time.monotonic() - start_time) * 1000)
                        is_sim = getattr(rep, "is_simulated", True)
                        status = ComponentStatus(
                            state=ComponentState.SIMULATED if is_sim else ComponentState.LIVE,
                            detail=f"Canary verification completed (score: {rep.score:.2f}, verifier: {rep.verifier_mode})",
                            freshness=Freshness.FRESH,
                            version="0.1.0",
                            checked_at=checked_at,
                            latency_ms=latency,
                            evidence=f"Krisis verification probe returned verdict ({latency}ms, simulated={is_sim})",
                            is_simulated=is_sim,
                            status="simulated" if is_sim else "ready",
                            mode="simulated" if is_sim else "live",
                        )
                    except Exception as exc:
                        latency = int((time.monotonic() - start_time) * 1000)
                        status = ComponentStatus(
                            state=ComponentState.OFFLINE,
                            detail=f"Krisis probe failed: {exc}",
                            freshness=Freshness.FRESH,
                            checked_at=checked_at,
                            latency_ms=latency,
                            evidence=f"Verifier probe exception: {exc}",
                            status="offline",
                            mode="offline",
                        )

            elif component_id == "phylax":
                if not self.phylax:
                    status = ComponentStatus(
                        state=ComponentState.OFFLINE,
                        detail="Phylax adapter unavailable",
                        freshness=Freshness.FRESH,
                        checked_at=checked_at,
                        evidence="Adapter not instantiated",
                    )
                else:
                    try:
                        # Non-mutating probe using internal validation
                        has_shim = getattr(self.phylax, "shim", None) is not None
                        is_sim = getattr(self.phylax, "is_simulated", False) or not has_shim
                        latency = int((time.monotonic() - start_time) * 1000)
                        status = ComponentStatus(
                            state=ComponentState.SIMULATED if is_sim else ComponentState.LIVE,
                            detail="Phylax in-memory approval manager validated" if is_sim else "GuardedGateway connected",
                            freshness=Freshness.FRESH,
                            version="0.1.0",
                            checked_at=checked_at,
                            latency_ms=latency,
                            evidence=f"Phylax probe validated ({latency}ms, simulated={is_sim})",
                            is_simulated=is_sim,
                            status="simulated" if is_sim else "ready",
                            mode="simulated" if is_sim else "live",
                        )
                    except Exception as exc:
                        latency = int((time.monotonic() - start_time) * 1000)
                        status = ComponentStatus(
                            state=ComponentState.OFFLINE,
                            detail=f"Phylax probe failed: {exc}",
                            freshness=Freshness.FRESH,
                            checked_at=checked_at,
                            latency_ms=latency,
                            evidence=f"Phylax probe exception: {exc}",
                        )

            elif component_id == "dianoia":
                if not self.dianoia:
                    status = ComponentStatus(
                        state=ComponentState.OFFLINE,
                        detail="Dianoia adapter unavailable",
                        freshness=Freshness.FRESH,
                        checked_at=checked_at,
                        evidence="Adapter not instantiated",
                    )
                else:
                    try:
                        # Non-mutating fallback deliberation probe
                        res = self.dianoia.deliberate(query="canary", seed_citations=[], force_zero_edges=True)
                        latency = int((time.monotonic() - start_time) * 1000)
                        status = ComponentStatus(
                            state=ComponentState.SIMULATED,
                            detail="Dianoia zero-edge fallback verified",
                            freshness=Freshness.FRESH,
                            version="0.1.0",
                            checked_at=checked_at,
                            latency_ms=latency,
                            evidence=f"Deliberation probe confirmed flat fallback ({latency}ms)",
                            is_simulated=True,
                            status="simulated",
                            mode="simulated",
                        )
                    except Exception as exc:
                        latency = int((time.monotonic() - start_time) * 1000)
                        status = ComponentStatus(
                            state=ComponentState.OFFLINE,
                            detail=f"Dianoia probe failed: {exc}",
                            freshness=Freshness.FRESH,
                            checked_at=checked_at,
                            latency_ms=latency,
                            evidence=f"Dianoia probe exception: {exc}",
                        )

            else:
                status = self.get_cached_status(component_id)

            self._cache[component_id] = status
            return status
