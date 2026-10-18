"""
Recoverable, Crash-Safe Model-Swap Lifecycle for Hestia (Metis serving runtime).
Implements the generation-checked CAS state machine:
  PREPARED -> VALIDATED -> STAGED -> ACTIVATING -> ACTIVE_PENDING -> COMMITTED
With explicit rollback, Mode A (side-by-side) and Mode B (maintenance replacement),
durable atomic persistence (POSIX replace + dir fsync), and inflight stream pinning.
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class SwapError(Exception):
    """Base exception for model swap failures."""
    pass


class StaleGenerationError(SwapError):
    """Raised when CAS active_generation != expected_generation (prevents ABA hazards)."""
    pass


class InvalidStateTransitionError(SwapError):
    """Raised when attempting an unauthorized state transition."""
    pass


class RecoveryExhaustedError(SwapError):
    """Raised when incumbent restoration fails, resulting in FAILED_CLOSED state."""
    pass


class SwapState(str, enum.Enum):
    PREPARED = "PREPARED"
    VALIDATED = "VALIDATED"
    STAGED = "STAGED"
    ACTIVATING = "ACTIVATING"
    ACTIVE_PENDING = "ACTIVE_PENDING"
    COMMITTED = "COMMITTED"
    REJECTED = "REJECTED"
    RECOVERING = "RECOVERING"
    ROLLING_BACK = "ROLLING_BACK"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED_CLOSED = "FAILED_CLOSED"


class StagingMode(str, enum.Enum):
    MODE_A_SIDE_BY_SIDE = "MODE_A_SIDE_BY_SIDE"
    MODE_B_MAINTENANCE = "MODE_B_MAINTENANCE"


@dataclass
class ModelBundle:
    """Immutable model bundle specification."""
    bundle_id: str
    schema_version: str = "1.0.0"
    model_name: str = ""
    asset_hashes: Dict[str, str] = field(default_factory=dict)
    tokenizer_config_hash: str = ""
    runtime_build_id: str = ""
    quantization: str = "none"
    generation_settings_hash: str = ""
    capabilities: List[str] = field(default_factory=list)
    embedding_space_id: Optional[str] = None
    index_build_id: Optional[str] = None
    compatibility_constraints: Dict[str, Any] = field(default_factory=dict)
    verified_recovery_instructions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelBundle:
        return cls(**data)


@dataclass
class ActiveRecord:
    """Represents the live active model pointer and generation in a slot."""
    slot: str
    generation: int
    bundle_id: str
    activated_change_id: str
    effective_runtime_instance: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ActiveRecord:
        return cls(**data)


@dataclass
class ChangeRecord:
    """Durable tracking record for a model swap change request."""
    change_id: str
    idempotency_key: str
    slot: str
    expected_generation: int
    prior_bundle_id: Optional[str]
    candidate_bundle_id: str
    state: SwapState
    staging_mode: StagingMode
    evidence_hashes: List[str] = field(default_factory=list)
    authorization_reference: str = ""
    readiness_result: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    retention_deadline: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value if isinstance(self.state, SwapState) else str(self.state)
        d["staging_mode"] = self.staging_mode.value if isinstance(self.staging_mode, StagingMode) else str(self.staging_mode)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChangeRecord:
        d = dict(data)
        d["state"] = SwapState(d["state"])
        d["staging_mode"] = StagingMode(d["staging_mode"])
        return cls(**d)


class ModelSwapManager:
    """
    Orchestrates crash-safe model swaps with monotonic generations,
    CAS transitions, Mode B maintenance downtime, and request pinning.
    """

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

        # In-memory maps persisted to disk
        self.bundles: Dict[str, ModelBundle] = {}
        self.active_records: Dict[str, ActiveRecord] = {}
        self.changes: Dict[str, ChangeRecord] = {}
        self.idempotency_map: Dict[str, str] = {}  # idempotency_key -> change_id

        # In-flight stream pinning: generation -> active request count
        self._generation_pins: Dict[int, int] = {}
        self._pin_events: Dict[int, asyncio.Event] = {}

        self._load_durable_state()

    def _load_durable_state(self) -> None:
        """Load durable state records from state_dir and reconcile on startup."""
        bundles_file = self.state_dir / "bundles.json"
        if bundles_file.is_file():
            try:
                with open(bundles_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.bundles[k] = ModelBundle.from_dict(v)
            except Exception as e:
                logger.error("Failed to load bundles.json: %s", e)

        active_file = self.state_dir / "active.json"
        if active_file.is_file():
            try:
                with open(active_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.active_records[k] = ActiveRecord.from_dict(v)
            except Exception as e:
                logger.error("Failed to load active.json: %s", e)

        changes_file = self.state_dir / "changes.json"
        if changes_file.is_file():
            try:
                with open(changes_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        rec = ChangeRecord.from_dict(v)
                        self.changes[k] = rec
                        self.idempotency_map[rec.idempotency_key] = rec.change_id
            except Exception as e:
                logger.error("Failed to load changes.json: %s", e)

        # Startup reconciliation
        self._reconcile_startup_states()

    def _reconcile_startup_states(self) -> None:
        """Reconcile in-flight changes left incomplete by a sudden crash."""
        for change_id, change in list(self.changes.items()):
            if change.state in (SwapState.ACTIVATING, SwapState.ROLLING_BACK, SwapState.RECOVERING):
                logger.warning(
                    "Startup reconciliation: change %s in non-terminal state %s. Failing closed.",
                    change_id, change.state
                )
                change.state = SwapState.FAILED_CLOSED
                change.failure_reason = "Process crashed mid-transition. Closed by startup reconciliation."
                change.updated_at = time.time()
        self._sync_to_disk()

    def _sync_to_disk(self) -> None:
        """Atomically persist state files with directory sync."""
        self._atomic_json_write("bundles.json", {k: v.to_dict() for k, v in self.bundles.items()})
        self._atomic_json_write("active.json", {k: v.to_dict() for k, v in self.active_records.items()})
        self._atomic_json_write("changes.json", {k: v.to_dict() for k, v in self.changes.items()})

    def _atomic_json_write(self, filename: str, data: Dict[str, Any]) -> None:
        target = self.state_dir / filename
        tmp = self.state_dir / f"{filename}.tmp.{os.getpid()}_{int(time.time()*1000)}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        # Sync parent directory to ensure entry is committed
        dir_fd = os.open(str(self.state_dir), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    # ── Stream Lease Pinning ───────────────────────────────────

    def acquire_stream_lease(self, generation: int) -> None:
        """Pin an admitted stream request to a generation."""
        self._generation_pins[generation] = self._generation_pins.get(generation, 0) + 1

    def release_stream_lease(self, generation: int) -> None:
        """Release a stream lease. Signals event when count reaches zero."""
        count = self._generation_pins.get(generation, 0) - 1
        if count <= 0:
            self._generation_pins.pop(generation, None)
            evt = self._pin_events.get(generation)
            if evt:
                evt.set()
        else:
            self._generation_pins[generation] = count

    async def wait_for_stream_drain(self, generation: int, timeout: float = 30.0) -> bool:
        """Wait until all active streams pinned to generation have drained."""
        if generation not in self._generation_pins or self._generation_pins[generation] <= 0:
            return True
        evt = self._pin_events.setdefault(generation, asyncio.Event())
        try:
            await asyncio.wait_for(evt.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning("Stream drain timed out after %.1fs for generation %d", timeout, generation)
            return False

    # ── Lifecycle API ──────────────────────────────────────────

    def register_bundle(self, bundle: ModelBundle) -> None:
        """Register an immutable ModelBundle."""
        self.bundles[bundle.bundle_id] = bundle
        self._sync_to_disk()

    async def prepare_change(
        self,
        slot: str,
        candidate_bundle_id: str,
        expected_generation: int,
        idempotency_key: str,
        authorization_reference: str,
        staging_mode: StagingMode = StagingMode.MODE_B_MAINTENANCE,
        evidence_hashes: Optional[List[str]] = None,
    ) -> ChangeRecord:
        """Initiate PREPARED state with CAS guard and idempotency."""
        async with self._lock:
            # Idempotency check: return existing record if already submitted
            if idempotency_key in self.idempotency_map:
                existing_id = self.idempotency_map[idempotency_key]
                return self.changes[existing_id]

            if candidate_bundle_id not in self.bundles:
                raise SwapError(f"Candidate bundle '{candidate_bundle_id}' not found.")

            active = self.active_records.get(slot)
            current_gen = active.generation if active else 0

            if current_gen != expected_generation:
                raise StaleGenerationError(
                    f"CAS generation check failed for slot '{slot}': current={current_gen}, "
                    f"expected={expected_generation}"
                )

            change_id = hashlib.sha256(
                f"{slot}:{idempotency_key}:{candidate_bundle_id}:{expected_generation}".encode("utf-8")
            ).hexdigest()[:16]

            rec = ChangeRecord(
                change_id=change_id,
                idempotency_key=idempotency_key,
                slot=slot,
                expected_generation=expected_generation,
                prior_bundle_id=active.bundle_id if active else None,
                candidate_bundle_id=candidate_bundle_id,
                state=SwapState.PREPARED,
                staging_mode=staging_mode,
                evidence_hashes=evidence_hashes or [],
                authorization_reference=authorization_reference,
            )

            self.changes[change_id] = rec
            self.idempotency_map[idempotency_key] = change_id
            self._sync_to_disk()
            return rec

    async def validate_change(
        self,
        change_id: str,
        validation_fn: Optional[Callable[[ModelBundle], bool]] = None,
    ) -> ChangeRecord:
        """Transition from PREPARED -> VALIDATED after verifying artifacts & constraints."""
        async with self._lock:
            rec = self.changes.get(change_id)
            if not rec:
                raise SwapError(f"Change '{change_id}' not found.")
            if rec.state != SwapState.PREPARED:
                raise InvalidStateTransitionError(f"Cannot validate from state {rec.state}")

            candidate = self.bundles[rec.candidate_bundle_id]
            if validation_fn and not validation_fn(candidate):
                rec.state = SwapState.REJECTED
                rec.failure_reason = "Candidate validation function failed"
                self._sync_to_disk()
                return rec

            rec.state = SwapState.VALIDATED
            rec.updated_at = time.time()
            self._sync_to_disk()
            return rec

    async def stage_change(self, change_id: str) -> ChangeRecord:
        """Transition from VALIDATED -> STAGED."""
        async with self._lock:
            rec = self.changes.get(change_id)
            if not rec:
                raise SwapError(f"Change '{change_id}' not found.")
            if rec.state != SwapState.VALIDATED:
                raise InvalidStateTransitionError(f"Cannot stage from state {rec.state}")

            rec.state = SwapState.STAGED
            rec.updated_at = time.time()
            self._sync_to_disk()
            return rec

    async def activate_change(
        self,
        change_id: str,
        start_process_fn: Callable[[ModelBundle], Tuple[bool, Dict[str, Any]]],
        stop_process_fn: Callable[[Dict[str, Any]], None],
        drain_timeout: float = 10.0,
    ) -> ChangeRecord:
        """
        Transition from STAGED -> ACTIVATING -> ACTIVE_PENDING.
        Enforces CAS generation check. In Mode B: drains streams, stops incumbent,
        starts candidate, and auto-rolls back if candidate fails.
        """
        async with self._lock:
            rec = self.changes.get(change_id)
            if not rec:
                raise SwapError(f"Change '{change_id}' not found.")
            if rec.state != SwapState.STAGED:
                raise InvalidStateTransitionError(f"Cannot activate from state {rec.state}")

            active = self.active_records.get(rec.slot)
            current_gen = active.generation if active else 0
            if current_gen != rec.expected_generation:
                rec.state = SwapState.REJECTED
                rec.failure_reason = f"CAS failure: current {current_gen} != expected {rec.expected_generation}"
                self._sync_to_disk()
                raise StaleGenerationError(rec.failure_reason)

            # 1. Persist ACTIVATING intent
            rec.state = SwapState.ACTIVATING
            rec.updated_at = time.time()
            self._sync_to_disk()

        # 2. Mode B: Drain streams and stop incumbent
        if rec.staging_mode == StagingMode.MODE_B_MAINTENANCE and active:
            logger.info("Mode B swap: draining active streams for generation %d", active.generation)
            await self.wait_for_stream_drain(active.generation, timeout=drain_timeout)
            logger.info("Stopping incumbent process for slot %s", rec.slot)
            try:
                stop_process_fn(active.effective_runtime_instance)
            except Exception as e:
                logger.error("Error stopping incumbent process: %s", e)

        # 3. Start candidate process & check served identity
        candidate = self.bundles[rec.candidate_bundle_id]
        success, runtime_instance = False, {}
        try:
            success, runtime_instance = start_process_fn(candidate)
        except Exception as e:
            logger.error("Exception starting candidate process: %s", e)
            success = False

        async with self._lock:
            if not success:
                logger.error("Candidate failed to start or pass health check. Triggering RECOVERING.")
                rec.state = SwapState.RECOVERING
                rec.failure_reason = "Candidate process failed identity/health validation"
                self._sync_to_disk()

                # Attempt incumbent recovery
                if active and rec.prior_bundle_id and rec.prior_bundle_id in self.bundles:
                    prior = self.bundles[rec.prior_bundle_id]
                    prior_ok, prior_instance = start_process_fn(prior)
                    if prior_ok:
                        logger.info("Incumbent successfully recovered for slot %s", rec.slot)
                        active.effective_runtime_instance = prior_instance
                        active.updated_at = time.time()
                        rec.state = SwapState.ROLLED_BACK
                        self._sync_to_disk()
                        return rec

                rec.state = SwapState.FAILED_CLOSED
                self._sync_to_disk()
                raise RecoveryExhaustedError("Candidate failed and incumbent could not be recovered.")

            # 4. Success: Advance generation and persist ACTIVE_PENDING
            new_generation = current_gen + 1
            new_active = ActiveRecord(
                slot=rec.slot,
                generation=new_generation,
                bundle_id=candidate.bundle_id,
                activated_change_id=rec.change_id,
                effective_runtime_instance=runtime_instance,
                updated_at=time.time(),
            )
            self.active_records[rec.slot] = new_active
            rec.state = SwapState.ACTIVE_PENDING
            rec.updated_at = time.time()
            self._sync_to_disk()
            return rec

    async def commit_change(self, change_id: str, expected_generation: int) -> ChangeRecord:
        """Transition from ACTIVE_PENDING -> COMMITTED with CAS check."""
        async with self._lock:
            rec = self.changes.get(change_id)
            if not rec:
                raise SwapError(f"Change '{change_id}' not found.")
            if rec.state != SwapState.ACTIVE_PENDING:
                raise InvalidStateTransitionError(f"Cannot commit from state {rec.state}")

            active = self.active_records.get(rec.slot)
            if not active or active.generation != expected_generation:
                raise StaleGenerationError(
                    f"CAS commit failed: active gen {getattr(active, 'generation', None)} != expected {expected_generation}"
                )

            rec.state = SwapState.COMMITTED
            rec.updated_at = time.time()
            self._sync_to_disk()
            return rec

    async def rollback_change(
        self,
        change_id: str,
        expected_generation: int,
        restore_process_fn: Callable[[ModelBundle], Tuple[bool, Dict[str, Any]]],
        stop_process_fn: Callable[[Dict[str, Any]], None],
        reason: str = "Rollback requested",
    ) -> ChangeRecord:
        """Rollback an ACTIVE_PENDING change to the prior bundle with CAS guard."""
        async with self._lock:
            rec = self.changes.get(change_id)
            if not rec:
                raise SwapError(f"Change '{change_id}' not found.")
            if rec.state != SwapState.ACTIVE_PENDING:
                raise InvalidStateTransitionError(f"Cannot rollback from state {rec.state}")

            active = self.active_records.get(rec.slot)
            if not active or active.generation != expected_generation:
                raise StaleGenerationError(
                    f"CAS rollback failed: active gen {getattr(active, 'generation', None)} != expected {expected_generation}"
                )

            if not rec.prior_bundle_id or rec.prior_bundle_id not in self.bundles:
                raise SwapError("No prior bundle available for rollback.")

            rec.state = SwapState.ROLLING_BACK
            rec.failure_reason = reason
            rec.updated_at = time.time()
            self._sync_to_disk()

        # Stop active process
        try:
            stop_process_fn(active.effective_runtime_instance)
        except Exception as e:
            logger.error("Error stopping process during rollback: %s", e)

        # Restore prior bundle
        prior_bundle = self.bundles[rec.prior_bundle_id]
        ok, runtime_instance = restore_process_fn(prior_bundle)

        async with self._lock:
            if not ok:
                rec.state = SwapState.FAILED_CLOSED
                self._sync_to_disk()
                raise RecoveryExhaustedError("Failed to restore prior model bundle during rollback.")

            # Rollback advances generation to prevent ABA hazards
            new_gen = active.generation + 1
            self.active_records[rec.slot] = ActiveRecord(
                slot=rec.slot,
                generation=new_gen,
                bundle_id=prior_bundle.bundle_id,
                activated_change_id=f"rollback:{rec.change_id}",
                effective_runtime_instance=runtime_instance,
                updated_at=time.time(),
            )
            rec.state = SwapState.ROLLED_BACK
            rec.updated_at = time.time()
            self._sync_to_disk()
            return rec
