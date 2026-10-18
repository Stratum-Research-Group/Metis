"""
Unit tests for Recoverable Model-Swap Lifecycle in Hestia
per specification 03-recoverable-model-swaps-metis-a2853c8a.txt.
"""

import asyncio
import os
import pytest
from pathlib import Path

from hestia.server.swap import (
    ActiveRecord,
    ChangeRecord,
    InvalidStateTransitionError,
    ModelBundle,
    ModelSwapManager,
    RecoveryExhaustedError,
    StagingMode,
    StaleGenerationError,
    SwapState,
)


@pytest.fixture
def swap_manager(tmp_path):
    """Fixture providing an isolated ModelSwapManager."""
    return ModelSwapManager(state_dir=tmp_path / "swap_state")


@pytest.fixture
def sample_bundles(swap_manager):
    """Fixture registering two sample ModelBundles."""
    bundle_a = ModelBundle(
        bundle_id="bundle_a_hash",
        model_name="qwen2.5-coder-7b",
        runtime_build_id="vllm_0.5",
    )
    bundle_b = ModelBundle(
        bundle_id="bundle_b_hash",
        model_name="qwen2.5-coder-14b",
        runtime_build_id="vllm_0.5",
    )
    swap_manager.register_bundle(bundle_a)
    swap_manager.register_bundle(bundle_b)
    return bundle_a, bundle_b


@pytest.mark.asyncio
async def test_successful_swap_lifecycle(swap_manager, sample_bundles):
    """Verify happy path: PREPARED -> VALIDATED -> STAGED -> ACTIVATING -> ACTIVE_PENDING -> COMMITTED."""
    bundle_a, bundle_b = sample_bundles

    # Prepare initial model (gen 0 -> 1)
    rec1 = await swap_manager.prepare_change(
        slot="primary",
        candidate_bundle_id=bundle_a.bundle_id,
        expected_generation=0,
        idempotency_key="init_key_1",
        authorization_reference="auth_admin_1",
        staging_mode=StagingMode.MODE_B_MAINTENANCE,
    )
    assert rec1.state == SwapState.PREPARED

    rec1 = await swap_manager.validate_change(rec1.change_id)
    assert rec1.state == SwapState.VALIDATED

    rec1 = await swap_manager.stage_change(rec1.change_id)
    assert rec1.state == SwapState.STAGED

    def start_mock(bundle):
        return True, {"port": 8000, "pid": 1234}

    def stop_mock(instance):
        pass

    rec1 = await swap_manager.activate_change(
        rec1.change_id,
        start_process_fn=start_mock,
        stop_process_fn=stop_mock,
    )
    assert rec1.state == SwapState.ACTIVE_PENDING
    active = swap_manager.active_records["primary"]
    assert active.generation == 1
    assert active.bundle_id == bundle_a.bundle_id

    # Commit change
    rec1 = await swap_manager.commit_change(rec1.change_id, expected_generation=1)
    assert rec1.state == SwapState.COMMITTED


@pytest.mark.asyncio
async def test_cas_generation_guard_prevents_stale_swap(swap_manager, sample_bundles):
    """Verify that wrong expected_generation triggers StaleGenerationError (ABA defense)."""
    bundle_a, bundle_b = sample_bundles

    # Establish gen 1
    rec1 = await swap_manager.prepare_change("primary", bundle_a.bundle_id, 0, "k1", "auth")
    await swap_manager.validate_change(rec1.change_id)
    await swap_manager.stage_change(rec1.change_id)
    await swap_manager.activate_change(rec1.change_id, lambda b: (True, {}), lambda i: None)
    await swap_manager.commit_change(rec1.change_id, 1)

    # Attempting to prepare a change with stale expected_generation=0 must fail loudly
    with pytest.raises(StaleGenerationError, match="CAS generation check failed"):
        await swap_manager.prepare_change(
            slot="primary",
            candidate_bundle_id=bundle_b.bundle_id,
            expected_generation=0,  # Stale! Current is 1
            idempotency_key="stale_key",
            authorization_reference="auth",
        )


@pytest.mark.asyncio
async def test_idempotency_key_returns_existing_record(swap_manager, sample_bundles):
    """Retrying with identical idempotency key returns the original ChangeRecord."""
    bundle_a, _ = sample_bundles

    rec1 = await swap_manager.prepare_change("primary", bundle_a.bundle_id, 0, "idemp_123", "auth")
    rec2 = await swap_manager.prepare_change("primary", bundle_a.bundle_id, 0, "idemp_123", "auth")

    assert rec1.change_id == rec2.change_id
    assert len(swap_manager.changes) == 1


@pytest.mark.asyncio
async def test_mode_b_candidate_failure_triggers_auto_recovery(swap_manager, sample_bundles):
    """If candidate fails health check during activation, incumbent is auto-restored."""
    bundle_a, bundle_b = sample_bundles

    # Establish initial incumbent at gen 1
    rec1 = await swap_manager.prepare_change("primary", bundle_a.bundle_id, 0, "k1", "auth")
    await swap_manager.validate_change(rec1.change_id)
    await swap_manager.stage_change(rec1.change_id)
    await swap_manager.activate_change(rec1.change_id, lambda b: (True, {"pid": 111}), lambda i: None)
    await swap_manager.commit_change(rec1.change_id, 1)

    # Prepare swap to bundle_b
    rec2 = await swap_manager.prepare_change("primary", bundle_b.bundle_id, 1, "k2", "auth")
    await swap_manager.validate_change(rec2.change_id)
    await swap_manager.stage_change(rec2.change_id)

    # Mock process: bundle_b fails to start; bundle_a successfully restarts
    def start_mock(bundle):
        if bundle.bundle_id == bundle_b.bundle_id:
            return False, {}  # OOM / failure
        return True, {"pid": 222}  # Recovered incumbent

    stopped_instances = []
    def stop_mock(instance):
        stopped_instances.append(instance)

    # Activation must auto-recover and return ROLLED_BACK
    rec2 = await swap_manager.activate_change(
        rec2.change_id,
        start_process_fn=start_mock,
        stop_process_fn=stop_mock,
    )
    assert rec2.state == SwapState.ROLLED_BACK
    assert "Candidate process failed" in rec2.failure_reason

    # Active model remains bundle_a with recovered instance
    active = swap_manager.active_records["primary"]
    assert active.bundle_id == bundle_a.bundle_id
    assert active.effective_runtime_instance == {"pid": 222}


@pytest.mark.asyncio
async def test_active_pending_manual_rollback(swap_manager, sample_bundles):
    """Rollback from ACTIVE_PENDING restores prior bundle and advances generation."""
    bundle_a, bundle_b = sample_bundles

    # Start with A (gen 1)
    rec1 = await swap_manager.prepare_change("primary", bundle_a.bundle_id, 0, "k1", "auth")
    await swap_manager.validate_change(rec1.change_id)
    await swap_manager.stage_change(rec1.change_id)
    await swap_manager.activate_change(rec1.change_id, lambda b: (True, {}), lambda i: None)
    await swap_manager.commit_change(rec1.change_id, 1)

    # Move to B (gen 2, ACTIVE_PENDING)
    rec2 = await swap_manager.prepare_change("primary", bundle_b.bundle_id, 1, "k2", "auth")
    await swap_manager.validate_change(rec2.change_id)
    await swap_manager.stage_change(rec2.change_id)
    await swap_manager.activate_change(rec2.change_id, lambda b: (True, {"pid": 333}), lambda i: None)

    assert swap_manager.active_records["primary"].generation == 2

    # Issue rollback
    rec2 = await swap_manager.rollback_change(
        rec2.change_id,
        expected_generation=2,
        restore_process_fn=lambda b: (True, {"pid": 444}),
        stop_process_fn=lambda i: None,
        reason="Quality regression in eval gate",
    )
    assert rec2.state == SwapState.ROLLED_BACK

    # Generation advanced to 3 on rollback (preventing ABA)
    active = swap_manager.active_records["primary"]
    assert active.generation == 3
    assert active.bundle_id == bundle_a.bundle_id


@pytest.mark.asyncio
async def test_stream_request_pinning_and_draining(swap_manager, sample_bundles):
    """Verify that streams pinned to generation prevent premature swap before draining."""
    bundle_a, bundle_b = sample_bundles

    # Establish gen 1
    rec1 = await swap_manager.prepare_change("primary", bundle_a.bundle_id, 0, "k1", "auth")
    await swap_manager.validate_change(rec1.change_id)
    await swap_manager.stage_change(rec1.change_id)
    await swap_manager.activate_change(rec1.change_id, lambda b: (True, {}), lambda i: None)
    await swap_manager.commit_change(rec1.change_id, 1)

    # Pin an inflight stream request to gen 1
    swap_manager.acquire_stream_lease(generation=1)

    # In background, release stream after 0.2s
    async def delayed_release():
        await asyncio.sleep(0.2)
        swap_manager.release_stream_lease(generation=1)

    asyncio.create_task(delayed_release())

    rec2 = await swap_manager.prepare_change("primary", bundle_b.bundle_id, 1, "k2", "auth")
    await swap_manager.validate_change(rec2.change_id)
    await swap_manager.stage_change(rec2.change_id)

    # Mode B activate will wait for drain and succeed cleanly
    rec2 = await swap_manager.activate_change(
        rec2.change_id,
        start_process_fn=lambda b: (True, {}),
        stop_process_fn=lambda i: None,
        drain_timeout=1.0,
    )
    assert rec2.state == SwapState.ACTIVE_PENDING


def test_crash_reconciliation_on_startup(tmp_path, sample_bundles):
    """Mid-flight changes left in non-terminal states are reconciled to FAILED_CLOSED on startup."""
    state_dir = tmp_path / "crash_swap_state"
    mgr1 = ModelSwapManager(state_dir=state_dir)
    bundle_a = ModelBundle(bundle_id="bundle_a", model_name="m1")
    mgr1.register_bundle(bundle_a)

    loop = asyncio.new_event_loop()
    rec = loop.run_until_complete(
        mgr1.prepare_change("primary", "bundle_a", 0, "crash_k", "auth")
    )
    # Simulate a crash right after setting state to ACTIVATING
    rec.state = SwapState.ACTIVATING
    mgr1._sync_to_disk()
    loop.close()

    # Restart a fresh ModelSwapManager pointing to same directory
    mgr2 = ModelSwapManager(state_dir=state_dir)
    reconciled_rec = mgr2.changes[rec.change_id]
    assert reconciled_rec.state == SwapState.FAILED_CLOSED
    assert "Process crashed mid-transition" in reconciled_rec.failure_reason
