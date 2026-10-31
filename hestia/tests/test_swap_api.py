"""
Comprehensive integration tests for Hestia ModelSwapManager,
dynamic LoRA adapter serving, Paideia export ingestion, and /manage/swap REST APIs.
"""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from hestia.api.app import create_app
from hestia.backend import BackendProcess, BackendStartError
from hestia.backend.llamacpp import LlamaCppBackend
from hestia.config import PortRange, ServiceConfig
from hestia.registry import ModelEntry, ModelRegistry, ModelState, ModelType, SamplingParams
from hestia.server import ServerManager
from hestia.server.swap import ModelBundle, SwapState


def _make_dummy_safetensors(path: Path) -> None:
    header = json.dumps({"weight_tensor": {"dtype": "F16", "shape": [16, 16], "data_offsets": [0, 512]}}).encode("utf-8")
    header_len = len(header)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header)
        f.write(b"\x00" * 512)


@pytest.fixture
def test_env(tmp_path: Path):
    """Setup isolated model files, configs, and registry."""
    base_gguf = tmp_path / "base_model.gguf"
    base_gguf.write_bytes(b"GGUF_MOCK_DATA")

    adapter_bin = tmp_path / "adapter_lora.bin"
    adapter_bin.write_bytes(b"LORA_MOCK_WEIGHTS")

    registry = ModelRegistry()
    base_model = ModelEntry(
        name="test-base-model",
        model_type=ModelType.CHAT,
        gguf_path=base_gguf,
        quantization="Q4_K_M",
        context_size=8192,
        vram_estimate_mb=4000,
        gpu_target="rtx-5060-ti",
        sampling=SamplingParams(),
    )
    registry.register(base_model)

    config = ServiceConfig(
        listen_port=8080,
        llamacpp_binary="/usr/local/bin/llama-server",
        backend_port_range=PortRange(start=9100, end=9105),
        idle_timeout_seconds=600,
        log_level="WARNING",
    )

    backend = AsyncMock()
    backend.start = AsyncMock(
        side_effect=lambda model, port: BackendProcess(model_name=model.name, pid=5555, port=port)
    )
    backend.stop = AsyncMock()
    backend.health_check = AsyncMock(return_value=True)

    swap_dir = tmp_path / "hestia_swap"
    swap_dir.mkdir(parents=True, exist_ok=True)

    # Attach swap_state_dir to config
    object.__setattr__(config, "swap_state_dir", str(swap_dir))

    manager = ServerManager(registry=registry, backend=backend, config=config)

    return {
        "tmp_path": tmp_path,
        "base_gguf": base_gguf,
        "adapter_bin": adapter_bin,
        "base_model": base_model,
        "registry": registry,
        "config": config,
        "backend": backend,
        "manager": manager,
        "swap_dir": swap_dir,
    }


def test_llamacpp_lora_flags_derivation(test_env):
    """Verify that LlamaCppBackend appends --lora and --lora-scaled flags."""
    tmp_path = test_env["tmp_path"]
    adapter_p = test_env["adapter_bin"]
    backend = LlamaCppBackend(binary_path="/bin/true")

    # 1. Unscaled LoRA
    model_lora = ModelEntry(
        name="model-with-lora",
        model_type=ModelType.CHAT,
        gguf_path=test_env["base_gguf"],
        quantization="Q4_K_M",
        context_size=4096,
        vram_estimate_mb=4500,
        lora_adapter_path=adapter_p,
    )
    args = backend.build_launch_args(model_lora, port=9001)
    assert "--lora" in args
    lora_idx = args.index("--lora")
    assert args[lora_idx + 1] == str(adapter_p)

    # 2. Scaled LoRA
    model_lora_scaled = ModelEntry(
        name="model-with-lora-scaled",
        model_type=ModelType.CHAT,
        gguf_path=test_env["base_gguf"],
        quantization="Q4_K_M",
        context_size=4096,
        vram_estimate_mb=4500,
        lora_adapter_path=adapter_p,
        lora_scale=0.75,
    )
    args_scaled = backend.build_launch_args(model_lora_scaled, port=9002)
    assert "--lora-scaled" in args_scaled
    scaled_idx = args_scaled.index("--lora-scaled")
    assert args_scaled[scaled_idx + 1] == str(adapter_p)
    assert args_scaled[scaled_idx + 2] == "0.75"


@pytest.mark.asyncio
async def test_llamacpp_lora_missing_file_fails(test_env):
    """Verify that LlamaCppBackend.start fails if LoRA file does not exist."""
    backend = LlamaCppBackend(binary_path="/bin/true")
    bad_model = ModelEntry(
        name="model-bad-lora",
        model_type=ModelType.CHAT,
        gguf_path=test_env["base_gguf"],
        quantization="Q4_K_M",
        context_size=4096,
        vram_estimate_mb=4500,
        lora_adapter_path=Path("/tmp/non_existent_lora.bin"),
    )
    with pytest.raises(BackendStartError, match="LoRA adapter weights not found"):
        await backend.start(bad_model, port=9001)


def test_paideia_entry_registration_into_registry(test_env):
    """Verify ModelRegistry.register_paideia_entry creates valid ModelEntry with LoRA path."""
    registry = test_env["registry"]
    adapter_p = test_env["adapter_bin"]

    paideia_entry = {
        "schema_version": "1.0.0",
        "adapter_id": "paideia-router-lora-v1",
        "base_model_name": "test-base-model",
        "base_model_digest": "sha256:1111",
        "adapter_digest": "sha256:2222",
        "tokenizer_digest": "sha256:3333",
        "template_digest": "sha256:4444",
        "task": "question_router",
        "compatible_backend": "llamacpp",
        "training_data_fingerprint": "train_data_hash_1",
        "evaluation_run_id": "eval_run_99",
        "rollback_target": None,
    }

    derived_entry = registry.register_paideia_entry(
        entry_data=paideia_entry,
        adapter_path=adapter_p,
        base_model_name="test-base-model",
        vram_overhead_mb=500,
    )

    assert derived_entry.name == "paideia-router-lora-v1"
    assert derived_entry.gguf_path == test_env["base_gguf"]
    assert derived_entry.lora_adapter_path == adapter_p
    assert derived_entry.vram_estimate_mb == 4500
    assert derived_entry.artifact_digest == "sha256:2222"
    assert registry.get("paideia-router-lora-v1") is derived_entry


@pytest.mark.asyncio
async def test_server_manager_swap_and_rollback_lifecycle(test_env):
    """Verify ServerManager.swap_model and rollback_model lifecycle."""
    manager: ServerManager = test_env["manager"]
    registry: ModelRegistry = test_env["registry"]

    # Register candidate model B
    model_b_gguf = test_env["tmp_path"] / "model_b.gguf"
    model_b_gguf.write_bytes(b"GGUF_B")
    model_b = ModelEntry(
        name="candidate-model-b",
        model_type=ModelType.CHAT,
        gguf_path=model_b_gguf,
        quantization="Q4_K_M",
        context_size=8192,
        vram_estimate_mb=4000,
    )
    registry.register(model_b)

    bundle_a = ModelBundle(bundle_id="b_initial", model_name="test-base-model")
    bundle_b = ModelBundle(bundle_id="b_candidate", model_name="candidate-model-b")
    manager.swap_manager.register_bundle(bundle_a)
    manager.swap_manager.register_bundle(bundle_b)

    # 1. Initial swap (gen 0 -> 1)
    rec1 = await manager.swap_model(
        slot="primary",
        candidate_bundle_id="b_initial",
        expected_generation=0,
        idempotency_key="init_swap",
    )
    assert rec1.state == SwapState.COMMITTED
    active = manager.swap_manager.active_records["primary"]
    assert active.generation == 1
    assert active.bundle_id == "b_initial"
    assert manager.is_loaded("test-base-model")

    # 2. Swap to bundle B (gen 1 -> 2)
    rec2 = await manager.swap_model(
        slot="primary",
        candidate_bundle_id="b_candidate",
        expected_generation=1,
        idempotency_key="swap_to_b",
    )
    assert rec2.state == SwapState.COMMITTED
    active = manager.swap_manager.active_records["primary"]
    assert active.generation == 2
    assert active.bundle_id == "b_candidate"
    assert manager.is_loaded("candidate-model-b")

    # 3. Rollback to bundle A (gen 2 -> 3)
    rec_rb = await manager.rollback_model(
        change_id=rec2.change_id,
        expected_generation=2,
        reason="Test rollback",
    )
    assert rec_rb.state == SwapState.ROLLED_BACK
    active = manager.swap_manager.active_records["primary"]
    assert active.generation == 3
    assert active.bundle_id == "b_initial"
    assert manager.is_loaded("test-base-model")


@pytest.mark.asyncio
async def test_rest_api_swap_and_rollback(test_env):
    """Test /manage/swap, /manage/rollback, and /manage/bundles endpoints."""
    manager: ServerManager = test_env["manager"]
    registry: ModelRegistry = test_env["registry"]

    app = create_app()

    # Register candidate model C
    model_c_gguf = test_env["tmp_path"] / "model_c.gguf"
    model_c_gguf.write_bytes(b"GGUF_C")
    model_c = ModelEntry(
        name="model-c",
        model_type=ModelType.CHAT,
        gguf_path=model_c_gguf,
        quantization="Q4_K_M",
        context_size=4096,
        vram_estimate_mb=3000,
    )
    registry.register(model_c)

    with patch("hestia.api.app._get_manager", return_value=manager), \
         patch("hestia.api.app._get_registry", return_value=registry):

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # 1. Register bundles
            resp_b1 = await client.post(
                "/manage/bundles",
                json={"bundle_id": "bundle_base", "model_name": "test-base-model"},
            )
            assert resp_b1.status_code == 200

            resp_b2 = await client.post(
                "/manage/bundles",
                json={"bundle_id": "bundle_c", "model_name": "model-c"},
            )
            assert resp_b2.status_code == 200

            # 2. Check /manage/swap status
            resp_st = await client.get("/manage/swap")
            assert resp_st.status_code == 200
            assert "bundle_base" in resp_st.json()["bundles"]

            # 3. Perform initial swap (gen 0 -> 1)
            resp_sw1 = await client.post(
                "/manage/swap",
                json={
                    "slot": "primary",
                    "candidate_bundle_id": "bundle_base",
                    "expected_generation": 0,
                    "idempotency_key": "api_swap_1",
                    "auto_commit": True,
                },
            )
            assert resp_sw1.status_code == 200
            data_sw1 = resp_sw1.json()
            assert data_sw1["status"] == "swapped"
            assert data_sw1["active"]["generation"] == 1

            # 4. CAS Conflict check (submitting expected_generation=0 when gen is 1)
            resp_conflict = await client.post(
                "/manage/swap",
                json={
                    "slot": "primary",
                    "candidate_bundle_id": "bundle_c",
                    "expected_generation": 0,  # Stale!
                    "idempotency_key": "stale_attempt",
                },
            )
            assert resp_conflict.status_code == 409
            assert "CAS Conflict" in resp_conflict.json()["detail"]

            # 5. Swap to bundle C (gen 1 -> 2)
            resp_sw2 = await client.post(
                "/manage/swap",
                json={
                    "slot": "primary",
                    "candidate_bundle_id": "bundle_c",
                    "expected_generation": 1,
                    "idempotency_key": "api_swap_2",
                    "auto_commit": True,
                },
            )
            assert resp_sw2.status_code == 200
            data_sw2 = resp_sw2.json()
            assert data_sw2["active"]["generation"] == 2
            change_id = data_sw2["change"]["change_id"]

            # 6. Rollback to bundle_base (gen 2 -> 3)
            resp_rb = await client.post(
                "/manage/rollback",
                json={
                    "change_id": change_id,
                    "expected_generation": 2,
                    "reason": "Test REST rollback",
                },
            )
            assert resp_rb.status_code == 200
            data_rb = resp_rb.json()
            assert data_rb["status"] == "rolled_back"
            assert data_rb["active"]["generation"] == 3
            assert data_rb["active"]["bundle_id"] == "bundle_base"


@pytest.mark.asyncio
async def test_end_to_end_paideia_export_to_hestia_swap(test_env):
    """
    End-to-end integration:
    Paideia exports hestia_registry_entry.json manifest ->
    Hestia ingests manifest via /manage/bundles ->
    Hestia executes crash-safe CAS model swap to activate the new adapter.
    """
    manager: ServerManager = test_env["manager"]
    registry: ModelRegistry = test_env["registry"]
    tmp_path = test_env["tmp_path"]

    # 1. Create simulated Paideia adapter directory with valid safetensors and manifest
    adapter_dir = tmp_path / "paideia_exported_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    _make_dummy_safetensors(adapter_file)

    manifest_file = adapter_dir / "manifest.json"
    manifest_file.write_text(json.dumps({
        "status": "COMPLETED",
        "is_simulated": False,
        "run_id": "run_paideia_001",
    }))

    # Paideia registry manifest entry (conforming to HestiaAdapterRegistryEntry)
    paideia_manifest = {
        "schema_version": "1.0.0",
        "adapter_id": "paideia-task-adapter-v1",
        "base_model_name": "test-base-model",
        "base_model_digest": "sha256:dummy_base_digest",
        "adapter_digest": "sha256:dummy_adapter_digest",
        "tokenizer_digest": "sha256:dummy_tok_digest",
        "template_digest": "sha256:dummy_tmpl_digest",
        "task": "question_router",
        "compatible_backend": "llamacpp",
        "training_data_fingerprint": "train_fp_12345",
        "evaluation_run_id": "eval_run_9999",
        "resource_footprint": {"peak_vram_mb": 4200},
    }

    app = create_app()

    with patch("hestia.api.app._get_manager", return_value=manager), \
         patch("hestia.api.app._get_registry", return_value=registry):

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # 2. Ingest Paideia entry via POST /manage/bundles
            resp = await client.post(
                "/manage/bundles",
                json={
                    "paideia_entry": paideia_manifest,
                    "adapter_path": str(adapter_file),
                    "base_model_name": "test-base-model",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "registered"
            assert data["model"] == "paideia-task-adapter-v1"

            # 3. Verify ModelRegistry now has the derived model with LoRA configured
            registered_model = registry.get("paideia-task-adapter-v1")
            assert registered_model is not None
            assert registered_model.lora_adapter_path == adapter_file

            # 4. Perform CAS hot-swap to the Paideia adapter
            swap_resp = await client.post(
                "/manage/swap",
                json={
                    "slot": "primary",
                    "candidate_bundle_id": "paideia-task-adapter-v1",
                    "expected_generation": 0,
                    "idempotency_key": "e2e_swap_key",
                    "auto_commit": True,
                },
            )
            assert swap_resp.status_code == 200
            swap_data = swap_resp.json()
            assert swap_data["status"] == "swapped"
            assert swap_data["active"]["bundle_id"] == "paideia-task-adapter-v1"
            assert swap_data["active"]["generation"] == 1
            assert manager.is_loaded("paideia-task-adapter-v1")
