"""Tests for Hestia Speculative Cascades serving engine and API endpoints."""

import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from hestia.api.app import create_app
from hestia.cascade.engine import CascadeConfig, CascadeDecision, SpeculativeCascadeEngine
from hestia.registry import ModelEntry, ModelRegistry, ModelType, SamplingParams


def test_cascade_config_validation():
    # Valid config
    cfg = CascadeConfig(
        draft_model="llama-3.2-1b",
        verifier_model="llama-3.1-8b",
        deferral_threshold=0.85,
        criterion="mean_prob",
    )
    assert cfg.deferral_threshold == 0.85

    # Invalid threshold
    with pytest.raises(ValueError, match="deferral_threshold must be in"):
        CascadeConfig(draft_model="a", verifier_model="b", deferral_threshold=1.5)

    # Invalid criterion
    with pytest.raises(ValueError, match="Unsupported criterion"):
        CascadeConfig(draft_model="a", verifier_model="b", criterion="unknown")


def test_extract_token_probs():
    # OpenAI format
    openai_logprobs = {
        "content": [
            {"token": "The", "logprob": 0.0},        # exp(0) = 1.0
            {"token": "capital", "logprob": -0.6931}, # exp(-0.6931) ~ 0.5
        ]
    }
    probs = SpeculativeCascadeEngine.extract_token_probs(openai_logprobs)
    assert len(probs) == 2
    assert probs[0] == pytest.approx(1.0, abs=1e-3)
    assert probs[1] == pytest.approx(0.5, abs=1e-3)

    # llama.cpp format
    llamacpp_logprobs = {
        "token_logprobs": [0.0, -0.6931]
    }
    probs2 = SpeculativeCascadeEngine.extract_token_probs(llamacpp_logprobs)
    assert len(probs2) == 2
    assert probs2[0] == pytest.approx(1.0, abs=1e-3)

    # None / Empty
    assert SpeculativeCascadeEngine.extract_token_probs(None) == []
    assert SpeculativeCascadeEngine.extract_token_probs({}) == []


@pytest.mark.asyncio
async def test_speculative_cascade_accepted_draft():
    """Verify that when the draft model produces high-confidence tokens, the draft is accepted."""
    cfg = CascadeConfig(
        draft_model="draft_model",
        verifier_model="verifier_model",
        deferral_threshold=0.80,
        criterion="mean_prob",
        cost_ratio=0.20,
    )
    engine = SpeculativeCascadeEngine(cfg)

    calls = []

    async def mock_forward(model_name: str, payload: dict) -> dict:
        calls.append(model_name)
        if model_name == "draft_model":
            return {
                "id": "chatcmpl-draft-123",
                "model": "draft_model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Paris is the capital."},
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {"token": "Paris", "logprob": -0.05},  # ~0.95
                                {"token": "is", "logprob": -0.02},     # ~0.98
                                {"token": "capital", "logprob": -0.05},# ~0.95
                            ]
                        },
                    }
                ],
                "usage": {"completion_tokens": 3, "prompt_tokens": 5, "total_tokens": 8},
            }
        else:
            return {"id": "chatcmpl-verifier-456", "model": "verifier_model"}

    req = {"messages": [{"role": "user", "content": "Capital of France?"}]}
    resp, decision = await engine.execute(req, mock_forward)

    # Only draft was queried; verifier was bypassed
    assert calls == ["draft_model"]
    assert decision.status == "ACCEPTED_DRAFT"
    assert decision.chosen_model == "draft_model"
    assert decision.deferral_score >= 0.80
    assert decision.draft_tokens == 3
    assert decision.verifier_tokens == 0
    assert decision.speedup_estimate == pytest.approx(5.0, abs=0.1)
    assert "cascade" in resp["usage"]
    assert resp["usage"]["cascade"]["status"] == "ACCEPTED_DRAFT"


@pytest.mark.asyncio
async def test_speculative_cascade_deferred_to_verifier():
    """Verify that when the draft model is uncertain, the request is deferred to the verifier."""
    cfg = CascadeConfig(
        draft_model="draft_model",
        verifier_model="verifier_model",
        deferral_threshold=0.80,
        criterion="mean_prob",
        cost_ratio=0.20,
    )
    engine = SpeculativeCascadeEngine(cfg)

    calls = []

    async def mock_forward(model_name: str, payload: dict) -> dict:
        calls.append(model_name)
        if model_name == "draft_model":
            return {
                "id": "chatcmpl-draft-123",
                "model": "draft_model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Uncertain answer."},
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {"token": "Uncertain", "logprob": -1.6},  # ~0.20
                                {"token": "answer", "logprob": -1.2},     # ~0.30
                            ]
                        },
                    }
                ],
                "usage": {"completion_tokens": 2, "prompt_tokens": 5, "total_tokens": 7},
            }
        else:
            return {
                "id": "chatcmpl-verifier-456",
                "model": "verifier_model",
                "choices": [
                    {"message": {"role": "assistant", "content": "Definitive verified response."}}
                ],
                "usage": {"completion_tokens": 4, "prompt_tokens": 5, "total_tokens": 9},
            }

    req = {"messages": [{"role": "user", "content": "Complex question?"}]}
    resp, decision = await engine.execute(req, mock_forward)

    # Both draft and verifier were queried
    assert calls == ["draft_model", "verifier_model"]
    assert decision.status == "DEFERRED_TO_VERIFIER"
    assert decision.chosen_model == "verifier_model"
    assert decision.deferral_score < 0.80
    assert decision.draft_tokens == 2
    assert decision.verifier_tokens == 4
    assert decision.total_tokens == 6
    assert resp["usage"]["cascade"]["status"] == "DEFERRED_TO_VERIFIER"


@pytest.fixture
def cascade_test_registry():
    registry = ModelRegistry()
    registry.register(
        ModelEntry(
            name="draft_model",
            model_type=ModelType.CHAT,
            gguf_path="/models/draft.gguf",
            quantization="Q4_K_M",
            context_size=4096,
            vram_estimate_mb=1500,
        )
    )
    registry.register(
        ModelEntry(
            name="verifier_model",
            model_type=ModelType.CHAT,
            gguf_path="/models/verifier.gguf",
            quantization="Q4_K_M",
            context_size=8192,
            vram_estimate_mb=6000,
        )
    )
    return registry


@pytest.mark.asyncio
async def test_api_cascade_endpoint(cascade_test_registry):
    """Test POST /v1/cascade/completions endpoint."""
    app = create_app(config_path="hestia.yaml", models_path="models.yaml")

    mock_manager = AsyncMock()
    mock_forward = AsyncMock(return_value={
        "id": "chatcmpl-test",
        "choices": [{
            "message": {"role": "assistant", "content": "High confidence draft."},
            "logprobs": {"content": [{"token": "High", "logprob": -0.01}]}
        }],
        "usage": {"completion_tokens": 3}
    })

    with patch("hestia.api.app._get_registry", return_value=cascade_test_registry), \
         patch("hestia.api.app._get_manager", return_value=mock_manager), \
         patch("hestia.api.app._forward_cascade_call", mock_forward):

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/v1/cascade/completions",
                json={
                    "draft_model": "draft_model",
                    "verifier_model": "verifier_model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "deferral_threshold": 0.70,
                },
                headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "usage" in data
            assert data["usage"]["cascade"]["status"] == "ACCEPTED_DRAFT"
            assert data["usage"]["cascade"]["chosen_model"] == "draft_model"
