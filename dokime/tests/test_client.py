"""Integration tests for HestiaClient against a mock OpenAI-compatible server."""

import json
import httpx
import pytest

from dokime.client import HestiaAPIError, HestiaClient


def test_list_models_openai_format():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "llama-3.1-8b-instruct", "object": "model"},
                        {"id": "nomic-embed-text", "object": "model"},
                    ],
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client_http = httpx.Client(transport=transport)
    client = HestiaClient(base_url="http://mock-hestia:8000", client=client_http)

    models = client.list_models()
    assert models == ["llama-3.1-8b-instruct", "nomic-embed-text"]


def test_list_models_hestia_native_fallback():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(404)
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "mock-model-7b", "state": "loaded"},
                    ],
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client_http = httpx.Client(transport=transport)
    client = HestiaClient(base_url="http://mock-hestia:8000", client=client_http)

    models = client.list_models()
    assert models == ["mock-model-7b"]


def test_chat_completion_success():
    captured_payload = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        if request.url.path == "/v1/chat/completions":
            captured_payload = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test123",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Answer: B"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    client_http = httpx.Client(transport=transport)
    client = HestiaClient(base_url="http://mock-hestia:8000", client=client_http)

    content, latency = client.chat_completion(
        model="test-model",
        prompt="Sample question?",
        seed=42,
        temperature=0.0,
        max_tokens=100,
    )

    assert content == "Answer: B"
    assert latency >= 0.0
    assert captured_payload["model"] == "test-model"
    assert captured_payload["seed"] == 42
    assert captured_payload["temperature"] == 0.0
    assert captured_payload["max_tokens"] == 100
    assert captured_payload["messages"][0]["content"] == "Sample question?"


def test_chat_completion_retry_and_backoff_then_success():
    attempt_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            return httpx.Response(503, text="Service temporarily overloaded")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "C"}}],
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client_http = httpx.Client(transport=transport)
    client = HestiaClient(
        base_url="http://mock-hestia:8000",
        max_retries=3,
        backoff_base=0.01,
        client=client_http,
    )

    content, _ = client.chat_completion(model="test-model", prompt="Q")
    assert content == "C"
    assert attempt_count == 3


def test_chat_completion_retry_exhaustion_raises():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(mock_handler)
    client_http = httpx.Client(transport=transport)
    client = HestiaClient(
        base_url="http://mock-hestia:8000",
        max_retries=2,
        backoff_base=0.01,
        client=client_http,
    )

    with pytest.raises(HestiaAPIError) as exc_info:
        client.chat_completion(model="failing-model", prompt="Q")

    assert "Chat completion failed after 2 retries" in str(exc_info.value)


def test_client_authorization_header():
    """Verify HestiaClient attaches Authorization Bearer token header (Finding A14)."""
    captured_headers = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_headers
        captured_headers = dict(request.headers)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Auth verified"}}]},
        )

    transport = httpx.MockTransport(mock_handler)
    client_http = httpx.Client(transport=transport)
    client = HestiaClient(
        base_url="http://mock-hestia:8000",
        token="test-secret-bearer-token-123",
        client=client_http,
    )

    content, _ = client.chat_completion(model="test-model", prompt="Hello")
    assert content == "Auth verified"
    assert captured_headers.get("authorization") == "Bearer test-secret-bearer-token-123"

