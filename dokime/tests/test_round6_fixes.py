"""Tests for Round 6 fixes: default Hestia base URL (localhost:8080), precedence, and URL construction."""

import httpx
import pytest

from dokime.client import DEFAULT_HESTIA_URL, HestiaClient
from dokime.config import RunConfig


def test_default_base_url_when_hestia_url_cleared(monkeypatch):
    """When HESTIA_URL is cleared from the environment, base_url defaults to http://localhost:8080."""
    monkeypatch.delenv("HESTIA_URL", raising=False)

    client = HestiaClient()
    assert client.base_url == "http://localhost:8080"
    assert DEFAULT_HESTIA_URL == "http://localhost:8080"
    assert client.models_url == "http://localhost:8080/models"
    assert client.chat_completions_url == "http://localhost:8080/v1/chat/completions"


def test_hestia_url_env_var_override(monkeypatch):
    """When HESTIA_URL is set in the environment, it overrides the default base_url."""
    monkeypatch.setenv("HESTIA_URL", "http://hestia-host:8080")

    client = HestiaClient()
    assert client.base_url == "http://hestia-host:8080"
    assert client.models_url == "http://hestia-host:8080/models"
    assert client.chat_completions_url == "http://hestia-host:8080/v1/chat/completions"

    # Trailing slash is stripped
    monkeypatch.setenv("HESTIA_URL", "http://hestia-host:8080/")
    client_slash = HestiaClient()
    assert client_slash.base_url == "http://hestia-host:8080"


def test_explicit_url_wins_over_env_var_and_default(monkeypatch):
    """Explicit base_url wins over both HESTIA_URL and the default."""
    monkeypatch.setenv("HESTIA_URL", "http://env-host:8080")

    client = HestiaClient(base_url="http://explicit-host:9090")
    assert client.base_url == "http://explicit-host:9090"
    assert client.models_url == "http://explicit-host:9090/models"
    assert client.chat_completions_url == "http://explicit-host:9090/v1/chat/completions"

    # Trailing slash stripped
    client_slash = HestiaClient(base_url="http://explicit-host:9090/")
    assert client_slash.base_url == "http://explicit-host:9090"


def test_config_and_cli_url_precedence(monkeypatch):
    """Verify precedence chain: CLI override -> config.url -> HESTIA_URL -> default."""
    monkeypatch.delenv("HESTIA_URL", raising=False)

    # 1. Default fallback
    cfg_default = RunConfig()
    c_default = HestiaClient(base_url=cfg_default.url)
    assert c_default.base_url == "http://localhost:8080"

    # 2. HESTIA_URL env var
    monkeypatch.setenv("HESTIA_URL", "http://env-server:8080")
    c_env = HestiaClient(base_url=cfg_default.url)
    assert c_env.base_url == "http://env-server:8080"

    # 3. config.url wins over HESTIA_URL
    cfg_file = RunConfig(url="http://config-server:8080")
    c_file = HestiaClient(base_url=cfg_file.url)
    assert c_file.base_url == "http://config-server:8080"

    # 4. CLI override wins over config.url
    cfg_cli = cfg_file.merge_cli_args(url="http://cli-override:8080")
    c_cli = HestiaClient(base_url=cfg_cli.url)
    assert c_cli.base_url == "http://cli-override:8080"


def test_url_construction_without_real_network_call(monkeypatch):
    """Verify endpoint URL construction for /models and /v1/chat/completions using MockTransport."""
    monkeypatch.delenv("HESTIA_URL", raising=False)

    requested_urls: list[str] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if request.url.path == "/models":
            return httpx.Response(200, json={"models": [{"name": "test-chat", "type": "chat", "state": "loaded"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
        return httpx.Response(404)

    mock_client = httpx.Client(transport=httpx.MockTransport(mock_handler))
    client = HestiaClient(client=mock_client)

    # Base URL should be default 8080
    assert client.base_url == "http://localhost:8080"

    # 1. /models call
    models = client.list_models()
    assert models == ["test-chat"]
    assert "http://localhost:8080/models" in requested_urls

    # 2. /v1/chat/completions call
    resp, _ = client.chat_completion(model="test-chat", prompt="Hello")
    assert resp == "ok"
    assert "http://localhost:8080/v1/chat/completions" in requested_urls
