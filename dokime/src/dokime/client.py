"""Hestia / OpenAI-compatible HTTP client with retry logic."""

from __future__ import annotations

import os
import time
from typing import Any
import httpx


class HestiaAPIError(Exception):
    """Raised when an API call fails after all retries."""
    pass


DEFAULT_HESTIA_URL = "http://localhost:8080"


class HestiaClient:
    """Client for querying models served via Hestia's OpenAI-compatible HTTP API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        client: httpx.Client | None = None,
    ) -> None:
        if base_url is None:
            base_url = os.environ.get("HESTIA_URL") or DEFAULT_HESTIA_URL
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client = client

    @property
    def models_url(self) -> str:
        """Construct full URL for Hestia native models endpoint."""
        return f"{self.base_url}/models"

    @property
    def chat_completions_url(self) -> str:
        """Construct full URL for chat completions endpoint."""
        return f"{self.base_url}/v1/chat/completions"

    def _get_client(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(timeout=self.timeout)

    def get_models_metadata(self) -> list[dict[str, Any]]:
        """Fetch all registered models with metadata (name, type, state) from Hestia.

        Prioritizes Hestia's native /models endpoint which exposes 'type' (chat vs embedding)
        and 'state' (loaded vs unloaded), falling back to /v1/models.
        """
        client = self._get_client()

        # 1. Try Hestia native /models
        try:
            resp = client.get(self.models_url)
            if resp.status_code == 200:
                data = resp.json()
                models_list = data.get("models", [])
                result = []
                for m in models_list:
                    if isinstance(m, dict):
                        result.append({
                            "name": m.get("name") or m.get("id") or "",
                            "type": m.get("type", "chat"),
                            "state": m.get("state", "unknown"),
                        })
                    else:
                        result.append({"name": str(m), "type": "chat", "state": "unknown"})
                if result:
                    return result
        except Exception:
            pass

        # 2. Try OpenAI standard /v1/models
        try:
            resp = client.get(f"{self.base_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data") or data.get("models") or []
                result = []
                for item in items:
                    if isinstance(item, dict):
                        result.append({
                            "name": item.get("id") or item.get("name") or "",
                            "type": item.get("type", "chat"),
                            "state": item.get("state", "unknown"),
                        })
                    else:
                        result.append({"name": str(item), "type": "chat", "state": "unknown"})
                return result
        except Exception as e:
            raise HestiaAPIError(f"Could not connect to Hestia serving layer at {self.base_url}: {e}") from e

        return []

    def list_models(self) -> list[str]:
        """Fetch list of all available model names from Hestia."""
        meta = self.get_models_metadata()
        return [m["name"] for m in meta if m.get("name")]

    def list_chat_models(self, exclude_failed: bool = True, prefer_loaded: bool = True) -> list[str]:
        """Fetch model identifiers explicitly capable of chat completions.

        Filters out embedding models or non-chat models.
        By default, excludes models in 'failed' state and prioritizes 'loaded' models
        over 'unloaded' models.
        """
        meta = self.get_models_metadata()
        chat_models = [
            m
            for m in meta
            if m.get("type") in ("chat", None) and m.get("name")
        ]
        if exclude_failed:
            chat_models = [m for m in chat_models if m.get("state") != "failed"]

        if prefer_loaded:
            loaded = [m for m in chat_models if m.get("state") == "loaded"]
            unloaded = [m for m in chat_models if m.get("state") != "loaded"]
            chat_models = loaded + unloaded

        return [m["name"] for m in chat_models]

    def get_auto_selectable_chat_models(self) -> list[str]:
        """Fetch chat models suitable for auto-selection.

        Excludes models with state == 'failed' (which Hestia cannot load).
        Orders models so that loaded models appear before unloaded models,
        while preserving relative order within each group.
        Unloaded models can load on demand if no loaded model exists.
        """
        return self.list_chat_models(exclude_failed=True, prefer_loaded=True)

    def chat_completion(
        self,
        model: str,
        prompt: str,
        seed: int = 42,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, float]:
        """Send chat completion request to /v1/chat/completions with retry and backoff.

        Returns:
            Tuple of (response_text, latency_ms)
        """
        client = self._get_client()
        url = self.chat_completions_url

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            t_start = time.perf_counter()
            try:
                resp = client.post(url, json=payload)
                latency_ms = (time.perf_counter() - t_start) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise HestiaAPIError("API returned empty choices list")
                    choice = choices[0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    return content, latency_ms

                # Non-200 status code
                err_text = resp.text
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"Server error {resp.status_code}: {err_text}",
                        request=resp.request,
                        response=resp,
                    )
                # Client errors (e.g. 400, 404) should not retry indefinitely
                raise HestiaAPIError(f"Hestia returned error {resp.status_code}: {err_text}")

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < self.max_retries:
                    sleep_time = self.backoff_base * (2**attempt)
                    time.sleep(sleep_time)
                else:
                    break
            except Exception as e:
                last_error = e
                break

        raise HestiaAPIError(
            f"Chat completion failed after {self.max_retries} retries for model '{model}': {last_error}"
        ) from last_error
