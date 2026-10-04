"""Hestia local inference server adapter for model discovery and streaming generation."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx


class HestiaAdapter:
    """
    Adapter communicating with local Hestia model server (OpenAI-compatible /v1 API).
    Includes model discovery, streaming token completion, and usage accounting.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        default_model: str = "llama-3.1-8b-instruct",
        timeout_seconds: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout_seconds

    async def check_health(self) -> Dict[str, Any]:
        """Verify Hestia reachability and return model list."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("id") for m in data.get("data", []) if "id" in m]
                    default_found = self.default_model in models if models else False
                    return {
                        "available": True,
                        "models": models,
                        "default_model": self.default_model,
                        "default_model_registered": default_found,
                        "url": self.base_url,
                    }
        except Exception as e:
            return {"available": False, "models": [], "error": str(e), "url": self.base_url}
        return {"available": False, "models": [], "url": self.base_url}

    async def list_models(self) -> List[str]:
        health = await self.check_health()
        return health.get("models", [])

    async def stream_chat(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful and truthful local AI assistant.",
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat completions from Hestia.
        Yields dicts with 'delta' (str) and final 'usage' (input_tokens, output_tokens).
        """
        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta_content = ""
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                delta_content = delta.get("content", "")

                            usage = chunk.get("usage")
                            yield {
                                "delta": delta_content,
                                "usage": usage,
                            }
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            # Yield error or mock fallback if local server is unreachable
            yield {
                "error": f"Hestia server error: {e}",
                "delta": f"[Local Hestia offline: {e}]",
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            }
