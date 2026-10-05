"""Hestia local inference server adapter for model discovery and streaming generation."""

import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx


class HestiaAdapter:
    """
    Adapter communicating with local Hestia model server (OpenAI-compatible /v1 API).
    Includes model discovery, streaming token completion, and usage accounting.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: str = "llama-3.1-8b-instruct",
        timeout_seconds: float = 30.0,
        service_token: Optional[str] = None,
    ):
        resolved_url = base_url or os.environ.get("HESTIA_BASE_URL", "http://127.0.0.1:8080/v1")
        self.base_url = resolved_url.rstrip("/")
        self.default_model = os.environ.get("HESTIA_DEFAULT_MODEL", default_model)
        self.timeout = timeout_seconds
        self.service_token = service_token if service_token is not None else os.environ.get("HESTIA_SERVICE_TOKEN")

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.service_token:
            headers["Authorization"] = f"Bearer {self.service_token}"
        return headers

    async def check_health(self) -> Dict[str, Any]:
        """Verify Hestia reachability and return model list."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self.base_url}/models", headers=self._get_headers())
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
        target_model = model or os.environ.get("HESTIA_DEFAULT_MODEL") or self.default_model
        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repeat_penalty": 1.15,
            "presence_penalty": 0.1,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._get_headers(),
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
        except httpx.HTTPStatusError as e:
            detail = str(e)
            try:
                raw_bytes = await e.response.aread()
                detail_text = raw_bytes.decode(errors="replace")
                err_json = json.loads(detail_text)
                if isinstance(err_json, dict) and "detail" in err_json:
                    detail = err_json["detail"]
                elif detail_text:
                    detail = detail_text
            except Exception:
                pass
            yield {
                "error": f"Hestia error ({e.response.status_code}): {detail}",
                "delta": f"[Local Hestia: {detail}]",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        except Exception as e:
            # Yield error or fallback if local server is unreachable
            yield {
                "error": f"Hestia server error: {e}",
                "delta": f"[Local Hestia offline: {e}]",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
