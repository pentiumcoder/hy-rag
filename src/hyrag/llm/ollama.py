"""Local chat models served by Ollama (https://ollama.com), the default provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from hyrag.llm.base import LLMError, LLMResponseError, LLMUnavailableError, Message, retry_call


class _RetryableError(Exception):
    """A transport failure or 5xx that is worth another attempt."""


class OllamaClient:
    """Calls ``POST /api/chat`` without streaming. Transient failures are retried with backoff.

    ``num_ctx`` is set explicitly because Ollama's default context window is small enough to
    silently truncate five retrieved passages plus the instructions.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        timeout: float = 180.0,
        num_ctx: int = 8192,
        seed: int = 0,
        attempts: int = 3,
        backoff: float = 1.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.seed = seed
        self.attempts = attempts
        self.backoff = backoff
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": self.num_ctx,
            "seed": self.seed,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"

        try:
            data = retry_call(
                lambda: self._post(payload),
                attempts=self.attempts,
                backoff=self.backoff,
                retry_on=(_RetryableError,),
            )
        except _RetryableError as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.base_url} ({exc.__cause__ or exc}). "
                "Start it with `ollama serve`."
            ) from exc
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise LLMResponseError(f"Ollama returned no message content: {str(data)[:200]}")
        return content

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post("/api/chat", json=payload)
        except httpx.TransportError as exc:
            raise _RetryableError(str(exc)) from exc
        if response.status_code == 404:
            raise LLMUnavailableError(
                f"Ollama has no model {self.model!r}. Pull it with `ollama pull {self.model}`."
            )
        if response.status_code >= 500:
            raise _RetryableError(f"HTTP {response.status_code}: {response.text[:200]}")
        if response.status_code >= 400:
            raise LLMError(f"Ollama rejected the request: HTTP {response.status_code}")
        body: dict[str, Any] = response.json()
        return body

    def close(self) -> None:
        self._http.close()
