"""Optional hosted chat models via the OpenAI API (``pip install hyrag[openai]``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hyrag.llm.base import LLMError, Message


class OpenAIChatClient:
    """Reads ``OPENAI_API_KEY`` from the environment unless a client is passed in."""

    def __init__(self, model: str = "gpt-4o-mini", client: Any = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self.model = model
        self.name = f"openai:{model}"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # the SDK's own error hierarchy is optional
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return str(response.choices[0].message.content or "")
