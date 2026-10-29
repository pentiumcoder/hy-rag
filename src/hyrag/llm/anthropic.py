"""Optional hosted Claude models via the Anthropic API (``pip install hyrag[anthropic]``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hyrag.llm.base import LLMError, Message

DEFAULT_MAX_TOKENS = 1024


class AnthropicClient:
    """Reads ``ANTHROPIC_API_KEY`` from the environment unless a client is passed in.

    The Messages API takes the system prompt as its own parameter and has no JSON mode, so
    system messages are lifted out and JSON replies rely on the prompt plus tolerant parsing.
    """

    def __init__(self, model: str = "claude-haiku-5-5", client: Any = None) -> None:
        if client is None:
            from anthropic import Anthropic

            client = Anthropic()
        self._client = client
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [dict(m) for m in messages if m["role"] != "system"]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:  # the SDK's own error hierarchy is optional
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
