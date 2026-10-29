"""Chat model adapters behind a single :class:`LLMClient` protocol."""

from __future__ import annotations

from hyrag.llm.base import (
    LLMClient,
    LLMError,
    LLMResponseError,
    LLMUnavailableError,
    Message,
    assistant,
    complete_json,
    parse_json_object,
    system,
    user,
)
from hyrag.llm.cache import CachedLLM
from hyrag.llm.stub import ScriptedLLM

__all__ = [
    "DEFAULT_MODELS",
    "CachedLLM",
    "LLMClient",
    "LLMError",
    "LLMResponseError",
    "LLMUnavailableError",
    "Message",
    "ScriptedLLM",
    "assistant",
    "complete_json",
    "default_model",
    "get_llm",
    "parse_json_object",
    "system",
    "user",
]

# (generator, judge) per provider. On Ollama the judge is a different model family than the
# generator, so it does not share the generator's blind spots or prefer its phrasing.
DEFAULT_MODELS: dict[str, tuple[str, str]] = {
    "ollama": ("qwen2.5:7b", "llama3.1:8b"),
    "openai": ("gpt-4o-mini", "gpt-4o-mini"),
    "anthropic": ("claude-sonnet-5-5", "claude-haiku-5-5"),
}


def default_model(provider: str, role: str) -> str:
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"No default model for provider {provider!r}")
    generator, judge = DEFAULT_MODELS[provider]
    return judge if role == "judge" else generator


def get_llm(
    provider: str,
    model: str | None = None,
    *,
    role: str = "generator",
    base_url: str = "http://localhost:11434",
    timeout: float = 180.0,
    num_ctx: int = 8192,
) -> LLMClient:
    """Build an uncached client for a hosted or local provider. SDKs are imported lazily."""
    model = model or default_model(provider, role)
    if provider == "ollama":
        from hyrag.llm.ollama import OllamaClient

        return OllamaClient(model, base_url, timeout=timeout, num_ctx=num_ctx)
    if provider == "openai":
        from hyrag.llm.openai import OpenAIChatClient

        return OpenAIChatClient(model)
    if provider == "anthropic":
        from hyrag.llm.anthropic import AnthropicClient

        return AnthropicClient(model)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
