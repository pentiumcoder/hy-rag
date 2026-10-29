from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from hyrag.llm import (
    DEFAULT_MODELS,
    LLMError,
    LLMResponseError,
    LLMUnavailableError,
    default_model,
    get_llm,
    system,
    user,
)
from hyrag.llm.anthropic import AnthropicClient
from hyrag.llm.ollama import OllamaClient
from hyrag.llm.openai import OpenAIChatClient

MESSAGES = [system("Be terse."), user("Say hi.")]


def _ollama(handler: Any, **kwargs: Any) -> OllamaClient:
    return OllamaClient("qwen2.5:7b", transport=httpx.MockTransport(handler), backoff=0, **kwargs)


def test_ollama_sends_a_non_streaming_chat_request() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}})

    client = _ollama(handler, num_ctx=4096)
    assert client.name == "ollama:qwen2.5:7b"
    assert client.complete(MESSAGES, max_tokens=50) == "hi"
    assert client.complete(MESSAGES, json_mode=True, temperature=0.3) == "hi"
    client.close()

    plain, as_json = seen
    assert plain["stream"] is False and plain["messages"] == MESSAGES
    assert plain["options"] == {"temperature": 0.0, "num_ctx": 4096, "seed": 0, "num_predict": 50}
    assert "format" not in plain
    assert as_json["format"] == "json" and as_json["options"]["temperature"] == 0.3
    assert "num_predict" not in as_json["options"]


def test_ollama_retries_server_errors_then_succeeds() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, text="loading model")
        return httpx.Response(200, json={"message": {"content": "ok"}})

    assert _ollama(handler, attempts=3).complete(MESSAGES) == "ok"
    assert len(calls) == 3


def test_ollama_reports_an_unreachable_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        _ollama(handler, attempts=2).complete(MESSAGES)


def test_ollama_reports_a_missing_model_without_retrying() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(LLMUnavailableError, match=r"ollama pull qwen2\.5:7b"):
        _ollama(handler).complete(MESSAGES)
    assert len(calls) == 1


def test_ollama_rejects_bad_requests_and_empty_replies() -> None:
    with pytest.raises(LLMError, match="HTTP 400"):
        _ollama(lambda r: httpx.Response(400, json={})).complete(MESSAGES)
    with pytest.raises(LLMResponseError):
        _ollama(lambda r: httpx.Response(200, json={"done": True})).complete(MESSAGES)


class _FakeOpenAI:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        message = SimpleNamespace(content='{"ok": true}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_openai_adapter_maps_json_mode_and_limits() -> None:
    fake = _FakeOpenAI()
    client = OpenAIChatClient("gpt-4o-mini", client=fake)
    assert client.name == "openai:gpt-4o-mini"
    assert client.complete(MESSAGES, json_mode=True, max_tokens=20) == '{"ok": true}'
    assert fake.kwargs["response_format"] == {"type": "json_object"}
    assert fake.kwargs["max_tokens"] == 20 and fake.kwargs["messages"] == MESSAGES


def test_openai_adapter_wraps_sdk_errors() -> None:
    def fail(**kwargs: Any) -> Any:
        raise RuntimeError("rate limited")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(LLMError, match="rate limited"):
        OpenAIChatClient(client=fake).complete(MESSAGES)


class _FakeAnthropic:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        blocks = [
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text="Hello "),
            SimpleNamespace(type="text", text="there"),
        ]
        return SimpleNamespace(content=blocks)


def test_anthropic_adapter_lifts_the_system_prompt() -> None:
    fake = _FakeAnthropic()
    client = AnthropicClient("claude-haiku-5-5", client=fake)
    assert client.name == "anthropic:claude-haiku-5-5"
    assert client.complete(MESSAGES) == "Hello there"
    assert fake.kwargs["system"] == "Be terse."
    assert fake.kwargs["messages"] == [{"role": "user", "content": "Say hi."}]
    assert fake.kwargs["max_tokens"] == 1024


def test_anthropic_adapter_wraps_sdk_errors() -> None:
    def fail(**kwargs: Any) -> Any:
        raise RuntimeError("overloaded")

    with pytest.raises(LLMError, match="overloaded"):
        AnthropicClient(client=SimpleNamespace(messages=SimpleNamespace(create=fail))).complete(
            [user("hi")]
        )


def test_default_models_put_the_judge_in_another_family_on_ollama() -> None:
    generator, judge = DEFAULT_MODELS["ollama"]
    assert generator.split(":")[0] != judge.split(":")[0]
    assert default_model("ollama", "judge") == judge
    assert default_model("ollama", "generator") == generator
    with pytest.raises(ValueError):
        default_model("nope", "judge")


def test_get_llm_builds_ollama_lazily_and_rejects_unknown_providers() -> None:
    client = get_llm("ollama", role="judge", base_url="http://gpu-box:11434/")
    assert isinstance(client, OllamaClient)
    assert client.name == "ollama:llama3.1:8b" and client.base_url == "http://gpu-box:11434"
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm("mystery", "m")


def test_get_llm_imports_hosted_sdks_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=_FakeAnthropic))
    assert get_llm("openai").name == "openai:gpt-4o-mini"
    assert get_llm("anthropic", role="judge").name == "anthropic:claude-haiku-5-5"
    assert get_llm("anthropic", "claude-opus-5-5").name == "anthropic:claude-opus-5-5"
