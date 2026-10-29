from __future__ import annotations

from pathlib import Path

import pytest

from hyrag.llm.base import (
    LLMError,
    LLMResponseError,
    complete_json,
    parse_json_object,
    retry_call,
    system,
    user,
)
from hyrag.llm.cache import CachedLLM
from hyrag.llm.stub import ScriptedLLM

MESSAGES = [system("Be terse."), user("Say hi.")]


@pytest.mark.parametrize(
    "reply",
    [
        '{"ok": true}',
        '```json\n{"ok": true}\n```',
        'Sure! Here it is: {"ok": true} Hope that helps.',
        '  {"ok": true}\n',
    ],
)
def test_parse_json_object_tolerates_fences_and_prose(reply: str) -> None:
    assert parse_json_object(reply) == {"ok": True}


@pytest.mark.parametrize("reply", ["not json", "[1, 2]", "{broken", ""])
def test_parse_json_object_rejects_non_objects(reply: str) -> None:
    with pytest.raises(LLMResponseError):
        parse_json_object(reply)


def test_complete_json_retries_once_with_the_bad_reply_shown() -> None:
    llm = ScriptedLLM(["I think the answer is yes", '{"answer": "yes"}'])
    assert complete_json(llm, MESSAGES) == {"answer": "yes"}
    assert len(llm.calls) == 2
    assert all(call.json_mode for call in llm.calls)
    retry = llm.calls[1].messages
    assert retry[-2] == {"role": "assistant", "content": "I think the answer is yes"}
    assert "not valid JSON" in retry[-1]["content"]


def test_complete_json_gives_up_after_the_retries() -> None:
    llm = ScriptedLLM(["nope", "still nope"])
    with pytest.raises(LLMResponseError):
        complete_json(llm, MESSAGES, retries=1)


def test_retry_call_retries_only_the_listed_errors() -> None:
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("down")
        return "up"

    assert retry_call(flaky, attempts=3, backoff=0, retry_on=(ConnectionError,)) == "up"
    assert len(attempts) == 3

    def broken() -> str:
        raise ValueError("bug")

    with pytest.raises(ValueError):
        retry_call(broken, attempts=3, backoff=0, retry_on=(ConnectionError,))


def test_retry_call_reraises_after_the_last_attempt() -> None:
    def down() -> str:
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        retry_call(down, attempts=2, backoff=0, retry_on=(ConnectionError,))


def test_scripted_llm_replays_raises_and_falls_back() -> None:
    llm = ScriptedLLM(["first", LLMError("boom")], responder=lambda msgs, js: f"json={js}")
    assert llm.complete(MESSAGES) == "first"
    with pytest.raises(LLMError, match="boom"):
        llm.complete(MESSAGES)
    assert llm.complete(MESSAGES, json_mode=True) == "json=True"
    assert ScriptedLLM(name="x").name == "x"
    with pytest.raises(LLMError, match="no replies"):
        ScriptedLLM().complete(MESSAGES)


def test_cached_llm_replays_deterministic_calls(tmp_path: Path) -> None:
    inner = ScriptedLLM(["one", "two", "three", "four"])
    cache = CachedLLM(inner, tmp_path / "llm.sqlite")
    assert cache.name == inner.name
    assert cache.complete(MESSAGES) == "one"
    assert cache.complete(MESSAGES) == "one"  # served from disk
    assert cache.complete(MESSAGES, json_mode=True) == "two"  # different options, new key
    assert cache.complete(MESSAGES, max_tokens=5) == "three"
    assert (cache.hits, cache.misses) == (1, 3)
    cache.close()

    reopened = CachedLLM(ScriptedLLM(), tmp_path / "llm.sqlite")
    assert reopened.complete(MESSAGES) == "one"  # survives a restart without calling the model
    reopened.close()


def test_cached_llm_never_caches_sampled_calls(tmp_path: Path) -> None:
    cache = CachedLLM(ScriptedLLM(["a", "b"]), tmp_path / "llm.sqlite")
    assert cache.complete(MESSAGES, temperature=0.7) == "a"
    assert cache.complete(MESSAGES, temperature=0.7) == "b"
    assert (cache.hits, cache.misses) == (0, 2)
    cache.close()
