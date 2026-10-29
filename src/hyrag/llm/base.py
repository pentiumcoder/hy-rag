"""The :class:`LLMClient` protocol every chat model adapter implements, plus shared helpers."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, TypedDict, TypeVar, runtime_checkable

T = TypeVar("T")


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(RuntimeError):
    """Base class for every model failure the pipeline knows how to report."""


class LLMUnavailableError(LLMError):
    """The model cannot be reached at all (server down, model not pulled, no API key)."""


class LLMResponseError(LLMError):
    """The model answered, but not in the shape that was asked for."""


@runtime_checkable
class LLMClient(Protocol):
    name: str  # "<provider>:<model>", recorded in responses and cache keys

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant's reply. ``json_mode`` asks for a single JSON object."""
        ...


def system(content: str) -> Message:
    return {"role": "system", "content": content}


def user(content: str) -> Message:
    return {"role": "user", "content": content}


def assistant(content: str) -> Message:
    return {"role": "assistant", "content": content}


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model reply, tolerating code fences and stray prose."""
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMResponseError(f"expected a JSON object, got: {text[:200]!r}")


def complete_json(
    llm: LLMClient,
    messages: Sequence[Message],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    retries: int = 1,
) -> dict[str, Any]:
    """Ask for JSON; on a malformed reply, show the model its output and ask again."""
    conversation = list(messages)
    for attempt in range(retries + 1):
        reply = llm.complete(
            conversation, json_mode=True, temperature=temperature, max_tokens=max_tokens
        )
        try:
            return parse_json_object(reply)
        except LLMResponseError:
            if attempt == retries:
                raise
            conversation += [
                assistant(reply),
                user("That was not valid JSON. Reply with only the JSON object."),
            ]
    raise AssertionError("unreachable")  # pragma: no cover


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    backoff: float = 0.5,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call ``fn`` until it succeeds, sleeping ``backoff * 2**i`` between attempts."""
    for attempt in range(attempts):
        try:
            return fn()
        except retry_on:
            if attempt == attempts - 1:
                raise
            time.sleep(backoff * 2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover
