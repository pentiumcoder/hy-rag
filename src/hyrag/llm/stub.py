"""A scripted LLM for tests: replays canned replies and records every call."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from hyrag.llm.base import LLMError, Message

Responder = Callable[[Sequence[Message], bool], str]


@dataclass
class LLMCall:
    messages: list[Message]
    json_mode: bool
    temperature: float
    max_tokens: int | None


@dataclass
class ScriptedLLM:
    """Returns ``replies`` in order, or asks ``responder(messages, json_mode)`` once they run out.

    A reply that is an exception instance is raised instead of returned, which is how tests
    simulate a model that times out or is not installed.
    """

    replies: list[str | Exception] = field(default_factory=list)
    responder: Responder | None = None
    name: str = "stub:scripted"
    calls: list[LLMCall] = field(default_factory=list)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(LLMCall(list(messages), json_mode, temperature, max_tokens))
        if self.replies:
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        if self.responder is not None:
            return self.responder(messages, json_mode)
        raise LLMError("ScriptedLLM has no replies left")
