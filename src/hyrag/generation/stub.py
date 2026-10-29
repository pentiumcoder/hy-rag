"""A deterministic, model-free stand-in for the generator and the judge.

Tests and CI cannot run a 7B model, but they should still exercise the real pipeline end to
end. This stub reads the same prompts the real models get:

* as a **generator** it copies the passage sentences that best overlap the question and
  cites them, or replies ``INSUFFICIENT_CONTEXT`` when nothing overlaps enough;
* as a **judge** it labels each check by how much of the claim's wording, and every number
  in it, appears in the cited passage.

It is a lexical baseline, not a language model: it cannot paraphrase or reason, which is
exactly why it makes a good floor for the Day 4 offline eval.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from hyrag.generation.claims import split_sentences
from hyrag.generation.prompts import INSUFFICIENT, JUDGE_SYSTEM
from hyrag.index.tokenize import key_terms, tokenize
from hyrag.llm import Message

_PASSAGE = re.compile(r'<passage id="(\d+)"[^\n]*\n(.*?)\n</passage>', re.DOTALL)
_CHECK = re.compile(r"^(\d+)\. Passage \[(\d+)\] \| Claim: (\".*\")$", re.MULTILINE)


def _passages(block: str) -> dict[int, str]:
    return {int(m.group(1)): m.group(2) for m in _PASSAGE.finditer(block)}


def _cite(sentence: str, number: int) -> str:
    sentence = sentence.rstrip()
    if sentence and sentence[-1] in ".!?":
        return f"{sentence[:-1]} [{number}]{sentence[-1]}"
    return f"{sentence} [{number}]."


class ExtractiveStubLLM:
    def __init__(
        self,
        min_overlap: float = 0.3,
        max_sentences: int = 2,
        supported: float = 0.75,
        partial: float = 0.5,
    ) -> None:
        self.name = "stub:extractive"
        self.min_overlap = min_overlap
        self.max_sentences = max_sentences
        self.supported = supported
        self.partial = partial

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        prompt = messages[-1]["content"] if messages else ""
        if messages and messages[0]["content"] == JUDGE_SYSTEM:
            return self._judge(prompt)
        if "<passage " in prompt:
            return self._answer(prompt)
        return "{}" if json_mode else ""

    def _answer(self, prompt: str) -> str:
        body, _, question = prompt.rpartition("\nQuestion:")
        terms = set(key_terms(question))
        scored: list[tuple[float, int, int, str]] = []
        for number, text in _passages(body).items():
            for position, sentence in enumerate(split_sentences(text)):
                if sentence.endswith("?") or not terms:
                    continue  # headings phrased as questions restate, they do not answer
                overlap = len(terms & set(tokenize(sentence))) / len(terms)
                scored.append((-overlap, number, position, sentence))
        scored.sort()
        if not scored or -scored[0][0] < self.min_overlap:
            missing = ", ".join(sorted(terms)) or "the question"
            return (
                f"{INSUFFICIENT}\nFound: The passages do not mention the key terms together.\n"
                f"Missing: Information about {missing}."
            )
        floor = max(self.min_overlap, -scored[0][0] * 0.6)
        picked = [(n, s) for neg, n, _, s in scored if -neg >= floor][: self.max_sentences]
        return " ".join(_cite(sentence, number) for number, sentence in picked)

    def _judge(self, prompt: str) -> str:
        head, _, checks_block = prompt.partition("\nChecks:\n")
        question = head.split("\n", 1)[0].removeprefix("Question:")
        answer, _, passage_block = head.partition("\nAnswer:\n")[2].partition("\n\nPassages:")
        passages = {n: set(tokenize(t)) for n, t in _passages(passage_block).items()}
        checks = []
        for match in _CHECK.finditer(checks_block):
            check_id, number = int(match.group(1)), int(match.group(2))
            claim_terms = set(tokenize(json.loads(match.group(3))))
            passage_terms = passages.get(number, set())
            numbers = {t for t in claim_terms if any(ch.isdigit() for ch in t)}
            overlap = len(claim_terms & passage_terms) / len(claim_terms) if claim_terms else 0.0
            if numbers - passage_terms:
                label, reason = "UNSUPPORTED", "a number in the claim is not in the passage"
            elif overlap >= self.supported:
                label, reason = "SUPPORTED", f"{overlap:.0%} of the claim's terms appear"
            elif overlap >= self.partial:
                label, reason = "PARTIAL", f"only {overlap:.0%} of the claim's terms appear"
            else:
                label, reason = "UNSUPPORTED", f"only {overlap:.0%} of the claim's terms appear"
            checks.append({"id": check_id, "label": label, "reason": reason})
        q_terms = set(key_terms(question))
        answered = len(q_terms & set(tokenize(answer))) / len(q_terms) if q_terms else 1.0
        completeness = (
            "COMPLETE" if answered >= 0.6 else "PARTIAL" if answered >= 0.3 else "INCOMPLETE"
        )
        return json.dumps({"checks": checks, "completeness": completeness})
