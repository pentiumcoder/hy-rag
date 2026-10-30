"""Answer correctness: does the answer say what the reference answer says?

Citation checks (Phase 3) ask whether an answer is *faithful* to its passages. That says
nothing about whether it is *right*: a faithful answer built on the wrong passage is still
wrong. Correctness compares the answer with the golden reference instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hyrag.evaluation.dataset import GoldenQuestion
from hyrag.evaluation.metrics import key_fact_recall
from hyrag.llm import LLMClient, LLMError, complete_json, system, user

Verdict = Literal["CORRECT", "PARTIAL", "INCORRECT"]
VERDICT_SCORES: dict[str, float] = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}

CORRECTNESS_SYSTEM = """\
You grade answers from a documentation assistant against a reference answer written by the \
documentation owners.

- CORRECT: the answer states the reference's key facts (numbers, names, conditions) and \
nothing that contradicts it. Extra correct detail and different wording are fine.
- PARTIAL: the answer gets some key facts right but misses or garbles others.
- INCORRECT: the answer contradicts the reference, gets the key facts wrong, or does not \
answer the question.

If the question is ambiguous and the reference lists several readings, an answer that \
covers one reading correctly is PARTIAL, and one that covers all of them (or correctly asks \
which one is meant) is CORRECT.

Reply with only a JSON object: {"verdict": "CORRECT", "reason": "<under 20 words>"}"""


@dataclass(frozen=True)
class Grade:
    verdict: Verdict
    reason: str
    source: Literal["judge", "key_facts"]

    @property
    def score(self) -> float:
        return VERDICT_SCORES[self.verdict]


def grade_by_key_facts(question: GoldenQuestion, answer: str) -> Grade:
    """The deterministic fallback, used offline and when the judge fails."""
    recall = key_fact_recall(answer, question.must_include)
    verdict: Verdict = "CORRECT" if recall == 1.0 else "PARTIAL" if recall > 0 else "INCORRECT"
    return Grade(verdict, f"{recall:.0%} of the key facts appear", "key_facts")


class CorrectnessJudge:
    """Grades with an LLM when one is given, otherwise by key facts."""

    def __init__(self, llm: LLMClient | None, *, max_tokens: int = 200) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def grade(self, question: GoldenQuestion, answer: str) -> Grade:
        if self.llm is None:
            return grade_by_key_facts(question, answer)
        messages = [
            system(CORRECTNESS_SYSTEM),
            user(
                f"Question: {question.question}\n\nReference answer: {question.reference}\n\n"
                f"Answer to grade:\n{answer}"
            ),
        ]
        try:
            data = complete_json(self.llm, messages, max_tokens=self.max_tokens)
        except LLMError:
            return grade_by_key_facts(question, answer)
        verdict = str(data.get("verdict") or "").strip().upper()
        if verdict not in VERDICT_SCORES:
            return grade_by_key_facts(question, answer)
        return Grade(verdict, str(data.get("reason") or "").strip(), "judge")  # type: ignore[arg-type]
