"""The golden question set: questions, reference answers and the evidence that answers them.

Relevance is defined by evidence, not by chunk ids: each evidence item is a short verbatim
quote plus the documents that contain it, and a retrieved chunk is relevant when it comes
from one of those documents and contains the quote. That keeps one gold set valid across all
three chunking strategies, whose chunk boundaries and ids differ.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from hyrag.models import Chunk, Document

QuestionType = Literal["lookup", "multi_hop", "exact_term", "unanswerable", "ambiguous"]
QUESTION_TYPES: tuple[QuestionType, ...] = (
    "lookup",
    "multi_hop",
    "exact_term",
    "unanswerable",
    "ambiguous",
)
DEFAULT_GOLDEN = Path("eval/golden.jsonl")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class Evidence(BaseModel):
    """A fact the answer needs: a verbatim quote and every document it may come from."""

    sources: list[str] = Field(min_length=1)
    quote: str = Field(min_length=2)

    def matches(self, chunk: Chunk) -> bool:
        return chunk.source in self.sources and normalize(self.quote) in normalize(chunk.text)


class GoldenQuestion(BaseModel):
    id: str
    type: QuestionType
    question: str
    answerable: bool
    reference: str
    # Strings a correct answer must contain, case-insensitive; "a|b" accepts either.
    must_include: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> GoldenQuestion:
        if self.answerable and not self.evidence:
            raise ValueError(f"{self.id}: an answerable question needs evidence")
        if not self.answerable and (self.evidence or self.type != "unanswerable"):
            raise ValueError(f"{self.id}: only unanswerable questions may lack evidence")
        return self


def load_golden(path: Path = DEFAULT_GOLDEN) -> list[GoldenQuestion]:
    questions = [
        GoldenQuestion.model_validate(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids in the golden set")
    return questions


def check_against_corpus(
    questions: Sequence[GoldenQuestion], docs: Iterable[Document]
) -> list[str]:
    """Problems that would make a question unscoreable: unknown sources, quotes not found."""
    texts = {d.source: normalize(d.text) for d in docs}
    problems: list[str] = []
    for q in questions:
        for e in q.evidence:
            unknown = [s for s in e.sources if s not in texts]
            if unknown:
                problems.append(f"{q.id}: unknown source(s) {', '.join(unknown)}")
            elif not any(normalize(e.quote) in texts[s] for s in e.sources):
                problems.append(f"{q.id}: quote not found: {e.quote!r}")
    return problems
