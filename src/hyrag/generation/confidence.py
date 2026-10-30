"""Confidence: ``0.4 * retrieval + 0.4 * citation coverage + 0.2 * completeness``.

* **retrieval** - how well the best passage matches the question. With a reranker this is
  ``sigmoid((top logit - center) / scale)``; the MiniLM cross-encoder's logits run from about
  -11 (unrelated) to +8 (direct answer), and the default center of -4 puts borderline
  passages near 0.5. Without rerank scores it falls back to the share of the question's key
  terms that appear in the passages.
* **citation coverage** - the mean verdict over the answer's factual sentences
  (supported 1, partial or unverified 0.5, unsupported or uncited 0).
* **completeness** - the judge's rating of whether the answer covers the whole question.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from hyrag.index.tokenize import key_terms, tokenize
from hyrag.models import Claim, Confidence, ScoredChunk

WEIGHTS = {"retrieval": 0.4, "citation_coverage": 0.4, "completeness": 0.2}
CLAIM_SCORES = {
    "SUPPORTED": 1.0,
    "PARTIAL": 0.5,
    "UNVERIFIED": 0.5,
    "UNSUPPORTED": 0.0,
    "UNCITED": 0.0,
}
RetrievalSignal = Literal["rerank", "lexical", "none"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def lexical_coverage(query: str, contexts: Sequence[ScoredChunk]) -> tuple[list[str], list[str]]:
    """Split the question's key terms into those the passages contain and those they lack."""
    present = set(tokenize(" ".join(sc.chunk.text for sc in contexts)))
    terms = key_terms(query)
    return [t for t in terms if t in present], [t for t in terms if t not in present]


def retrieval_confidence(
    query: str,
    contexts: Sequence[ScoredChunk],
    *,
    center: float = -4.0,
    scale: float = 2.0,
) -> tuple[float, RetrievalSignal]:
    if not contexts:
        return 0.0, "none"
    reranked = [sc.rerank_score for sc in contexts if sc.rerank_score is not None]
    if reranked:
        return _sigmoid((max(reranked) - center) / scale), "rerank"
    found, missing = lexical_coverage(query, contexts)
    total = len(found) + len(missing)
    return (len(found) / total if total else 0.0), "lexical"


def citation_coverage(claims: Sequence[Claim]) -> float:
    facts = [c for c in claims if c.kind == "fact"]
    if not facts:
        return 0.0
    return sum(CLAIM_SCORES[c.label or "UNVERIFIED"] for c in facts) / len(facts)


def combine(
    retrieval: float,
    coverage: float,
    completeness: float,
    signal: RetrievalSignal,
    answerability: float | None = None,
) -> Confidence:
    score = (
        WEIGHTS["retrieval"] * retrieval
        + WEIGHTS["citation_coverage"] * coverage
        + WEIGHTS["completeness"] * completeness
    )
    return Confidence(
        score=round(score, 4),
        retrieval=round(retrieval, 4),
        citation_coverage=round(coverage, 4),
        completeness=round(completeness, 4),
        retrieval_signal=signal,
        answerability=answerability,
    )
