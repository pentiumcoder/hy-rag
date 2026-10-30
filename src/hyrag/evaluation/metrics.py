"""Deterministic metrics: retrieval ranks, key-fact recall, abstention scores and AUROC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hyrag.evaluation.dataset import GoldenQuestion, normalize
from hyrag.models import Chunk


def evidence_ranks(question: GoldenQuestion, chunks: Sequence[Chunk]) -> list[int | None]:
    """For each evidence item, the 1-based rank of the first chunk containing it (or None)."""
    ranks: list[int | None] = []
    for evidence in question.evidence:
        rank = next((i for i, c in enumerate(chunks, start=1) if evidence.matches(c)), None)
        ranks.append(rank)
    return ranks


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Share of the question's evidence items found in the top ``k``."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """1 / rank of the first relevant chunk (any evidence item), 0 if none was retrieved."""
    found = [r for r in ranks if r is not None]
    return 1.0 / min(found) if found else 0.0


def key_fact_recall(answer: str | None, must_include: Sequence[str]) -> float:
    """Share of the required strings in the answer. ``"a|b"`` accepts either alternative."""
    if not must_include:
        return 1.0 if answer else 0.0
    if not answer:
        return 0.0
    text = normalize(answer)
    hits = sum(1 for item in must_include if any(normalize(a) in text for a in item.split("|")))
    return hits / len(must_include)


@dataclass(frozen=True)
class AbstentionScores:
    """Abstaining is the positive class; a question should be abstained on if unanswerable."""

    precision: float
    recall: float
    f1: float
    abstained: int
    unanswerable: int


def abstention_scores(outcomes: Sequence[tuple[bool, bool]]) -> AbstentionScores:
    """``outcomes`` holds (should_abstain, abstained) pairs."""
    tp = sum(1 for should, did in outcomes if should and did)
    abstained = sum(1 for _, did in outcomes if did)
    unanswerable = sum(1 for should, _ in outcomes if should)
    precision = tp / abstained if abstained else 1.0
    recall = tp / unanswerable if unanswerable else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return AbstentionScores(precision, recall, f1, abstained, unanswerable)


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
