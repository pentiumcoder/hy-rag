from __future__ import annotations

from collections.abc import Iterable, Iterator

import pytest

from hyrag.retrieval import CrossEncoderReranker, Reranker, get_reranker


class _StubCrossEncoder:
    """Scores by how many query words the passage contains, like a (very) cheap cross-encoder."""

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> Iterator[float]:
        words = set(query.lower().split())
        for doc in documents:
            yield float(len(words & set(doc.lower().split())))


def test_cross_encoder_reranker_scores_each_text() -> None:
    reranker = CrossEncoderReranker("some/model", model=_StubCrossEncoder())
    assert isinstance(reranker, Reranker)
    assert reranker.name == "cross-encoder:some/model"
    assert reranker.score("rotate certs", ["rotate the certs", "pto policy", "rotate"]) == [
        2.0,
        0.0,
        1.0,
    ]
    assert reranker.score("anything", []) == []


def test_get_reranker() -> None:
    assert get_reranker("none") is None
    with pytest.raises(ValueError, match="Unknown reranker"):
        get_reranker("llm")
