from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pytest

from hyrag.config import Settings
from hyrag.embeddings import FakeEmbedder
from hyrag.index import IndexManager, IndexMismatchError
from hyrag.models import ChunkStrategy, Document, RetrievalMode, Section
from hyrag.retrieval import CrossEncoderReranker, HybridRetriever
from tests.conftest import make_document
from tests.test_rerank import _StubCrossEncoder

_IDENTIFIER = re.compile(r"\S*[._\-]\S*")


class IdentifierBlindEmbedder(FakeEmbedder):
    """A dense model that understands words but not identifiers.

    Real embedding models split ``ingest.max_batch_size`` into sub-word pieces that carry
    little meaning, so its exact match barely moves the vector. This stub makes that
    failure mode deterministic by dropping identifier tokens before embedding.
    """

    name = "identifier-blind"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return super().embed_documents([_IDENTIFIER.sub(" ", t) for t in texts])


def doc(source: str, text: str) -> Document:
    d = make_document([Section(heading="Notes", text=text)], source)
    d.doc_id = source
    return d


KEY_DOC = "config.md"
DOCS = [
    doc(
        KEY_DOC, "ingest.max_batch_size: default 500, max 5000. Larger requests fail with NW-4012."
    ),
    doc("ingest-a.md", "Send events to the ingest service in batches; the ingest service is fast."),
    doc("ingest-b.md", "The ingest service default timeout is 30 seconds for every ingest batch."),
    doc("ingest-c.md", "Scale the ingest service horizontally when the ingest queue grows."),
    doc("ingest-d.md", "Ingest service dashboards show the default ingest throughput per region."),
    doc("pto.md", "Employees accrue 25 days of paid time off per year."),
]
QUERY = "What is the default ingest.max_batch_size for the ingest service?"


@pytest.fixture
def settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={"chunk_size": 300, "chunk_overlap": 20, "dense_k": 3, "sparse_k": 3}
    )


@pytest.fixture
def retriever(settings: Settings) -> HybridRetriever:
    embedder = IdentifierBlindEmbedder()
    index = IndexManager(settings, ChunkStrategy.RECURSIVE, embedder)
    index.sync(DOCS)
    return HybridRetriever(index, embedder, reranker=None, settings=settings)


def sources(results: list) -> list[str]:  # type: ignore[type-arg]
    return [r.chunk.source for r in results]


def test_bm25_finds_the_exact_config_key_that_dense_misses(retriever: HybridRetriever) -> None:
    dense = retriever.retrieve(QUERY, RetrievalMode.DENSE, k=3)
    sparse = retriever.retrieve(QUERY, RetrievalMode.SPARSE, k=3)
    hybrid = retriever.retrieve(QUERY, RetrievalMode.HYBRID, k=3)

    assert KEY_DOC not in sources(dense)  # the dense model drowns in "ingest service" chunks
    assert sources(sparse)[0] == KEY_DOC  # the exact identifier is a rare, high-idf token
    assert KEY_DOC in sources(hybrid)  # fusion keeps the best of both

    key = next(r for r in hybrid if r.chunk.source == KEY_DOC)
    assert key.sparse_rank == 1 and key.rrf_score is not None
    assert key.rerank_score is None and key.score == key.rrf_score


def test_scores_are_reported_per_stage(retriever: HybridRetriever) -> None:
    dense = retriever.retrieve(QUERY, "dense", k=2)
    assert [r.rank for r in dense] == [1, 2]
    assert all(r.dense_rank and r.sparse_rank is None and r.rrf_score is None for r in dense)
    assert dense[0].score == dense[0].dense_score >= dense[1].score

    sparse = retriever.retrieve(QUERY, "sparse", k=2)
    assert all(r.sparse_score and r.dense_score is None for r in sparse)

    hybrid = retriever.retrieve(QUERY, "hybrid", k=10)
    rrf = [r.rrf_score for r in hybrid]
    assert rrf == sorted(rrf, reverse=True)
    assert len(hybrid) <= retriever.settings.fusion_top_n
    assert "pto.md" not in sources(hybrid)


def test_reranker_reorders_candidates(retriever: HybridRetriever) -> None:
    retriever.reranker = CrossEncoderReranker("stub", model=_StubCrossEncoder())
    query = "paid time off per year"
    reranked = retriever.retrieve(query, "hybrid", k=3)
    assert reranked[0].chunk.source == "pto.md"
    assert all(r.rerank_score is not None and r.score == r.rerank_score for r in reranked)
    scores = [r.rerank_score for r in reranked]
    assert scores == sorted(scores, reverse=True)

    plain = retriever.retrieve(query, "hybrid", k=3, rerank=False)
    assert all(r.rerank_score is None for r in plain)


def test_default_k_and_empty_index(settings: Settings) -> None:
    embedder = FakeEmbedder()
    empty = HybridRetriever(IndexManager(settings, "fixed", embedder), embedder, settings=settings)
    assert empty.retrieve("anything") == []

    settings = settings.model_copy(update={"top_k": 2})
    index = IndexManager(settings, "fixed", embedder)
    index.sync(DOCS)
    assert len(HybridRetriever(index, embedder, settings=settings).retrieve("ingest service")) == 2


def test_refuses_an_index_built_with_another_embedder(settings: Settings) -> None:
    index = IndexManager(settings, ChunkStrategy.RECURSIVE, FakeEmbedder())
    index.sync(DOCS)
    with pytest.raises(IndexMismatchError):
        HybridRetriever(index, IdentifierBlindEmbedder(), settings=settings)


def test_from_settings_uses_configured_components(settings: Settings) -> None:
    IndexManager(settings, settings.chunk_strategy, FakeEmbedder(settings.embedding_dim)).sync(DOCS)
    retriever = HybridRetriever.from_settings(settings)
    assert retriever.reranker is None  # HYRAG_RERANKER=none in tests
    assert retriever.index.strategy is settings.chunk_strategy
    assert retriever.retrieve("NW-4012", "sparse", k=1)[0].chunk.source == KEY_DOC


def test_settings_reject_all_zero_fusion_weights() -> None:
    with pytest.raises(ValueError, match="cannot both be zero"):
        Settings(dense_weight=0, sparse_weight=0, _env_file=None)  # type: ignore[call-arg]
