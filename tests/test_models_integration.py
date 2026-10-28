"""End-to-end checks with the real local models (bge-small + MiniLM cross-encoder).

Skipped by default so CI never downloads anything. Run locally with:

    HYRAG_TEST_MODELS=1 pytest -m models
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from hyrag.config import Settings
from hyrag.embeddings import embedder_from_settings
from hyrag.index import IndexManager
from hyrag.ingestion import load_corpus
from hyrag.models import ChunkStrategy
from hyrag.retrieval import HybridRetriever, get_reranker
from tests.conftest import CORPUS_DIR, REPO_ROOT

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(
        os.environ.get("HYRAG_TEST_MODELS") != "1", reason="set HYRAG_TEST_MODELS=1 to run"
    ),
]


@pytest.fixture(scope="module")
def real_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        corpus_dir=CORPUS_DIR,
        data_dir=tmp_path_factory.mktemp("data"),
        embedding_provider="fastembed",
        reranker="cross-encoder",
        model_cache_dir=REPO_ROOT / ".cache" / "models",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture(scope="module")
def retriever(real_settings: Settings) -> Iterator[HybridRetriever]:
    embedder = embedder_from_settings(real_settings)
    index = IndexManager(real_settings, ChunkStrategy.RECURSIVE, embedder)
    index.sync(load_corpus(CORPUS_DIR))
    reranker = get_reranker("cross-encoder", cache_dir=real_settings.model_cache_dir)
    yield HybridRetriever(index, embedder, reranker, real_settings)


def test_bge_embeddings_capture_paraphrase(real_settings: Settings) -> None:
    embedder = embedder_from_settings(real_settings)
    assert embedder.dim == 384
    vecs = embedder.embed_documents(
        [
            "How many vacation days can I carry into next year?",
            "Up to 5 unused PTO days roll over to the following year.",
            "Rotate the TLS certificate on the edge proxy.",
        ]
    )
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-4)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2] + 0.1


def test_cross_encoder_prefers_the_answering_passage() -> None:
    reranker = get_reranker("cross-encoder", cache_dir=REPO_ROOT / ".cache" / "models")
    assert reranker is not None
    scores = reranker.score(
        "what happens if a batch is too large",
        [
            "Requests whose batch exceeds ingest.max_batch_size are rejected with NW-4012.",
            "The release train departs every Tuesday at 09:00 UTC.",
        ],
    )
    assert scores[0] > scores[1]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What error do I get when a batch exceeds ingest.max_batch_size?", "error-codes.md"),
        ("How many PTO days carry over to next year?", "hr/"),
        ("How do I rotate the edge certificates?", "certificate-rotation.md"),
        ("Which signature header do webhooks carry?", "webhooks.html"),
    ],
)
def test_hybrid_finds_the_right_document(
    retriever: HybridRetriever, query: str, expected: str
) -> None:
    results = retriever.retrieve(query, "hybrid", k=3)
    assert any(expected in r.chunk.source for r in results), [r.chunk.source for r in results]
    assert all(r.rerank_score is not None for r in results)


def test_reingest_hits_the_embedding_cache(real_settings: Settings, tmp_path: Path) -> None:
    settings = real_settings.model_copy(update={"data_dir": tmp_path})
    settings.cache_dir.mkdir(parents=True)
    (settings.cache_dir / "embeddings.sqlite").write_bytes(
        (real_settings.cache_dir / "embeddings.sqlite").read_bytes()
    )
    embedder = embedder_from_settings(settings)
    IndexManager(settings, ChunkStrategy.RECURSIVE, embedder).sync(load_corpus(CORPUS_DIR))
    assert embedder.misses == 0 and embedder.hits > 50  # type: ignore[attr-defined]
