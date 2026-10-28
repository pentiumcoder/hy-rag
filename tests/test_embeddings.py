from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hyrag.config import Settings
from hyrag.embeddings import (
    CachedEmbedder,
    Embedder,
    FakeEmbedder,
    cosine_similarity_matrix,
    embedder_from_settings,
    get_embedder,
)
from hyrag.embeddings.fastembed import FastEmbedEmbedder
from hyrag.embeddings.openai import OpenAIEmbedder


def test_fake_embedder_is_normalized_and_deterministic() -> None:
    emb = FakeEmbedder(dim=64)
    a = emb.embed_documents(["rate limit exceeded", "rate limit exceeded"])
    assert a.shape == (2, 64)
    assert a.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(a[0], a[1])
    np.testing.assert_array_equal(FakeEmbedder(dim=64).embed_query("rate limit exceeded"), a[0])


def test_fake_embedder_similarity_tracks_vocabulary_overlap() -> None:
    emb = FakeEmbedder()
    vecs = emb.embed_documents(
        [
            "ingest batch size limit for events",
            "the maximum batch size for ingest events",
            "paid time off carryover policy",
        ]
    )
    sims = cosine_similarity_matrix(vecs, vecs)
    assert sims[0, 1] > 0.3
    assert sims[0, 1] > sims[0, 2] + 0.2


def test_fake_embedder_handles_empty_input() -> None:
    emb = FakeEmbedder(dim=8)
    assert emb.embed_documents([]).shape == (0, 8)
    assert np.linalg.norm(emb.embed_query("")) == 0.0  # no tokens -> zero vector, no NaN


def test_get_embedder() -> None:
    assert isinstance(get_embedder("fake"), Embedder)
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedder("nope")


# ---------- fastembed / OpenAI adapters (stub models, no downloads) ----------


class _StubTextModel:
    """Stands in for fastembed.TextEmbedding: records inputs, returns length-based vectors."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, documents: Iterable[str], batch_size: int = 64) -> Iterator[np.ndarray]:
        docs = list(documents)
        self.calls.append(docs)
        for d in docs:
            yield np.array([len(d), 1.0, 0.0], dtype=np.float32)


def test_fastembed_embedder_normalizes_and_prefixes_queries() -> None:
    model = _StubTextModel()
    emb = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", model=model, dim=3)
    docs = emb.embed_documents(["abc", "abcdef"])
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-5)
    emb.embed_query("rate limits")
    assert model.calls[0] == ["abc", "abcdef"]  # passages are embedded as-is
    assert model.calls[1][0].startswith("Represent this sentence")  # BGE query instruction
    assert emb.name == "fastembed:BAAI/bge-small-en-v1.5"
    assert emb.embed_documents([]).shape == (0, 3)


def test_openai_embedder_batches_and_restores_order() -> None:
    class _Embeddings:
        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def create(self, model: str, input: list[str]) -> SimpleNamespace:
            self.batches.append(input)
            data = [
                SimpleNamespace(index=i, embedding=[float(len(t)), 1.0])
                for i, t in enumerate(input)
            ]
            return SimpleNamespace(data=list(reversed(data)))  # API order is not guaranteed

    client = SimpleNamespace(embeddings=_Embeddings())
    emb = OpenAIEmbedder(client=client, batch_size=2)
    out = emb.embed_documents(["a", "bbb", "cc"])
    assert client.embeddings.batches == [["a", "bbb"], ["cc"]]
    expected = np.array([[1, 1], [3, 1], [2, 1]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(out, expected, rtol=1e-5)
    assert emb.embed_query("x").shape == (2,)
    assert emb.embed_documents([]).shape == (0, emb.dim)


# ---------- disk cache ----------


class _CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__(dim=16)
        self.embedded: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.embedded.extend(texts)
        return super().embed_documents(texts)


def test_cached_embedder_only_embeds_new_texts(tmp_path: Path) -> None:
    inner = _CountingEmbedder()
    cache = CachedEmbedder(inner, tmp_path / "emb.sqlite")
    first = cache.embed_documents(["alpha beta", "gamma", "alpha beta"])
    assert inner.embedded == ["alpha beta", "gamma"]  # in-call duplicates embedded once
    second = cache.embed_documents(["gamma", "delta"])
    assert inner.embedded == ["alpha beta", "gamma", "delta"]
    np.testing.assert_array_equal(first[1], second[0])
    assert (cache.hits, cache.misses) == (2, 3)
    cache.close()

    reopened = CachedEmbedder(_CountingEmbedder(), tmp_path / "emb.sqlite")
    np.testing.assert_array_equal(reopened.embed_documents(["alpha beta"])[0], first[0])
    assert reopened.hits == 1 and reopened.misses == 0
    reopened.close()


def test_cached_embedder_keeps_queries_separate(tmp_path: Path) -> None:
    inner = _CountingEmbedder()
    cache = CachedEmbedder(inner, tmp_path / "emb.sqlite")
    cache.embed_documents(["rate limit"])
    np.testing.assert_allclose(cache.embed_query("rate limit"), inner.embed_query("rate limit"))
    assert cache.misses == 2  # same text, different kind: not a cache hit
    assert cache.embed_documents([]).shape == (0, 16)
    cache.close()


def test_embedder_from_settings(settings: Settings, tmp_path: Path) -> None:
    assert isinstance(embedder_from_settings(settings), FakeEmbedder)  # fake is never cached
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "hyrag.embeddings.fastembed.FastEmbedEmbedder.__init__",
            lambda self, *a, **k: setattr(self, "name", "stub") or setattr(self, "dim", 4),
        )
        real = settings.model_copy(update={"embedding_provider": "fastembed"})
        wrapped = embedder_from_settings(real)
        assert isinstance(wrapped, CachedEmbedder) and wrapped.name == "stub"
        wrapped.close()
        uncached = settings.model_copy(
            update={"embedding_provider": "fastembed", "embedding_cache": False}
        )
        assert not isinstance(embedder_from_settings(uncached), CachedEmbedder)
