from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hyrag.embeddings import FakeEmbedder
from hyrag.index.vector_store import VectorStore
from tests.conftest import make_chunk

TEXTS = [
    "ingest.max_batch_size limits events per batch",
    "certificate rotation runbook for the edge proxy",
    "paid time off carryover rules",
]


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", "test")


def test_upsert_query_get_roundtrip(store: VectorStore, embedder: FakeEmbedder) -> None:
    chunks = [
        make_chunk(TEXTS[0], source="doc0.md", index=0),
        make_chunk(TEXTS[1], source="doc1.md", index=1),
        make_chunk(TEXTS[2], source="hr/pto.pdf", index=2, heading=None, page=3),
    ]
    store.upsert(chunks, embedder.embed_documents(TEXTS))
    assert store.count() == 3
    assert store.ids() == {c.chunk_id for c in chunks}

    hits = store.query(embedder.embed_query("certificate rotation for the edge proxy"), k=2)
    assert hits[0].chunk_id == chunks[1].chunk_id
    assert hits[0].source == "doc1.md"
    assert -1.0 <= hits[1].similarity <= hits[0].similarity <= 1.0

    got = store.get([chunks[2].chunk_id, chunks[0].chunk_id])
    assert got[chunks[2].chunk_id] == chunks[2]  # page kept, None heading survives
    assert got[chunks[0].chunk_id] == chunks[0]
    assert store.get([]) == {}
    assert sorted(c.chunk_id for c in store.all_chunks()) == sorted(c.chunk_id for c in chunks)


def test_exact_text_has_similarity_one(store: VectorStore, embedder: FakeEmbedder) -> None:
    chunk = make_chunk(TEXTS[0])
    store.upsert([chunk], embedder.embed_documents([TEXTS[0]]))
    [hit] = store.query(embedder.embed_query(TEXTS[0]), k=5)
    assert hit.similarity == pytest.approx(1.0, abs=1e-5)


def test_delete_reset_and_empty_queries(store: VectorStore, embedder: FakeEmbedder) -> None:
    assert store.query_many(embedder.embed_documents(["x"]), k=3) == [[]]
    assert store.query_many(np.zeros((0, 256), dtype=np.float32), k=3) == []

    chunks = [make_chunk(t, index=i) for i, t in enumerate(TEXTS)]
    store.upsert(chunks, embedder.embed_documents(TEXTS))
    store.delete([chunks[0].chunk_id])
    assert store.count() == 2
    store.reset()
    assert store.count() == 0


def test_collections_are_isolated(tmp_path: Path, embedder: FakeEmbedder) -> None:
    a = VectorStore(tmp_path / "chroma", "col_a")
    b = VectorStore(tmp_path / "chroma", "col_b")
    a.upsert([make_chunk(TEXTS[0])], embedder.embed_documents(TEXTS[:1]))
    assert (a.count(), b.count()) == (1, 0)


def test_stores_on_one_directory_share_a_client_and_see_each_others_writes(
    tmp_path: Path,
) -> None:
    """A second client on the same directory used to miss unflushed HNSW segments."""
    writer = VectorStore(tmp_path / "chroma", "col_a")
    chunks = [make_chunk(f"text number {i}", index=i) for i in range(5)]
    embedder = FakeEmbedder(dim=32)
    writer.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    reader = VectorStore(tmp_path / "chroma", "col_a")
    assert reader._client is writer._client
    hits = reader.query(embedder.embed_query("text number 3"), 2)
    assert hits[0].chunk_id == chunks[3].chunk_id


@pytest.mark.parametrize("reembed", [False, True])
def test_a_failing_hnsw_reader_falls_back_to_exact_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reembed: bool
) -> None:
    """Chroma 1.x sometimes reports a fresh collection's HNSW segment as missing.

    With an embedder the fallback re-embeds the stored texts, because reading the stored
    vectors fails the same way; without one it reads the stored vectors.
    """
    import chromadb.errors

    embedder = FakeEmbedder(dim=32)
    store = VectorStore(
        tmp_path / "chroma", "col_fallback", embed=embedder.embed_documents if reembed else None
    )
    chunks = [make_chunk(f"text number {i}", index=i) for i in range(8)]
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    queries = embedder.embed_documents(["text number 3", "text number 6"])
    expected = store.query_many(queries, 4)

    calls: list[int] = []

    def broken(**kwargs: object) -> None:
        calls.append(1)
        raise chromadb.errors.InternalError(
            "Error creating hnsw segment reader: Nothing found on disk"
        )

    monkeypatch.setattr(store._collection, "query", broken)
    monkeypatch.setattr("hyrag.index.vector_store.time.sleep", lambda s: None)
    if reembed:
        real_get = store._collection.get

        def get_without_vectors(**kwargs: list[str]) -> object:
            assert "embeddings" not in kwargs.get("include", []), "vectors are unreadable"
            return real_get(**kwargs)

        monkeypatch.setattr(store._collection, "get", get_without_vectors)
    fallback = store.query_many(queries, 4)
    assert len(calls) == 2  # retried, then answered exactly
    for got, want in zip(fallback, expected, strict=True):
        assert got[0].chunk_id == want[0].chunk_id
        sims = [h.similarity for h in got]
        assert sims == sorted(sims, reverse=True)
        # HNSW is approximate; the exact search can only match or beat it at each rank.
        assert all(g >= w - 1e-4 for g, w in zip(sims, [h.similarity for h in want], strict=True))
        assert all(h.source == "test/doc.md" for h in got)


def test_other_chroma_errors_are_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chromadb.errors

    store = VectorStore(tmp_path / "chroma", "col_errors")
    chunk = make_chunk("only one")
    store.upsert([chunk], FakeEmbedder(dim=32).embed_documents([chunk.text]))

    def broken(**kwargs: object) -> None:
        raise chromadb.errors.InternalError("disk is full")

    monkeypatch.setattr(store._collection, "query", broken)
    with pytest.raises(chromadb.errors.InternalError, match="disk is full"):
        store.query(FakeEmbedder(dim=32).embed_query("only one"), 1)
