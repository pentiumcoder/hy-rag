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
