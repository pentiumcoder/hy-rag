from __future__ import annotations

from pathlib import Path

import pytest

from hyrag.embeddings import FakeEmbedder
from hyrag.index.dedup import find_duplicates
from hyrag.index.vector_store import VectorStore
from tests.conftest import make_chunk

FREEZE = (
    "Deploy freeze: no production deploys after 15:00 UTC on Fridays or during the last "
    "two business days of a quarter without approval from the release manager."
)


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", "dedup")


def test_skips_copy_of_an_indexed_chunk(store: VectorStore, embedder: FakeEmbedder) -> None:
    original = make_chunk(FREEZE, source="engineering/handbook.md")
    store.upsert([original], embedder.embed_documents([FREEZE]))

    copy = make_chunk(FREEZE, source="engineering/release-process.html")
    other = make_chunk("Rotate certificates every 90 days.", source="runbooks/certs.md")
    vecs = embedder.embed_documents([copy.text, other.text])
    kept, dups = find_duplicates([copy, other], vecs, store, 0.95)

    assert kept == [1]
    [dup] = dups
    assert dup.chunk_id == copy.chunk_id
    assert dup.duplicate_of == original.chunk_id
    assert dup.duplicate_of_source == "engineering/handbook.md"
    assert dup.similarity == pytest.approx(1.0, abs=1e-4)


def test_catches_duplicates_within_one_batch(store: VectorStore, embedder: FakeEmbedder) -> None:
    a = make_chunk(FREEZE, source="a.md")
    b = make_chunk(FREEZE + " ", source="b.md")  # whitespace differs, tokens identical
    kept, dups = find_duplicates([a, b], embedder.embed_documents([a.text, b.text]), store, 0.95)
    assert kept == [0]
    assert dups[0].duplicate_of == a.chunk_id and dups[0].duplicate_of_source == "a.md"


def test_threshold_is_strict_and_tunable(store: VectorStore, embedder: FakeEmbedder) -> None:
    a = make_chunk(FREEZE, source="a.md")
    b = make_chunk(FREEZE.replace("Fridays", "Thursdays"), source="b.md")  # near-duplicate
    vecs = embedder.embed_documents([a.text, b.text])
    similarity = float(vecs[0] @ vecs[1])
    assert 0.8 < similarity < 1.0

    assert find_duplicates([a, b], vecs, store, threshold=similarity - 0.01)[0] == [0]
    assert find_duplicates([a, b], vecs, store, threshold=similarity)[0] == [0, 1]  # strictly >


def test_reindexing_the_same_chunk_is_not_a_duplicate(
    store: VectorStore, embedder: FakeEmbedder
) -> None:
    chunk = make_chunk(FREEZE)
    vecs = embedder.embed_documents([FREEZE])
    store.upsert([chunk], vecs)
    assert find_duplicates([chunk], vecs, store, 0.95) == ([0], [])
