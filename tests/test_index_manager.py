from __future__ import annotations

import json

import pytest

from hyrag.config import Settings
from hyrag.embeddings import FakeEmbedder
from hyrag.index import IndexManager, IndexMismatchError
from hyrag.models import ChunkStrategy, Document, Section
from tests.conftest import make_document

FREEZE = (
    "Deploy freeze: no production deploys after 15:00 UTC on Fridays or during the last "
    "two business days of a quarter without approval from the release manager."
)


def doc(source: str, *texts: str) -> Document:
    d = make_document([Section(heading=f"H{i}", text=t) for i, t in enumerate(texts)], source)
    d.doc_id = source
    return d


DOCS = [
    doc("handbook.md", FREEZE, "On-call engineers acknowledge SEV1 pages within 5 minutes."),
    doc("release.html", FREEZE, "Releases are cut every Tuesday from the main branch."),
    doc("config.md", "ingest.max_batch_size defaults to 500 events and caps at 5000."),
]


@pytest.fixture
def settings(settings: Settings) -> Settings:
    # chunk_size small enough that every section becomes its own chunk
    return settings.model_copy(update={"chunk_size": 200, "chunk_overlap": 20})


def manager(settings: Settings, embedder: FakeEmbedder | None = None) -> IndexManager:
    return IndexManager(settings, ChunkStrategy.RECURSIVE, embedder or FakeEmbedder())


def test_first_sync_indexes_and_skips_duplicates(settings: Settings) -> None:
    m = manager(settings)
    report = m.sync(DOCS)
    assert report.added == 4 and report.removed == 0 and report.unchanged == 0
    [dup] = report.new_duplicates
    assert (dup.source, dup.duplicate_of_source) == ("release.html", "handbook.md")
    assert report.total_chunks == 4 and report.total_duplicates == 1
    assert m.vectors.count() == len(m.bm25) == len(m) == 4
    assert m.consistent()

    saved = json.loads(m.dedup_report_path.read_text(encoding="utf-8"))
    assert saved["threshold"] == 0.95 and len(saved["duplicates"]) == 1


def test_resync_is_a_noop_and_survives_reopen(settings: Settings) -> None:
    manager(settings).sync(DOCS)
    m = manager(settings)
    assert not m.repaired
    report = m.sync(DOCS)
    assert (report.added, report.removed, report.unchanged) == (0, 0, 4)
    assert report.total_duplicates == 1 and not report.new_duplicates


def test_changed_and_removed_documents(settings: Settings) -> None:
    m = manager(settings)
    m.sync(DOCS)
    edited = doc("config.md", "ingest.max_batch_size defaults to 1000 events and caps at 5000.")
    report = m.sync([DOCS[0], DOCS[1], edited])
    assert (report.added, report.removed) == (1, 1)
    assert any("1000" in c.text for c in m.vectors.all_chunks())

    report = m.sync([DOCS[0], DOCS[1]])
    assert report.removed == 1
    assert not m.bm25.search("max_batch_size")


def test_duplicate_is_revived_when_its_original_disappears(settings: Settings) -> None:
    m = manager(settings)
    m.sync(DOCS)
    report = m.sync(DOCS[1:])  # the handbook (holder of the original) is gone
    assert report.removed == 2
    assert report.added == 1  # release.html's copy of the freeze paragraph is now indexed
    assert report.total_duplicates == 0
    assert any(
        c.source == "release.html" and "Deploy freeze" in c.text for c in m.vectors.all_chunks()
    )


def test_bm25_is_rebuilt_from_chroma_after_drift(settings: Settings) -> None:
    manager(settings).sync(DOCS)
    bm25_file = settings.index_dir / "recursive" / "bm25.json"
    bm25_file.unlink()  # simulate a crash between the Chroma write and the BM25 write

    m = manager(settings)
    assert m.repaired and m.consistent()
    assert m.bm25.search("NW-4012 max_batch_size")
    assert manager(settings).repaired is False  # repair was persisted
    assert m.sync(DOCS).repaired is True  # the next sync report says the index was repaired
    assert m.sync(DOCS).repaired is False


def test_manifest_follows_chroma_when_a_chunk_is_lost(settings: Settings) -> None:
    m = manager(settings)
    m.sync(DOCS)
    lost = sorted(m.vectors.ids())[0]
    m.vectors.delete([lost])

    reopened = manager(settings)
    assert reopened.repaired and lost not in reopened.manifest.chunks
    report = reopened.sync(DOCS)  # the lost chunk is simply re-added
    assert report.added == 1 and reopened.consistent()


def test_embedder_change_forces_rebuild_and_retriever_check(settings: Settings) -> None:
    manager(settings).sync(DOCS)
    other = FakeEmbedder(dim=128)
    other.name = "fake-128"
    m = manager(settings, other)
    with pytest.raises(IndexMismatchError, match="hyrag ingest"):
        m.check_embedder()
    report = m.sync(DOCS)
    assert report.rebuilt and report.added == 4
    m.check_embedder()


def test_strategies_get_separate_indexes(settings: Settings) -> None:
    rec = manager(settings)
    rec.sync(DOCS)
    fixed = IndexManager(settings, ChunkStrategy.FIXED, FakeEmbedder())
    assert len(fixed) == 0 and fixed.vectors.count() == 0
    fixed.sync(DOCS)
    assert {c.strategy for c in fixed.vectors.all_chunks()} == {ChunkStrategy.FIXED}
    assert {c.strategy for c in rec.vectors.all_chunks()} == {ChunkStrategy.RECURSIVE}
