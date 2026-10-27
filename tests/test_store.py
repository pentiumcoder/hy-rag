from __future__ import annotations

from pathlib import Path

from hyrag.ingestion import DocumentStore


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("# A\nalpha text\n")
    (root / "sub" / "b.txt").write_text("B\n=\nbeta text\n")
    return root


def test_ingest_copies_raw_and_writes_processed(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    store = DocumentStore(tmp_path / "data")
    report = store.ingest_directory(root)

    assert sorted(report.added) == ["a.md", "sub/b.txt"]
    assert (tmp_path / "data" / "raw" / "sub" / "b.txt").read_text() == "B\n=\nbeta text\n"
    docs = store.load_documents()
    assert [d.source for d in docs] == ["a.md", "sub/b.txt"]
    assert docs[0].sections[0].text == "alpha text"


def test_reingest_skips_unchanged_and_detects_updates(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    store = DocumentStore(tmp_path / "data")
    store.ingest_directory(root)

    (root / "a.md").write_text("# A\nalpha text, revised\n")
    report = store.ingest_directory(root)
    assert report.updated == ["a.md"]
    assert report.unchanged == ["sub/b.txt"]
    assert report.changed == ["a.md"]
    assert "revised" in store.load_documents()[0].text


def test_reprocess_reads_from_raw_without_source_files(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    store = DocumentStore(tmp_path / "data")
    store.ingest_directory(root)
    for f in root.rglob("*.*"):
        f.unlink()  # originals gone: only data/raw remains

    docs = store.reprocess()
    assert [d.source for d in docs] == ["a.md", "sub/b.txt"]


def test_empty_files_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "blank.md").write_text("\n\n")
    report = DocumentStore(tmp_path / "data").ingest_directory(root)
    assert report.skipped == ["blank.md"]


def test_remove(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    store = DocumentStore(tmp_path / "data")
    store.ingest_directory(root)
    assert store.remove("a.md") is True
    assert store.remove("a.md") is False
    assert [d.source for d in store.load_documents()] == ["sub/b.txt"]
    assert not (tmp_path / "data" / "raw" / "a.md").exists()
