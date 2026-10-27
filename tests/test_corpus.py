"""Integration checks over the bundled Northwind sample corpus."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from hyrag.chunking import get_chunker
from hyrag.config import Settings
from hyrag.ingestion import load_corpus
from hyrag.models import ChunkStrategy, Document


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    return load_corpus(Path(__file__).resolve().parents[1] / "data" / "corpus")


def test_corpus_covers_all_formats(docs: list[Document]) -> None:
    assert len(docs) >= 25
    formats = Counter(d.format for d in docs)
    assert set(formats) == {"markdown", "text", "html", "pdf"}
    assert all(d.sections and d.title for d in docs)


def test_pdf_documents_have_page_numbers(docs: list[Document]) -> None:
    pdfs = [d for d in docs if d.format == "pdf"]
    assert len(pdfs) >= 4
    assert all(s.page is not None for d in pdfs for s in d.sections)


def test_exact_identifiers_survive_normalization(docs: list[Document]) -> None:
    text = "\n".join(d.text for d in docs)
    for term in ("ingest.max_batch_size", "NW-4012", "X-NW-Signature", "nwctl rollback"):
        assert term in text


@pytest.mark.parametrize("strategy", list(ChunkStrategy))
def test_every_document_produces_chunks(docs: list[Document], strategy: ChunkStrategy) -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    chunks = get_chunker(strategy, settings).chunk(docs)
    assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.text.strip() for c in chunks)
