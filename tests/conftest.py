from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from hyrag.config import Settings, get_settings
from hyrag.embeddings import FakeEmbedder
from hyrag.models import Chunk, ChunkStrategy, Document, Section, stable_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"


@pytest.fixture(autouse=True)
def _offline_models(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests never download or call models: fake embedder, no reranker and the stub LLM."""
    monkeypatch.setenv("HYRAG_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("HYRAG_RERANKER", "none")
    monkeypatch.setenv("HYRAG_LLM_PROVIDER", "stub")
    monkeypatch.delenv("HYRAG_JUDGE_PROVIDER", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def corpus_dir() -> Path:
    return CORPUS_DIR


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(corpus_dir=CORPUS_DIR, data_dir=tmp_path / "data", _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=256)


def make_document(sections: list[Section], source: str = "test/doc.md") -> Document:
    return Document(
        doc_id="doc123",
        source=source,
        title="Test Document",
        format="markdown",
        sections=sections,
        doc_hash="0" * 64,
    )


def write_pdf(path: Path, title: str, lines: list[str]) -> Path:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for line in lines:
        if line == "<page>":
            pdf.add_page()
            continue
        pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))
    return path


def make_chunk(
    text: str,
    source: str = "test/doc.md",
    index: int = 0,
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    heading: str | None = "Section",
    page: int | None = None,
) -> Chunk:
    doc = make_document([Section(text=text)], source=source)
    doc.doc_id = stable_hash(source, length=12)
    return Chunk.create(doc, strategy, index, text, section_heading=heading, page=page)
