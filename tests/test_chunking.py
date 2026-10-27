from __future__ import annotations

import itertools

import pytest

from hyrag.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    get_chunker,
)
from hyrag.chunking.fixed import sliding_windows
from hyrag.chunking.semantic import split_sentences
from hyrag.config import Settings
from hyrag.embeddings import FakeEmbedder
from hyrag.models import ChunkStrategy, Section
from tests.conftest import make_document

LOREM = "Northwind ingests product events through the gateway and stores them in ClickHouse. " * 30


# ---------- fixed ----------


def test_sliding_windows_cover_text_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    windows = sliding_windows(text, size=200, overlap=40)
    assert windows[0][0] == 0 and windows[-1][1] == len(text)
    for (s1, e1), (s2, _) in itertools.pairwise(windows):
        assert e1 - s1 <= 200
        assert s2 < e1  # overlap
        assert s2 > s1  # always progresses
        assert text[s2 - 1] == " "  # starts on a word boundary


def test_sliding_windows_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        sliding_windows("abc", size=10, overlap=10)


def test_sliding_windows_handles_text_without_spaces() -> None:
    windows = sliding_windows("x" * 250, size=100, overlap=10)
    assert windows[-1][1] == 250


def test_fixed_chunker_sizes_and_metadata() -> None:
    doc = make_document(
        [
            Section(heading="Doc > Intro", text=LOREM[:300]),
            Section(heading="Doc > Details", text=LOREM, page=2),
        ]
    )
    chunks = FixedSizeChunker(chunk_size=400, chunk_overlap=50).chunk_document(doc)
    assert len(chunks) > 3
    assert all(c.char_count <= 400 for c in chunks)
    assert all(c.strategy is ChunkStrategy.FIXED for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].section_heading == "Doc > Intro"
    assert chunks[-1].section_heading == "Doc > Details" and chunks[-1].page == 2


# ---------- recursive ----------


def test_recursive_packs_small_sections_and_prefixes_headings() -> None:
    doc = make_document(
        [
            Section(heading="Ref > key.a", text="Default: 1"),
            Section(heading="Ref > key.b", text="Default: 2"),
        ]
    )
    chunks = RecursiveChunker(chunk_size=500, chunk_overlap=50).chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Ref > key.a\nDefault: 1\n\nRef > key.b\nDefault: 2"
    assert chunks[0].section_heading == "Ref > key.a"


def test_recursive_splits_large_sections_and_keeps_heading_on_every_piece() -> None:
    doc = make_document(
        [
            Section(heading="Small", text="tiny"),
            Section(heading="Big > Part", text=LOREM),
            Section(heading="After", text="tail"),
        ]
    )
    chunks = RecursiveChunker(chunk_size=300, chunk_overlap=30).chunk_document(doc)
    big = [c for c in chunks if c.section_heading == "Big > Part"]
    assert len(big) >= 5
    assert all(c.text.startswith("Big > Part\n") for c in big)
    assert all(c.char_count <= 300 + len("Big > Part\n") for c in big)
    # Sections before and after an oversized one are not merged into it.
    assert chunks[0].text == "Small\ntiny" and chunks[-1].text == "After\ntail"


# ---------- semantic ----------


def test_split_sentences_keeps_dotted_identifiers() -> None:
    text = "Set ingest.max_batch_size to 500. Then retry.\n- bullet item"
    assert split_sentences(text) == [
        "Set ingest.max_batch_size to 500.",
        "Then retry.",
        "- bullet item",
    ]


def test_semantic_chunker_splits_on_topic_change(embedder: FakeEmbedder) -> None:
    ingest = [
        f"The ingest pipeline batch size for events is limited to {n} events per batch."
        for n in range(8)
    ]
    pto = [
        f"Employees receive paid time off days and carryover expires in month {n}."
        for n in range(8)
    ]
    doc = make_document([Section(heading="Mixed", text=" ".join(ingest + pto))])
    chunker = SemanticChunker(embedder, breakpoint_percentile=10, min_chars=100, max_chars=2000)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 2
    assert "paid time off" not in chunks[0].text
    assert "ingest" not in chunks[1].text
    assert all(c.strategy is ChunkStrategy.SEMANTIC for c in chunks)


def test_semantic_chunker_respects_max_chars(embedder: FakeEmbedder) -> None:
    doc = make_document([Section(heading="H", text=LOREM)])
    chunks = SemanticChunker(embedder, min_chars=100, max_chars=400).chunk_document(doc)
    assert len(chunks) > 3
    assert all(c.char_count <= 400 + len("H\n") for c in chunks)


def test_semantic_chunker_folds_tiny_tail(embedder: FakeEmbedder) -> None:
    doc = make_document([Section(heading=None, text=LOREM[:600] + " End.")])
    chunks = SemanticChunker(embedder, min_chars=100, max_chars=700).chunk_document(doc)
    assert chunks[-1].text.endswith("End.")
    assert all(c.char_count >= 100 for c in chunks)


def test_semantic_chunker_short_and_empty_docs(embedder: FakeEmbedder) -> None:
    chunker = SemanticChunker(embedder)
    assert chunker.chunk_document(make_document([])) == []
    one = chunker.chunk_document(make_document([Section(text="Just one sentence.")]))
    assert [c.text for c in one] == ["Just one sentence."]


# ---------- shared behaviour ----------


@pytest.mark.parametrize("strategy", list(ChunkStrategy))
def test_chunk_ids_are_deterministic_and_unique(strategy: ChunkStrategy) -> None:
    settings = Settings(chunk_size=300, chunk_overlap=30, _env_file=None)  # type: ignore[call-arg]
    doc = make_document([Section(heading="A", text=LOREM), Section(heading="B", text=LOREM)])
    first = get_chunker(strategy, settings).chunk([doc])
    second = get_chunker(strategy, settings).chunk([doc])
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    assert all(c.strategy is strategy for c in first)


def test_get_chunker_builds_each_strategy(settings: Settings) -> None:
    assert isinstance(get_chunker("fixed", settings), FixedSizeChunker)
    assert isinstance(get_chunker(ChunkStrategy.RECURSIVE, settings), RecursiveChunker)
    semantic = get_chunker("semantic", settings, embedder=FakeEmbedder(dim=32))
    assert isinstance(semantic, SemanticChunker) and semantic.embedder.dim == 32
    with pytest.raises(ValueError):
        get_chunker("paragraph", settings)


def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        Settings(chunk_size=200, chunk_overlap=200, _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Settings(semantic_min_chars=900, semantic_max_chars=800, _env_file=None)  # type: ignore[call-arg]
