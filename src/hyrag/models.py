"""Domain models shared by every layer of the pipeline."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

DocFormat = Literal["markdown", "text", "html", "pdf"]


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


def stable_hash(*parts: str, length: int = 16) -> str:
    """Deterministic short hash; Python's built-in hash() is salted per process."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


class Section(BaseModel):
    """A contiguous run of text under one heading (and, for PDFs, on one page)."""

    heading: str | None = None
    text: str
    page: int | None = None
    level: int = 0


class Document(BaseModel):
    """A source file normalized into plain-text sections plus metadata."""

    doc_id: str
    source: str  # POSIX path relative to the corpus root
    title: str
    format: DocFormat
    sections: list[Section]
    doc_hash: str  # hash of the raw bytes, used to skip unchanged files
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text)

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


class Chunk(BaseModel):
    """A retrievable unit of text. The id is stable for identical inputs."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    strategy: ChunkStrategy
    chunk_index: int
    section_heading: str | None = None
    page: int | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def char_count(self) -> int:
        return len(self.text)

    @classmethod
    def create(
        cls,
        doc: Document,
        strategy: ChunkStrategy,
        chunk_index: int,
        text: str,
        section_heading: str | None = None,
        page: int | None = None,
    ) -> Chunk:
        return cls(
            chunk_id=stable_hash(doc.source, strategy.value, str(chunk_index), text),
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            text=text,
            strategy=strategy,
            chunk_index=chunk_index,
            section_heading=section_heading,
            page=page,
        )


class ScoredChunk(BaseModel):
    """A retrieved chunk plus every score that contributed to its final rank.

    Ranks are 1-based; a ``None`` score means that stage did not see the chunk
    (e.g. ``sparse_*`` is ``None`` for a chunk only the dense retriever found).
    """

    chunk: Chunk
    rank: int
    score: float  # what the final order is sorted by: rerank score if reranked, else first stage
    dense_score: float | None = None  # cosine similarity
    dense_rank: int | None = None
    sparse_score: float | None = None  # BM25
    sparse_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None  # cross-encoder logit
