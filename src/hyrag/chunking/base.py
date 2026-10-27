"""Shared chunker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from hyrag.models import Chunk, ChunkStrategy, Document, Section


class Chunker(ABC):
    strategy: ChunkStrategy

    @abstractmethod
    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split one document; chunk indexes start at 0 per document."""

    def chunk(self, docs: Sequence[Document]) -> list[Chunk]:
        return [chunk for doc in docs for chunk in self.chunk_document(doc)]


def heading_leaf(section: Section) -> str | None:
    """Innermost heading only: 'Config Reference > Ingest' -> 'Ingest'."""
    return section.heading.split(" > ")[-1] if section.heading else None


def with_heading(heading: str | None, text: str) -> str:
    """Prefix a chunk with its heading path so it stays meaningful out of context."""
    return f"{heading}\n{text}" if heading else text
