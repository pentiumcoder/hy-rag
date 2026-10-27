"""Structure-aware chunking: sections are the unit.

Small neighbouring sections are packed together up to ``chunk_size``; a section that is too
large on its own is split recursively (paragraphs, then lines, then sentences, then words)
with LangChain's ``RecursiveCharacterTextSplitter``. Every chunk carries its heading path as
a prefix, so a chunk like "Default: 500" still says which config key it describes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from hyrag.chunking.base import Chunker, with_heading
from hyrag.models import Chunk, ChunkStrategy, Document, Section


@dataclass
class _Pack:
    blocks: list[str] = field(default_factory=list)
    first: Section | None = None
    length: int = 0


class RecursiveChunker(Chunker):
    strategy = ChunkStrategy.RECURSIVE

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator="end",
            strip_whitespace=True,
        )

    def chunk_document(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        pack = _Pack()

        def emit(text: str, section: Section) -> None:
            chunks.append(
                Chunk.create(
                    doc,
                    self.strategy,
                    len(chunks),
                    text,
                    section_heading=section.heading,
                    page=section.page,
                )
            )

        def flush() -> None:
            nonlocal pack
            if pack.blocks and pack.first is not None:
                emit("\n\n".join(pack.blocks), pack.first)
            pack = _Pack()

        for section in doc.sections:
            block = with_heading(section.heading, section.text)
            if len(block) > self.chunk_size:
                flush()
                for piece in self._splitter.split_text(section.text):
                    emit(with_heading(section.heading, piece), section)
                continue
            if pack.blocks and pack.length + len(block) + 2 > self.chunk_size:
                flush()
            if pack.first is None:
                pack.first = section
            pack.blocks.append(block)
            pack.length += len(block) + 2
        flush()
        return chunks
