"""Baseline: fixed-size character windows with overlap, blind to document structure."""

from __future__ import annotations

import bisect

from hyrag.chunking.base import Chunker, heading_leaf
from hyrag.models import Chunk, ChunkStrategy, Document, Section


def _last_space(text: str, lo: int, hi: int) -> int:
    for i in range(hi - 1, lo - 1, -1):
        if text[i].isspace():
            return i
    return -1


def _next_space(text: str, lo: int, hi: int) -> int:
    for i in range(lo, hi):
        if text[i].isspace():
            return i
    return -1


def sliding_windows(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) windows of at most ``size`` chars that avoid cutting words in half.

    Cuts are moved back to the last whitespace in the final 20% of the window, and each new
    window starts ``overlap`` chars before the previous end (moved forward to a word start).
    """
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    windows: list[tuple[int, int]] = []
    n, start = len(text), 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = _last_space(text, start + int(size * 0.8), end)
            if cut > start:
                end = cut
        windows.append((start, end))
        if end >= n:
            break
        nxt = end - overlap
        space = _next_space(text, nxt, end)
        if space != -1:
            nxt = space + 1
        start = max(nxt, start + 1)
    return windows


class FixedSizeChunker(Chunker):
    strategy = ChunkStrategy.FIXED

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Document) -> list[Chunk]:
        blocks: list[str] = []
        offsets: list[int] = []
        sections: list[Section] = []
        offset = 0
        for section in doc.sections:
            leaf = heading_leaf(section)
            block = f"{leaf}\n{section.text}" if leaf else section.text
            offsets.append(offset)
            sections.append(section)
            blocks.append(block)
            offset += len(block) + 2
        text = "\n\n".join(blocks)

        chunks: list[Chunk] = []
        for start, end in sliding_windows(text, self.chunk_size, self.chunk_overlap):
            piece = text[start:end].strip()
            if not piece:
                continue
            # Attribute the chunk to the section where its first character sits.
            first_char = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            section = sections[max(bisect.bisect_right(offsets, first_char) - 1, 0)]
            chunks.append(
                Chunk.create(
                    doc,
                    self.strategy,
                    len(chunks),
                    piece,
                    section_heading=section.heading,
                    page=section.page,
                )
            )
        return chunks
