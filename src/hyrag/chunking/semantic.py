"""Semantic chunking: start a new chunk where the topic shifts.

Sentences are embedded and every gap between two sentences is scored by the cosine
similarity of the ``window`` sentences before it and the ``window`` sentences after it
(averaging a few sentences smooths out noise while keeping the score centred on the gap).
Gaps whose similarity falls below the ``breakpoint_percentile`` of the document's own
distribution become candidate boundaries. ``min_chars`` / ``max_chars`` keep chunks within
usable sizes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from hyrag.chunking.base import Chunker, with_heading
from hyrag.embeddings.base import Embedder, l2_normalize
from hyrag.models import Chunk, ChunkStrategy, Document, Section

# Split after sentence punctuation when the next sentence starts like one; dotted
# identifiers such as "ingest.max_batch_size" contain no space and are never split.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9`-])")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.split("\n"):
        sentences.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return sentences


@dataclass(frozen=True)
class _Unit:
    text: str
    section: Section


class SemanticChunker(Chunker):
    strategy = ChunkStrategy.SEMANTIC

    def __init__(
        self,
        embedder: Embedder,
        breakpoint_percentile: float = 20.0,
        min_chars: int = 200,
        max_chars: int = 1200,
        window: int = 2,
    ) -> None:
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.window = window

    def breakpoints(self, sentences: list[str]) -> set[int]:
        """Indexes ``i`` such that a topic boundary falls *before* sentence ``i``."""
        if len(sentences) < 3:
            return set()
        emb = self.embedder.embed_documents(sentences)
        gaps = range(1, len(sentences))
        before = l2_normalize(np.stack([emb[max(0, i - self.window) : i].mean(0) for i in gaps]))
        after = l2_normalize(np.stack([emb[i : i + self.window].mean(0) for i in gaps]))
        sims = np.sum(before * after, axis=1)
        threshold = float(np.percentile(sims, self.breakpoint_percentile))
        # A real boundary also depresses the scores of its neighbouring gaps (their windows
        # straddle it), so keep only gaps that are the local minimum within the window.
        breaks: set[int] = set()
        for k, sim in enumerate(sims):
            lo, hi = max(0, k - self.window + 1), k + self.window
            if sim <= threshold and sim <= sims[lo:hi].min():
                breaks.add(gaps[k])
        return breaks

    def chunk_document(self, doc: Document) -> list[Chunk]:
        units = [_Unit(s, sec) for sec in doc.sections for s in split_sentences(sec.text)]
        if not units:
            return []
        breaks = self.breakpoints([u.text for u in units])

        groups: list[list[_Unit]] = []
        current: list[_Unit] = []
        length = 0
        for i, unit in enumerate(units):
            topic_shift = i in breaks and length >= self.min_chars
            too_long = length + len(unit.text) + 1 > self.max_chars
            if current and (topic_shift or too_long):
                groups.append(current)
                current, length = [], 0
            current.append(unit)
            length += len(unit.text) + 1
        if current:
            # Fold a tiny tail into the previous chunk rather than emitting a fragment.
            tail_len = sum(len(u.text) + 1 for u in current)
            prev_len = sum(len(u.text) + 1 for u in groups[-1]) if groups else 0
            if groups and tail_len < self.min_chars and prev_len + tail_len <= self.max_chars:
                groups[-1].extend(current)
            else:
                groups.append(current)

        chunks: list[Chunk] = []
        for group in groups:
            first = group[0].section
            body = "\n".join(u.text for u in group)
            chunks.append(
                Chunk.create(
                    doc,
                    self.strategy,
                    len(chunks),
                    with_heading(first.heading, body),
                    section_heading=first.heading,
                    page=first.page,
                )
            )
        return chunks
