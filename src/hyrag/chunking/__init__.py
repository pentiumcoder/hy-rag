"""Three switchable chunking strategies behind one :class:`Chunker` interface."""

from __future__ import annotations

from hyrag.chunking.base import Chunker
from hyrag.chunking.fixed import FixedSizeChunker
from hyrag.chunking.recursive import RecursiveChunker
from hyrag.chunking.semantic import SemanticChunker
from hyrag.config import Settings, get_settings
from hyrag.embeddings import Embedder, get_embedder
from hyrag.models import ChunkStrategy

__all__ = [
    "Chunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "get_chunker",
]


def get_chunker(
    strategy: ChunkStrategy | str,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> Chunker:
    """Build the chunker for ``strategy`` using sizes from ``settings``."""
    settings = settings or get_settings()
    strategy = ChunkStrategy(strategy)
    if strategy is ChunkStrategy.FIXED:
        return FixedSizeChunker(settings.chunk_size, settings.chunk_overlap)
    if strategy is ChunkStrategy.RECURSIVE:
        return RecursiveChunker(settings.chunk_size, settings.chunk_overlap)
    return SemanticChunker(
        embedder or get_embedder(settings.embedding_provider, settings.embedding_dim),
        breakpoint_percentile=settings.semantic_breakpoint_percentile,
        min_chars=settings.semantic_min_chars,
        max_chars=settings.semantic_max_chars,
    )
