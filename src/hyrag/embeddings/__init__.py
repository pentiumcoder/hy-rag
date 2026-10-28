"""Embedding providers behind a single :class:`Embedder` protocol."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hyrag.embeddings.base import Embedder, cosine_similarity_matrix, l2_normalize
from hyrag.embeddings.cache import CachedEmbedder
from hyrag.embeddings.fake import FakeEmbedder

if TYPE_CHECKING:
    from hyrag.config import Settings

__all__ = [
    "CachedEmbedder",
    "Embedder",
    "FakeEmbedder",
    "cosine_similarity_matrix",
    "embedder_from_settings",
    "get_embedder",
    "l2_normalize",
]


def get_embedder(
    provider: str = "fake",
    dim: int = 256,
    model: str | None = None,
    cache_dir: Path | None = None,
) -> Embedder:
    """Build an uncached embedder. Heavy providers are imported lazily."""
    if provider == "fake":
        return FakeEmbedder(dim=dim)
    if provider == "fastembed":
        from hyrag.embeddings.fastembed import DEFAULT_MODEL, FastEmbedEmbedder

        return FastEmbedEmbedder(model or DEFAULT_MODEL, cache_dir=cache_dir)
    if provider == "openai":
        from hyrag.embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder(model or "text-embedding-3-small")
    raise ValueError(f"Unknown embedding provider: {provider!r}")


def embedder_from_settings(settings: Settings) -> Embedder:
    """The embedder the pipeline uses: configured provider, wrapped in the disk cache."""
    inner = get_embedder(
        settings.embedding_provider,
        dim=settings.embedding_dim,
        model=settings.embedding_model,
        cache_dir=settings.model_cache_dir,
    )
    if settings.embedding_cache and settings.embedding_provider != "fake":
        return CachedEmbedder(inner, settings.cache_dir / "embeddings.sqlite")
    return inner
