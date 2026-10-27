"""Embedding providers behind a single :class:`Embedder` protocol."""

from __future__ import annotations

from hyrag.embeddings.base import Embedder, cosine_similarity_matrix, l2_normalize
from hyrag.embeddings.fake import FakeEmbedder

__all__ = ["Embedder", "FakeEmbedder", "cosine_similarity_matrix", "get_embedder", "l2_normalize"]


def get_embedder(provider: str = "fake", dim: int = 256) -> Embedder:
    if provider == "fake":
        return FakeEmbedder(dim=dim)
    raise ValueError(f"Unknown embedding provider: {provider!r}")
