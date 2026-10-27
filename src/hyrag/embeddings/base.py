"""Embedding provider interface. All vectors returned are L2-normalized float32."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        """Embed passages; returns shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a search query; some models use a different prefix for queries."""
        ...


def l2_normalize(matrix: NDArray[np.floating]) -> Matrix:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: Matrix = (matrix / norms).astype(np.float32)
    return normalized


def cosine_similarity_matrix(a: Matrix, b: Matrix) -> Matrix:
    """Pairwise cosine similarity for already-normalized rows."""
    return (a @ b.T).astype(np.float32)
