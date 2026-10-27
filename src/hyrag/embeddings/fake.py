"""Deterministic, dependency-free embedder for tests and CI.

It hashes word unigrams and bigrams into a fixed number of signed buckets. Texts that share
vocabulary land close together, which is enough to exercise similarity-driven code
(semantic chunking, dedup, dense retrieval) without downloading a model.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Sequence

import numpy as np

from hyrag.embeddings.base import Matrix, Vector, l2_normalize

_TOKEN = re.compile(r"[a-z0-9_]+(?:[.\-][a-z0-9_]+)*")
_STOPWORDS = frozenset(
    {
        *("a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for"),
        *("from", "has", "have", "how", "if", "in", "is", "it", "its", "not", "of", "on"),
        *("or", "that", "the", "this", "to", "was", "were", "what", "when", "which"),
        *("who", "will", "with"),
    }
)


def _bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
    index = int.from_bytes(digest[:4], "little") % dim
    sign = 1.0 if digest[4] & 1 else -1.0
    return index, sign


class FakeEmbedder:
    name = "fake"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]
        features = tokens + [f"{a} {b}" for a, b in itertools.pairwise(tokens)]
        for feature in features:
            index, sign = _bucket(feature, self.dim)
            vec[index] += sign
        return vec

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._embed_one(t) for t in texts]))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector
