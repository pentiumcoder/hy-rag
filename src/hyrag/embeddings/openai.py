"""Optional hosted embeddings via the OpenAI API (``pip install hyrag[openai]``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from hyrag.embeddings.base import Matrix, Vector, l2_normalize

DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class OpenAIEmbedder:
    """Reads ``OPENAI_API_KEY`` from the environment unless a client is passed in."""

    def __init__(
        self, model_name: str = "text-embedding-3-small", client: Any = None, batch_size: int = 256
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self.model_name = model_name
        self.name = f"openai:{model_name}"
        self.dim = DIMENSIONS.get(model_name, 1536)
        self.batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._client.embeddings.create(model=self.model_name, input=batch)
            rows.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return l2_normalize(np.asarray(rows, dtype=np.float32))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector
