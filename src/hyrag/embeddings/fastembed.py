"""Local ONNX embeddings via fastembed. The default model is BAAI/bge-small-en-v1.5 (384-dim)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hyrag.embeddings.base import Matrix, Vector, l2_normalize, quiet_model_downloads

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models retrieve better when short queries carry this instruction; passages get none.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class _TextModel(Protocol):
    def embed(self, documents: Iterable[str], batch_size: int = ...) -> Iterable[Any]: ...


class FastEmbedEmbedder:
    """Wraps ``fastembed.TextEmbedding``. The model is downloaded on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
        batch_size: int = 64,
        model: _TextModel | None = None,
        dim: int | None = None,
    ) -> None:
        self.name = f"fastembed:{model_name}"
        self.batch_size = batch_size
        self.query_prefix = QUERY_PREFIXES.get(model_name, "")
        if model is None:
            quiet_model_downloads()
            from fastembed import TextEmbedding

            model = TextEmbedding(
                model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
            )
            dim = dim or TextEmbedding.get_embedding_size(model_name)
        self._model = model
        self.dim = dim or 384

    def _embed(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows = list(self._model.embed(list(texts), batch_size=self.batch_size))
        return l2_normalize(np.asarray(rows, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return self._embed(texts)

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self._embed([self.query_prefix + text])[0]
        return vector
