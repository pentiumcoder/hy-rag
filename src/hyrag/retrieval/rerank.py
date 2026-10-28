"""Second-stage reranking.

First-stage retrievers embed query and passage separately; a cross-encoder reads them
together, which is far more precise but too slow to run over the whole corpus. So it only
reorders the ~20 fused candidates.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from hyrag.embeddings.base import quiet_model_downloads

DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One relevance score per text; higher is more relevant."""
        ...


class _CrossEncoderModel(Protocol):
    def rerank(
        self, query: str, documents: Iterable[str], batch_size: int = ...
    ) -> Iterable[float]: ...


class CrossEncoderReranker:
    """fastembed ``TextCrossEncoder``; scores are raw logits. Downloads the model on first use."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        cache_dir: Path | None = None,
        model: _CrossEncoderModel | None = None,
    ) -> None:
        self.name = f"cross-encoder:{model_name}"
        if model is None:
            quiet_model_downloads()
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            model = TextCrossEncoder(
                model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
            )
        self._model = model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.rerank(query, list(texts), batch_size=32)]


def get_reranker(
    kind: str, model_name: str = DEFAULT_RERANKER, cache_dir: Path | None = None
) -> Reranker | None:
    if kind == "none":
        return None
    if kind == "cross-encoder":
        return CrossEncoderReranker(model_name, cache_dir=cache_dir)
    raise ValueError(f"Unknown reranker: {kind!r}")
