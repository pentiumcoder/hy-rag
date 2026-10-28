"""Persistent BM25 (Okapi) index over chunk texts.

Only the tokenized corpus is persisted, as JSON; the scoring tables are rebuilt on load,
which takes milliseconds at this scale and avoids unpickling anything from disk.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from pathlib import Path

from rank_bm25 import BM25Okapi

from hyrag.index.tokenize import tokenize


class LuceneBM25(BM25Okapi):  # type: ignore[misc]
    """BM25Okapi with Lucene's always-positive idf: ``ln(1 + (N - n + 0.5) / (n + 0.5))``.

    rank_bm25's classic Okapi idf is zero for a term in exactly half the chunks and negative
    above that (then floored to a small epsilon). On a small corpus that silently erases
    useful terms: with four chunks, "batch" appearing in two of them scores 0.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


class BM25Store:
    VERSION = 1

    def __init__(self, path: Path | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.path = path
        self.k1 = k1
        self.b = b
        self._docs: dict[str, list[str]] = {}
        self._bm25: LuceneBM25 | None = None
        self._order: list[str] = []

    @classmethod
    def load(cls, path: Path) -> BM25Store:
        store = cls(path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            store.k1, store.b = data["k1"], data["b"]
            store._docs = {cid: tokens for cid, tokens in data["docs"]}
        return store

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.VERSION,
            "k1": self.k1,
            "b": self.b,
            "docs": sorted(self._docs.items()),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic, so a crash never leaves a half-written index

    def add(self, items: Iterable[tuple[str, str]]) -> None:
        """Insert or replace ``(chunk_id, text)`` pairs."""
        for chunk_id, text in items:
            self._docs[chunk_id] = tokenize(text)
        self._bm25 = None

    def remove(self, chunk_ids: Iterable[str]) -> None:
        for chunk_id in chunk_ids:
            self._docs.pop(chunk_id, None)
        self._bm25 = None

    def clear(self) -> None:
        self._docs.clear()
        self._bm25 = None

    def ids(self) -> set[str]:
        return set(self._docs)

    def __len__(self) -> int:
        return len(self._docs)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top ``k`` chunk ids by BM25 score. Chunks sharing no query term are never returned."""
        terms = tokenize(query)
        if not terms or not self._docs:
            return []
        if self._bm25 is None:
            self._order = sorted(self._docs)
            self._bm25 = LuceneBM25([self._docs[c] for c in self._order], k1=self.k1, b=self.b)
        scores = self._bm25.get_scores(terms)
        query_terms = set(terms)
        ranked = sorted(
            (
                (cid, float(score))
                for cid, score in zip(self._order, scores, strict=True)
                if query_terms.intersection(self._docs[cid])
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:k]
