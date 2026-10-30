"""ChromaDB-backed dense index: one collection per chunking strategy, cosine space."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import chromadb
import chromadb.errors
import numpy as np
from chromadb.config import Settings as ChromaSettings

from hyrag.embeddings.base import Matrix, Vector
from hyrag.models import Chunk, ChunkStrategy

logger = logging.getLogger(__name__)

_BATCH = 1000  # Chroma rejects very large single upserts
_QUERY_ATTEMPTS = 2  # HNSW retries before falling back to exact search


class VectorHit(NamedTuple):
    chunk_id: str
    similarity: float  # cosine similarity, 1.0 = identical direction
    source: str


def _to_metadata(chunk: Chunk) -> dict[str, str | int]:
    meta: dict[str, str | int] = {
        "doc_id": chunk.doc_id,
        "source": chunk.source,
        "title": chunk.title,
        "strategy": chunk.strategy.value,
        "chunk_index": chunk.chunk_index,
    }
    # Chroma metadata values cannot be None, so optional fields are simply omitted.
    if chunk.section_heading is not None:
        meta["section_heading"] = chunk.section_heading
    if chunk.page is not None:
        meta["page"] = chunk.page
    return meta


def _from_record(chunk_id: str, text: str, meta: Mapping[str, Any]) -> Chunk:
    page = meta.get("page")
    heading = meta.get("section_heading")
    return Chunk(
        chunk_id=chunk_id,
        doc_id=str(meta["doc_id"]),
        source=str(meta["source"]),
        title=str(meta["title"]),
        text=text,
        strategy=ChunkStrategy(str(meta["strategy"])),
        chunk_index=int(meta["chunk_index"]),
        section_heading=str(heading) if heading is not None else None,
        page=int(page) if page is not None else None,
    )


_CLIENTS: dict[str, Any] = {}
_CLIENTS_LOCK = threading.Lock()


def _client_for(path: Path) -> Any:
    """One Chroma client per directory per process.

    A second ``PersistentClient`` on the same directory loads the HNSW segment from disk,
    which the first client may not have flushed yet; queries then fail with "Nothing found
    on disk". Sharing the client keeps every reader on the writer's in-memory state.
    """
    key = str(path.resolve())
    with _CLIENTS_LOCK:
        if key not in _CLIENTS:
            _CLIENTS[key] = chromadb.PersistentClient(
                path=key, settings=ChromaSettings(anonymized_telemetry=False)
            )
        return _CLIENTS[key]


Embed = Callable[[Sequence[str]], Matrix]


class VectorStore:
    """``embed`` re-embeds stored texts for the exact-search fallback (see :meth:`_query`)."""

    def __init__(self, path: Path, collection: str, embed: Embed | None = None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._client = _client_for(path)
        self.collection_name = collection
        self._embed = embed
        self._collection = self._open()

    def _open(self) -> Any:
        # embedding_function=None: we always pass vectors, so Chroma must never load its own model.
        return self._client.get_or_create_collection(
            self.collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def count(self) -> int:
        return int(self._collection.count())

    def ids(self) -> set[str]:
        return set(self._collection.get(include=[])["ids"])

    def upsert(self, chunks: Sequence[Chunk], embeddings: Matrix) -> None:
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[start : start + _BATCH],
                documents=[c.text for c in batch],
                metadatas=[_to_metadata(c) for c in batch],
            )

    def delete(self, chunk_ids: Iterable[str]) -> None:
        ids = list(chunk_ids)
        for start in range(0, len(ids), _BATCH):
            self._collection.delete(ids=ids[start : start + _BATCH])

    def get(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        result = self._collection.get(ids=list(chunk_ids), include=["documents", "metadatas"])
        return {
            cid: _from_record(cid, text or "", meta or {})
            for cid, text, meta in zip(
                result["ids"], result["documents"] or [], result["metadatas"] or [], strict=True
            )
        }

    def all_chunks(self) -> list[Chunk]:
        return list(self.get(sorted(self.ids())).values())

    def query(self, embedding: Vector, k: int) -> list[VectorHit]:
        return self.query_many(np.asarray([embedding], dtype=np.float32), k)[0]

    def query_many(self, embeddings: Matrix, k: int) -> list[list[VectorHit]]:
        """Nearest ``k`` indexed chunks for each row of ``embeddings``."""
        n = min(k, self.count())
        if n == 0 or len(embeddings) == 0:
            return [[] for _ in range(len(embeddings))]
        result = self._query(embeddings, n)
        hits: list[list[VectorHit]] = []
        for ids, dists, metas in zip(
            result["ids"], result["distances"] or [], result["metadatas"] or [], strict=True
        ):
            hits.append(
                [
                    VectorHit(cid, 1.0 - float(dist), str((meta or {}).get("source", "")))
                    for cid, dist, meta in zip(ids, dists, metas, strict=True)
                ]
            )
        return hits

    def _query(self, embeddings: Matrix, n: int) -> Any:
        """HNSW query, with an exact search as the fallback for a Chroma 1.x failure.

        Rarely (about once in 50 freshly built indexes in one process, in plain Chroma too),
        a collection's vector segment fails with "Nothing found on disk" and keeps failing;
        reading its stored vectors fails the same way. Documents and metadata stay readable.
        So after two quick retries the stored texts are re-embedded (the embedding cache
        makes that nearly free) and searched exactly. At this corpus size that is as fast as
        HNSW and returns the true neighbours. Any other error is raised at once.
        """
        for attempt in range(_QUERY_ATTEMPTS):
            try:
                return self._collection.query(
                    query_embeddings=embeddings, n_results=n, include=["distances", "metadatas"]
                )
            except chromadb.errors.InternalError as exc:
                if "Nothing found on disk" not in str(exc):
                    raise
                time.sleep(0.05 * 2**attempt)
        logger.warning("Chroma HNSW reader failed for %s; using exact search", self.collection_name)
        return self._exact_query(embeddings, n)

    def _exact_query(self, embeddings: Matrix, n: int) -> dict[str, Any]:
        """Brute-force cosine search over every stored record, shaped like a Chroma result."""
        if self._embed is not None:
            data = self._collection.get(include=["documents", "metadatas"])
            stored = np.asarray(self._embed([d or "" for d in data["documents"] or []]))
        else:
            data = self._collection.get(include=["embeddings", "metadatas"])
            stored = np.asarray(data["embeddings"])
        ids: list[str] = list(data["ids"])
        stored = stored.astype(np.float32).reshape(len(ids), -1)
        stored /= np.maximum(np.linalg.norm(stored, axis=1, keepdims=True), 1e-12)
        queries = np.asarray(embeddings, dtype=np.float32)
        queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
        sims = queries @ stored.T
        metas = list(data["metadatas"] or [{}] * len(ids))
        result: dict[str, Any] = {"ids": [], "distances": [], "metadatas": []}
        for row in sims:
            top = np.argsort(-row, kind="stable")[:n]
            result["ids"].append([ids[i] for i in top])
            result["distances"].append([1.0 - float(row[i]) for i in top])
            result["metadatas"].append([metas[i] for i in top])
        return result

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._open()
